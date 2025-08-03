##### CARLOS CODE FILE ########
### This code was never used for my final experiments

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from modules.net import conv_bn, conv_bn1X1

class DirectAttentionArchitecture(nn.Module):
    def __init__(self, in_channels_list, out_channels, phase, cfg):
        super(DirectAttentionArchitecture,self).__init__()
        leaky = 0
        if (out_channels <= 64):
            leaky = 0.1

        d_model = out_channels
        self.d_model = out_channels
        self.phase = phase

        self.n_heads = cfg['attention_heads']  # rename for brevity
        assert d_model % self.n_heads == 0, \
            "d_model must be divisible by n_heads"
        self.head_dim = d_model // self.n_heads

        self.upperandlower = cfg['upperandlower']

        # in_channels_list = [128,256,512,2048]
        self.num_levels = len(in_channels_list)

        # Linear projections for Q,K,V
        self.q_projs = nn.ModuleList([nn.Linear(in_ch, d_model, bias=False) for in_ch in in_channels_list[:-1]])

        self.final_linear = nn.ModuleList([nn.Linear(d_model, d_model, bias=False) for _ in in_channels_list[1:]])

        self.k_projs_upper = nn.ModuleList([nn.Linear(in_ch, d_model, bias=False) for in_ch in in_channels_list[1:-1]])
        self.k_projs_upper.append(nn.Linear(d_model, d_model, bias = False))

        self.v_projs_upper = nn.ModuleList([nn.Linear(in_ch, d_model, bias=False) for in_ch in in_channels_list[1:-1]])
        self.v_projs_upper.append(nn.Linear(d_model, d_model, bias = False))

        if self.upperandlower:
            self.k_proj_c2 = nn.Linear(in_channels_list[0], d_model, bias=False)
            self.v_proj_c2 = nn.Linear(in_channels_list[0], d_model, bias=False)

        self.conv1x1_for_c5 = conv_bn1X1(in_channels_list[-1], out_channels, stride = 1, leaky = leaky)

        self.merge_list = nn.ModuleList()

        for _ in in_channels_list[:-1]:
            self.merge_list.append(conv_bn(out_channels, out_channels, leaky=leaky))

    def forward(self, x):
        x, orig_sizes = self.preprocessing(x)
        neck_computation = self.forward_detail(x)
        if self.phase == 'train':
            return neck_computation

        self.crop_back(neck_computation, orig_sizes)
        return neck_computation

    def preprocessing(self, x):
        # x = OrderedDict, 128,256,512,2048
        x = list(x.values())
        # x = List, 128,256,512,2048 = C2,...C5

        orig_sizes = [(f.size(2), f.size(3)) for f in x]

        if self.phase == 'train':
            return x, orig_sizes

        # Preprocessing: Bottom-up pad/crop so that each query is exactly 2× its next key (Especially necessary at Inference / Evaluation time!)
        div = 2  # 1 or 2
        top = x[-1]
        _, _, H, W = top.shape
        pad_h = (div - H % div) % div
        pad_w = (div - W % div) % div
        if pad_h or pad_w:
            top = F.pad(top, (0, pad_w, 0, pad_h), mode='replicate')
        x[-1] = top

        for i in range(self.num_levels - 2, -1, -1):
            q = x[i]  # e.g. C2, C3, C4, C5 in turn
            k = x[i + 1]  # the immediately finer map (C3, C4, C5)
            B, C, Hq, Wq = q.shape
            _, _, Hk, Wk = k.shape
            Ht, Wt = 2 * Hk, 2 * Wk
            # pad bottom/right if under‐sized, important at evaluation / inference, since we cannot assume, that our input is 640 x 640 and that we hence have nice, square feature map sizes
            pad_h = max(0, Ht - Hq)
            pad_w = max(0, Wt - Wq)
            if pad_h or pad_w:
                q = F.pad(q, (0, pad_w, 0, pad_h), mode='replicate')

            # crop bottom/right if over‐sized, just to be safe at evaluation / inference, since we cannot assume, that our input is 640 x 640 and that we hence have nice, square feature map sizes
            crop_h = max(0, q.size(2) - Ht)
            crop_w = max(0, q.size(3) - Wt)
            if crop_h or crop_w:
                q = q[:, :, :Ht, :Wt]
            x[i] = q
        return x, orig_sizes

    def reshape_attentioned(self, output_attentioned, batch, H_querylayer, W_querylayer, d_model, device, idx_query):
        # Rearrange output of attention to have correct position
        output_attentioned = output_attentioned.reshape(batch, int(H_querylayer * W_querylayer), d_model)  # (B, N, d_model) in group-order

        # Create inverse mapping
        scatter_idx = torch.empty_like(idx_query.view(-1))
        scatter_idx[idx_query.view(-1)] = torch.arange(len(scatter_idx), device=device)
        #torch.set_printoptions(profile="full")
        #print(scatter_idx)
        output_attentioned = output_attentioned[:, scatter_idx, :]

        # reshape to spatial map
        output_attentioned = output_attentioned.transpose(1, 2).view(batch, d_model, H_querylayer, W_querylayer)  # (B, d_model, Hf, Wf)
        return output_attentioned

    def crop_back(self, outputs, orig_sizes):
        # cropping back: important at evaluation / inference, since we cannot assume, that our input is 640 x 640 and that we hence have nice, square feature map sizes
        # We need to have feature maps exactly of the sizes as the ResNet gave us, in order to have correct anchor matching, after the network produces the predictions
        for i, out in enumerate(outputs):
            H_orig, W_orig = orig_sizes[i]
            outputs[i] = out[:, :, :H_orig, :W_orig]

    def group(self, fmap, i, u_c, v_c, W_current, batch, number_of_groups, member_per_group, d_model):
        # compute index groups for 2x2 mapping
        base = i * u_c * W_current + i * v_c  # (Nc,)

        offs = torch.tensor(
            [ii + jj * W_current for jj in range(i) for ii in range(i)],
            device=fmap.device
        )
        idx = base.unsqueeze(1) + offs.unsqueeze(0)  # (Nc, 4)
        # prepare Queries via index_select()
        fmap = fmap.index_select(1, idx.view(-1))
        fmap = fmap.view(batch, number_of_groups, member_per_group, d_model)
        return fmap, idx

    def forward_detail(self, x):
        outputs = []
        idx_dict = dict()
        c5 = self.conv1x1_for_c5(x[-1])
        x[-1] = c5

        Q_list, K_list, V_list  = [], [], []
        device = x[-1].device
        batch, _ , _, _ = x[-1].shape

        #/2 because the 1x1 case is C6, but C6 is not provided as input, only C5
        H_most_upper_keylayer, W_most_upper_keylayer = x[-1].shape[2:]
        Number_in_most_upper_keylayer = int(H_most_upper_keylayer * W_most_upper_keylayer)

        u_c = torch.arange(H_most_upper_keylayer//2, device=device).unsqueeze(1).expand(
            H_most_upper_keylayer//2, W_most_upper_keylayer//2).reshape(-1)
        v_c = torch.arange(W_most_upper_keylayer//2, device=device).unsqueeze(0).expand(
            H_most_upper_keylayer//2, W_most_upper_keylayer//2).reshape(-1)
        number_of_groups = Number_in_most_upper_keylayer // 4

        query_member_per_group = dict()
        query_width = dict()
        query_height = dict()

        # From 0 ... 2
        for i in range(self.num_levels - 1):
            H_current, W_current = x[i].shape[2:]
            query_width[i] = W_current
            query_height[i] = H_current
            q = x[i].flatten(2).transpose(1, 2)  # (B, Nf, Cin_q)
            q = self.q_projs[i](q)
            _, N_q, _ = q.shape
            member_per_group = N_q // number_of_groups
            query_member_per_group[i] = member_per_group

            q, idx = self.group(q, (2 ** (self.num_levels - i)), u_c, v_c, W_current, batch, number_of_groups, member_per_group, self.d_model)
            idx_dict[i] = idx
            Q_list.append(q.view(*q.shape[:-1], self.n_heads, self.head_dim))  # (B, Nf, d)

            W_kv = x[i+1].shape[-1]
            kv_flat_upper = x[i + 1].flatten(2).transpose(1, 2)
            _, N_kv, _ = kv_flat_upper.shape
            member_per_group = N_kv // number_of_groups

            k_flat_upper = self.k_projs_upper[i](kv_flat_upper)
            v_flat_upper = self.v_projs_upper[i](kv_flat_upper)

            k_flat_upper, _ = self.group(k_flat_upper, (2 ** (self.num_levels - i - 1)), u_c, v_c, W_kv, batch, number_of_groups, member_per_group, self.d_model)
            v_flat_upper, _ = self.group(v_flat_upper, (2 ** (self.num_levels - i - 1)), u_c, v_c, W_kv, batch, number_of_groups, member_per_group, self.d_model)

            K_list.append(k_flat_upper.view(*k_flat_upper.shape[:-1], self.n_heads, self.head_dim))
            V_list.append(v_flat_upper.view(*v_flat_upper.shape[:-1], self.n_heads, self.head_dim))

        k_flat_c2 = None
        v_flat_c2 = None
        if self.upperandlower:
            W_kv = x[0].shape[-1]
            kv_flat_c2 = x[0].flatten(2).transpose(1, 2)
            _, N_kv, _ = kv_flat_c2.shape
            member_per_group = N_kv // number_of_groups
            k_flat_c2 = self.k_proj_c2(kv_flat_c2)
            v_flat_c2 = self.v_proj_c2(kv_flat_c2)

            k_flat_c2,_ = self.group(k_flat_c2, (2 ** (self.num_levels)), u_c, v_c, W_kv, batch, number_of_groups, member_per_group, self.d_model)
            v_flat_c2,_ = self.group(v_flat_c2, (2 ** (self.num_levels)), u_c, v_c, W_kv, batch, number_of_groups, member_per_group, self.d_model)

            k_flat_c2 = k_flat_c2.view(*k_flat_c2.shape[:-1], self.n_heads, self.head_dim)
            v_flat_c2 = v_flat_c2.view(*v_flat_c2.shape[:-1], self.n_heads, self.head_dim)


        # From 0 ... 2
        for i in range(self.num_levels - 1):
            Q_grouped = Q_list[i]

            k_all = list()
            v_all = list()

            for j in range(self.num_levels - 2, i-1, -1):
                k_all.append(K_list[j])
                v_all.append(V_list[j])

            if self.upperandlower: #Implement this if the only upper pathway showed that the direct attention idea is somewhat promising.
                for j in range(i-2, -1, -1):
                    k_all.append(K_list[j])
                    v_all.append(V_list[j])
                if i > 0:
                    k_all.append(k_flat_c2)
                    v_all.append(v_flat_c2)

            K_grouped = torch.cat(k_all, dim=2)
            V_grouped = torch.cat(v_all, dim=2)

            scores = torch.einsum('bnqhd,bnkhd->bnqkh', Q_grouped, K_grouped)
            scores = scores / math.sqrt(self.head_dim)
            scores = F.softmax(scores, dim=3)
            output_attentioned = torch.einsum('bnqkh,bnkhd->bnqhd', scores, V_grouped)

            # ── [MHA] merge heads back to d_model ─────────────────────
            member_per_group = query_member_per_group[i]
            output_attentioned = output_attentioned.reshape(batch, number_of_groups, member_per_group, self.n_heads * self.head_dim)  # (B,Nk,4,d_model)  # [MHA]

            output_attentioned = self.reshape_attentioned(output_attentioned, batch, query_height[i],
                                                          query_width[i], self.d_model, device, idx_dict[i]) # B, d_model, H_query, W_query

            output_attentioned = output_attentioned.flatten(2).transpose(1, 2) # B, (H_query * W_query =) N, d_model
            output_attentioned = self.final_linear[i](output_attentioned)
            output_attentioned = output_attentioned.transpose(1, 2).contiguous().view(batch, self.d_model, query_height[i], query_width[i])

            output_attentioned = self.merge_list[i](output_attentioned)
            outputs.append(output_attentioned)
        outputs.append(c5)
        return outputs