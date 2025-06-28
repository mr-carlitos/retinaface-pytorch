##### CARLOS CODE FILE ########

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

        # in_channels_list = [128,256,512,2048]
        self.num_levels = len(in_channels_list)

        # Linear projections for Q,K,V
        self.q_projs = nn.ModuleList([nn.Linear(in_ch, d_model, bias=False) for in_ch in in_channels_list[:-1]])

        self.final_linear = nn.ModuleList([nn.Linear(d_model, d_model, bias=False) for _ in in_channels_list[1:]])

        self.k_projs_upper = nn.ModuleList([nn.Linear(in_ch, d_model, bias=False) for in_ch in in_channels_list[1:]])
        self.v_projs_upper = nn.ModuleList([nn.Linear(in_ch, d_model, bias=False) for in_ch in in_channels_list[1:]])

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

    def reshape_attentioned(self, output_attentioned, batch, Number_in_keylayer, H_querylayer, W_querylayer, d_model, device, idx_query):
        # Rearrange output of attention to have correct position
        output_attentioned = output_attentioned.reshape(batch, Number_in_keylayer * 4, d_model)  # (B, N, d_model) in group-order

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
        fmap = fmap.index_select(1, idx.view(-1)).view(batch, number_of_groups, member_per_group, d_model)
        return fmap, idx

    def forward_detail(self, x):
        outputs = []
        idx_dict = dict()
        c5 = self.conv1x1_for_c5(x[-1])

        Q_list, K_list_upper, V_list_upper  = [], [], []
        device = x[-1].device
        batch, _ , _, _ = x[-1].shape

        #/2 because the 1x1 case is C6, but C6 is not provided as input, only C5
        H_most_upper_keylayer, W_most_upper_keylayer = x[-1].shape[2:]
        _, Number_in_most_upper_keylayer, _, _ = x[-1].shape
        u_c = torch.arange(H_most_upper_keylayer/2, device=device).unsqueeze(1).expand(
            H_most_upper_keylayer/2, W_most_upper_keylayer/2).reshape(-1)
        v_c = torch.arange(W_most_upper_keylayer/2, device=device).unsqueeze(0).expand(
            H_most_upper_keylayer/2, W_most_upper_keylayer/2).reshape(-1)
        number_of_groups = Number_in_most_upper_keylayer // 4

        # From 2 to 0 -> 2, 1, 0
        for i in range(self.num_levels - 2, -1, -1):
            W_current = x[i].shape[2:]
            q = x[i].flatten(2).transpose(1, 2)  # (B, Nf, Cin_q)
            q = self.q_projs[i](q)
            _, N_q, _ = q.shape
            member_per_group = N_q // number_of_groups

            q, idx = self.group(q, (2 ** (i + 1)), u_c, v_c, W_current, batch, number_of_groups, member_per_group, self.d_model)
            idx_dict[i] = idx
            Q_list.append(q.view(*q.shape[:-1], self.n_heads, self.head_dim))  # (B, Nf, d)

            W_current = x[i+1].shape
            kv_flat_upper = x[i + 1].flatten(2).transpose(1, 2)
            _, N_kv, _ = kv_flat_upper.shape
            member_per_group = N_kv // number_of_groups
            kv_flat_upper, _ = self.group(kv_flat_upper, (2 ** (i + 2)), u_c, v_c, W_current, batch, number_of_groups, member_per_group, self.d_model)
            K_list_upper.append(self.k_projs_upper[i](kv_flat_upper).view(*kv_flat_upper.shape[:-1], self.n_heads, self.head_dim))
            V_list_upper.append(self.v_projs_upper[i](kv_flat_upper).view(*kv_flat_upper.shape[:-1], self.n_heads, self.head_dim))

        # 3 Queries, 3 K/Vs, not grouped yet
        #TODO: Continue here

        # From 2 to 0 -> 2, 1, 0
        for i in range(self.num_levels - 2, -1, -1):
            Q_grouped = Q_list[i]

            k_all_upper = list()
            v_all_upper = list()

            for j in range(self.num_levels - 2, i, -1):
                k_all_upper.append(K_list_upper[j])
                v_all_upper.append(V_list_upper[j])

            K_grouped = torch.cat(k_all_upper, dim=2)
            V_grouped = torch.cat(v_all_upper, dim=2)

            scores = torch.einsum('bnqhd,bnkhd->bnqkh', Q_grouped, K_grouped)  # (B, Nk, 4, K, h)
            scores = scores / math.sqrt(self.head_dim)
            attn = F.softmax(scores, dim=3)
            output_attentioned = torch.einsum('bnqkh,bnkhd->bnqhd', attn, V_grouped)  # (B,Nk,4,h,d_h)  # [MHA]

            # ── [MHA] merge heads back to d_model ─────────────────────
            output_attentioned = output_attentioned.reshape(batch, int(Number_in_upper_keylayer/self.quadratic_base), 4*self.quadratic_base, self.n_heads * self.head_dim)  # (B,Nk,4,d_model)  # [MHA]

            output_attentioned = self.reshape_attentioned(output_attentioned, batch, Number_in_upper_keylayer, H_querylayer,
                                                          W_querylayer, d_model, device, idx_dict[i]) # B, d_model, H_query, W_query

            output_attentioned = output_attentioned.flatten(2).transpose(1, 2) # B, (H_query * W_query =) N, d_model
            output_attentioned = self.final_linear[i](output_attentioned)
            output_attentioned = output_attentioned.transpose(1, 2).contiguous().view(batch, self.d_model, H_querylayer, W_querylayer)


            output_attentioned = self.merge_list[i](output_attentioned)
            outputs.append(output_attentioned)
        outputs = list(reversed(outputs))
        outputs.append(c5)
        return outputs