##### CARLOS CODE FILE ########

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from data import PositionalMode
from modules.net import conv_bn
from modules.positionalencoding_QKV import FixedSinePositionalEncodingQKV, LearnedPositionalEncodingQKV

class AttentionArchitecture(nn.Module):
    def __init__(self, in_channels_list, out_channels, phase, cfg):
        super(AttentionArchitecture,self).__init__()
        leaky = 0
        if (out_channels <= 64):
            leaky = 0.1

        d_model = out_channels
        self.d_model = out_channels
        self.phase = phase
        self.position_awareness = cfg['pos_embedding']  # PositionalMode Enum
        self.query_focused_residualconn = cfg['query_focused_residualconn']
        self.upperandlower = cfg['upperandlower']
        self.pyramidial = cfg['pyramidial']
        self.increase_receptive_field = cfg['increase_receptive_field']

        if self.increase_receptive_field:
            self.base_number = 2
        else:
            self.base_number = 1

        self.quadratic_base = self.base_number ** 2 #1 or 4

        self.n_heads = cfg['attention_heads']  # rename for brevity
        assert d_model % self.n_heads == 0, \
            "d_model must be divisible by n_heads"
        self.head_dim = d_model // self.n_heads

        # in_channels_list = [128,256,512,2048,256] -> Which are C2, C3, C4, C5, and C6/P6
        self.num_levels = len(in_channels_list)

        # Linear projections for Q,K,V
        self.q_projs = nn.ModuleList([nn.Linear(in_ch, d_model, bias=False) for in_ch in in_channels_list[:-1]])

        self.final_linear = nn.ModuleList([nn.Linear(d_model, d_model, bias=False) for _ in in_channels_list[1:]])

        #PYRAMIDIAL -> First set of queries depends on C5, while the rest depends on the calculated (reused) intermediate feature maps, which per definition have channels = 256
        if self.pyramidial:
            self.k_projs_upper = nn.ModuleList()
            self.v_projs_upper = nn.ModuleList()

            self.k_projs_upper.append(nn.Linear(in_channels_list[-1], d_model, bias=False))
            self.v_projs_upper.append(nn.Linear(in_channels_list[-1], d_model, bias=False))

            for step in range(len(in_channels_list)-2):
                self.k_projs_upper.append(nn.Linear(d_model, d_model, bias=False))
                self.v_projs_upper.append(nn.Linear(d_model, d_model, bias=False))

        #HORIZONTAL -> All sets of queries depend on the ResNet backbone, and hence, each feature map has a different channel size
        else:
            self.k_projs_upper = nn.ModuleList([nn.Linear(in_ch, d_model, bias=False) for in_ch in in_channels_list[1:]])
            self.v_projs_upper = nn.ModuleList([nn.Linear(in_ch, d_model, bias=False) for in_ch in in_channels_list[1:]])

        #Own set of linear projections for the LOWER feature maps
        if self.upperandlower:
            self.k_projs_lower = nn.ModuleList([nn.Linear(in_ch, d_model, bias=False) for in_ch in in_channels_list[:-2]])
            self.v_projs_lower = nn.ModuleList([nn.Linear(in_ch, d_model, bias=False) for in_ch in in_channels_list[:-2]])

        #Learn positional embedding while training
        #Now: All levels share the learned embeddings
        self.posembed = False
        if self.position_awareness == PositionalMode.POS_EMBEDDING_QKV:
            self.posembed = True
            if self.increase_receptive_field:
                self.embed_upper = LearnedPositionalEncodingQKV(self.head_dim, 2,2)
                self.embed_q = LearnedPositionalEncodingQKV(self.head_dim, 4,4)
                if self.upperandlower:
                    self.embed_lower = LearnedPositionalEncodingQKV(self.head_dim, 8, 8)
            else:
                self.embed_q = LearnedPositionalEncodingQKV(self.head_dim, 2, 2)
                if self.upperandlower:
                    self.embed_lower = LearnedPositionalEncodingQKV(self.head_dim, 4, 4)

        elif self.position_awareness == PositionalMode.POS_ENCODING_QKV:
            self.posembed = True
            if self.increase_receptive_field:
                self.embed_upper = FixedSinePositionalEncodingQKV(self.head_dim, 2, 2)
                self.embed_q = FixedSinePositionalEncodingQKV(self.head_dim, 4, 4)
                if self.upperandlower:
                    self.embed_lower = FixedSinePositionalEncodingQKV(self.head_dim, 8, 8)
            else:
                self.embed_q = FixedSinePositionalEncodingQKV(self.head_dim, 2, 2)
                if self.upperandlower:
                    self.embed_lower = FixedSinePositionalEncodingQKV(self.head_dim, 4, 4)

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
        # x = OrderedDict, 128,256,512,2048(,256)
        x = list(x.values())
        # x = List, 128,256,512,2048(,256)

        orig_sizes = [(f.size(2), f.size(3)) for f in x]

        if self.phase == 'train':
            return x, orig_sizes

        # Preprocessing: Bottom-up pad/crop so that each query is exactly 2× its next key (Especially necessary at Inference / Evaluation time!)
        div = self.base_number  # 1 or 2
        top = x[-1]
        _, _, H, W = top.shape
        pad_h = (div - H % div) % div
        pad_w = (div - W % div) % div
        if pad_h or pad_w:
            top = F.pad(top, (0, pad_w, 0, pad_h), mode='constant')
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
                q = F.pad(q, (0, pad_w, 0, pad_h), mode='constant')

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

    def prepare_queries(self, Q, device, H_keylayer, W_keylayer, W_querylayer, batch, Number_in_keylayer, d_model):
        # compute index groups for 2x2 mapping
        u_c = torch.arange(int(H_keylayer/self.base_number), device=device).unsqueeze(1).expand(int(H_keylayer/self.base_number), int(W_keylayer/self.base_number)).reshape(-1)
        v_c = torch.arange(int(W_keylayer/self.base_number), device=device).unsqueeze(0).expand(int(H_keylayer/self.base_number), int(W_keylayer/self.base_number)).reshape(-1)
        base = 2 * self.base_number * u_c * W_querylayer + 2 * self.base_number * v_c  # (Nc,)

        offs_q = torch.tensor(
            [ii + jj * W_querylayer for jj in range(2*self.base_number) for ii in range(2*self.base_number)],
            device=device
        )
        idx_query = base.unsqueeze(1) + offs_q.unsqueeze(0)  # (Nc, 4)
        # prepare Queries via index_select()
        Q_grouped = Q.index_select(1, idx_query.view(-1)).view(batch, int(Number_in_keylayer/self.quadratic_base), 4*self.quadratic_base, d_model)  # (B, Nc, 4, d)
        return Q_grouped, idx_query, u_c, v_c

    def forward_detail(self, x):
        outputs = []

        # 1) Project Q, K_lower, V_lower for each level, K_upper and V_upper can only be calculated from the highest level for now
        Q_list, K_list_upper, V_list_upper, K_list_lower, V_list_lower = [], [], [], [], []

        if self.pyramidial:
            kv_flat = x[-1].flatten(2).transpose(1, 2)  # (B, Nc, Cin_kv)
            K_highestlevel = self.k_projs_upper[0](kv_flat)
            V_highestlevel = self.v_projs_upper[0](kv_flat)

        for i in range(self.num_levels-1):
            q_flat = x[i].flatten(2).transpose(1, 2)  # (B, Nf, Cin_q)
            Q_list.append(self.q_projs[i](q_flat))  # (B, Nf, d)

            if self.upperandlower:
                if i > 0:
                    kv_flat_lower = x[i - 1].flatten(2).transpose(1, 2)
                    K_list_lower.append(self.k_projs_lower[i-1](kv_flat_lower))
                    V_list_lower.append(self.v_projs_lower[i-1](kv_flat_lower))

            if not self.pyramidial:
                kv_flat_upper = x[i + 1].flatten(2).transpose(1, 2)
                K_list_upper.append(self.k_projs_upper[i](kv_flat_upper))
                V_list_upper.append(self.v_projs_upper[i](kv_flat_upper))

        # From 3 to 0 -> 3, 2, 1, 0
        for i in range(self.num_levels - 2, -1, -1):
            # from (batch ,C_in ,H, W) to (B, Nf, d_model)
            Q = Q_list[i]  # (B, Nf, d_model)

            if self.pyramidial:

                if i == self.num_levels - 2:
                    K_upper = K_highestlevel
                    V_upper = V_highestlevel
                else:
                    K_upper = outputs[-1].flatten(2).transpose(1, 2)  # (B, Nc, Cin_kv)
                    k_upp_ind = self.num_levels-2-i
                    K_upper = self.k_projs_upper[k_upp_ind](K_upper)
                    V_upper = outputs[-1].flatten(2).transpose(1, 2)  # (B, Nc, Cin_kv)
                    v_upp_ind = self.num_levels-2-i
                    V_upper = self.v_projs_upper[v_upp_ind](V_upper)
            else:
                K_upper = K_list_upper[i]  # (B, Nc, d)
                V_upper = V_list_upper[i]  # (B, Nc, d)

            #[MHA] split into h heads
            Q = Q.view(*Q.shape[:-1], self.n_heads, self.head_dim)  # (B,N,h,d_h)  # [MHA]
            K_upper = K_upper.view(*K_upper.shape[:-1], self.n_heads, self.head_dim)  # [MHA]
            V_upper = V_upper.view(*V_upper.shape[:-1], self.n_heads, self.head_dim)  # [MHA]

            d_model = self.d_model

            batch, Number_in_querylayer, _, _ = Q.shape
            _, Number_in_upper_keylayer, _, _ = K_upper.shape

            H_querylayer, W_querylayer = x[i].shape[2:] # From C2 to C4 -> from 160 × 160 to 40 × 40
            H_upper_keylayer, W_upper_keylayer = x[i + 1].shape[2:] # From C3 to C5 -> from 80 × 80 to 20 × 20
            device = Q.device

            Q_grouped, idx_query, u, v = self.prepare_queries(Q.view(batch, Number_in_querylayer, self.n_heads * self.head_dim) # flatten back temporarily
                                                              ,device, H_upper_keylayer, W_upper_keylayer, W_querylayer, batch, Number_in_upper_keylayer, d_model)
            # ── [MHA] restore (h,d_h) split after grouping ─────────────
            Q_grouped = Q_grouped.view(batch, int(Number_in_upper_keylayer/self.quadratic_base), 4*self.quadratic_base, self.n_heads, self.head_dim)

            if self.posembed:
                Q_grouped = Q_grouped.view(batch, int(Number_in_upper_keylayer/self.quadratic_base), 2*self.base_number , 2*self.base_number , self.n_heads, self.head_dim)
                q_pos_enc = self.embed_q().to(device).unsqueeze(0).unsqueeze(0).unsqueeze(4).expand(batch,
                                                                                           int(Number_in_upper_keylayer / self.quadratic_base),
                                                                                           -1, -1, self.n_heads, -1)
                Q_grouped = Q_grouped + q_pos_enc
                Q_grouped = Q_grouped.view(batch, int(Number_in_upper_keylayer/self.quadratic_base), 4*self.quadratic_base, self.n_heads, self.head_dim)

            # [MHA]: (B,Nk,upper-per-group,h,d_h)
            if self.increase_receptive_field:
                H_upper, W_upper = x[i + 1].shape[2:]
                base_lo = (self.base_number * u) * W_upper + (self.base_number * v)
                # Create neighborhood offsets for upper keys
                offs_up = torch.tensor(
                    [ii + jj * W_upper for jj in range(self.base_number) for ii in range(self.base_number)],
                    device=device
                )
                idx_up = (base_lo.unsqueeze(1) + offs_up).reshape(-1)


                K_grouped = K_upper.index_select(1, idx_up).view(batch, int(Number_in_upper_keylayer/self.quadratic_base), self.quadratic_base,
                                                                                   self.n_heads, self.head_dim)
                V_grouped = V_upper.index_select(1, idx_up).view(batch, int(Number_in_upper_keylayer/self.quadratic_base), self.quadratic_base,
                                                                                   self.n_heads, self.head_dim)
                if self.posembed:
                    K_grouped = K_grouped.view(batch, int(Number_in_upper_keylayer / self.quadratic_base),
                                                           self.base_number, self.base_number, self.n_heads,
                                                           self.head_dim)
                    kv_pos_enc = self.embed_upper().to(device).unsqueeze(0).unsqueeze(0).unsqueeze(4).expand(batch,
                                                                                                int(Number_in_upper_keylayer / self.quadratic_base),
                                                                                                -1, -1, self.n_heads,
                                                                                                -1)
                    K_grouped = K_grouped + kv_pos_enc
                    ## I decided to only positionally encode the Keys, not the Values
                    K_grouped = K_grouped.view(batch, int(Number_in_upper_keylayer/self.quadratic_base), self.quadratic_base,
                                                                                   self.n_heads, self.head_dim)
            else:
                K_grouped = K_upper.unsqueeze(2)
                V_grouped = V_upper.unsqueeze(2)

            if self.upperandlower:
                #Keys/Values from lower layer need their own grouping -> hence also their own index vector
                if i > 0:
                    H_lower, W_lower = x[i - 1].shape[2:]
                    base_lo = (4 * u * self.base_number) * W_lower + (4 * v * self.base_number)
                    # Create 4x4 neighborhood offsets for lower resolution keys
                    offs_l = torch.tensor(
                        [ii + jj*W_lower for jj in range(4*self.base_number) for ii in range(4*self.base_number)],
                        device=device
                    )
                    idx_lo = (base_lo.unsqueeze(1) + offs_l).reshape(-1)

                    # gather lower keys per coarse cell
                    K_lower_grouped = K_list_lower[i-1].index_select(1, idx_lo).view(batch, int(Number_in_upper_keylayer/self.quadratic_base), 16*self.quadratic_base, self.n_heads, self.head_dim)
                    V_lower_grouped = V_list_lower[i-1].index_select(1, idx_lo).view(batch, int(Number_in_upper_keylayer/self.quadratic_base), 16*self.quadratic_base, self.n_heads, self.head_dim)

                    if self.posembed:
                        K_lower_grouped = K_lower_grouped.view(batch, int(Number_in_upper_keylayer/self.quadratic_base), 4*self.base_number,4*self.base_number , self.n_heads, self.head_dim)
                        kv_pos_enc = self.embed_lower().to(device).unsqueeze(0).unsqueeze(0).unsqueeze(4).expand(batch,
                                                                                                    int(Number_in_upper_keylayer / self.quadratic_base),
                                                                                                    -1, -1,
                                                                                                    self.n_heads,
                                                                                                    -1)
                        K_lower_grouped = K_lower_grouped + kv_pos_enc
                        ## I decided to only positionally encode the Keys, not the Values
                        K_lower_grouped = K_lower_grouped.view(batch, int(Number_in_upper_keylayer/self.quadratic_base), 16*self.quadratic_base, self.n_heads, self.head_dim)

                    K_grouped = torch.cat([K_grouped, K_lower_grouped], dim = 2)
                    V_grouped = torch.cat([V_grouped, V_lower_grouped], dim = 2)

            # c) compute scores and column-wise softmax (over queries)

            # Qg : (B, Nk, 4,  h, d_h)
            # Kg : (B, Nk, K,  h, d_h)   (K = 1 or 17)
            scores = torch.einsum('bnqhd,bnkhd->bnqkh', Q_grouped, K_grouped)  # (B, Nk, 4, K, h)
            scores = scores / math.sqrt(self.head_dim)

            if (self.upperandlower and i>0) or (self.increase_receptive_field):
                attn = F.softmax(scores, dim=3)
            else:
                attn = F.softmax(scores, dim=2)

            output_attentioned = torch.einsum('bnqkh,bnkhd->bnqhd', attn, V_grouped)  # (B,Nk,4,h,d_h)  # [MHA]

            # ── [MHA] merge heads back to d_model ─────────────────────
            output_attentioned = output_attentioned.reshape(batch, int(Number_in_upper_keylayer/self.quadratic_base), 4*self.quadratic_base, self.n_heads * self.head_dim)  # (B,Nk,4,d_model)  # [MHA]

            output_attentioned = self.reshape_attentioned(output_attentioned, batch, Number_in_upper_keylayer, H_querylayer,
                                                          W_querylayer, d_model, device, idx_query) # B, d_model, H_query, W_query

            output_attentioned = output_attentioned.flatten(2).transpose(1, 2) # B, (H_query * W_query =) N, d_model
            output_attentioned = self.final_linear[i](output_attentioned)
            output_attentioned = output_attentioned.transpose(1, 2).contiguous().view(batch, d_model, H_querylayer, W_querylayer)

            if self.query_focused_residualconn:
                Q_new = Q.view(batch, Number_in_querylayer, self.n_heads * self.head_dim).transpose(1, 2).contiguous().view(batch, d_model, H_querylayer, W_querylayer)
                output_attentioned = output_attentioned + Q_new

            output_attentioned = self.merge_list[i](output_attentioned)
            outputs.append(output_attentioned)
        outputs = list(reversed(outputs))
        return outputs