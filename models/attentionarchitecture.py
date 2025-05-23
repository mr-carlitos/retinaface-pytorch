##### CARLOS CODE FILE ########

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class AttentionArchitecture(nn.Module):
    def __init__(self, in_channels_list, out_channels, d_model=256):
        super(AttentionArchitecture,self).__init__()
        leaky = 0
        if (out_channels <= 64):
            leaky = 0.1

        # in_channels_list = [128,256,512,2048,256]
        self.num_levels = len(in_channels_list)

        # Linear projections for Q,K,V
        self.q_projs = nn.ModuleList([nn.Linear(in_ch, d_model) for in_ch in in_channels_list])
        self.k_projs = nn.ModuleList([nn.Linear(in_ch, d_model) for in_ch in in_channels_list])
        self.v_projs = nn.ModuleList([nn.Linear(in_ch, d_model) for in_ch in in_channels_list])


    def forward(self, input):
        # input = OrderedDict, 128,256,512,2048(,256)
        input = list(input.values())
        # input = List, 128,256,512,2048(,256)

        orig_sizes = [(f.size(2), f.size(3)) for f in input]

        # Preprocessing: Bottom-up pad/crop so that each query is exactly 2× its next key (Especially necessary at Inference / Evaluation time!)
        for i in range(self.num_levels - 2, -1, -1):
            q = input[i]  # e.g. C2, C3, C4, C5 in turn
            k = input[i + 1]  # the immediately finer map (C3, C4, C5)
            B, C, Hq, Wq = q.shape
            _, _, Hk, Wk = k.shape

            Ht, Wt = 2 * Hk, 2 * Wk

            # pad bottom/right if under‐sized
            pad_h = max(0, Ht - Hq)
            pad_w = max(0, Wt - Wq)
            if pad_h or pad_w:
                # replicate so we don’t introduce zeros
                q = F.pad(q, (0, pad_w, 0, pad_h), mode='replicate')

            # crop bottom/right if over‐sized
            crop_h = max(0, q.size(2) - Ht)
            crop_w = max(0, q.size(3) - Wt)
            if crop_h or crop_w:
                q = q[:, :, :Ht, :Wt]

            input[i] = q

        outputs = []

        # 1) Project Q, K, V for each adjacent pair
        Q_list, K_list, V_list = [], [], []

        for i in range(self.num_levels):
            # Q from level i

            # C5 will not act as query layer -> but all below will
            if i < (self.num_levels-1):
                q_flat = input[i].flatten(2).transpose(1, 2)  # (B, Nf, Cin_q)
                Q_list.append(self.q_projs[i](q_flat))  # (B, Nf, d)

            # C2 will not act as key/value layer -> but all above will
            # K, V from level i
            if i > 0:
                kv_flat = input[i].flatten(2).transpose(1, 2)  # (B, Nc, Cin_kv)
                K_list.append(self.k_projs[i](kv_flat))  # (B, Nc, d)
                V_list.append(self.v_projs[i](kv_flat))  # (B, Nc, d)

        # 2) Perform local cross-attention per level
        for i in range(self.num_levels - 1):
            # from (batch ,C_in ,H, W) to (B, Nf, d)
            Q = Q_list[i]  # (B, Nf, d)
            K = K_list[i]  # (B, Nc, d)
            V = V_list[i]  # (B, Nc, d)

            batch, Number_in_querylayer, d_model = Q.shape
            _, Number_in_keylayer, _ = K.shape

            H_querylayer, W_querylayer = input[i].shape[2:] # From C2 to C4 -> from 160 × 160 to 40 × 40
            H_keylayer, W_keylayer = input[i + 1].shape[2:] # From C3 to C5 -> from 80 × 80 to 20 × 20
            device = Q.device

            # a) compute index groups for 2x2 mapping
            u_c = torch.arange(H_keylayer, device=device).unsqueeze(1).expand(H_keylayer, W_keylayer).reshape(-1)
            v_c = torch.arange(W_keylayer, device=device).unsqueeze(0).expand(H_keylayer, W_keylayer).reshape(-1)
            base = 2 * u_c * W_querylayer + 2 * v_c  # (Nc,)
            del u_c, v_c
            offs = torch.tensor([0, 1, W_querylayer, W_querylayer + 1], device=device)  # 4 offsets
            idx_query = base.unsqueeze(1) + offs.unsqueeze(0)  # (Nc, 4)
            del base, offs

            #torch.set_printoptions(profile="full")
            #print(idx_query)

            # b) prepare Queries, Keys and Values
            Qg = Q.index_select(1, idx_query.view(-1)).view(batch, Number_in_keylayer, 4, d_model)  # (B, Nc, 4, d)
            Kg = K.unsqueeze(2).expand(-1, -1, 4, -1)  # (B, Nc, 4, d)
            Vg = V.unsqueeze(2).expand(-1, -1, 4, -1)  # (B, Nc, 4, d)

            # c) compute scores and column-wise softmax (over queries)
            scores = (Qg * Kg).sum(-1) / math.sqrt(d_model)  # (B, Nc, 4)
            del Qg, Kg
            scores = F.softmax(scores, dim=2)  # (B, Nc, 4)

            # d) aggregate
            output_attentioned = scores.unsqueeze(-1) * Vg  # (B, Nc, 4, d)
            del Vg, scores

            # e) Rearrange output of attention to have correct position
            output_attentioned = output_attentioned.reshape(batch, Number_in_keylayer * 4, d_model)  # (B, Nf, d) in group-order
            output_attentioned = output_attentioned[:, idx_query.reshape(-1), :]
            del idx_query

            # f) reshape to spatial map
            output_attentioned = output_attentioned.transpose(1, 2).view(batch, d_model, H_querylayer, W_querylayer) # (B, d, Hf, Wf)
            outputs.append(output_attentioned)

        # cropping back: (Especially necessary at Inference / Evaluation time!)
        for i, out in enumerate(outputs):
            H_orig, W_orig = orig_sizes[i]
            outputs[i] = out[:, :, :H_orig, :W_orig]

        return outputs
