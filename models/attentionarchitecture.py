##### CARLOS CODE FILE ########

from models.net import conv_bn1X1, conv_bn, conv_bn_no_relu
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

        # in_channels_list = [128,256,512,2048]
        self.num_levels = len(in_channels_list)

        # Linear projections for Q,K,V
        self.q_projs = nn.ModuleList([nn.Linear(in_ch, d_model) for in_ch in in_channels_list])
        self.k_projs = nn.ModuleList([nn.Linear(in_ch, d_model) for in_ch in in_channels_list])
        self.v_projs = nn.ModuleList([nn.Linear(in_ch, d_model) for in_ch in in_channels_list])


    def forward(self, input):
        # input = OrderedDict, 128,256,512,2048
        input = list(input.values())
        # input = List, 128,256,512,2048

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
            Q = Q_list[i]  # (B, Nf, d)
            K = K_list[i]  # (B, Nc, d)
            V = V_list[i]  # (B, Nc, d)

            B, Nf, d = Q.shape
            _, Nc, _ = K.shape

            Hf, Wf = input[i].shape[2:] # From C2 to C4 -> from 160 × 160 to 40 × 40
            Hc, Wc = input[i + 1].shape[2:] # From C3 to C5 -> from 80 × 80 to 20 × 20
            device = Q.device

            #TODO: Continue here

            # a) compute index groups for 2x2 mapping
            u_c = torch.arange(Hc, device=device).unsqueeze(1).expand(Hc, Wc).reshape(-1)
            v_c = torch.arange(Wc, device=device).unsqueeze(0).expand(Hc, Wc).reshape(-1)
            base = 2 * u_c * Wf + 2 * v_c  # (Nc,)
            offs = torch.tensor([0, 1, Wf, Wf + 1], device=device)  # 4 offsets
            idx_q = base.unsqueeze(1) + offs.unsqueeze(0)  # (Nc, 4)

            # b) group Q, broadcast K and V to same shape
            Qg = Q.index_select(1, idx_q.view(-1)).view(B, Nc, 4, d)  # (B, Nc, 4, d)
            Kg = K.unsqueeze(2).expand(-1, -1, 4, -1)  # (B, Nc, 4, d)
            Vg = V.unsqueeze(2).expand(-1, -1, 4, -1)  # (B, Nc, 4, d)

            # c) compute scores and column-wise softmax (over queries)
            scores = (Qg * Kg).sum(-1) / math.sqrt(d)  # (B, Nc, 4)
            attn = F.softmax(scores, dim=2)  # (B, Nc, 4)

            # d) aggregate
            Og = attn.unsqueeze(-1) * Vg  # (B, Nc, 4, d)

            # e) scatter back into flat output
            Og_flat = Og.reshape(B, Nc * 4, d)  # (B, Nf, d) in group-order
            idx_flat = idx_q.reshape(-1)  # (Nf,)
            out_flat = torch.zeros(B, Nf, d, device=device)
            out_flat = out_flat.scatter(1, idx_flat.unsqueeze(0).expand(B, -1), Og_flat)

            # f) reshape to spatial map
            P = out_flat.transpose(1, 2).view(B, d, Hf, Wf)  # (B, d, Hf, Wf)
            outputs.append(P)

        return outputs



        #return final_outputs