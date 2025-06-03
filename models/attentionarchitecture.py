##### CARLOS CODE FILE ########

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from data import NeckMode

class AttentionArchitecture(nn.Module):
    def __init__(self, in_channels_list, out_channels, mode, phase, position_awareness, residualconn = False, layernorm = False, feedforward = False, ):
        super(AttentionArchitecture,self).__init__()
        d_model = out_channels
        self.mode = mode
        self.phase = phase
        self.position_awareness = position_awareness

        self.residualconn = residualconn
        self.layernorm = layernorm
        self.feedforward = feedforward

        # in_channels_list = [128,256,512,2048,256] -> Which are C2, C3, C4, C5, and C6/P6
        self.num_levels = len(in_channels_list)

        # Linear projections for Q,K,V
        self.q_projs = nn.ModuleList([nn.Linear(in_ch, d_model) for in_ch in in_channels_list[:-1]])

        #PYRAMIDIAL -> First set of queries depends on C5, while the rest depends on the calculated (reused) intermediate feature maps, which per definition have channels = 256
        if mode == NeckMode.CROSSATTENTION_FROMUPPER_PYRAMIDIAL or mode == NeckMode.CROSSATTENTION_FROMUPPERANDLOWER_PYRAMIDIAL:
            self.k_projs_upper = nn.ModuleList()
            self.v_projs_upper = nn.ModuleList()

            self.k_projs_upper.append(nn.Linear(in_channels_list[-1], d_model))
            self.v_projs_upper.append(nn.Linear(in_channels_list[-1], d_model))

            for step in range(len(in_channels_list)-2):
                self.k_projs_upper.append(nn.Linear(d_model, d_model))
                self.v_projs_upper.append(nn.Linear(d_model, d_model))
        #HORIZONTAL -> All sets of queries depend on the ResNet backbone, and hence, each feature map has a different channel size
        elif mode == NeckMode.CROSSATTENTION_FROMUPPERANDLOWER_HORIZONTAL or mode == NeckMode.CROSSATTENTION_FROMUPPER_HORIZONTAL:
            self.k_projs_upper = nn.ModuleList([nn.Linear(in_ch, d_model) for in_ch in in_channels_list[1:]])
            self.v_projs_upper = nn.ModuleList([nn.Linear(in_ch, d_model) for in_ch in in_channels_list[1:]])

        else:
            raise Exception("NeckMode is not set to a CROSSATTENTION mode (coming from Attention constructor)")

        #Own set of linear projections for the LOWER feature maps
        if mode == NeckMode.CROSSATTENTION_FROMUPPERANDLOWER_HORIZONTAL or mode == NeckMode.CROSSATTENTION_FROMUPPERANDLOWER_PYRAMIDIAL:
            self.k_projs_lower = nn.ModuleList([nn.Linear(in_ch, d_model) for in_ch in in_channels_list[:-2]])
            self.v_projs_lower = nn.ModuleList([nn.Linear(in_ch, d_model) for in_ch in in_channels_list[:-2]])

        #Learn positional embedding while training
        if self.position_awareness:
            if mode == NeckMode.CROSSATTENTION_FROMUPPERANDLOWER_HORIZONTAL or mode == NeckMode.CROSSATTENTION_FROMUPPERANDLOWER_PYRAMIDIAL:
                # For layers that only have upper keys (the highest level)
                self.pos_bias_onlyupper = nn.Parameter(torch.zeros(4))  # 4 queries attend to 1 upper key

                # For layers that have both upper and lower keys
                self.pos_bias_upperandlower = nn.ParameterList([
                    nn.Parameter(torch.zeros(17))  # 4 queries attend to 1 upper + 16 lower keys
                    for _ in range(self.num_levels - 2)
                ])
            else:
                # For upper-only modes: each layer has 4 queries attending to 1 upper key
                self.pos_bias_onlyupper = nn.ParameterList([
                    nn.Parameter(torch.zeros(4))  # 4 queries attend to 1 upper key
                    for _ in range(self.num_levels - 1)
                ])


    def forward(self, x):
        #torch.set_printoptions(profile="full")

        x, orig_sizes = self.preprocessing(x)

        if self.mode == NeckMode.CROSSATTENTION_FROMUPPER_PYRAMIDIAL:
            neck_computation = self.forward_onlyupper_pyramidial(x)

        elif self.mode == NeckMode.CROSSATTENTION_FROMUPPERANDLOWER_PYRAMIDIAL:
            neck_computation = self.forward_upperandlower_pyramidial(x)

        elif self.mode == NeckMode.CROSSATTENTION_FROMUPPERANDLOWER_HORIZONTAL:
            neck_computation = self.forward_upperandlower_horizontal(x)

        elif self.mode == NeckMode.CROSSATTENTION_FROMUPPER_HORIZONTAL:
            neck_computation = self.forward_onlyupper_horizontal(x)
        else:
            raise Exception("illegal (coming from Attention forward)")

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
        u_c = torch.arange(H_keylayer, device=device).unsqueeze(1).expand(H_keylayer, W_keylayer).reshape(-1)
        v_c = torch.arange(W_keylayer, device=device).unsqueeze(0).expand(H_keylayer, W_keylayer).reshape(-1)
        base = 2 * u_c * W_querylayer + 2 * v_c  # (Nc,)

        offs = torch.tensor([0, 1, W_querylayer, W_querylayer + 1], device=device)  # 4 offsets
        idx_query = base.unsqueeze(1) + offs.unsqueeze(0)  # (Nc, 4)

        # prepare Queries via index_select()
        Q_grouped = Q.index_select(1, idx_query.view(-1)).view(batch, Number_in_keylayer, 4, d_model)  # (B, Nc, 4, d)
        return Q_grouped, idx_query, u_c, v_c


    # Attention in a top-down/pyramidial manner, down passing information from C6/P6 until P2
    def forward_onlyupper_pyramidial(self, x):
        outputs = []

        # 1) Project Q for each level, K and V can only be calculated from the highest level for now
        Q_list = []

        kv_flat = x[-1].flatten(2).transpose(1, 2)  # (B, Nc, Cin_kv)
        K_highestlevel = self.k_projs_upper[0](kv_flat)
        V_highestlevel = self.v_projs_upper[0](kv_flat)

        for i in range(self.num_levels-1):
            # Q from level i
            # C6/P6 will not act as query layer -> but all below will
            q_flat = x[i].flatten(2).transpose(1, 2)  # (B, Nf, Cin_q)
            Q_list.append(self.q_projs[i](q_flat))  # (B, Nf, d)

        # 2) Perform local cross-attention per level
        for i in range(self.num_levels - 2, -1, -1):
            # from (batch ,C_in ,H, W) to (B, Nf, d)
            Q = Q_list[i]  # (B, Nf, d)

            if i == self.num_levels - 2:
                K = K_highestlevel
                V = V_highestlevel
            else:
                K = outputs[-1].flatten(2).transpose(1, 2)  # (B, Nc, Cin_kv)
                k_proj_ind = self.num_levels-2-i
                K = self.k_projs_upper[k_proj_ind](K)
                V = outputs[-1].flatten(2).transpose(1, 2)  # (B, Nc, Cin_kv)
                v_proj_ind = self.num_levels-2-i
                V = self.v_projs_upper[v_proj_ind](V)

            batch, Number_in_querylayer, d_model = Q.shape
            _, Number_in_keylayer, _ = K.shape

            H_querylayer, W_querylayer = x[i].shape[2:] # From C2 to C4 -> from 160 × 160 to 40 × 40
            H_keylayer, W_keylayer = x[i + 1].shape[2:] # From C3 to C5 -> from 80 × 80 to 20 × 20
            device = Q.device

            Q_grouped,idx_query,_,_ = self.prepare_queries(Q, device, H_keylayer, W_keylayer, W_querylayer, batch, Number_in_keylayer, d_model)

            Kg = K.unsqueeze(2).expand(-1, -1, 4, -1)  # (B, Nc, 4, d)
            Vg = V.unsqueeze(2).expand(-1, -1, 4, -1)  # (B, Nc, 4, d)

            # c) compute scores and column-wise softmax (over queries)
            scores = (Q_grouped * Kg).sum(-1) / math.sqrt(d_model)  # (B, Nc, 4)
            if self.position_awareness:
                scores = scores + self.pos_bias_onlyupper[i].view(1, 1, -1)
            scores = F.softmax(scores, dim=2)  # (B, Nc, 4)

            # d) aggregate
            output_attentioned = scores.unsqueeze(-1) * Vg  # (B, Nc, 4, d)

            # e) Rearrange output of attention to have correct position
            output_attentioned = self.reshape_attentioned(output_attentioned, batch, Number_in_keylayer, H_querylayer, W_querylayer, d_model, device, idx_query)
            outputs.append(output_attentioned)

        outputs = list(reversed(outputs))
        return outputs


    def forward_upperandlower_pyramidial(self, x):
        outputs = []

        # 1) Project Q, K_lower, V_lower for each level, K_upper and V_upper can only be calculated from the highest level for now
        Q_list, K_list_lower, V_list_lower = [], [], []

        kv_flat = x[-1].flatten(2).transpose(1, 2)  # (B, Nc, Cin_kv)
        K_highestlevel = self.k_projs_upper[0](kv_flat)
        V_highestlevel = self.v_projs_upper[0](kv_flat)

        for i in range(self.num_levels-1):
            q_flat = x[i].flatten(2).transpose(1, 2)  # (B, Nf, Cin_q)
            Q_list.append(self.q_projs[i](q_flat))  # (B, Nf, d)

            if i > 0:
                kv_flat_lower = x[i - 1].flatten(2).transpose(1, 2)
                K_list_lower.append(self.k_projs_lower[i-1](kv_flat_lower))
                V_list_lower.append(self.v_projs_lower[i-1](kv_flat_lower))

        for i in range(self.num_levels - 2, -1, -1):
            # from (batch ,C_in ,H, W) to (B, Nf, d_model)
            Q = Q_list[i]  # (B, Nf, d_model)

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

            batch, Number_in_querylayer, d_model = Q.shape
            _, Number_in_upper_keylayer, _ = K_upper.shape

            H_querylayer, W_querylayer = x[i].shape[2:] # From C2 to C4 -> from 160 × 160 to 40 × 40
            H_upper_keylayer, W_upper_keylayer = x[i + 1].shape[2:] # From C3 to C5 -> from 80 × 80 to 20 × 20
            device = Q.device

            Q_grouped, idx_query, u, v = self.prepare_queries(Q, device, H_upper_keylayer, W_upper_keylayer, W_querylayer, batch, Number_in_upper_keylayer,
                                      d_model)
            K_grouped = K_upper.unsqueeze(2)
            V_grouped = V_upper.unsqueeze(2)

            K_lower_grouped = None

            #Keys/Values from lower layer need their own grouping -> hence also their own index vector
            if i > 0:
                H_lower, W_lower = x[i - 1].shape[2:]
                base_lo = (4 * u) * W_lower + (4 * v)
                offs_l = torch.tensor(
                    [ii + jj*W_lower for jj in range(4) for ii in range(4)],
                    device=device
                )
                idx_lo = (base_lo.unsqueeze(1) + offs_l).reshape(-1)

                # gather 16 lower keys per coarse cell
                K_lower_grouped = K_list_lower[i-1].index_select(1, idx_lo).view(batch, Number_in_upper_keylayer, 16, d_model)
                V_lower_grouped = V_list_lower[i-1].index_select(1, idx_lo).view(batch, Number_in_upper_keylayer, 16, d_model)
                K_grouped = torch.cat([K_grouped, K_lower_grouped], dim = 2)
                V_grouped = torch.cat([V_grouped, V_lower_grouped], dim = 2)

            # c) compute scores and column-wise softmax (over queries)
            scores = torch.einsum('bnmd,bnkd->bnmk', Q_grouped, K_grouped) / math.sqrt(d_model)
            if K_lower_grouped is not None:
                if self.position_awareness:
                    scores = scores + self.pos_bias_upperandlower[i-1].view(1, 1, 1, -1)
                attn = F.softmax(scores, dim=3)
            else:
                if self.position_awareness:
                    scores = scores + self.pos_bias_onlyupper.view(1, 1, -1, 1)
                attn = F.softmax(scores, dim=2)
            output_attentioned = torch.einsum('bnmk,bnkd->bnmd', attn, V_grouped)

            output_attentioned = self.reshape_attentioned(output_attentioned, batch, Number_in_upper_keylayer, H_querylayer,
                                                          W_querylayer, d_model, device, idx_query)
            outputs.append(output_attentioned)
        outputs = list(reversed(outputs))
        return outputs

    ## horizontal Version of forward()
    def forward_onlyupper_horizontal(self, x,):
        # input = OrderedDict, 128,256,512,2048(,256)
        outputs = []

        # 1) Project Q, K, V for each adjacent pair
        Q_list, K_list, V_list = [], [], []

        for i in range(self.num_levels):
            # Q from level i

            # C5 will not act as query layer -> but all below will
            if i < (self.num_levels-1):
                q_flat = x[i].flatten(2).transpose(1, 2)  # (B, Nf, Cin_q)
                Q_list.append(self.q_projs[i](q_flat))  # (B, Nf, d)

            # C2 will not act as key/value layer -> but all above will
            # K, V from level i
            if i > 0:
                kv_flat = x[i].flatten(2).transpose(1, 2)  # (B, Nc, Cin_kv)
                K_list.append(self.k_projs_upper[i-1](kv_flat))  # (B, Nc, d)
                V_list.append(self.v_projs_upper[i-1](kv_flat))  # (B, Nc, d)

        # 2) Perform local cross-attention per level
        for i in range(self.num_levels - 1):
            # from (batch ,C_in ,H, W) to (B, Nf, d)
            Q = Q_list[i]  # (B, Nf, d)
            K = K_list[i]  # (B, Nc, d)
            V = V_list[i]  # (B, Nc, d)

            batch, Number_in_querylayer, d_model = Q.shape
            _, Number_in_keylayer, _ = K.shape

            H_querylayer, W_querylayer = x[i].shape[2:] # From C2 to C4 -> from 160 × 160 to 40 × 40
            H_keylayer, W_keylayer = x[i + 1].shape[2:] # From C3 to C5 -> from 80 × 80 to 20 × 20
            device = Q.device

            Qg,idx_query,_,_ = self.prepare_queries(Q, device, H_keylayer, W_keylayer, W_querylayer, batch, Number_in_keylayer,
                                      d_model)
            Kg = K.unsqueeze(2).expand(-1, -1, 4, -1)  # (B, Nc, 4, d)
            Vg = V.unsqueeze(2).expand(-1, -1, 4, -1)  # (B, Nc, 4, d)

            # c) compute scores and column-wise softmax (over queries)
            scores = (Qg * Kg).sum(-1) / math.sqrt(d_model)  # (B, Nc, 4)
            if self.position_awareness:
                scores = scores + self.pos_bias_onlyupper[i].view(1, 1, -1, 1)

            scores = F.softmax(scores, dim=2)  # (B, Nc, 4)

            # d) aggregate
            output_attentioned = scores.unsqueeze(-1) * Vg  # (B, Nc, 4, d)
            output_attentioned = self.reshape_attentioned(output_attentioned, batch, Number_in_keylayer, H_querylayer,
                                                          W_querylayer, d_model, device, idx_query)
            outputs.append(output_attentioned)

        return outputs

    ## horizontal Version of forward()
    def forward_upperandlower_horizontal(self, x):
        outputs = []

        # 1) Project Q, K, V for each adjacent pair
        Q_list, K_list_upper, V_list_upper, K_list_lower, V_list_lower = [], [], [], [], []

        for i in range(self.num_levels-1):
            q_flat = x[i].flatten(2).transpose(1, 2)  # (B, Nf, Cin_q)
            Q_list.append(self.q_projs[i](q_flat))  # (B, Nf, d)

            kv_flat_upper = x[i + 1].flatten(2).transpose(1, 2)
            K_list_upper.append(self.k_projs_upper[i](kv_flat_upper))
            V_list_upper.append(self.v_projs_upper[i](kv_flat_upper))
            if i > 0:
                kv_flat_lower = x[i - 1].flatten(2).transpose(1, 2)
                K_list_lower.append(self.k_projs_lower[i-1](kv_flat_lower))
                V_list_lower.append(self.v_projs_lower[i-1](kv_flat_lower))

        # 2) Perform local cross-attention per level
        for i in range(self.num_levels - 1):
            # from (batch ,C_in ,H, W) to (B, Nf, d)
            Q = Q_list[i]  # (B, Nf, d)
            K_upper = K_list_upper[i]  # (B, Nc, d)
            V_upper = V_list_upper[i]  # (B, Nc, d)

            batch, Number_in_querylayer, d_model = Q.shape
            _, Number_in_upper_keylayer, _ = K_upper.shape

            H_querylayer, W_querylayer = x[i].shape[2:] # From C2 to C4 -> from 160 × 160 to 40 × 40
            H_upper_keylayer, W_upper_keylayer = x[i + 1].shape[2:] # From C3 to C5 -> from 80 × 80 to 20 × 20
            device = Q.device

            Q_grouped, idx_query, u, v = self.prepare_queries(Q, device, H_upper_keylayer, W_upper_keylayer, W_querylayer, batch, Number_in_upper_keylayer,
                                      d_model)

            K_grouped = K_upper.unsqueeze(2)
            V_grouped = V_upper.unsqueeze(2)

            K_lower_grouped = None

            #Keys/Values from lower layer need their own grouping -> hence also their own index vector
            if i > 0:
                H_lower, W_lower = x[i - 1].shape[2:]
                base_lo = (4 * u) * W_lower + (4 * v)
                #torch.set_printoptions(profile="full")
                #print(base_lo)
                offs_l = torch.tensor(
                    [ii + jj*W_lower for jj in range(4) for ii in range(4)],
                    device=device
                )
                idx_lo = (base_lo.unsqueeze(1) + offs_l).reshape(-1)  # (Nc*4*4,)

                # gather 16 lower keys per coarse cell
                K_lower_grouped = K_list_lower[i-1].index_select(1, idx_lo).view(batch, Number_in_upper_keylayer, 16, d_model)
                V_lower_grouped = V_list_lower[i-1].index_select(1, idx_lo).view(batch, Number_in_upper_keylayer, 16, d_model)
                K_grouped = torch.cat([K_grouped, K_lower_grouped], dim = 2)
                V_grouped = torch.cat([V_grouped, V_lower_grouped], dim = 2)

            # c) compute scores and column-wise softmax (over queries)
            scores = torch.einsum('bnmd,bnkd->bnmk', Q_grouped, K_grouped) / math.sqrt(d_model)
            if K_lower_grouped is not None:
                if self.position_awareness:
                    scores = scores + self.pos_bias_upperandlower[i - 1].view(1, 1, 1, -1)
                attn = F.softmax(scores, dim=3)
            else:
                if self.position_awareness:
                    scores = scores + self.pos_bias_onlyupper.view(1, 1, -1, 1)
                attn = F.softmax(scores, dim=2)
            output_attentioned = torch.einsum('bnmk,bnkd->bnmd', attn, V_grouped)

            output_attentioned = self.reshape_attentioned(output_attentioned, batch, Number_in_upper_keylayer, H_querylayer,
                                                          W_querylayer, d_model, device, idx_query)
            outputs.append(output_attentioned)

        return outputs