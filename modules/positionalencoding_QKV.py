import torch
import math
import torch.nn as nn


class FixedSinePositionalEncodingQKV(nn.Module):
    """
    Resolution-agnostic 2-D sine/cos positional encoding for Q (2x2) and K/V(4x4) maps.
    """
    def __init__(self, channels, temperature=10000):
        super().__init__()
        assert channels % 4 == 0, "channels must be divisible by 4"
        self.channels = channels
        self.temperature = temperature

    def forward(self):
        posenc_q = self.compute_posenc(2,2)
        posenc_kv= self.compute_posenc(4,4)

        return (posenc_q, posenc_kv)

    def compute_posenc(self, H, W):
        pos_y = torch.arange(H).unsqueeze(1)     # (H,1)
        pos_x = torch.arange(W).unsqueeze(1)     # (W,1)
        div_term = torch.exp(
            torch.arange(0, self.channels // 2, 2) *
            (-math.log(self.temperature) / (self.channels // 2))
        )                                                   # (C/4,)
        pos_y = pos_y * div_term                                # (H,C/4)
        pos_x = pos_x * div_term                                # (W,C/4)
        pos_y = torch.stack((pos_y.sin(), pos_y.cos()), dim=2).flatten(1)  # (H, C/2)
        pos_x = torch.stack((pos_x.sin(), pos_x.cos()), dim=2).flatten(1)  # (W, C/2)
        pe = torch.cat((
            pos_y.unsqueeze(0).expand(W,-1,-1).transpose(0,1),
            pos_x.unsqueeze(0).expand(H,-1,-1)
        ), dim=2).contiguous()

        return pe #(H,W,C)
