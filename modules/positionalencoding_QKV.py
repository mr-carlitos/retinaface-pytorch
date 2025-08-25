### CARLOS CODE FILE. Inspired by DETR: https://github.com/facebookresearch/detr/blob/main/models/position_encoding.py
'''

This file provides fixed sinus/cosinus positional encodings (FixedSinePositionalEncodingQKV) and learned positional embeddings (LearnedPositionalEncodingQKV)
'''
import torch
import math
import torch.nn as nn

class FixedSinePositionalEncodingQKV(nn.Module):
    """
    2D sine/cos positional encoding for Q and K maps.
    """
    def __init__(self, channels, height, width, temperature=10000):
        super().__init__()
        assert channels % 4 == 0, "channels must be divisible by 4"
        self.channels = channels
        self.temperature = temperature
        self.height = height
        self.width = width

        self.div_term = torch.exp(
            torch.arange(0, self.channels // 2, 2) *
            (-math.log(self.temperature) / (self.channels // 2))
        )

    def forward(self):
        device = self.div_term.device
        pos_y = torch.arange(self.height, device = device).unsqueeze(1)     # (H,1)
        pos_x = torch.arange(self.width, device = device).unsqueeze(1)     # (W,1)
                                     # (C/4,)
        pos_y = pos_y * self.div_term                                # (H,C/4)
        pos_x = pos_x * self.div_term                                # (W,C/4)
        pos_y = torch.stack((pos_y.sin(), pos_y.cos()), dim=2).flatten(1)  # (H, C/2)
        pos_x = torch.stack((pos_x.sin(), pos_x.cos()), dim=2).flatten(1)  # (W, C/2)

        pos_y = pos_y[:, None, :].expand(-1, self.width, -1)
        pos_x = pos_x[None, :, :].expand(self.height, -1, -1)
        pe = torch.cat((pos_y, pos_x), dim=-1).contiguous()  # (H, W, C)

        return pe #(H,W,C)

class LearnedPositionalEncodingQKV(nn.Module):
    """
    Learned 2-D positional encoding for
    Q and K  feature maps."""

    def __init__(self, channels, height, width):
        super().__init__()
        assert channels % 2 == 0, "channels must be even (half for rows, half for cols)"
        self.channels = channels

        self.height = height
        self.width = width
        # Learned lookup tables
        self.row_embed = nn.Embedding(height, channels // 2)
        self.col_embed = nn.Embedding(width,  channels // 2)

    def forward(self):

        device = self.row_embed.weight.device

        # Indices for this spatial size
        rows = torch.arange(self.height, device=device)           # (H,)
        cols = torch.arange(self.width, device=device)           # (W,)

        # Look them up
        row_feat = self.row_embed(rows)                 # (H, C/2)
        col_feat = self.col_embed(cols)                 # (W, C/2)

        # Broadcast to an (H, W, C/2) grid and concatenate
        pos_y = row_feat[:, None, :].expand(-1, self.width, -1)  # (H, W, C/2)
        pos_x = col_feat[None, :, :].expand(self.height, -1, -1)  # (H, W, C/2)

        pe = torch.cat((pos_x, pos_y), dim=-1).contiguous()  # (H, W, C)
        return pe
