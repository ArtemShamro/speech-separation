import torch
from math import log

import torch.nn as nn
from torch.nn import functional as F

from src.model.DTTwRTFSNet.layers.conv import Conv



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=10000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-log(max_len) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        return self.dropout(x + self.pe[None, :seq_len, :])


class VideoAttention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.norm = nn.RMSNorm(dim)
        self.pos_enc = PositionalEncoding(dim, dropout)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        
        self.ffn = nn.Sequential(
            Conv(dim, dim * 2, 1, dim=1, activation=nn.PReLU),
            Conv(dim * 2, dim * 2, 5, groups=dim * 2, dim=1, activation=nn.PReLU),
            Conv(dim * 2, dim, 1, dim=1, activation=nn.PReLU),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x_t = x.transpose(1, 2)
        x_pe = self.pos_enc(x_t)
        
        res = x_pe
        x_norm = self.norm(x_pe)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x_t = self.dropout(attn_out) + res
        
        x_c = x_t.transpose(1, 2)
        x_c = self.ffn(x_c) + x_c
        return x_c


class Compressor(nn.Module):
    def __init__(self, num_levels, dim):
        super().__init__()
        self.num_levels = num_levels
        
        layers = []
        for i in range(num_levels):
            stride = 2 if i > 0 else 1
            pad = "same" if i == 0 else None
            layers.append(
                Conv(dim, dim, 4, stride=stride, padding=pad, 
                                 groups=dim, dim=1, activation=nn.PReLU)
            )
        self.layers = nn.ModuleList(layers)
        self.pool = F.adaptive_avg_pool1d

    def forward(self, x):
        pyramid = [self.layers[0](x)]
        
        for i in range(self.num_levels - 1):
            prev_feat = pyramid[-1]
            pyramid.append(self.layers[i + 1](prev_feat))
            
        bottleneck = pyramid[-1]
        for i in range(len(pyramid) - 1):
            bottleneck = bottleneck + self.pool(pyramid[i], bottleneck.shape[2:])
            
        return pyramid, bottleneck


class Interpolation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate_net = Conv(dim, dim, 4, padding="same", groups=dim, dim=1, activation=nn.Sigmoid)
        self.content_net = Conv(dim, dim, 4, padding="same", groups=dim, dim=1)
        self.state_net = Conv(dim, dim, 4, padding="same", groups=dim, dim=1)

    def forward(self, current_state, skip_connection):
        target_size = current_state.shape[2:]
        
        transformed_state = self.state_net(current_state)
        
        gate = F.interpolate(self.gate_net(skip_connection), size=target_size, mode="nearest")
        content = F.interpolate(self.content_net(skip_connection), size=target_size, mode="nearest")
        
        return (gate * transformed_state) + content


class Decompressor(nn.Module):
    def __init__(self, num_levels, dim):
        super().__init__()
        self.pre_mix = nn.ModuleList([Interpolation(dim) for _ in range(num_levels)])
        self.recursive_mix = nn.ModuleList([Interpolation(dim) for _ in range(num_levels - 1)])

    def forward(self, bottleneck, pyramid):
        mixed_states = [
            block(p_feat, bottleneck) 
            for block, p_feat in zip(self.pre_mix, pyramid)
        ]
        
        x = self.recursive_mix[-1](mixed_states[-2], mixed_states[-1]) + pyramid[-2]
        
        for i in range(len(pyramid) - 3, -1, -1):
            x = self.recursive_mix[i](mixed_states[i], x) + pyramid[i]
            
        return x


class VPBlock(nn.Module):
    def __init__(self, num_levels, in_dim, internal_dim):
        super().__init__()
        
        self.input_proj = Conv(in_dim, internal_dim, 1, dim=1, activation=nn.PReLU)
        
        self.encoder = Compressor(num_levels, internal_dim)
        self.processor = VideoAttention(internal_dim)
        self.decoder = Decompressor(num_levels, internal_dim)
        
        self.output_proj = Conv(internal_dim, in_dim, 1, dim=1, normalization=False)

    def forward(self, x):
        residual = x
        
        x = self.input_proj(x)
        
        pyramid, bottleneck = self.encoder(x)
        processed_bottleneck = self.processor(bottleneck)
        reconstructed = self.decoder(processed_bottleneck, pyramid)
        
        out = self.output_proj(reconstructed)
        return out + residual