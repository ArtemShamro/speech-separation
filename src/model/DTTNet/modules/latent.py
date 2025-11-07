import torch
from torch import nn
from torch.nn import functional as F
from src.model.DTTNet.blocks.block_tfc_tdf import TFC_TDF_Block


class LatentModule(nn.Module):
    def __init__(self, fc_dim, n_heads=1, n_channels=256, n_layers=3):
        super().__init__()
        assert n_channels % n_heads == 0, "n_channels must be divisible by n_heads"

        self.tfc_tdf = TFC_TDF_Block(
            channels=n_channels,
            fc_dim=fc_dim
        )

        self.idp_modules = nn.ModuleList([
            IDPModule(n_heads, n_channels) for _ in range(n_layers)
        ])

    def forward(self, x):
        x = self.tfc_tdf(x)
        for module in self.idp_modules:
            x = module(x)

        return x


class IDPModule(nn.Module):
    def __init__(self, n_heads, in_channels):
        super().__init__()
        self.n_heads = n_heads
        channels = int(in_channels // n_heads)

        self.tc_rnn = TC_FC_RNN(n_heads, channels)

        self.fc_rnn = TC_FC_RNN(n_heads, channels)

    def forward(self, x: torch.Tensor):
        # x [B, C, F, T]
        B, C, F, T = x.shape
        B_hat, C_hat = B * self.n_heads, C // self.n_heads

        # split channels
        x = self.split_heads(x)  # [B', F, T, C']

        x = x.reshape(B_hat * F, T, C_hat)
        x = self.tc_rnn(x)
        x = x.reshape(B_hat, F, T, C_hat)

        x = x.permute(0, 2, 1, 3)  # [B', T, F, C']
        x = x.reshape(B_hat * T, F, C_hat)  # [B' * T, F, C']
        x = self.fc_rnn(x)
        x = x.reshape(B_hat, T, F, C_hat)  # [B', T, F, C']

        # merge channels
        x = self.merge_heads(x, (B, C, F, T))  # [B, C, F, T]
        return x

    def split_heads(self, x: torch.Tensor):
        B, C, F, T = x.shape
        x = x.view(B, self.n_heads, C // self.n_heads, F, T)
        x = x.permute(0, 1, 3, 4, 2)  # [B, heads, F, T, C_hat]
        x = x.reshape(B * self.n_heads, F, T, C // self.n_heads)

        return x

    def merge_heads(self, x: torch.Tensor, shape):
        B, C, F, T = shape
        C_hat = C // self.n_heads
        # x shape is [B_hat, T, F, C_hat]
        x = x.view(B, self.n_heads, F, T, C_hat)
        x = x.permute(0, 1, 4, 2, 3).reshape(B, C, F, T)
        return x


class TC_FC_RNN(nn.Module):
    def __init__(self, n_heads, in_chanels):
        super().__init__()

        self.norm = nn.GroupNorm(
            num_groups=n_heads,
            num_channels=in_chanels,
        )

        self.blstm = nn.LSTM(
            input_size=in_chanels,
            hidden_size=2 * in_chanels,
            batch_first=True,
            bidirectional=True,
        )

        self.fc = nn.Linear(
            in_chanels * 4, in_chanels
        )

    def forward(self, x):
        # x.shape is [B x H x F, T, C // H] or [B x H x T, F, C // H]
        out = self.norm(x.transpose(-1, -2)).transpose(-1, -2)
        out, _ = self.blstm(out)
        out = self.fc(out)
        out += x
        return out
