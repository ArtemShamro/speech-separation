import torch
from torch import nn
from torch.nn import functional as F
from src.model.DTTNet.blocks.block_tfc_tdf import TFC_TDF_Block
from src.model.DTTNet.blocks.block_sampling import UpSamplingBlock


class Decoder(nn.Module):
    def __init__(self, fc_dim, out_channels=32, n_sources=2, n_layers=3):
        super().__init__()
        self.out_conv = nn.Conv2d(out_channels, n_sources * 2, 1)

        self.decoder_layers = nn.ModuleList()
        for layer_idx in range(n_layers):
            dec_dim = (fc_dim + (2 ** layer_idx)) if layer_idx > 0 else fc_dim
            self.decoder_layers.insert(
                0,
                DecoderBlock(
                    fc_dim=dec_dim // (2 ** layer_idx),
                    channels=out_channels * 2 ** (layer_idx),
                )
            )

    def forward(self, x, skip):
        for layer_idx, layer in enumerate(self.decoder_layers):
            x = layer(x, skip[layer_idx])
        x = self.out_conv(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, fc_dim, channels=32):
        super().__init__()
        # print(f"Decoder Block : init : channels {channels}")
        self.tfc_tdf = TFC_TDF_Block(channels, fc_dim, bf=2)
        self.upsampling = UpSamplingBlock(channels)

    def forward(self, x, skip):
        out = self.upsampling(x)
        out = self.pad_out(out, skip)
        out = self.tfc_tdf(out)
        out *= skip
        return out

    def pad_out(self, x, skip):
        _, _, Ht, Wt = skip.shape
        _, _, H, W = x.shape
        diff_H = Ht - H
        diff_W = Wt - W
        out = F.pad(x, (0, diff_W, 0, diff_H))
        return out
