import torch
from torch import nn
from torch.nn import functional as F
from src.model.DTTNet.blocks.block_tfc_tdf import TFC_TDF_Block
from src.model.DTTNet.blocks.block_sampling import DownSamplingBlock


class Encoder(nn.Module):
    def __init__(
        self, fc_dim, in_channels=2, out_channels=32, n_layers=3, use_checkpoints=False
    ):
        super().__init__()
        self.use_checkpoints = use_checkpoints
        self.init_conv = nn.Conv2d(in_channels, out_channels, 1)

        self.encoder_layers = nn.ModuleList()
        for layer_idx in range(n_layers):
            enc_dim = (fc_dim + (2**layer_idx)) if layer_idx > 0 else fc_dim
            self.encoder_layers.append(
                EncoderBlock(
                    fc_dim=enc_dim // (2**layer_idx),
                    channels=out_channels * 2**layer_idx,
                    use_checkpoints=use_checkpoints,
                )
            )

    def forward(self, spectrogram, phase):
        x = torch.stack([spectrogram, phase], dim=1)
        x = self.init_conv(x)
        skip_results = []
        for layer_idx, layer in enumerate(self.encoder_layers):
            x, skip = layer(x)
            skip_results.insert(0, skip)

        return x, skip_results


class EncoderBlock(nn.Module):
    def __init__(self, fc_dim, channels=32, use_checkpoints=False):
        super().__init__()

        self.tfc_tdf = TFC_TDF_Block(
            channels, fc_dim, bf=2, use_checkpoints=use_checkpoints
        )
        self.downsampling = DownSamplingBlock(channels)

    def forward(self, x):
        skip = self.tfc_tdf(x)
        out = self.downsampling(skip)
        return out, skip
