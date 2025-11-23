import torch
from torch import nn, Tensor
from torch.nn import functional as F


class AudioEncoderModule(nn.Module):
    def __init__(
        self,
        out_channels,
        in_channels=2,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding="same",
        )

    def forward(self, spectrogram: Tensor, phase: Tensor) -> Tensor:
        x = torch.stack([spectrogram, phase], dim=1)
        x = self.conv(x)
        return x
