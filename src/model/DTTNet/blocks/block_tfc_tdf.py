import torch
from torch import nn
from torch.nn import functional as F


class TFC_TDF_Block(nn.Module):
    def __init__(self, channels, fc_dim, bf=2):
        super().__init__()

        def conv_block():
            return nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding="same"),
                nn.InstanceNorm2d(channels),
                nn.GELU(),
            )

        self.conv1 = nn.Sequential(
            *[conv_block() for _ in range(3)]
        )

        self.fc = nn.Sequential(
            nn.Linear(fc_dim, fc_dim // bf),
            nn.Linear(fc_dim // bf, fc_dim),
        )

        self.conv2 = nn.Sequential(
            *[conv_block() for _ in range(3)]
        )

        self.skip_conv = nn.Conv2d(channels, channels, 3, padding="same")

    def forward(self, x: torch.Tensor):
        x_skip = self.skip_conv(x)
        x_fc_skip = self.conv1(x)
        out = self.fc(x_fc_skip.transpose(-1, -2)).transpose(-1, -2)
        out += x_fc_skip
        out = self.conv2(out)
        out += x_skip

        return out
