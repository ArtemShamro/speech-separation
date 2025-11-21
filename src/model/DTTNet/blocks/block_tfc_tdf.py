import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class TFC_TDF_Block(nn.Module):
    def __init__(self, channels, fc_dim, bf=2, use_checkpoints=False):
        super().__init__()
        self.use_checkpoints = use_checkpoints

        def conv_block():
            return nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding="same"),
                nn.InstanceNorm2d(channels),
                nn.GELU(),
            )

        self.conv1 = nn.Sequential(*[conv_block() for _ in range(3)])

        self.fc = nn.Sequential(
            nn.Linear(fc_dim, fc_dim // bf),
            nn.Linear(fc_dim // bf, fc_dim),
        )

        self.conv2 = nn.Sequential(*[conv_block() for _ in range(3)])

        self.skip_conv = nn.Conv2d(channels, channels, 3, padding="same")

    def forward(self, x: torch.Tensor):
        x_skip = self.skip_conv(x)
        if self.use_checkpoints:
            x_fc_skip = checkpoint(self.conv1, x)
        else:
            x_fc_skip = self.conv1(x)

        out = self.fc(x_fc_skip.transpose(-1, -2)).transpose(-1, -2)
        out += x_fc_skip

        if self.use_checkpoints:
            out = checkpoint(self.conv2, out)
        else:
            out = self.conv2(out)

        out += x_skip

        return out
