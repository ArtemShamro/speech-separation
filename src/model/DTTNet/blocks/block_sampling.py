from torch import nn


class DownSamplingBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.downsampling = nn.Sequential(
            nn.Conv2d(in_channels, 2 * in_channels, 3, stride=2, padding=1),
            nn.InstanceNorm2d(2 * in_channels),
            nn.GELU(),
        )

    def forward(self, x):
        out = self.downsampling(x)
        return out


class UpSamplingBlock(nn.Module):
    def __init__(self, out_chanels):
        super().__init__()

        self.upsampling = nn.Sequential(
            nn.ConvTranspose2d(
                out_chanels * 2, out_chanels, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(out_chanels),
            nn.GELU(),
        )

    def forward(self, x):
        out = self.upsampling(x)
        return out
