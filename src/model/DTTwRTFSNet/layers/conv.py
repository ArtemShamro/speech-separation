import torch
import torch.nn as nn


class Conv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=None,
        groups=1,
        normalization=True,
        activation=None,
        dim=1,  # 1 для 1D, 2 для 2D
    ):
        super().__init__()
        conv_layer = nn.Conv2d if dim == 2 else nn.Conv1d
        pad_val = (kernel_size - 1) // 2 if padding is None else padding
        
        self.op = nn.Sequential()
 
        conv = conv_layer(
            in_channels, out_channels, kernel_size, 
            stride=stride, padding=pad_val, groups=groups
        )
        self._init_weights(conv)
        self.op.add_module("conv", conv)
        
        if normalization:
            self.op.add_module("norm", nn.GroupNorm(1, out_channels))
            
        if activation:
            self.op.add_module("act", activation())

    def _init_weights(self, layer):
        nn.init.xavier_uniform_(layer.weight, nn.init.calculate_gain("leaky_relu"))

    def forward(self, x):
        return self.op(x)