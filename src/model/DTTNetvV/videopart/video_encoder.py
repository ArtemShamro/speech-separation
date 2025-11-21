"""
Source: https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks/blob/master/lipreading/model.py
"""

import torch
import torch.nn as nn
import math
import numpy as np
from src.model.DTTNetvV.videopart.resnet import Swish, BasicBlock, ResNet

def threeD_to_2D_tensor(x):
    n_batch, n_channels, s_time, sx, sy = x.shape
    x = x.transpose(1, 2)
    return x.reshape(n_batch*s_time, n_channels, sx, sy)

class VideoEncoder(nn.Module):
    def __init__(self, relu_type = 'swish'):
        super(VideoEncoder, self).__init__()
        self.frontend_nout = 64
        self.backend_out = 512
        self.trunk = ResNet(BasicBlock, [2, 2, 2, 2], relu_type=relu_type)

        if relu_type == 'relu':
            frontend_relu = nn.ReLU(True)
        elif relu_type == 'prelu':
            frontend_relu = nn.PReLU( self.frontend_nout )
        elif relu_type == 'swish':
            frontend_relu = Swish()

        self.frontend3D = nn.Sequential(
            nn.Conv3d(1, self.frontend_nout, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
            nn.BatchNorm3d(self.frontend_nout),
            frontend_relu,
            nn.MaxPool3d( kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)))
        
    def forward(self, x):
        B, N, T, H, W = x.size()
        x = x.view(B * N, 1, T, H, W)
        x = self.frontend3D(x) # (B * N, C', T', H', W')
        Tnew = x.shape[2]
        x = threeD_to_2D_tensor(x) # (B * N * T', C', H', W')
        x = self.trunk(x) # (B * N * T', embed_dim)
        x = x.view(B * N, Tnew, -1) # (B * N, T', embed_dim)
        x = x.view(B, N, Tnew, -1).transpose(2, 3)  # (B, 2, embed_dim, T')

        return x
