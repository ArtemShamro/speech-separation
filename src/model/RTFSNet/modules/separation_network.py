import torch
from torch import nn, Tensor
from torch.nn import functional as F
from .separation_module.audio_preprocess import AudioPreprocessModule
from .separation_module.video_preprocess import VideoPreprocessModule
from .separation_module.CAF_block import CAFBlock
from .separation_module.RTFS_module import RTFSModule


class SeparationNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.audio_preprocess = AudioPreprocessModule()  # single RTFS Block
        self.video_preprocess = VideoPreprocessModule()  # TDANet Block

        self.CAF_block = CAFBlock()  # Fusion Block

        self.RTFS_module = RTFSModule()  # Fused data processor

    def forward(self, audio_encoded: Tensor, video_encoded: Tensor) -> Tensor:

        audio_preprocessed = self.audio_preprocess(audio_encoded)  # [C_a, T_a, F]
        video_preprocessed = self.video_preprocess(video_encoded)  # [B, n_sources, T_v, 1024] ->

        fused_data = self.CAF_block(audio_preprocessed, video_preprocessed)  # [C_a, T_a, F]

        out = self.RTFS_module(fused_data, audio_encoded)  # [C_a, T_a, F]

        return out
