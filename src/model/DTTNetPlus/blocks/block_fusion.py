import torch
from torch import nn
from torch.nn import functional as Fun


class FusionBlock(nn.Module):
    def __init__(self, audio_channels, audio_freq, video_channels=256):
        super().__init__()

        self.w_gamma_channels = nn.Linear(video_channels, audio_channels)
        self.w_beta_channels = nn.Linear(video_channels, audio_channels)

        self.w_gamma_freq = nn.Linear(video_channels, audio_freq)
        self.w_beta_freq = nn.Linear(video_channels, audio_freq)

    def forward(self, audio_data: torch.Tensor, video_data: torch.Tensor):
        B, C, F, T = audio_data.shape
        video_interpolated = Fun.interpolate(
            video_data.transpose(-1, -2), size=T, mode="linear").transpose(-1, -2)

        gamma_c = self.w_gamma_channels(video_interpolated)
        beta_c = self.w_beta_channels(video_interpolated)

        gamma_f = self.w_gamma_freq(video_interpolated)
        beta_f = self.w_beta_freq(video_interpolated)

        gamma_c = gamma_c.permute(0, 2, 1).unsqueeze(2)  # [B, C, 1, T]
        gamma_f = gamma_f.permute(0, 2, 1).unsqueeze(1)  # [B, 1, F, T]
        beta_c = beta_c.permute(0, 2, 1).unsqueeze(2)  # [B, C, 1, T]
        beta_f = beta_f.permute(0, 2, 1).unsqueeze(1)  # [B, 1, F, T]

        gamma = gamma_c + gamma_f   # [B, C, F, T]
        beta = beta_c + beta_f  # [B, C, F, T]

        fused_signal = (1 + gamma) * audio_data + beta

        return fused_signal
