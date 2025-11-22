import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAligner(nn.Module):
    def __init__(self, audio_channels, video_channels):
        super().__init__()
        self.audio_projection = nn.Conv2d(audio_channels, audio_channels, 1)
        self.video_projection = nn.Conv2d(video_channels, audio_channels, 1)

    def forward(self, audio_features, video_features):
        B, C_a, F_a, T_a = audio_features.shape
        _, C_v, emb_dim, T_v = video_features.shape

        video_features = video_features.reshape(B, C_v * emb_dim, 1, T_v)
        video_proj = self.video_projection(video_features)
        video_proj = video_proj.repeat(1, 1, F_a, 1)

        if T_v != T_a:
            video_proj = F.interpolate(video_proj, size=(F_a, T_a), mode='bilinear', align_corners=False)

        audio_proj = self.audio_projection(audio_features)
        combined = torch.cat([audio_proj, video_proj], dim=1)
        return combined
