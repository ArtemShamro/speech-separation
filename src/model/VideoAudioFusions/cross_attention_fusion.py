import torch
import torch.nn as nn


class CrossAttentionFusion(nn.Module):
    def __init__(self, audio_channels: int, video_channels: int, n_heads: int = 2):
        super().__init__()
        self.audio_channels = audio_channels
        self.video_channels = video_channels

        self.video_proj = nn.Linear(video_channels, audio_channels)
        self.cross_q_norm = nn.LayerNorm(audio_channels)
        self.cross_kv_norm = nn.LayerNorm(audio_channels)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=audio_channels,
            num_heads=n_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.cross_ffn_norm = nn.LayerNorm(audio_channels)
        self.cross_ffn = nn.Sequential(
            nn.Linear(audio_channels, 4 * audio_channels),
            nn.GELU(),
            nn.Linear(4 * audio_channels, audio_channels),
        )

    def forward(self, audio_features: torch.Tensor, video_features: torch.Tensor) -> torch.Tensor:
        B, C_a, F_a, T_a = audio_features.shape
        audio_tokens = audio_features.flatten(2).transpose(1, 2)

        B, C_v, E_v, T_v = video_features.shape
        video_tokens = video_features.permute(0, 3, 1, 2).reshape(B, T_v, C_v * E_v)
        video_tokens = self.video_proj(video_tokens)

        q = self.cross_q_norm(audio_tokens)
        kv = self.cross_kv_norm(video_tokens)
        attn_out, _ = self.cross_attn(q, kv, kv)
        audio_tokens = audio_tokens + attn_out
        audio_tokens = audio_tokens + self.cross_ffn(self.cross_ffn_norm(audio_tokens))

        audio_enh = audio_tokens.transpose(1, 2).reshape(B, C_a, F_a, T_a)

        fused = torch.cat([audio_features, audio_enh], dim=1)

        return fused
