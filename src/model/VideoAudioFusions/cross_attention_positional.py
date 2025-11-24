import math
import torch
import torch.nn as nn


def _build_sinusoids(length: int, dim: int, device: torch.device):
    position = torch.arange(length, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, device=device) * (-math.log(10000.0) / dim))
    pe = torch.zeros(length, dim, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class CrossAttentionPositional(nn.Module):
    def __init__(self, audio_channels: int, video_channels: int, n_heads: int = 2):
        super().__init__()
        self.audio_channels = audio_channels
        self.video_channels = video_channels

        self.video_proj = nn.Sequential(
            nn.Linear(video_channels, audio_channels),
            nn.GELU(),
        )

        self.q_audio = nn.Linear(audio_channels, audio_channels)
        self.k_video = nn.Linear(audio_channels, audio_channels)
        self.v_video = nn.Linear(audio_channels, audio_channels)

        self.inp_layernorm = nn.LayerNorm(audio_channels)
        self.attn_layernorm = nn.LayerNorm(audio_channels)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=audio_channels,
            num_heads=n_heads,
            dropout=0.0,
            batch_first=True,
        )

        self.cross_ffn = nn.Sequential(
            nn.Linear(audio_channels, 4 * audio_channels),
            nn.GELU(),
            nn.Linear(4 * audio_channels, audio_channels),
        )

        self.audio_gate = nn.Sequential(
            nn.Linear(audio_channels, audio_channels),
            nn.Sigmoid(),
        )

    def forward(self, audio_features: torch.Tensor, video_features: torch.Tensor) -> torch.Tensor:
        B, C_a, F_a, T_a = audio_features.shape
        B, C_v, E_v, T_v = video_features.shape

        audio_tokens = audio_features.mean(dim=2).transpose(1, 2)

        video_tokens = video_features.permute(0, 3, 1, 2).reshape(B, T_v, C_v * E_v)
        video_tokens = self.video_proj(video_tokens)

        pe_audio = _build_sinusoids(T_a, C_a, audio_tokens.device)
        pe_video = _build_sinusoids(T_v, C_a, video_tokens.device)
        audio_tokens = audio_tokens + pe_audio.unsqueeze(0)
        video_tokens = video_tokens + pe_video.unsqueeze(0)

        q = self.q_audio(self.inp_layernorm(audio_tokens))
        k = self.k_video(self.inp_layernorm(video_tokens))
        v = self.v_video(self.inp_layernorm(video_tokens))

        attn_out, _ = self.cross_attn(q, k, v)
        audio_tokens = self.attn_layernorm(audio_tokens + attn_out)

        audio_tokens = audio_tokens + self.cross_ffn(audio_tokens)

        gate = self.audio_gate(audio_tokens)
        audio_tokens = audio_tokens * gate

        audio_enh = audio_tokens.transpose(1, 2).unsqueeze(2).expand(B, C_a, F_a, T_a)
        fused = torch.cat([audio_features, audio_enh], dim=1)

        return fused
