import torch
import torch.nn.functional as F
from torch import nn

from src.model.DTTwRTFSNet.layers.conv import Conv


class CAFBlock(nn.Module):
    def __init__(self, audio_dim, video_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.audio_dim = audio_dim
        
        self.a_gate_op = nn.Sequential(
            nn.Conv2d(audio_dim, audio_dim, 1, groups=audio_dim),
            nn.BatchNorm2d(audio_dim),
            nn.ReLU()
        )
        
        self.a_val_op = nn.Sequential(
            nn.Conv2d(audio_dim, audio_dim, 1, groups=audio_dim),
            nn.BatchNorm2d(audio_dim)
        )

        self.v_attn_conv = Conv(video_dim, audio_dim * num_heads, 1, groups=1, dim=1)
        self.v_key_conv = Conv(video_dim, audio_dim, 1, groups=1, dim=1)

    def forward(self, audio_feat, video_feat):
        batch, c_a, t_a, freq = audio_feat.shape
        
        a_value = self.a_val_op(audio_feat)
        a_gating = self.a_gate_op(audio_feat)
        
        v_key = self.v_key_conv(video_feat)
        v_key = F.interpolate(v_key, size=t_a, mode="nearest")
        
        v_raw_attn = self.v_attn_conv(video_feat) # (B, C_a * Heads, T_v)
        
        v_raw_attn = v_raw_attn.view(batch, c_a, self.num_heads, -1)
        v_mean_attn = torch.mean(v_raw_attn, dim=2) 
        
        v_attn_probs = F.softmax(v_mean_attn, dim=-1)
        v_attn_probs = F.interpolate(v_attn_probs, size=t_a, mode="nearest")

        term1 = v_attn_probs.unsqueeze(-1) * a_value
        term2 = v_key.unsqueeze(-1) * a_gating
        
        return term1 + term2
    