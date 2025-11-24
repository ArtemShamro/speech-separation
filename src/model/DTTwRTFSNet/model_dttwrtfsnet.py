import torch
import torch.nn as nn

from src.model.DTTNet.model_dttnet import DTTNetModel
from src.model.VideoEncoders.resnet_video_encoder import VideoEncoder
from src.model.DTTwRTFSNet.blocks.vp_block import VPBlock
from src.model.DTTwRTFSNet.blocks.caf_block import CAFBlock


class DTTwRTFSNet(DTTNetModel):
    def __init__(
        self,
        g=32,
        n_sources=2,
        n_layers=3,
        n_idp_layers=3,
        n_fft=512,
        hop_length=128,
        n_heads=2,
        use_checkpoints=False,
        q_video=4,
        video_embed_dim=512,
        hidden_dim=64,
        caf_n_heads=4,
    ):
        super().__init__(
            g=g,
            n_sources=n_sources,
            n_layers=n_layers,
            n_idp_layers=n_idp_layers,
            n_fft=n_fft,
            hop_length=hop_length,
            n_heads=n_heads,
            use_checkpoints=use_checkpoints,
        )

        self.video_encoder = VideoEncoder()
        self.video_encoder.eval().requires_grad_(False)
        self.vp_block = VPBlock(q_video, video_embed_dim, hidden_dim)
        audio_channels = g * 2 ** n_layers
        self.caf_block = CAFBlock(audio_channels, video_embed_dim, caf_n_heads)
            

    def forward(self, spectrogram, phase, audio_length, video, **batch):
        audio_embed, skip_results = self.encoder(spectrogram, phase)
        audio_embed = audio_embed.transpose(2, 3) # (B, g*2^n_layers, T/8, F/8)

        video_embed = self.video_encoder(video) # (B, 2, embed_dim, T')
        B, N, C, T = video_embed.size()
        video_embed = video_embed.view(B * N, C, T)
        video_embed = self.vp_block(video_embed) # (B*N, C, T)
        video_embed = video_embed.view(B, N, C, T).mean(dim=1) # (B, C, T)

        fused = self.caf_block(audio_embed, video_embed) # (B, C_a, T_a, F_a)
        fused = fused.transpose(2, 3) # (B, C_a, F_a, T_a)

        latent_out = self.latent(fused)
        x = self.decoder(latent_out, skip_results)

        return self.postprocess(x, spectrogram, phase, audio_length)
