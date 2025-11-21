import torch
from torch import nn

from .modules.decoder import Decoder
from .modules.encoder import Encoder
from .modules.latent import LatentModule


class DTTNetModel(nn.Module):
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
    ):
        super().__init__()
        self.n_sources = n_sources
        self.n_fft = n_fft
        self.hop_length = hop_length
        fc_dim = n_fft // 2 + 1

        self.encoder = Encoder(
            fc_dim,
            in_channels=2,
            out_channels=g,
            n_layers=n_layers,
            use_checkpoints=use_checkpoints,
        )

        self.decoder = Decoder(
            fc_dim,
            out_channels=g,
            n_sources=n_sources,
            n_layers=n_layers,
            use_checkpoints=use_checkpoints,
        )

        self.latent = LatentModule(
            fc_dim=(fc_dim + 2**n_layers) // 2**n_layers,
            n_heads=n_heads,
            n_channels=g * 2**n_layers,
            n_layers=n_idp_layers,
        )

    def forward(self, spectrogram, phase, audio_length, **batch):
        x, skip_results = self.encoder(spectrogram, phase)
        x = self.latent(x)
        x = self.decoder(x, skip_results)

        B, _, F, T = x.shape
        masks = x.view(B, self.n_sources, 2, F, T)

        # [B, n_sources * 2, F, T] -> List[{"audio", "spec", "phase"}]
        outs = []
        for i in range(self.n_sources):
            mask_spec = masks[:, i, 0]  # [B, F, T]
            mask_phase = masks[:, i, 1]  # [B, F, T]

            source_spec, source_phase = self.recon_signal_spectral(
                spectrogram, phase, mask_spec, mask_phase
            )
            source_audio = self.recon_audio(source_spec, source_phase, audio_length)

            outs.append(
                {
                    "audio": source_audio,
                    "spectrogram": source_spec,
                    "phase": source_phase,
                }
            )

        return {"preds": outs}

    def recon_signal_spectral(self, mix_spec, mix_phase, mask_spec, mask_phase):
        source_spec = (mix_spec * mask_spec) - (mix_phase * mask_phase)
        source_phase = (mix_spec * mask_phase) + (mix_phase * mask_spec)
        return source_spec, source_phase

    def recon_audio(self, source_spec, source_phase, audio_length):
        complex_spec = torch.polar(source_spec, source_phase)
        source_audio = torch.istft(
            complex_spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            length=audio_length,
        )
        return source_audio
