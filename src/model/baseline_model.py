import torch
from torch import nn


class BaselineModelBlock(nn.Module):
    """
    A simple feed-forward network block for predicting
    magnitude and phase masks from input spectrogram features.
    """

    def __init__(self, n_feats, fc_hidden):
        """
        Initialize a BaselineModelBlock with two small MLPs
        for spectral and phase feature prediction.

        Args:
            n_feats (int): Number of input features per frame.
            fc_hidden (int): Number of hidden units in the fully connected layers.
        """
        super().__init__()

        self.net_spec = nn.Sequential(
            nn.Linear(n_feats, fc_hidden),
            nn.ReLU(),
            nn.Linear(fc_hidden, n_feats),
        )

        self.net_phase = nn.Sequential(
            nn.Linear(n_feats, fc_hidden),
            nn.ReLU(),
            nn.Linear(fc_hidden, n_feats),
        )

    def forward(self, spectrogram, phase, **batch):
        """
        Forward pass of the block to estimate magnitude and phase masks.

        Args:
            spectrogram (Tensor): Input magnitude spectrogram of shape [B, T, F].
            phase (Tensor): Input phase tensor of shape [B, T, F].
            **batch: Additional data passed through for compatibility.

        Returns:
            tuple(Tensor, Tensor): Predicted magnitude and phase masks, each of shape [B, T, F].
        """
        predicted_spec = self.net_spec(spectrogram)
        predicted_phase = self.net_spec(spectrogram)

        return predicted_spec, predicted_phase


class BaselineModel(nn.Module):
    """
    A simple baseline multi-layer perceptron (MLP) model for
    two-speaker audio source separation.
    """

    def __init__(self, n_feats, fc_hidden=512, sources=2, n_fft=512, hop_length=128):
        """
        Initialize the baseline model with multiple source-specific sub-networks.

        Args:
            n_feats (int): Number of input features per frame.
            fc_hidden (int, optional): Number of hidden units in each subnetwork. Defaults to 512.
            sources (int, optional): Number of target sources to separate. Defaults to 2.
            n_fft (int, optional): FFT window size for reconstruction. Defaults to 512.
            hop_length (int, optional): Hop length for inverse STFT. Defaults to 128.
        """
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length

        self.source_nets = nn.ModuleList(
            [BaselineModelBlock(n_feats, fc_hidden) for _ in range(sources)]
        )

    def forward(self, spectrogram, phase, audio_length, **batch):
        """
        Forward pass for predicting separated sources in waveform and spectral domains.

        Args:
            spectrogram (Tensor): Input magnitude spectrogram of shape [B, F, T].
            phase (Tensor): Input phase tensor of shape [B, F, T].
            audio_length (int): Target length of reconstructed waveform.
            **batch: Additional arguments for compatibility.

        Returns:
            dict: A dictionary with key 'preds' containing a list of source predictions.
                  Each prediction is a dict with keys 'audio', 'spectrogram', and 'phase'.
        """
        out = []
        for source_network in self.source_nets:
            mask_spec, mask_phase = source_network(
                spectrogram.transpose(-1, -2), phase.transpose(-1, -2)
            )
            mask_spec, mask_phase = mask_spec.transpose(-1, -2), mask_phase.transpose(
                -1, -2
            )

            source_spec, source_phase = self.recon_signal_spectral(
                spectrogram, phase, mask_spec, mask_phase
            )

            source_audio = self.recon_audio(source_spec, source_phase, audio_length)

            out.append(
                {
                    "audio": source_audio,
                    "spectrogram": source_spec,
                    "phase": source_phase,
                }
            )

        return {"preds": out}

    def recon_signal_spectral(self, mix_spec, mix_phase, mask_spec, mask_phase):
        """
        Reconstruct the source spectrogram using predicted masks.

        Args:
            mix_spec (Tensor): Input mixture magnitude spectrogram [B, F, T].
            mix_phase (Tensor): Input mixture phase [B, F, T].
            mask_spec (Tensor): Predicted magnitude mask [B, F, T].
            mask_phase (Tensor): Predicted phase mask [B, F, T].

        Returns:
            tuple(Tensor, Tensor): Reconstructed source magnitude and phase tensors.
        """
        source_spec = (mix_spec * mask_spec) - (mix_phase * mask_phase)
        source_phase = (mix_spec * mask_phase) + (mix_phase * mask_spec)
        return source_spec, source_phase

    def recon_audio(self, source_spec, source_phase, audio_length):
        """
        Reconstruct the time-domain waveform from magnitude and phase tensors.

        Args:
            source_spec (Tensor): Source magnitude spectrogram [B, F, T].
            source_phase (Tensor): Source phase tensor [B, F, T].
            audio_length (int): Desired output waveform length.

        Returns:
            Tensor: Time-domain reconstructed waveform of shape [B, T].
        """
        complex_spec = torch.polar(source_spec, source_phase)

        source_audio = torch.istft(
            complex_spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            length=audio_length,
        )

        return source_audio

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info
