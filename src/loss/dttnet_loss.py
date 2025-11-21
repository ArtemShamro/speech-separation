import torch
from torch import nn


class DTTNetLoss(nn.Module):
    """
    Compute the DTTNet loss, combining waveform and spectrogram reconstruction errors.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        audio_true,
        spectrogram_true,
        phase_true,
        audio_pred,
        spectrogram_pred,
        phase_pred,
    ):
        """
        Compute the loss for audio and spectrogram reconstruction.

        Args:
            audio_true (Tensor): Ground truth waveform tensor of shape [B, T].
            spectrogram_true (Tensor): Ground truth magnitude spectrogram tensor of shape [B, F, T].
            phase_true (Tensor): Ground truth phase tensor of shape [B, F, T].
            audio_pred (Tensor): Predicted waveform tensor of shape [B, T].
            spectrogram_pred (Tensor): Predicted magnitude spectrogram tensor of shape [B, F, T].
            phase_pred (Tensor): Predicted phase tensor of shape [B, F, T].

        Returns:
            Tensor: Per-sample loss tensor of shape [B], combining spectrum and waveform losses.
        """
        loss_spectrum = torch.abs(spectrogram_true - spectrogram_pred) + torch.abs(
            phase_true - phase_pred
        )

        loss_wave = torch.abs(audio_true - audio_pred)

        batch_loss = loss_spectrum.mean(dim=(1, 2)) + loss_wave.mean(dim=(-1))
        return batch_loss  # [B, 1]
