import torch
from torch import nn
from src.metrics.base_metric import BaseMetric


class SiSnr(BaseMetric):
    """
    Compute the Scale-Invariant Signal-to-Noise Ratio (SI-SNR) metric.
    """

    def __init__(self):
        super().__init__()
        self.eps = 1e-5

    def __call__(self, pred_audio_batch, true_audio_batch):
        """
        Compute the SI-SNR between predicted and reference audio signals.

        Args:
            pred_audio_batch (Tensor): Predicted audio batch of shape [B, T].
            true_audio_batch (Tensor): Ground truth audio batch of shape [B, T].

        Returns:
            Tensor: SI-SNR values for each sample in the batch, shape [B].
        """
        s_target = ((pred_audio_batch * true_audio_batch).sum(dim=1, keepdim=True)
                    / (true_audio_batch.pow(2).sum(dim=1, keepdim=True) + self.eps)) * true_audio_batch

        e_noise = pred_audio_batch - s_target

        ratio = (s_target.pow(2).sum(dim=1)) / (e_noise.pow(2).sum(dim=1) + self.eps)
        metric = 10 * torch.log10(ratio)

        return metric
