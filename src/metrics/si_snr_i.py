import torch
from torch import nn
from src.metrics.base_metric import BaseMetric
from src.metrics.si_snr import SiSnr


class SiSnrI(BaseMetric):
    """
    Compute the improvement in Scale-Invariant Signal-to-Noise Ratio (SI-SNRi)
    between the separated signal and the input mixture.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.si_snr = SiSnr(*args, **kwargs)

    def __call__(self, pred_audio_batch, true_audio_batch, mix_audio_batch):
        """
        Compute the SI-SNR improvement (SI-SNRi) for each sample in the batch.

        Args:
            pred_audio_batch (Tensor): Predicted separated audio batch of shape [B, T].
            true_audio_batch (Tensor): Ground truth clean audio batch of shape [B, T].
            mix_audio_batch (Tensor): Mixture audio batch of shape [B, T].

        Returns:
            Tensor: SI-SNR improvement values per sample, shape [B].
        """
        metric = self.si_snr(pred_audio_batch, true_audio_batch) - \
            self.si_snr(mix_audio_batch, true_audio_batch)
        return metric
