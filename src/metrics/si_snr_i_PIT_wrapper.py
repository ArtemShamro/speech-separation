import torch

from src.metrics.base_metric import BaseMetric
from src.metrics.si_snr_i import SiSnrI


class SiSnrIPITWrapper(BaseMetric):
    """
    Permutation Invariant Training (PIT) wrapper for the SI-SNRi metric.
    Computes the mean SI-SNR improvement across the best source permutation
    for each sample in the batch.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.si_snr_i = SiSnrI()

    def __call__(self, audio, preds, sources, batch_permuts, **batch):
        """
        Compute the mean SI-SNR improvement (SI-SNRi) across the batch,
        taking into account optimal source permutations determined by PIT.

        Args:
            audio (Tensor): Mixture waveform tensor of shape [B, T].
            preds (list[dict]): List of prediction dictionaries,
                each containing key 'audio' with tensor [B, T].
            sources (list[dict]): List of source dictionaries,
                each containing key 'audio' with tensor [B, T].
            batch_permuts (Tensor): Tensor of shape [B, n_sources]
                containing the optimal permutation indices per sample.
            **batch: Additional data passed through for compatibility.

        Returns:
            Tensor: Scalar tensor with the mean SI-SNRi value over the batch.
        """
        B = audio.shape[0]
        batch_snr_i = torch.zeros(B, device=audio.device)
        stacked_sources = torch.stack([source["audio"] for source in sources])
        for idx, pred in enumerate(preds):
            pred_audio = pred["audio"]
            source_choice = stacked_sources[batch_permuts[:, idx], torch.arange(B), ...]
            source_snr_i = self.si_snr_i(pred_audio, source_choice, audio)
            batch_snr_i += source_snr_i
        return batch_snr_i.detach().cpu().mean()
