import torch
from torch import nn
from itertools import permutations
from src.metrics.base_metric import BaseMetric
from src.metrics.si_snr_i import SiSnrI


class SiSnrIPITWrapperInference(BaseMetric):
    """
    Permutation Invariant Training (PIT) wrapper for SI-SNRi metric used during inference.
    Evaluates all possible source permutations and selects the one
    that yields the highest total SI-SNR improvement (SI-SNRi).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.si_snr_i = SiSnrI()

    def __call__(self, audio, preds, sources, **batch):
        """
        Compute the maximum achievable SI-SNRi by evaluating all permutations
        between predicted and reference sources.

        Args:
            audio (Tensor): Mixture waveform tensor of shape [B, T].
            preds (list[dict]): List of prediction dictionaries,
                each containing 'audio' tensor of shape [B, T].
            sources (list[dict]): List of ground truth source dictionaries,
                each containing 'audio' tensor of shape [B, T].
            **batch: Additional data passed through for compatibility.

        Returns:
            Tensor: Scalar tensor with the mean of the best SI-SNRi values across the batch.
        """
        B = audio.shape[0]
        n_sources = len(sources)
        device = audio.device

        # [n_sources, B, 1, T]
        stacked_sources = torch.stack([s["audio"] for s in sources])
        # [n_sources, B, 1, T]
        stacked_preds = torch.stack([p["audio"] for p in preds])

        perm_indices = list(permutations(range(n_sources), n_sources))
        perm_scores = torch.zeros(len(perm_indices), B, device=device)

        # SiSNRi for each permutation
        for p_idx, perm in enumerate(perm_indices):
            total_score = torch.zeros(B, device=device)
            for src_idx, pred_idx in enumerate(perm):
                true_audio = stacked_sources[src_idx]
                pred_audio = stacked_preds[pred_idx]
                total_score += self.si_snr_i(pred_audio, true_audio, audio)
            perm_scores[p_idx] = total_score

        # max SiSNRi
        best_perm_idx = perm_scores.argmax(dim=0)  # [B]
        best_scores = perm_scores[best_perm_idx, torch.arange(B)]

        return best_scores.mean().cpu() / n_sources
