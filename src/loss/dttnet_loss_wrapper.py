import torch
from torch import nn
from itertools import permutations
from hydra.utils import instantiate


class DTTNetLossWrapper(nn.Module):
    """
    Permutation Invariant Training (PIT) wrapper for source separation losses.
    It computes all possible permutations of source-target pairs and selects
    the one yielding the minimum total loss per batch.
    """

    def __init__(self, loss_function):
        """
        Initialize the PIT loss wrapper.

        Args:
            loss_function (nn.Module): Base loss function used to compute pairwise source losses.
        """
        super().__init__()
        self.loss_function = loss_function

    def forward(self, audio, sources, preds, **batch):
        """
        Compute the permutation invariant loss across all source-target orderings.

        Args:
            audio (Tensor): Input mixture waveform tensor of shape [B, T].
            sources (list[dict]): List of dictionaries with ground truth source data
                (each containing 'audio', 'spectrogram', 'phase' tensors).
            preds (list[dict]): List of dictionaries with model predictions
                (each containing 'audio', 'spectrogram', 'phase' tensors).
            **batch: Additional data passed through for compatibility.

        Returns:
            dict:
                - loss (Tensor): Mean permutation invariant loss over the batch.
                - batch_permuts (Tensor): Tensor of shape [B, n_sources] with optimal source permutations.
        """
        n_sources = len(sources)

        loss_values = torch.zeros(audio.shape[0], device=audio.device)
        for idx, pred in enumerate(preds):
            source = sources[idx]
            batch_loss = self.loss_function(
                audio_pred=pred["audio"],
                spectrogram_pred=pred["spectrogram"],
                phase_pred=pred["phase"],
                audio_true=source["audio"],
                spectrogram_true=source["spectrogram"],
                phase_true=source["phase"],
            )
            loss_values += batch_loss

        return {
            "loss": loss_values.mean(),
            "batch_permuts": torch.arange(n_sources, device=audio.device, dtype=torch.long),
        }
