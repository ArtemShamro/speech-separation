import torch
from torch import nn
from itertools import permutations
from hydra.utils import instantiate


class PITLossWrapper(nn.Module):
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
        perm_set = list(range(n_sources))
        permuts_sources = list(permutations(perm_set, n_sources))
        batch_losses = []

        for permut in permuts_sources:
            permut_loss = torch.zeros(audio.shape[0], device=audio.device)
            for idx, pred in enumerate(preds):
                source = sources[permut[idx]]
                batch_loss = self.loss_function(
                    audio_pred=pred["audio"],
                    spectrogram_pred=pred["spectrogram"],
                    phase_pred=pred["phase"],
                    audio_true=source["audio"],
                    spectrogram_true=source["spectrogram"],
                    phase_true=source["phase"],
                )
                permut_loss += batch_loss

            batch_losses.append(permut_loss)

        batch_losses = torch.stack(batch_losses)
        loss_values, batch_perm_idx = torch.min(batch_losses, dim=0)

        batch_permuts = [permuts_sources[idx] for idx in batch_perm_idx]
        batch_permuts = torch.tensor(batch_permuts, device=audio.device, dtype=torch.long)

        return {
            "loss": loss_values.mean(),
            "batch_permuts": batch_permuts,
        }
