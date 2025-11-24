from abc import abstractmethod
from itertools import permutations

import torch


class BaseMetric:
    """
    Base class for all metrics
    """

    def __init__(self, name=None, *args, **kwargs):
        """
        Args:
            name (str | None): metric name to use in logger and writer.
        """
        self.name = name if name is not None else type(self).__name__

    @abstractmethod
    def __call__(self, **batch):
        """
        Defines metric calculation logic for a given batch.
        Can use external functions (like TorchMetrics) or custom ones.
        """
        raise NotImplementedError()


def PIT_wrapper_training(metric_fn, audio, preds, sources, batch_permuts, **batch):
    B = audio.shape[0]
    n_sources = len(sources)
    device = audio.device

    batch_scores = torch.zeros(B, device=device)
    stacked_sources = torch.stack([source["audio"] for source in sources])

    for idx, pred in enumerate(preds):
        pred_audio = pred["audio"]
        source_choice = stacked_sources[batch_permuts[:, idx], torch.arange(B), ...]
        source_score = metric_fn(pred_audio, source_choice, audio)
        batch_scores += source_score

    return (batch_scores / n_sources).detach().cpu().mean()


def PIT_wrapper_inference(metric_fn, audio, preds, sources, **batch):
    B = audio.shape[0]
    n_sources = len(sources)
    device = audio.device

    stacked_sources = torch.stack([s["audio"] for s in sources])
    stacked_preds = torch.stack([p["audio"] for p in preds])

    perm_indices = list(permutations(range(n_sources), n_sources))
    perm_scores = torch.zeros(len(perm_indices), B, device=device)

    for p_idx, perm in enumerate(perm_indices):
        total_score = torch.zeros(B, device=device)
        for src_idx, pred_idx in enumerate(perm):
            true_audio = stacked_sources[src_idx]
            pred_audio = stacked_preds[pred_idx]
            total_score += metric_fn(pred_audio, true_audio, audio)
        perm_scores[p_idx] = total_score

    best_perm_idx = perm_scores.argmax(dim=0)
    best_scores = perm_scores[best_perm_idx, torch.arange(B)]

    return (best_scores / n_sources).mean().cpu()
