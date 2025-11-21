import torch
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility

from src.metrics.base_metric import BaseMetric


class StoiMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stoi_metric = ShortTimeObjectiveIntelligibility(16000)

    def __call__(self, preds, target, mix_audio=None):
        preds = preds.detach()
        target = target.detach()
        scores = []

        for i in range(preds.shape[0]):
            scores.append(self.stoi_metric(preds[i], target[i]))

        return torch.tensor(scores, device=preds.device)
