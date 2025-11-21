import torch
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality

from src.metrics.base_metric import BaseMetric


class PesqMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pesq_metric = PerceptualEvaluationSpeechQuality(16000, "wb")

    def __call__(self, preds, target, mix_audio=None):
        preds = preds.detach()
        target = target.detach()
        scores = []

        for i in range(preds.shape[0]):
            scores.append(self.pesq_metric(preds[i], target[i]))

        return torch.tensor(scores, device=preds.device)
