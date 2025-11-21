from src.metrics.base_metric import BaseMetric, PIT_wrapper_training, PIT_wrapper_inference
from src.metrics.stoi import StoiMetric


class StoiPITTraining(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stoi = StoiMetric(*args, **kwargs)

    def __call__(self, audio, preds, sources, batch_permuts, **batch):
        return PIT_wrapper_training(self.stoi, audio, preds, sources, batch_permuts, **batch)


class StoiPITInference(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stoi = StoiMetric(*args, **kwargs)

    def __call__(self, audio, preds, sources, **batch):
        return PIT_wrapper_inference(self.stoi, audio, preds, sources, **batch)

