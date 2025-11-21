from src.metrics.base_metric import (
    BaseMetric,
    PIT_wrapper_inference,
    PIT_wrapper_training,
)
from src.metrics.pesq import PesqMetric


class PesqPITTraining(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pesq = PesqMetric(*args, **kwargs)

    def __call__(self, audio, preds, sources, batch_permuts, **batch):
        return PIT_wrapper_training(
            self.pesq, audio, preds, sources, batch_permuts, **batch
        )


class PesqPITInference(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pesq = PesqMetric(*args, **kwargs)

    def __call__(self, audio, preds, sources, **batch):
        return PIT_wrapper_inference(self.pesq, audio, preds, sources, **batch)
