from src.metrics.base_metric import (
    BaseMetric,
    PIT_wrapper_inference,
    PIT_wrapper_training,
)
from src.metrics.si_snr import SiSnr


class SiSnrPITTraining(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.si_snr = SiSnr()

    def __call__(self, audio, preds, sources, batch_permuts, **batch):
        return PIT_wrapper_training(
            self.si_snr, audio, preds, sources, batch_permuts, **batch
        )


class SiSnrPITInference(BaseMetric):
    """
    Permutation Invariant Training (PIT) wrapper for SI-SNRi metric used during inference.
    Evaluates all possible source permutations and selects the one
    that yields the highest total SI-SNR improvement (SI-SNRi).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.si_snr = SiSnr(*args, **kwargs)

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
        return PIT_wrapper_inference(self.si_snr, audio, preds, sources, **batch)
