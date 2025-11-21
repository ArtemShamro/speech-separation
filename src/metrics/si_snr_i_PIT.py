from src.metrics.base_metric import BaseMetric, PIT_wrapper_training, PIT_wrapper_inference
from src.metrics.si_snr_i import SiSnrI


class SiSnrIPITTraining(BaseMetric):
    """
    Permutation Invariant Training (PIT) wrapper for the SI-SNRi metric.
    Computes the mean SI-SNR improvement across the best source permutation
    for each sample in the batch.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.si_snr_i = SiSnrI(*args, **kwargs)

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
        return PIT_wrapper_training(self.si_snr_i, audio, preds, sources, batch_permuts, **batch)


class SiSnrIPITInference(BaseMetric):
    """
    Permutation Invariant Training (PIT) wrapper for SI-SNRi metric used during inference.
    Evaluates all possible source permutations and selects the one
    that yields the highest total SI-SNR improvement (SI-SNRi).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.si_snr_i = SiSnrI(*args, **kwargs)

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
        return PIT_wrapper_inference(self.si_snr_i, audio, preds, sources, **batch)
