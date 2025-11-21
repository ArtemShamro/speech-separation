import torch_audiomentations
from torch import Tensor

from .base_aug import BaseAugmentation


class AddColoredNoise(BaseAugmentation):
    """
    Add colored noise (e.g., white, pink, brown) to the input signal.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the colored noise augmentation.

        Args:
            *args, **kwargs: Parameters for torch_audiomentations.AddColoredNoise.
        """
        super().__init__()
        self._aug = torch_audiomentations.AddColoredNoise(*args, **kwargs)

    def __call__(self, data: Tensor) -> Tensor:
        """
        Apply colored noise to the input audio tensor.

        Args:
            data (Tensor): Input waveform tensor of shape [B, T].

        Returns:
            Tensor: Augmented waveform tensor of shape [B, T].
        """
        x = data.unsqueeze(1)
        return self._aug(x).squeeze(1)
