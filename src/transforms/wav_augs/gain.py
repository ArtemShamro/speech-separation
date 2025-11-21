import torch_audiomentations
from torch import Tensor

from .base_aug import BaseAugmentation


class Gain(BaseAugmentation):
    """
    Randomly change the amplitude (gain) of the input signal.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the gain augmentation.

        Args:
            *args, **kwargs: Parameters for torch_audiomentations.Gain.
        """
        super().__init__()
        self._aug = torch_audiomentations.Gain(*args, **kwargs)

    def __call__(self, data: Tensor):
        """
        Apply random gain to the input audio.

        Args:
            data (Tensor): Input waveform tensor [B, T].

        Returns:
            Tensor: Augmented waveform tensor [B, T].
        """
        x = data.unsqueeze(1)
        return self._aug(x).squeeze(1)
