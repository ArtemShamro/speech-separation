import random

import torch
from torchaudio.transforms import Speed

from .base_aug import BaseAugmentation


class SpeedPerturb(BaseAugmentation):
    """
    Apply random speed perturbation to an audio signal by resampling.
    """

    def __init__(self, speeds=[0.9, 1.0, 1.1], sample_rate=16000, p=0.5):
        """
        Initialize the speed perturbation module.

        Args:
            speeds (list[float]): List of possible speed factors.
            sample_rate (int): Sampling rate of input audio.
            p (float): Probability of applying speed perturbation.
        """
        super().__init__()
        self.speeds = [Speed(sample_rate, factor) for factor in speeds]
        self.p = p
        self.are_parameters_frozen = False

        self._should_apply = None
        self._speed_idx = None

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """
        Apply random speed perturbation to the input audio.

        Args:
            data (Tensor): Input waveform tensor [B, T].

        Returns:
            Tensor: Perturbed waveform tensor [B, T].
        """
        if not self.are_parameters_frozen:
            self._should_apply = random.random() <= self.p
            self._speed_idx = random.randint(0, len(self.speeds) - 1)

        if not self._should_apply:
            return data

        if self._speed_idx is None or self._should_apply is None:
            raise Exception()
        resampled, _ = self.speeds[self._speed_idx](data)
        return resampled

    def freeze_parameters(self):
        """
        Freeze current random speed parameters (apply same effect consistently).
        """
        if not self.are_parameters_frozen:
            self.are_parameters_frozen = True

    def unfreeze_parameters(self):
        """
        Allow parameters to randomize again.
        """
        self.are_parameters_frozen = False
        self._should_apply = None
        self._speed_idx = None
