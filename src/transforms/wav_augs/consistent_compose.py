import torch
from torchvision.transforms.v2 import Compose
from typing import List
from .base_aug import BaseAugmentation


class ConsistentCompose(Compose):
    """
    Compose multiple augmentations and apply them consistently
    across related audio signals (e.g., mixture and sources).
    """

    transforms: List[BaseAugmentation]

    def __init__(self, transforms: List[BaseAugmentation]):
        """
        Initialize the consistent composition of augmentations.

        Args:
            transforms (List[BaseAugmentation]): List of augmentation objects to apply.
        """
        super().__init__(transforms)

    def apply_and_freeze(self, audio, freeze_parameters=False):
        """
        Apply all augmentations sequentially to an audio tensor,
        optionally freezing random parameters for consistency.

        Args:
            audio (Tensor): Input waveform tensor.
            freeze_parameters (bool): Whether to fix augmentation randomness.

        Returns:
            Tensor: Augmented audio tensor.
        """
        augmented = audio
        if freeze_parameters:
            for t in self.transforms:
                if hasattr(t, "unfreeze_parameters"):
                    t.unfreeze_parameters()
                augmented = t(augmented)
                if hasattr(t, "freeze_parameters"):
                    t.freeze_parameters()
        else:
            for t in self.transforms:
                augmented = t(augmented)

        return augmented

    def unfreeze_parameters(self):
        """
        Unfreeze all augmentations inside the composition.
        """
        for t in self.transforms:
            if hasattr(t, "unfreeze_parameters"):
                t.unfreeze_parameters()
