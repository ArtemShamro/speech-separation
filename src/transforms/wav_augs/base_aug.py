from torch import nn


class BaseAugmentation(nn.Module):
    """
    Base class for all audio augmentations.
    Wraps torch_audiomentations augmentations with a consistent interface.
    """

    def __init__(self):
        super().__init__()
        self._aug = None

    def freeze_parameters(self):
        """
        Freeze internal random parameters (if supported by the augmentation).
        This allows applying the same augmentation to multiple audio samples.
        """
        if hasattr(self._aug, "freeze_parameters"):
            self._aug.freeze_parameters()

    def unfreeze_parameters(self):
        """
        Unfreeze internal random parameters (if supported by the augmentation).
        Allows generating new random parameters on the next call.
        """
        if hasattr(self._aug, "unfreeze_parameters"):
            self._aug.unfreeze_parameters()
