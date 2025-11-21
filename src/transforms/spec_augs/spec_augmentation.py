import torch
import torch.nn as nn
import torchlibrosa.augmentation as A
import random


class SpecAugmentation(nn.Module):
    def __init__(self, p, *args, **kwargs):
        super().__init__()
        self.p = p
        self.spec_aug = A.SpecAugmentation(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return x
        x = x.unsqueeze(1)
        return self.spec_aug(x).squeeze(1)
