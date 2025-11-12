import torch
from torch import nn, Tensor
from torch.nn import functional as F
from src.model.lipreading.model import Lipreading
import os


class VideoEncoderModule(nn.Module):
    def __init__(self, load_weights_path=None):
        super().__init__()
        tcn_options = {
            'num_layers': 4,
            'kernel_size': [
                3,
                5,
                7
            ],
            'dropout': 0.2,
            'dwpw': True,
            'width_mult': 1,
        }
        self.encoder = Lipreading(modality="video",
                                  num_classes=500,
                                  tcn_options=tcn_options,
                                  densetcn_options={},
                                  backbone_type="shufflenet",
                                  relu_type="relu",
                                  width_mult=1.0,
                                  use_boundary=False,
                                  extract_feats=True
                                  )

        if load_weights_path is not None:
            self._load_weights(load_weights_path)

    def forward(self, x):
        return self.encoder(x)

    def _load_weights(self, load_path):
        assert os.path.isfile(
            load_path), "Error when loading the model, provided path not found: {}".format(load_path)
        checkpoint = torch.load(load_path)
        loaded_state_dict = checkpoint['model_state_dict']

        # -- copy loaded state into current model and, optionally, optimizer
        self.encoder.load_state_dict(loaded_state_dict)
