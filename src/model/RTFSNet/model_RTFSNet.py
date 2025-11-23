import torch
from torch import nn, Tensor
from torch.nn import functional as F
from .modules.video_encoder import VideoEncoderModule
from .modules.audio_encoder import AudioEncoderModule
from .modules.separation_network import SeparationNetwork
from .modules.S3_module import S3Block
from .modules.audio_decoder import AudioDecoderModule


class RTFSNetModel(nn.Module):
    def __init__(
        self,
        c_a=32,
        n_sources=2,
        n_fft=512,
        hop_length=128,
        video_encoder_weights_path: str = "/media/atem/Data/HSE_videos/4_DLA/hw_2_SeppechSep/git_speech_separation/src/model/lipreading/lrw_snv1x_dsmstcn3x.pth",  # modek name from repo
        use_checkpoints=False,
    ):
        """
        inputs:
        c_a - C_a - audio encoder out channels
        """
        super().__init__()

        self.n_sources = n_sources
        self.n_fft = n_fft
        self.hop_length = hop_length
        fc_dim = n_fft // 2 + 1

        self.video_encoder = VideoEncoderModule(
            load_weights_path=video_encoder_weights_path)

        self.audio_encoder = AudioEncoderModule(
            in_channels=2,
            out_channels=c_a,
        )  # just convolution

        self.separation_network = SeparationNetwork()

        self.s3_block = S3Block()

        self.audio_decoder = AudioDecoderModule()

    def forward(self, spectrogram, phase, audio_length, video, **batch):

        video_encoded = self.video_encoder(video)  # [B, n_sources, T_v, 1024]

        audio_encoded = self.audio_encoder(spectrogram, phase)  # [C_a, T_a, F]

        out = self.separation_network(audio_encoded, video_encoded)  # [C_a, T_a, F]

        out = self.s3_block(fused_data=out,
                            audio_encoded=audio_encoded)

        out = self.audio_decoder(out)

        return out
