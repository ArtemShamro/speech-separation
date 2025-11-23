import torch
from torch import nn, Tensor
from torch.nn import functional as Fun


class RTFSBlock(nn.Module):
    def __init__(self, F_dim, c_a, d, q=3):
        """
        c_a : int - input dim
        d : int - inner dim
        q : int - layers in compression and decompression units
        """
        super().__init__()

        self.compression_unit = CompressionUnit(c_a, d, q)

        self.freq_process = OnePathUnit(input_channels=d)
        self.time_process = OnePathUnit(input_channels=d)

        self.time_freq_process = TFAttention(
            D_dim=d,
            E_dim=128,
            F_dim=F_dim,
            n_heads=8,
        )

        self.recon_unit = ReconstructionUnit()

    def forward(self, audio_batch):
        # audio batch shape is B, C_a, F, T_a
        x = self.compression_unit(audio_batch)  # B, D, F', T_a'

        x = self.freq_process(x)

        x = self.time_process(x.transpose(-1, -2)).transpose(-1, -2)

        x = self.time_freq_process(x)

        x = self.recon_unit(x)

        return x


class CompressionUnit(nn.Module):
    def __init__(self, c_a, d, q=3):
        super().__init__()

        self.D = d
        self.q = q

        self.init_conv = nn.Conv2d(in_channels=c_a, out_channels=d, kernel_size=(1, 1))

        self.compression_layers = nn.ModuleList([
            nn.Conv2d(in_channels=d, out_channels=d, kernel_size=(4, 4), stride=2) for _ in range(q)
        ])

    def forward(self, audio_batch):
        x = self.init_conv(audio_batch)
        B, C, F, T = audio_batch.shape
        output_size = (F // 2 ** (self.q - 1), T // 2 ** (self.q - 1))
        compression_outs = [Fun.adaptive_avg_pool2d(x, output_size)]
        for layer in self.compression_layers:
            x = layer(x)
            compression_outs.append(Fun.adaptive_avg_pool2d(x, output_size))

        out = torch.stack(compression_outs).sum(dim=0)

        return out


class OnePathUnit(nn.Module):
    def __init__(self, input_channels, kernel_size=8):
        super().__init__()
        self.kernel_size = kernel_size
        self.rnn_inp_dim = kernel_size * input_channels
        self.rnn_hid_dim = self.rnn_inp_dim * 2
        self.rnn_out_dim = self.rnn_hid_dim * 2

        self.norm = nn.LayerNorm(kernel_size * input_channels)

        self.conv_transpose = nn.ConvTranspose2d(
            self.rnn_out_dim, input_channels, (kernel_size, 1))

        self.rnn = RnnUnit(
            input_size=self.rnn_inp_dim,
            hidden_size=self.rnn_hid_dim,
        )

    def forward(self, x: Tensor):
        B, D, F, T = x.shape
        F_hat = F - self.kernel_size + 1
        # compressed x shape [B, D, F', T']
        skip = x
        out = self._unfold(x)  # [B * T', F', 8*D]
        out = self.norm(out)
        out = self.rnn(out)  # [B * T', F', 2 (bidirectional) * 2(scale hidden dim) * 8 * D]
        out = out.view(B, T, F_hat, self.rnn_out_dim).permute(0, 3, 2, 1)
        out = self.conv_transpose(out)  # [B, D, F, T]
        out += skip
        # permute freq and time and do it again
        return out

    def _unfold(self, x: Tensor):
        B, D, F, T = x.shape
        D_hat = self.kernel_size * D
        x = torch.nn.functional.unfold(x, (self.kernel_size, 1))  # [B, 8 * D, F_hat]
        F_hat = F - self.kernel_size + 1
        x = x.view(B, D_hat, F_hat, T).permute(0, 3, 2, 1).reshape(B * T, F_hat, D_hat)
        return x


class RnnUnit(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x):
        out, _ = self.rnn(x)
        return out


class TFAttention(nn.Module):
    def __init__(self, D_dim, E_dim, F_dim, n_heads):
        """
        D_dim - input channels
        E_dim - attention Q, K dimension
        F_dim - freq dim
        """
        super().__init__()
        assert D_dim % n_heads == 0, "TFAttention : D_dim % n_heads != 0"
        self.D_dim = D_dim
        self.F_dim = F_dim
        self.E_dim = E_dim
        self.V_dim = D_dim // n_heads
        self.n_heads = n_heads

        #
        self.conv = nn.Conv2d(D_dim, (E_dim * 2 + self.V_dim) * n_heads, 1).cuda()
        self.prelu = torch.nn.PReLU()
        self.norm_q = torch.nn.LayerNorm(E_dim * F_dim)
        self.norm_k = torch.nn.LayerNorm(E_dim * F_dim)
        self.norm_v = torch.nn.LayerNorm(self.V_dim * F_dim)

    def forward(self, x: Tensor):
        # x shape is [B, D, F, T]
        n_batch, _, F, T = x.shape
        skip = x
        # prepare Q, K, V by conv
        x = self.conv(x)

        # split Q, K, V
        q = x[:, : self.E_dim * self.n_heads, ...].unflatten(1, (self.n_heads, -1))
        k = x[:, self.E_dim * self.n_heads: 2 * self.E_dim
              * self.n_heads, ...].unflatten(1, (self.n_heads, -1))
        v = x[:, 2 * self.E_dim * self.n_heads:, ...].unflatten(1, (self.n_heads, -1))

        # act and norma
        q = self.norm_q(self.prelu(q.permute(0, 1, -1, 2, 3).flatten(0, 2).flatten(1, 2)))\
            .unflatten(0, (n_batch, self.n_heads, T))
        k = self.norm_k(self.prelu(k.permute(0, 1, -1, 2, 3).flatten(0, 2).flatten(1, 2)))\
            .unflatten(0, (n_batch, self.n_heads, T))
        v = self.norm_v(self.prelu(v.permute(0, 1, -1, 2, 3).flatten(0, 2).flatten(1, 2)))\
            .unflatten(0, (n_batch, self.n_heads, T))

        attn = torch.nn.functional.softmax(torch.matmul(
            q, k.transpose(-1, -2)) / (self.E_dim * F) ** (1 / 2), dim=-1)

        res = torch.matmul(attn, v)
        res += skip

        return res


class ReconstructionUnit(nn.Module):
    def __init__(self, smt):
        super().__init__()

    def forward(self, x: Tensor):
        # x shape is [B, D, F, T]
        # out shape is [B, C_a, F_a, T_a]
