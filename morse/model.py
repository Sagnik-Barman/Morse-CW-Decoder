"""CTC acoustic models: log-spectrogram front-end + CRNN or Conformer encoder.

Input : raw audio (B, N) at 8 kHz and its valid lengths
Output: log-probabilities (B, T', C) over the Morse vocabulary (index 0 = blank)
        and output lengths T'.

The spectrogram is computed inside the model (on the GPU, in float32) so the
data loader only has to synthesise audio.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .alphabet import NUM_CLASSES


# --------------------------------------------------------------------------
# front-end
# --------------------------------------------------------------------------

class LogSpectrogram(nn.Module):
    def __init__(self, sample_rate=8000, n_fft=256, win_length=200, hop_length=64,
                 fmin=200.0, fmax=1400.0):
        super().__init__()
        self.n_fft, self.win_length, self.hop_length = n_fft, win_length, hop_length
        self.register_buffer("window", torch.hann_window(win_length), persistent=False)
        bin_hz = sample_rate / n_fft
        self.lo = int(math.floor(fmin / bin_hz))
        self.hi = int(math.ceil(fmax / bin_hz)) + 1
        self.n_bins = self.hi - self.lo

    def frame_lengths(self, audio_lengths: torch.Tensor) -> torch.Tensor:
        return audio_lengths // self.hop_length + 1  # center=True

    def forward(self, audio: torch.Tensor, audio_lengths: torch.Tensor):
        with torch.autocast(device_type=audio.device.type, enabled=False):
            spec = torch.stft(audio.float(), self.n_fft, self.hop_length, self.win_length,
                              window=self.window, center=True, return_complex=True)
            x = torch.log(spec[:, self.lo:self.hi].abs().pow(2) + 1e-8)  # (B, F, T)
            lengths = self.frame_lengths(audio_lengths).clamp(max=x.shape[-1])
            # per-utterance normalisation over valid frames (acts like an AGC)
            mask = (torch.arange(x.shape[-1], device=x.device)[None, :] < lengths[:, None]).float()
            m = mask[:, None, :]
            count = m.sum(dim=(1, 2), keepdim=True) * x.shape[1]
            mean = (x * m).sum(dim=(1, 2), keepdim=True) / count
            var = (((x - mean) * m) ** 2).sum(dim=(1, 2), keepdim=True) / count
            x = (x - mean) / (var.sqrt() + 1e-5) * m
        return x, lengths


class ConvFrontEnd(nn.Module):
    """2-D conv stack over (freq, time); downsamples time by 2, freq by 8."""

    def __init__(self, n_bins: int, channels=(32, 64, 128), out_dim: int = 256, dropout=0.1):
        super().__init__()
        c1, c2, c3 = channels

        def block(ci, co):
            return nn.Sequential(nn.Conv2d(ci, co, 3, padding=1, bias=False), nn.BatchNorm2d(co), nn.GELU())

        self.net = nn.Sequential(
            block(1, c1), block(c1, c1), nn.MaxPool2d((2, 1)),
            block(c1, c2), block(c2, c2), nn.MaxPool2d((2, 2)),
            block(c2, c3), block(c3, c3), nn.MaxPool2d((2, 1)),
        )
        f_out = n_bins // 8
        self.proj = nn.Sequential(nn.Linear(c3 * f_out, out_dim), nn.Dropout(dropout))

    @staticmethod
    def output_lengths(lengths):
        return lengths // 2

    def forward(self, x):  # x: (B, F, T)
        x = self.net(x.unsqueeze(1))  # (B, C, F', T')
        b, c, f, t = x.shape
        x = x.permute(0, 3, 1, 2).reshape(b, t, c * f)
        return self.proj(x)  # (B, T', D)


# --------------------------------------------------------------------------
# encoders
# --------------------------------------------------------------------------

class BiGRUEncoder(nn.Module):
    def __init__(self, d_in, hidden=256, layers=3, dropout=0.2):
        super().__init__()
        self.rnn = nn.GRU(d_in, hidden, num_layers=layers, batch_first=True, bidirectional=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.out_dim = 2 * hidden

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.rnn(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=x.shape[1])
        return out


class FeedForward(nn.Module):
    def __init__(self, d, mult=4, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d * mult), nn.SiLU(), nn.Dropout(dropout),
                                 nn.Linear(d * mult, d), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class ConvModule(nn.Module):
    def __init__(self, d, kernel=15, dropout=0.1):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.pw1 = nn.Conv1d(d, 2 * d, 1)
        self.dw = nn.Conv1d(d, d, kernel, padding=kernel // 2, groups=d)
        self.bn = nn.BatchNorm1d(d)
        self.pw2 = nn.Conv1d(d, d, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, pad_mask):
        y = self.ln(x).masked_fill(pad_mask[..., None], 0.0).transpose(1, 2)
        y = F.glu(self.pw1(y), dim=1)
        y = F.silu(self.bn(self.dw(y)))
        y = self.drop(self.pw2(y)).transpose(1, 2)
        return y


class SelfAttention(nn.Module):
    """Multi-head self-attention via F.scaled_dot_product_attention. Its fused kernels avoid
    materialising the full (B, heads, T, T) attention matrix, but need a head dim that is a multiple
    of 8; the default Conformer (d_model 144, 4 heads -> 36) falls back to the plain "math" path.
    Keep attention dropout at 0 there: on the math path it adds a full-size tensor per layer, which
    pushed training past a 6 GB GPU's memory (see scripts/speed_dropout.py)."""

    def __init__(self, d, heads=4, dropout=0.1):
        super().__init__()
        assert d % heads == 0
        self.h, self.dk, self.p = heads, d // heads, dropout
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)

    def forward(self, x, pad_mask):
        b, t, d = x.shape
        q, k, v = self.qkv(x).view(b, t, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)
        keep = ~pad_mask[:, None, None, :]  # True = attend; (B, 1, 1, T) broadcast over heads/queries
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=keep, dropout_p=self.p if self.training else 0.0)
        return self.out(y.transpose(1, 2).reshape(b, t, d))


class ConformerBlock(nn.Module):
    def __init__(self, d, heads=4, ff_mult=4, kernel=15, dropout=0.1, attn_dropout=None):
        super().__init__()
        self.ff1 = FeedForward(d, ff_mult, dropout)
        self.ln_att = nn.LayerNorm(d)
        self.att = SelfAttention(d, heads, dropout if attn_dropout is None else attn_dropout)
        self.drop = nn.Dropout(dropout)
        self.conv = ConvModule(d, kernel, dropout)
        self.ff2 = FeedForward(d, ff_mult, dropout)
        self.ln_out = nn.LayerNorm(d)

    def forward(self, x, pad_mask):
        x = x + 0.5 * self.ff1(x)
        y = self.ln_att(x)
        y = self.att(y, pad_mask)
        x = x + self.drop(y)
        x = x + self.conv(x, pad_mask)
        x = x + 0.5 * self.ff2(x)
        return self.ln_out(x)


def sinusoidal_pe(t, d, device):
    pos = torch.arange(t, device=device, dtype=torch.float32)[:, None]
    div = torch.exp(torch.arange(0, d, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / d))
    pe = torch.zeros(t, d, device=device)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class ConformerEncoder(nn.Module):
    def __init__(self, d_in, d_model=144, layers=8, heads=4, ff_mult=4, kernel=15, dropout=0.1,
                 attn_dropout=None):
        super().__init__()
        self.inp = nn.Linear(d_in, d_model) if d_in != d_model else nn.Identity()
        self.blocks = nn.ModuleList([ConformerBlock(d_model, heads, ff_mult, kernel, dropout, attn_dropout)
                                     for _ in range(layers)])
        self.out_dim = d_model
        self.d_model = d_model

    def forward(self, x, lengths):
        x = self.inp(x)
        x = x + sinusoidal_pe(x.shape[1], self.d_model, x.device)[None].to(x.dtype)
        pad_mask = torch.arange(x.shape[1], device=x.device)[None, :] >= lengths[:, None]
        for blk in self.blocks:
            x = blk(x, pad_mask)
        return x


# --------------------------------------------------------------------------
# full model
# --------------------------------------------------------------------------

class MorseCTC(nn.Module):
    def __init__(self, arch: str = "crnn", sample_rate: int = 8000, frontend: dict | None = None,
                 encoder: dict | None = None, conv_dim: int = 256, dropout: float = 0.1,
                 conv_channels=(32, 64, 128)):
        super().__init__()
        self.arch = arch
        self.spec = LogSpectrogram(sample_rate=sample_rate, **(frontend or {}))
        self.conv = ConvFrontEnd(self.spec.n_bins, channels=tuple(conv_channels), out_dim=conv_dim, dropout=dropout)
        enc = dict(encoder or {})
        if arch == "crnn":
            self.encoder = BiGRUEncoder(conv_dim, **enc)
        elif arch == "conformer":
            self.encoder = ConformerEncoder(conv_dim, **enc)
        else:
            raise ValueError(f"unknown arch {arch}")
        self.head = nn.Linear(self.encoder.out_dim, NUM_CLASSES)

    def output_lengths(self, audio_lengths):
        return self.conv.output_lengths(self.spec.frame_lengths(audio_lengths))

    def forward(self, audio, audio_lengths):
        x, frame_lengths = self.spec(audio, audio_lengths)
        x = self.conv(x)
        lengths = self.conv.output_lengths(frame_lengths).clamp(max=x.shape[1], min=1)
        x = self.encoder(x, lengths)
        logits = self.head(x)
        return logits.float().log_softmax(dim=-1), lengths


def build_model(cfg: dict) -> MorseCTC:
    m = cfg.get("model", cfg)
    return MorseCTC(arch=m.get("arch", "crnn"), sample_rate=cfg.get("synth", {}).get("sample_rate", 8000),
                    frontend=m.get("frontend"), encoder=m.get("encoder"),
                    conv_dim=m.get("conv_dim", 256), dropout=m.get("dropout", 0.1),
                    conv_channels=m.get("conv_channels", (32, 64, 128)))


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
