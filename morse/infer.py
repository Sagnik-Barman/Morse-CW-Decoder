"""Shared inference helpers: checkpoint loading, batched transcription, audio I/O."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import collate
from .decode import beam_decode, greedy_decode
from .model import build_model


def pick_device(name: str | None = None) -> torch.device:
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint(path: str, device=None):
    device = device or pick_device()
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt["config"], ckpt


@torch.no_grad()
def transcribe_dataset(model, dataset, device, batch_size=64, decoder="greedy", beam=16, lm=None,
                       lm_weight=0.3, char_bonus=0.5, num_workers=0):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=num_workers)
    hyps, refs, metas = [], [], []
    for b in loader:
        lp, ln = model(b["audio"].to(device), b["audio_lengths"].to(device))
        if decoder == "greedy":
            hyps += greedy_decode(lp, ln)
        else:
            hyps += beam_decode(lp, ln, beam=beam, lm=lm, lm_weight=lm_weight, char_bonus=char_bonus)
        refs += b["texts"]
        metas += b["meta"]
    return hyps, refs, metas


@torch.no_grad()
def transcribe_audio(model, audio: np.ndarray, device, decoder="greedy", beam=16, lm=None, lm_weight=0.3,
                     char_bonus=0.5) -> str:
    x = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))[None].to(device)
    ln = torch.tensor([x.shape[1]], device=device)
    lp, oln = model(x, ln)
    if decoder == "greedy":
        return greedy_decode(lp, oln)[0]
    return beam_decode(lp, oln, beam=beam, lm=lm, lm_weight=lm_weight, char_bonus=char_bonus)[0]


def load_audio(path: str, target_sr: int = 8000) -> np.ndarray:
    """Read a wav/flac/ogg file, mix to mono, resample to target_sr, peak-normalise."""
    import soundfile as sf
    from math import gcd
    from scipy.signal import resample_poly

    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != target_sr:
        g = gcd(sr, target_sr)
        x = resample_poly(x, target_sr // g, sr // g).astype(np.float32)
    peak = np.percentile(np.abs(x), 99.5) + 1e-9
    return np.clip(x / peak * 0.5, -1, 1).astype(np.float32)


def shift_tone(x: np.ndarray, sr: int, f_from: float, f_to: float) -> np.ndarray:
    """Move a CW tone from f_from to f_to Hz (single-sideband frequency shift)."""
    from scipy.signal import hilbert

    t = np.arange(len(x)) / sr
    return np.real(hilbert(x) * np.exp(2j * np.pi * (f_to - f_from) * t)).astype(np.float32)


def split_quiet(x: np.ndarray, sr: int, chunk_s: float, search_s: float = 2.0):
    """Yield chunks of ~chunk_s seconds, each cut at the lowest-energy point in
    the last search_s seconds, so characters are (almost) never cut in half."""
    n, chunk, search = len(x), int(chunk_s * sr), int(search_s * sr)
    hop = max(1, sr // 100)
    start = 0
    while start < n:
        end = start + chunk
        if end >= n:
            yield x[start:]
            return
        seg = x[end - search:end]
        m = len(seg) // hop
        energy = (seg[: m * hop].reshape(m, hop) ** 2).mean(axis=1)
        energy = np.convolve(energy, np.ones(5) / 5, mode="same")  # cut inside a gap
        cut = end - search + int(np.argmin(energy)) * hop
        yield x[start:cut]
        start = cut


def pad_with_noise(x: np.ndarray, sr: int, total: int, seed: int = 0) -> np.ndarray:
    """Append noise that matches the clip's own background (same spectrum, tone notched out)
    until it is `total` samples long. The models were trained only on 8 s clips; on short real
    clips (e.g. 3 s beacon callsigns) this cut CER from about 31% to 21% (see RESULTS.md)."""
    if len(x) >= total:
        return x
    from scipy.signal import butter, sosfiltfilt, welch

    f, p = welch(x, fs=sr, nperseg=min(1024, len(x)))
    band = (f >= 200) & (f <= 2000)
    f0 = float(f[band][np.argmax(p[band])]) if band.any() else 700.0
    lo, hi = max(50.0, f0 - 100.0), min(sr / 2 - 50.0, f0 + 100.0)
    bg = sosfiltfilt(butter(4, [lo, hi], btype="bandstop", fs=sr, output="sos"), x)
    n = total - len(x)
    spec = np.abs(np.fft.rfft(np.tile(bg, n // len(bg) + 2)[:n]))
    rng = np.random.default_rng(seed + len(x))
    noise = np.fft.irfft(spec * np.exp(2j * np.pi * rng.random(len(spec))), n)
    return np.concatenate([x, noise]).astype(np.float32)


@torch.no_grad()
def transcribe_long(model, audio: np.ndarray, device, sr: int = 8000, chunk_seconds: float = 8.0,
                    decoder="greedy", beam=16, lm=None, lm_weight=0.3, char_bonus=0.5,
                    pad_short: bool = True) -> str:
    """Decode audio of any length in quiet-point chunks (the models are trained on
    8 s clips, so chunks of about that size match training conditions best).
    With pad_short, chunks shorter than chunk_seconds are padded with matched noise."""
    parts = []
    for c in split_quiet(audio, sr, chunk_seconds):
        if len(c) < sr // 4:
            continue
        if pad_short:
            c = pad_with_noise(c, sr, int(chunk_seconds * sr))
        parts.append(transcribe_audio(model, c, device, decoder=decoder, beam=beam, lm=lm,
                                      lm_weight=lm_weight, char_bonus=char_bonus))
    return " ".join(" ".join(p for p in parts if p).split())


def baseline_long(audio: np.ndarray, sr: int = 8000, chunk_seconds: float = 16.0) -> str:
    from .baseline import decode_audio
    parts = [decode_audio(c, sr) for c in split_quiet(audio, sr, chunk_seconds) if len(c) >= sr // 4]
    return " ".join(" ".join(p for p in parts if p).split())
