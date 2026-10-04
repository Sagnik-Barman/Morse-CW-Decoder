"""Classical (non-learned) CW decoder -- the baseline every model must beat.

Pipeline (what most traditional software decoders do):
  1. find the tone: peak of the Welch power spectrum in 200-1500 Hz
  2. mix to baseband and low-pass -> envelope |x(t)| at 200 Hz frame rate
  3. threshold the log-envelope with 2-means (Otsu-like), with hysteresis
  4. run-length encode into marks / spaces
  5. cluster mark durations into dot / dash, estimate the unit length
  6. classify spaces (element / character / word) with 2- and 5-unit cut-offs
  7. look up each dot-dash pattern in the Morse table
"""
from __future__ import annotations

import numpy as np
from scipy import signal as sps

from .alphabet import REVERSE


def _two_means_1d(v: np.ndarray, iters: int = 30):
    lo, hi = np.percentile(v, 10), np.percentile(v, 90)
    for _ in range(iters):
        thr = 0.5 * (lo + hi)
        a, b = v[v <= thr], v[v > thr]
        if len(a) == 0 or len(b) == 0:
            break
        lo_new, hi_new = a.mean(), b.mean()
        if abs(lo_new - lo) < 1e-6 and abs(hi_new - hi) < 1e-6:
            break
        lo, hi = lo_new, hi_new
    return lo, hi


def estimate_tone(x: np.ndarray, sr: int, fmin=200.0, fmax=1500.0) -> float:
    f, p = sps.welch(x, fs=sr, nperseg=2048)
    band = (f >= fmin) & (f <= fmax)
    return float(f[band][np.argmax(p[band])])


def envelope(x: np.ndarray, sr: int, f0: float, bw_hz: float = 120.0, frame_rate: int = 200) -> np.ndarray:
    t = np.arange(len(x)) / sr
    bb = x * np.exp(-2j * np.pi * f0 * t)
    sos = sps.butter(4, bw_hz / (sr / 2), output="sos")
    bb = sps.sosfiltfilt(sos, bb.real) + 1j * sps.sosfiltfilt(sos, bb.imag)
    env = np.abs(bb)
    hop = sr // frame_rate
    n = len(env) // hop
    return env[: n * hop].reshape(n, hop).mean(axis=1)


def binarize(env: np.ndarray, min_run: int = 4, squelch: float = 1.2) -> np.ndarray:
    le = np.log(env + 1e-9)
    lo, hi = _two_means_1d(le)
    if hi - lo < squelch:  # no clear on/off separation (log-amplitude) -> squelch
        return np.zeros_like(env, dtype=bool)
    # threshold halfway in log domain, but never more than ~9 dB below the mark
    # level (at high SNR filter tails would otherwise bridge the gaps)
    mid = max(lo + 0.5 * (hi - lo), hi - 1.0)
    hyst = 0.05 * (hi - lo)
    t_on, t_off = mid + hyst, mid - hyst
    key = np.zeros(len(le), dtype=bool)
    state = False
    for i, v in enumerate(le):
        if state and v < t_off:
            state = False
        elif not state and v > t_on:
            state = True
        key[i] = state
    # remove glitches shorter than min_run frames
    if min_run > 1:
        runs = _runs(key)
        for val, start, length in runs:
            if length < min_run:
                key[start:start + length] = not val
    return key


def _runs(key: np.ndarray):
    out = []
    if len(key) == 0:
        return out
    start = 0
    for i in range(1, len(key) + 1):
        if i == len(key) or key[i] != key[start]:
            out.append((bool(key[start]), start, i - start))
            start = i
    return out


def _trim_runs(key):
    runs = _runs(key)
    while runs and not runs[0][0]:
        runs.pop(0)
    while runs and not runs[-1][0]:
        runs.pop()
    return runs


def estimate_unit(runs):
    """Return (unit length, dot/dash threshold) in frames, or (None, None)."""
    marks = np.array([l for v, _, l in runs if v], dtype=float)
    if len(marks) == 0:
        return None, None
    lo, hi = _two_means_1d(np.log(marks))
    if hi - lo > np.log(1.8):  # dots and dashes both present
        return float(np.exp(lo)), float(np.exp(0.5 * (lo + hi)))
    # only one mark type: decide using spaces (element gaps ~ 1 unit)
    spaces = np.array([l for v, _, l in runs if not v], dtype=float)
    unit = np.median(spaces) if len(spaces) else np.median(marks)
    unit = float(min(unit, np.median(marks)))
    return unit, 2.0 * unit


def decode_key(key: np.ndarray) -> str:
    runs = _trim_runs(key)
    unit, dash_thr = estimate_unit(runs)
    if unit is None:
        return ""
    # word-gap threshold: 5 units, unless the long gaps are clearly bimodal
    # (then split them in the log domain -- handles Farnsworth / slow spacing)
    word_thr = 5.0
    long_gaps = np.array([l / unit for v, _, l in runs if not v and l / unit >= 2.0])
    if len(long_gaps) >= 3:
        glo, ghi = _two_means_1d(np.log(long_gaps))
        if ghi - glo > np.log(1.6):
            word_thr = float(np.exp(0.5 * (glo + ghi)))
    out, sym = [], ""
    for v, _, l in runs:
        if v:
            sym += "-" if l > dash_thr else "."
        else:
            u = l / unit
            if u >= 2.0:
                if sym:
                    out.append(REVERSE.get(sym, ""))
                    sym = ""
                if u >= word_thr:
                    out.append(" ")
    if sym:
        out.append(REVERSE.get(sym, ""))
    return " ".join("".join(out).split())


def decode_audio(x: np.ndarray, sr: int = 8000, f0: float | None = None, frame_rate: int = 200,
                 adaptive: bool = True) -> str:
    """Two passes: a wide envelope filter to estimate the speed, then a filter
    and glitch length matched to that speed (narrower = less noise at low WPM)."""
    f0 = f0 if f0 is not None else estimate_tone(x, sr)
    env = envelope(x, sr, f0, bw_hz=120.0, frame_rate=frame_rate)
    key = binarize(env)
    if adaptive:
        unit, _ = estimate_unit(_trim_runs(key))
        if unit is not None:
            unit_s = max(unit, 2) / frame_rate
            bw = float(np.clip(2.5 / unit_s, 20.0, 150.0))
            env = envelope(x, sr, f0, bw_hz=bw, frame_rate=frame_rate)
            key = binarize(env, min_run=max(2, int(round(0.35 * unit_s * frame_rate))))
    return decode_key(key)
