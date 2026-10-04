"""Synthetic CW (Morse) audio generator with realistic impairments.

Everything a real receiver hears is modelled as a random variable, so the
network cannot overfit to a clean "machine" signal:

  timing   : speed (WPM), Farnsworth spacing, dash/dot ratio, keying weight,
             per-element jitter (hand-keyed "fist"), slow speed drift
  tone     : pitch, slow frequency drift, key-down chirp, edge rise time
  channel  : Rayleigh/Rician fading (QSB), interfering CW signal (QRM),
             impulsive static (QRN), AWGN at a target SNR, receiver bandpass

SNR convention: key-down carrier power vs noise power in a 2500 Hz bandwidth
(the usual SSB-filter reference used by ham-radio decoders). Humans copy CW
down to roughly -15 dB with this definition.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np
from scipy import signal as sps

from .alphabet import MORSE, normalize_text
from .text_gen import sample_text


@dataclass
class SynthConfig:
    sample_rate: int = 8000
    clip_seconds: float = 8.0
    wpm_range: tuple = (12.0, 45.0)
    snr_db_range: tuple = (-12.0, 30.0)
    tone_hz_range: tuple = (450.0, 950.0)
    lead_silence_range: tuple = (0.0, 1.0)     # seconds before first element
    p_fill: float = 0.75                       # keep appending text until the clip is full
    # timing / fist
    p_hand: float = 0.6                        # hand-keyed (irregular) vs keyer-perfect
    jitter_range: tuple = (0.0, 0.18)          # std of per-element duration, fraction of length
    dash_ratio_range: tuple = (2.6, 3.8)       # hand-keyed dash/dot ratio
    weight_range: tuple = (-0.15, 0.15)        # units added to marks / removed from spaces
    char_gap_scale_range: tuple = (0.8, 1.6)   # hand-keyed inter-character gap scale
    word_gap_scale_range: tuple = (0.8, 1.8)
    p_farnsworth: float = 0.15
    farnsworth_range: tuple = (1.3, 2.5)       # extra stretch of char/word gaps
    speed_drift_range: tuple = (0.0, 0.12)     # +- fractional speed wander
    # tone
    rise_ms_range: tuple = (2.0, 8.0)
    freq_drift_hz_range: tuple = (0.0, 8.0)
    p_chirp: float = 0.1
    chirp_hz_range: tuple = (5.0, 40.0)
    # channel
    p_fading: float = 0.3
    fade_rate_hz_range: tuple = (0.05, 1.5)
    fade_rate_dist: str = "uniform"            # "loguniform" spreads rates evenly across decades (v2)
    rician_k_range: tuple = (0.0, 5.0)
    p_qrm: float = 0.3
    qrm_rel_db_range: tuple = (-20.0, -3.0)
    qrm_offset_hz_range: tuple = (60.0, 450.0)
    p_qrn: float = 0.2
    p_narrow_filter: float = 0.25              # 500 Hz CW filter instead of 2.4 kHz SSB filter
    text_mix: dict = field(default_factory=lambda: {"qso": 0.3, "contest": 0.15, "words": 0.25, "random": 0.3})
    # v2 additions (defaults reproduce v1 exactly; configs/*_ft.yaml turn them on)
    p_empty: float = 0.0                       # noise-only clips with an empty label (teaches "say nothing")
    rx_lowcut_range: tuple = (200.0, 200.0)    # receiver filter low edge (Hz) for the wide filter
    # v3 additions (defaults reproduce v1/v2 exactly; configs/*_ft3.yaml turn them on).
    # Real busy-band audio has several other stations near the target, often as strong or stronger.
    # v3 keeps the target at the centre pitch (tone_hz_range narrow) and adds 1..N interferers anywhere
    # around it, so the model learns "decode the signal in the middle, ignore the rest".
    qrm_count_range: tuple = (1, 1)            # number of interfering stations when QRM is on
    qrm_on_empty: bool = False                 # interferers also in noise-only clips (label stays empty)
    qrm_fading: bool = False                   # interferers fade independently of the target

    def to_dict(self):
        return asdict(self)


def _u(rng, lo_hi):
    lo, hi = lo_hi
    return float(rng.uniform(lo, hi))


# --------------------------------------------------------------------------
# keying: text -> list of (is_mark, seconds, char_index)
# --------------------------------------------------------------------------

def _keying_params(rng: np.random.Generator, cfg: SynthConfig, wpm: float | None = None) -> dict:
    hand = rng.random() < cfg.p_hand
    p = {
        "wpm": wpm if wpm is not None else _u(rng, cfg.wpm_range),
        "hand": hand,
        "jitter": _u(rng, cfg.jitter_range) if hand else 0.0,
        "dash_ratio": _u(rng, cfg.dash_ratio_range) if hand else 3.0,
        "weight": _u(rng, cfg.weight_range) if hand else 0.0,
        "char_gap": _u(rng, cfg.char_gap_scale_range) if hand else 1.0,
        "word_gap": _u(rng, cfg.word_gap_scale_range) if hand else 1.0,
        "farnsworth": _u(rng, cfg.farnsworth_range) if rng.random() < cfg.p_farnsworth else 1.0,
        "drift": _u(rng, cfg.speed_drift_range),
        "drift_hz": float(rng.uniform(0.02, 0.2)),
        "drift_phase": float(rng.uniform(0, 2 * np.pi)),
    }
    return p


def text_to_segments(text: str, p: dict, rng: np.random.Generator):
    """Return [(is_mark, duration_s, char_idx)], char_idx = index into text of
    the character the segment belongs to (gaps after a char belong to it)."""
    unit0 = 1.2 / p["wpm"]
    segs = []
    t = 0.0

    def dur(units, mark):
        nonlocal t
        if mark:
            units = units + p["weight"]
        else:
            units = max(units - p["weight"], 0.3)
        speed = 1.0 + p["drift"] * np.sin(2 * np.pi * p["drift_hz"] * t + p["drift_phase"])
        d = units * unit0 / speed
        if p["jitter"] > 0:
            d *= max(0.35, 1.0 + p["jitter"] * rng.standard_normal())
        t += d
        return d

    for ci, ch in enumerate(text):
        if ch == " ":
            # a word gap is 7 units; 3 were already added after the previous char
            segs.append((False, dur(4 * p["word_gap"] * p["farnsworth"], False), ci))
            continue
        code = MORSE[ch]
        for si, sym in enumerate(code):
            segs.append((True, dur(1.0 if sym == "." else p["dash_ratio"], True), ci))
            if si < len(code) - 1:
                segs.append((False, dur(1.0, False), ci))
        segs.append((False, dur(3 * p["char_gap"] * p["farnsworth"], False), ci))
    return segs


def segments_to_key(segs, sample_rate: int) -> np.ndarray:
    """Boolean key-down array (one value per audio sample)."""
    lens = np.maximum(1, np.round(np.array([d for _, d, _ in segs]) * sample_rate).astype(int))
    vals = np.array([m for m, _, _ in segs], dtype=bool)
    return np.repeat(vals, lens)


# --------------------------------------------------------------------------
# signal building blocks
# --------------------------------------------------------------------------

def shape_edges(key: np.ndarray, rise_s: float, sample_rate: int) -> np.ndarray:
    """Raised-cosine keying edges (avoids key clicks, like a real transmitter)."""
    n = max(1, int(rise_s * sample_rate))
    if n <= 1:
        return key.astype(np.float32)
    win = np.hanning(2 * n + 1)
    win /= win.sum()
    out = sps.oaconvolve(key.astype(np.float32), win.astype(np.float32), mode="same")
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def tone(env: np.ndarray, key: np.ndarray, f0: float, sample_rate: int, rng, cfg: SynthConfig,
         allow_chirp: bool = True) -> np.ndarray:
    n = len(env)
    t = np.arange(n) / sample_rate
    freq = np.full(n, f0, dtype=np.float64)
    drift = _u(rng, cfg.freq_drift_hz_range)
    if drift > 0:
        freq += drift * np.sin(2 * np.pi * rng.uniform(0.02, 0.3) * t + rng.uniform(0, 2 * np.pi))
    if allow_chirp and rng.random() < cfg.p_chirp:
        # frequency jumps at key-down then settles (poorly regulated oscillator)
        onsets = np.flatnonzero(np.diff(key.astype(np.int8), prepend=0) == 1)
        last_on = np.zeros(n, dtype=np.int64)
        if len(onsets):
            last_on[onsets] = onsets
            last_on = np.maximum.accumulate(last_on)
        since = (np.arange(n) - last_on) / sample_rate
        freq += _u(rng, cfg.chirp_hz_range) * np.exp(-since / rng.uniform(0.005, 0.03))
    phase = 2 * np.pi * np.cumsum(freq) / sample_rate + rng.uniform(0, 2 * np.pi)
    return env * np.sin(np.mod(phase, 2 * np.pi).astype(np.float32))


def fading(n: int, sample_rate: int, rng, cfg: SynthConfig) -> np.ndarray:
    """Rician/Rayleigh amplitude fading, normalised to unit mean power."""
    lo, hi = cfg.fade_rate_hz_range
    rate = float(np.exp(rng.uniform(np.log(lo), np.log(hi)))) if cfg.fade_rate_dist == "loguniform" else _u(rng, (lo, hi))
    k = _u(rng, cfg.rician_k_range)
    fs_low = max(50.0, 6.0 * rate)
    m = int(n / sample_rate * fs_low) + 8
    h = rng.standard_normal(m) + 1j * rng.standard_normal(m)
    b, a = sps.butter(2, min(rate / (fs_low / 2), 0.99))
    h = sps.filtfilt(b, a, h.real) + 1j * sps.filtfilt(b, a, h.imag)
    h /= np.sqrt(np.mean(np.abs(h) ** 2)) + 1e-12
    los = np.sqrt(k / (k + 1))
    scat = np.sqrt(1 / (k + 1))
    amp = np.abs(los + scat * h)
    amp /= np.sqrt(np.mean(amp ** 2)) + 1e-12
    x_low = np.arange(m) / fs_low
    x = np.arange(n) / sample_rate
    return np.interp(x, x_low, amp)


def qrn(n: int, sample_rate: int, noise_std: float, rng) -> np.ndarray:
    """Impulsive static crashes."""
    out = np.zeros(n)
    n_bursts = rng.poisson(n / sample_rate * rng.uniform(0.3, 3.0))
    for _ in range(n_bursts):
        length = int(sample_rate * rng.uniform(0.001, 0.03))
        start = int(rng.integers(0, max(1, n - length)))
        decay = np.exp(-np.arange(length) / max(1, length / 4))
        out[start:start + length] += rng.uniform(3, 25) * noise_std * decay * rng.standard_normal(length)
    return out


def receiver_filter(x: np.ndarray, sample_rate: int, center: float, narrow: bool, lowcut: float = 200.0) -> np.ndarray:
    nyq = sample_rate / 2
    if narrow:
        lo, hi = max(50.0, center - 250), min(nyq - 50, center + 250)
    else:
        lo, hi = lowcut, min(2700.0, nyq - 100)
    sos = sps.butter(4, [lo / nyq, hi / nyq], btype="band", output="sos")
    return sps.sosfilt(sos, x)


# --------------------------------------------------------------------------
# main entry point
# --------------------------------------------------------------------------

def _fill_segments(rng, cfg, p, n_target_s, first_text=None):
    """Build keying for text(s) until n_target_s seconds are covered; returns
    (segments, full_text). Texts are joined with a single space."""
    texts = []
    segs = []
    total = 0.0
    fill = rng.random() < cfg.p_fill
    while True:
        txt = normalize_text(first_text if (first_text and not texts) else sample_text(rng, cfg.text_mix))
        if not txt:
            continue
        if texts:
            txt = " " + txt
        offset = sum(len(t) for t in texts)
        s = text_to_segments(txt, p, rng)
        segs.extend((m, d, ci + offset) for m, d, ci in s)
        texts.append(txt)
        total += sum(d for _, d, _ in s)
        if total >= n_target_s or not fill or first_text:
            break
    return segs, "".join(texts)


def _truncate(segs, full_text, max_s):
    """Keep whole characters whose last segment ends before max_s."""
    t = 0.0
    last_ok = -1
    keep = 0
    for i, (m, d, ci) in enumerate(segs):
        t += d
        if t > max_s:
            break
        # a character is complete once its trailing (char-gap) segment is placed
        nxt = segs[i + 1][2] if i + 1 < len(segs) else None
        if nxt != ci:
            last_ok = ci
            keep = i + 1
    return segs[:keep], full_text[: last_ok + 1].strip()


def generate_sample(rng: np.random.Generator, cfg: SynthConfig | None = None, text: str | None = None,
                    snr_db: float | None = None, wpm: float | None = None):
    """Generate one clip.

    Returns (audio float32 [n], label str, meta dict). audio length is always
    cfg.clip_seconds * cfg.sample_rate.
    """
    cfg = cfg or SynthConfig()
    sr = cfg.sample_rate
    n = int(cfg.clip_seconds * sr)
    lead = _u(rng, cfg.lead_silence_range)

    p = _keying_params(rng, cfg, wpm)
    segs, full_text = _fill_segments(rng, cfg, p, cfg.clip_seconds - lead, first_text=text)
    segs, label = _truncate(segs, full_text, cfg.clip_seconds - lead - 0.05)
    # retry once with a fresh text if nothing fitted (very slow speed + long first char)
    if not label and text is None:
        segs, full_text = _fill_segments(rng, cfg, p, cfg.clip_seconds - lead)
        segs, label = _truncate(segs, full_text, cfg.clip_seconds - lead - 0.05)

    empty = cfg.p_empty > 0 and text is None and rng.random() < cfg.p_empty
    if empty:  # noise (+ static) only: the correct transcription is nothing at all
        segs, label = [], ""
    key = np.zeros(n, dtype=bool)
    if segs:
        k = segments_to_key(segs, sr)
        start = int(lead * sr)
        k = k[: n - start]
        key[start:start + len(k)] = k

    f0 = _u(rng, cfg.tone_hz_range)
    env = shape_edges(key, _u(rng, cfg.rise_ms_range) / 1000.0, sr)
    x = tone(env, key, f0, sr, rng, cfg)

    meta = {"wpm": p["wpm"], "tone_hz": f0, "hand": p["hand"], "jitter": p["jitter"],
            "farnsworth": p["farnsworth"], "fading": False, "qrm": False, "qrn": False, "empty": empty}

    if rng.random() < cfg.p_fading:
        x *= fading(n, sr, rng, cfg)
        meta["fading"] = True

    if rng.random() < cfg.p_qrm and (not empty or cfg.qrm_on_empty):
        c_lo, c_hi = cfg.qrm_count_range
        n_qrm = c_lo if c_lo == c_hi else int(rng.integers(c_lo, c_hi + 1))  # no RNG draw in v1/v2 mode
        for _ in range(n_qrm):
            q_p = _keying_params(rng, cfg)
            q_segs, _ = _fill_segments(rng, cfg, q_p, cfg.clip_seconds + 2)
            q_key = segments_to_key(q_segs, sr)
            shift = int(rng.integers(0, max(1, len(q_key) - n))) if len(q_key) > n else 0
            q_key = q_key[shift:shift + n]
            q_key = np.pad(q_key, (int(rng.integers(0, max(1, n - len(q_key) + 1))), 0))[:n]
            q_key = np.pad(q_key, (0, n - len(q_key)))
            offset = _u(rng, cfg.qrm_offset_hz_range) * (1 if rng.random() < 0.5 else -1)
            q_f = float(np.clip(f0 + offset, 250, 2400))
            q_env = shape_edges(q_key, _u(rng, cfg.rise_ms_range) / 1000.0, sr)
            rel = 10 ** (_u(rng, cfg.qrm_rel_db_range) / 20)
            q = rel * tone(q_env, q_key, q_f, sr, rng, cfg, allow_chirp=False)
            if cfg.qrm_fading and rng.random() < cfg.p_fading:
                q = q * fading(n, sr, rng, cfg)
            x = x + q
        meta["qrm"] = True
        meta["n_qrm"] = n_qrm

    snr = snr_db if snr_db is not None else _u(rng, cfg.snr_db_range)
    # carrier amplitude 1 -> key-down power 0.5 ; noise measured in 2500 Hz
    noise_var = 0.5 / (10 ** (snr / 10)) * (sr / 2) / 2500.0
    noise_std = float(np.sqrt(noise_var))
    x = x + noise_std * rng.standard_normal(n)
    if rng.random() < cfg.p_qrn:
        x = x + qrn(n, sr, noise_std, rng)
        meta["qrn"] = True

    narrow = rng.random() < cfg.p_narrow_filter
    lc_lo, lc_hi = cfg.rx_lowcut_range
    lowcut = lc_lo if lc_lo == lc_hi else _u(rng, cfg.rx_lowcut_range)  # no RNG draw in v1 mode
    x = receiver_filter(x, sr, f0, narrow, lowcut=lowcut)

    # AGC-ish normalisation with random gain
    x = x / (np.percentile(np.abs(x), 99.5) + 1e-9) * rng.uniform(0.2, 0.8)
    x = np.clip(x, -1, 1).astype(np.float32)

    meta.update({"snr_db": snr, "narrow_filter": narrow})
    return x, label, meta
