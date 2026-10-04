"""NCDXF/IARU International Beacon Project: schedule, slicing and SNR measurement.

Every beacon sends, in its 10-second slot:
    <callsign> at 22 WPM, then four 1-second dashes at 100 W, 10 W, 1 W, 0.1 W
The 18 beacons rotate through 5 bands on a 3-minute cycle aligned to the UTC
hour (https://www.ncdxf.org/beacon/beaconschedule.html). So from the UTC time
of a recording we know exactly which callsign should be on the air -> free
labels for real, off-air audio, plus a built-in 10 dB/step power ladder.
"""
from __future__ import annotations

import numpy as np
from scipy import signal as sps

from .synth import segments_to_key, text_to_segments

BEACONS = ["4U1UN", "VE8AT", "W6WX", "KH6RS", "ZL6B", "VK6RBP", "JA2IGY", "RR9O", "VR2B",
           "4S7B", "ZS6DN", "5Z4B", "4X6TU", "OH2B", "CS3B", "LU4AA", "OA4B", "YV5B"]
BANDS_KHZ = [14100, 18110, 21150, 24930, 28200]
SLOT_S = 10.0
CYCLE_S = 180.0
BEACON_WPM = 22.0

_PERFECT = {"wpm": BEACON_WPM, "hand": False, "jitter": 0.0, "dash_ratio": 3.0, "weight": 0.0,
            "char_gap": 1.0, "word_gap": 1.0, "farnsworth": 1.0, "drift": 0.0, "drift_hz": 0.1,
            "drift_phase": 0.0}


def beacon_for_slot(slot_index: int, band_khz: int) -> str:
    """slot_index = number of 10 s slots since the start of the UTC hour."""
    b = BANDS_KHZ.index(band_khz)
    return BEACONS[(slot_index % 18 - b) % 18]


def parse_utc_seconds_in_hour(s: str) -> float:
    """Accepts '2026-10-02T14:03:17Z', '2026-10-02 14:03:17', '14:03:17', '03:17' (MM:SS).
    Only minutes/seconds within the hour matter for the schedule."""
    s = s.strip().rstrip("Z")
    if "T" in s:
        s = s.split("T", 1)[1]
    elif " " in s:
        s = s.split(" ", 1)[1]
    s = s.split("+")[0]
    parts = [float(p) for p in s.split(":")]
    if len(parts) == 3:
        _, m, sec = parts
    elif len(parts) == 2:
        m, sec = parts
    else:
        raise ValueError(f"cannot parse UTC time '{s}'")
    return m * 60 + sec


def callsign_key(callsign: str, sr: int) -> np.ndarray:
    """Key-down array of a perfectly timed callsign at 22 WPM (no trailing gap)."""
    segs = text_to_segments(callsign, _PERFECT, np.random.default_rng(0))
    while segs and not segs[-1][0]:
        segs.pop()
    return segments_to_key(segs, sr)


# --------------------------------------------------------------------------
# narrow-band power envelope (complex baseband) and its noise calibration
# --------------------------------------------------------------------------

def baseband_power(x: np.ndarray, sr: int, f0: float, bw_hz: float = 40.0, frame_rate: int = 100) -> np.ndarray:
    t = np.arange(len(x)) / sr
    bb = x * np.exp(-2j * np.pi * f0 * t)
    sos = sps.butter(4, bw_hz / (sr / 2), output="sos")
    bb = sps.sosfiltfilt(sos, bb.real) + 1j * sps.sosfiltfilt(sos, bb.imag)
    p = np.abs(bb) ** 2
    hop = sr // frame_rate
    n = len(p) // hop
    return p[: n * hop].reshape(n, hop).mean(axis=1)


_CAL = {}


def noise_gain(sr: int, bw_hz: float) -> float:
    """E[baseband power] / noise variance for white noise (computed once, numerically)."""
    key = (sr, bw_hz)
    if key not in _CAL:
        rng = np.random.default_rng(0)
        x = rng.standard_normal(sr * 20)
        _CAL[key] = float(np.mean(baseband_power(x, sr, 700.0, bw_hz)[50:-50]))
    return _CAL[key]


def snr_2500_db(p_signal_plus_noise: float, p_noise: float, sr: int, bw_hz: float) -> float:
    """Carrier SNR in a 2500 Hz reference bandwidth -- the same definition the
    synthetic generator uses (carrier power A^2/2 vs noise power in 2500 Hz).
    A tone of amplitude A gives baseband power A^2/4, so carrier power = 2 * P_bb."""
    s = p_signal_plus_noise - p_noise
    if not np.isfinite(s) or s <= 0:
        return float("nan")  # nothing above the noise floor
    carrier = 2.0 * s
    noise_var = p_noise / noise_gain(sr, bw_hz)
    return float(10 * np.log10(carrier / (noise_var * 2500.0 / (sr / 2))))


# --------------------------------------------------------------------------
# alignment of the schedule to a recording
# --------------------------------------------------------------------------

def slot_plan(start_s_in_hour: float, duration_s: float, band_khz: int, sr: int):
    """All slots whose start lies inside the recording (times relative to recording start)."""
    first = int(np.ceil(start_s_in_hour / SLOT_S))
    plan = []
    k = first
    while True:
        t_rel = k * SLOT_S - start_s_in_hour
        if t_rel > duration_s:
            break
        call = beacon_for_slot(k, band_khz)
        plan.append({"slot": k, "t_rel": t_rel, "callsign": call, "key": callsign_key(call, sr)})
        k += 1
    return plan


def align(power: np.ndarray, plan, frame_rate: int, search_s: float, sr: int):
    """Find the time offset (s) between the nominal schedule and the audio that best
    matches the expected callsign keying. Returns (offset_s, confidence)."""
    z = np.log(power + 1e-20)
    med = np.median(z)
    mad = np.median(np.abs(z - med)) + 1e-9
    z = np.clip((z - med) / (1.4826 * mad), -3, 6)

    pad = int(search_s * frame_rate)
    n = len(z)
    tmpl = np.zeros(n + 2 * pad)
    hop = sr // frame_rate
    for p in plan:
        k = p["key"][: (len(p["key"]) // hop) * hop].reshape(-1, hop).mean(axis=1)
        w = np.where(k > 0.5, 1.0, -1.0)
        w -= w.mean()  # zero-mean per callsign -> insensitive to overall level
        s = int(round(p["t_rel"] * frame_rate)) + pad
        e = min(len(tmpl), s + len(w))
        if s < len(tmpl) and e > 0:
            lo = max(0, s)
            tmpl[lo:e] += w[lo - s:e - s]
    # score(lag) = sum_n z[n] * tmpl[n + pad - lag]  for lag in [-pad, pad]
    corr = sps.correlate(tmpl, z, mode="valid")  # length 2*pad+1, index i <-> lag = pad - i
    lags = (pad - np.arange(len(corr))) / frame_rate
    best = int(np.argmax(corr))
    # confidence: best peak vs the best peak more than 0.5 s away
    far = np.abs(lags - lags[best]) > 0.5
    second = corr[far].max() if far.any() else 0.0
    conf = float(corr[best] / (abs(second) + 1e-9))
    return float(lags[best]), conf


# --------------------------------------------------------------------------
# simulation of a beacon recording (for testing the slicer end-to-end)
# --------------------------------------------------------------------------

def simulate_recording(start_s_in_hour: float, duration_s: float, band_khz: int = 14100, sr: int = 8000,
                       delay_s: float = 2.0, tone_hz: float = 700.0, snr_db: dict | None = None,
                       dash_gap_s: float = 0.25, seed: int = 0, off_air=("YV5B",)):
    """Synthetic off-air recording of the beacon band. snr_db maps callsign -> SNR
    (2500 Hz ref, for the 100 W part). Returns (audio, truth list)."""
    rng = np.random.default_rng(seed)
    n = int(duration_s * sr)
    amp = np.zeros(n)
    truth = []
    plan = slot_plan(start_s_in_hour, duration_s, band_khz, sr)
    unit = 1.2 / BEACON_WPM
    for p in plan:
        call = p["callsign"]
        if call in off_air:
            continue
        snr = snr_db.get(call, rng.uniform(-20, 20)) if snr_db else rng.uniform(-20, 20)
        a = 10 ** (snr / 20)  # carrier amplitude so that SNR matches the synth definition
        key = p["key"].astype(float)
        seq = [key, np.zeros(int(7 * unit * sr))]
        for lvl_db in (0, -10, -20, -30):
            seq += [np.full(sr, 10 ** (lvl_db / 20)), np.zeros(int(dash_gap_s * sr))]
        env = np.concatenate(seq) * a
        s = int(round((p["t_rel"] + delay_s) * sr))
        lo, hi = max(0, s), min(n, s + len(env))
        if hi > lo:
            amp[lo:hi] = env[lo - s:hi - s]
        truth.append({"slot": p["slot"], "callsign": call, "snr_db": snr})
    # soft edges, tone, slow fading, noise
    win = np.hanning(int(0.005 * sr) * 2 + 1)
    amp = np.convolve(amp, win / win.sum(), mode="same")
    fade = 1 + 0.2 * np.sin(2 * np.pi * 0.07 * np.arange(n) / sr)
    x = amp * fade * np.sin(2 * np.pi * tone_hz * np.arange(n) / sr)
    noise_var = 0.5 * (sr / 2) / 2500.0
    x = x + np.sqrt(noise_var) * rng.standard_normal(n)
    sos = sps.butter(4, [200 / (sr / 2), 2700 / (sr / 2)], btype="band", output="sos")
    x = sps.sosfilt(sos, x)
    x = (x / (np.percentile(np.abs(x), 99.9) + 1e-9) * 0.5).clip(-1, 1).astype(np.float32)
    return x, truth
