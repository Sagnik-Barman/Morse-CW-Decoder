"""Cut an off-air NCDXF/IARU beacon recording into labelled, SNR-measured clips.

    python scripts/slice_beacons.py rec_14100.wav --start-utc 2026-10-02T14:03:17Z --band 14100 --out real_data/beacons/rec1

How to record: open a WebSDR / KiwiSDR, tune to the beacon frequency in CW
mode (14.100, 18.110, 21.150, 24.930 or 28.200 MHz), press record, and note
the UTC time at which the recording started (a few seconds of error is fine:
the script searches +-search-seconds and aligns on the callsign keying).

For every 10-second slot it writes:
  clips/<n>_<CALL>.wav   the callsign only (the power-step dashes are cut off)
  labels.csv             file, text (=callsign), snr_db (measured), detection flags,
                         per-dash SNRs (100 W / 10 W / 1 W / 0.1 W ladder), slot time
The `use` column is 1 when the beacon was detected; scripts/benchmark_real.py
skips rows with use=0. Beacons that are off-air get use=0 automatically.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import soundfile as sf

from morse.baseline import estimate_tone
from morse.beacons import (BANDS_KHZ, BEACON_WPM, align, baseband_power, parse_utc_seconds_in_hour,
                           slot_plan, snr_2500_db)
from morse.infer import load_audio, shift_tone

SR = 8000
FR = 100          # envelope frame rate (frames / s)
BW = 40.0         # baseband low-pass (Hz) -> ~80 Hz wide measurement filter


def erode(mask: np.ndarray, k: int) -> np.ndarray:
    """Drop k frames at every edge of each True run (keeps only stable parts)."""
    out = mask.copy()
    for _ in range(k):
        out = out & np.roll(out, 1) & np.roll(out, -1)
    return out


def fmt(v: float) -> str:
    return f"{v:.1f}" if np.isfinite(v) else ""


def best_window(power: np.ndarray, start: int, stop: int, length: int):
    """Start index of the length-frame window with the highest mean power in [start, stop)."""
    stop = min(stop, len(power) - length)
    if stop <= start:
        return None
    c = np.convolve(power[start:stop + length], np.ones(length) / length, mode="valid")
    return start + int(np.argmax(c[: stop - start]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("recording")
    ap.add_argument("--start-utc", default=None,
                    help="UTC time the recording started, e.g. 2026-10-02T14:03:17Z or 14:03:17. "
                         "Optional for kiwirecorder files: their names contain it (20261002T140317Z_...)")
    ap.add_argument("--band", type=int, default=None, choices=BANDS_KHZ,
                    help="kHz (optional for kiwirecorder files: read from the file name)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tone-hz", type=float, default=None, help="beacon audio tone (default: auto-detect)")
    ap.add_argument("--tone-search", type=float, nargs=2, default=(500.0, 900.0), metavar=("LO", "HI"),
                    help="auto-detect the tone in this range (Hz); widen it, e.g. 200 2000, for recordings "
                         "not made with record_beacons.py")
    ap.add_argument("--center-hz", type=float, default=700.0, help="shift the tone here (model trained on 450-950 Hz)")
    ap.add_argument("--no-shift", action="store_true")
    ap.add_argument("--search-seconds", type=float, default=8.0, help="+- search range for the time offset")
    ap.add_argument("--offset", type=float, default=None, help="skip the search and use this offset (s)")
    ap.add_argument("--min-snr", type=float, default=-20.0, help="SNR below which a slot is marked use=0")
    ap.add_argument("--pre", type=float, default=0.4, help="seconds kept before the callsign")
    ap.add_argument("--post", type=float, default=0.12, help="seconds kept after the last callsign element")
    args = ap.parse_args()

    x = load_audio(args.recording, SR)
    dur = len(x) / SR
    # record_beacons.py tunes 0.7 kHz below the beacon, so its tone is ~700 Hz: search only near there,
    # or a stronger unrelated carrier elsewhere in the passband can be picked instead (seen on a JA receiver).
    lo, hi = args.tone_search
    f0 = args.tone_hz or estimate_tone(x, SR, lo, hi)

    if args.start_utc is None:
        m = re.search(r"(\d{8})T(\d{2})(\d{2})(\d{2})Z", os.path.basename(args.recording))
        if not m:
            sys.exit("no --start-utc given and the file name has no kiwirecorder timestamp (YYYYMMDDTHHMMSSZ)")
        args.start_utc = f"{m.group(2)}:{m.group(3)}:{m.group(4)}"
        print(f"start time from file name: {args.start_utc} UTC")
    if args.band is None:
        m = re.search(r"Z_(\d{7,8})_", os.path.basename(args.recording))
        if m:
            khz = int(m.group(1)) / 1000
            args.band = min(BANDS_KHZ, key=lambda b: abs(b - khz))
            if abs(args.band - khz) > 5:
                sys.exit(f"file was recorded at {khz:.1f} kHz, not near a beacon frequency {BANDS_KHZ}")
        else:
            args.band = 14100
    print(f"{args.recording}: {dur:.1f} s, beacon tone {f0:.0f} Hz, band {args.band} kHz")
    start = parse_utc_seconds_in_hour(args.start_utc)
    plan = slot_plan(start, dur, args.band, SR)
    power = baseband_power(x, SR, f0, BW, FR)

    if args.offset is None:
        offset, conf = align(power, plan, FR, args.search_seconds, SR)
        print(f"alignment: audio is {offset:+.2f} s relative to the schedule (confidence {conf:.2f}; "
              f"< 1.3 means weak/unclear -- check the UTC start time and the band)")
    else:
        offset, conf = args.offset, float("nan")

    if not args.no_shift and abs(f0 - args.center_hz) > 5:
        x = shift_tone(x, SR, f0, args.center_hz)

    os.makedirs(os.path.join(args.out, "clips"), exist_ok=True)
    rows = []
    hop = SR // FR
    for i, p in enumerate(plan):
        t0 = p["t_rel"] + offset
        key = p["key"]
        kf = key[: (len(key) // hop) * hop].reshape(-1, hop).mean(axis=1)
        s = int(round(t0 * FR))
        e = s + len(kf)
        if s - int(args.pre * FR) < 0 or e + 5 * FR > len(power):
            continue  # slot not fully inside the recording
        seg = power[s:e]
        marks = erode(kf > 0.9, 2)
        spaces = erode(kf < 0.1, 2)
        p_mark = float(seg[marks].mean()) if marks.any() else np.nan
        p_space = float(seg[spaces].mean()) if spaces.any() else np.nan
        snr_cs = snr_2500_db(p_mark, p_space, SR, BW)
        detected = bool(p_mark > 1.3 * p_space and np.isfinite(snr_cs) and snr_cs >= args.min_snr)

        # power-step ladder: four 1 s dashes after the callsign
        dash_snr, dash_start = [], []
        cur = e
        for k in range(4):
            w = best_window(power, cur, cur + int((1.2 if k == 0 else 0.8) * FR), int(0.9 * FR))
            if w is None:
                break
            dash_snr.append(snr_2500_db(float(power[w:w + int(0.9 * FR)].mean()), p_space, SR, BW))
            dash_start.append(w)
            cur = w + FR
        ladder_ok = bool(len(dash_snr) >= 2 and np.isfinite(dash_snr[:2]).all() and dash_snr[0] > -5
                         and 6 <= dash_snr[0] - dash_snr[1] <= 14)

        clip_end = e / FR + args.post
        if dash_start and np.isfinite(dash_snr[0]) and dash_snr[0] > -5:
            clip_end = min(clip_end, dash_start[0] / FR - 0.03)
        a, b = int((t0 - args.pre) * SR), int(clip_end * SR)
        name = f"{i:04d}_{p['callsign']}.wav"
        sf.write(os.path.join(args.out, "clips", name), x[a:b], SR)
        hh = int(p["slot"] * 10 // 60)
        rows.append({
            "file": f"clips/{name}", "text": p["callsign"], "use": int(detected),
            "snr_db": fmt(snr_cs), "beacon": p["callsign"], "band_khz": args.band,
            "slot_mmss": f"{hh:02d}:{int(p['slot'] * 10 % 60):02d}",
            "snr_dash_100w": fmt(dash_snr[0]) if len(dash_snr) > 0 else "",
            "snr_dash_10w": fmt(dash_snr[1]) if len(dash_snr) > 1 else "",
            "snr_dash_1w": fmt(dash_snr[2]) if len(dash_snr) > 2 else "",
            "snr_dash_100mw": fmt(dash_snr[3]) if len(dash_snr) > 3 else "",
            "ladder_ok": int(ladder_ok), "offset_s": f"{offset:.2f}", "source": os.path.basename(args.recording),
        })

    with open(os.path.join(args.out, "labels.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["file", "text"])
        w.writeheader()
        w.writerows(rows)
    used = [r for r in rows if r["use"]]
    print(f"{len(rows)} slots -> {len(used)} detected (use=1), {sum(r['ladder_ok'] for r in rows)} with a clean power ladder")
    for r in rows:
        flag = "ok " if r["use"] else "-- "
        print(f"  {flag}{r['slot_mmss']}  {r['beacon']:<7} SNR {r['snr_db']:>6} dB   "
              f"ladder {r['snr_dash_100w']:>6} {r['snr_dash_10w']:>6} {r['snr_dash_1w']:>6} {r['snr_dash_100mw']:>6}")
    print(f"wrote {args.out}/labels.csv and {len(rows)} clips  (callsigns at {BEACON_WPM:.0f} WPM)")
    n_ladder = sum(r["ladder_ok"] for r in rows)
    if n_ladder == 0 or (conf == conf and conf < 1.3):
        print("\nWARNING: no beacon was clearly heard (weak alignment and/or no clean power ladder).\n"
              "The band was probably closed at this receiver, and the 'detected' clips are likely just noise.\n"
              "Don't use this recording for testing; try another receiver or another time of day.")


if __name__ == "__main__":
    main()
