"""Make a fake off-air beacon recording (known truth) to test slice_beacons.py.

    python scripts/simulate_beacons.py --out sim_beacons.wav --start-utc 14:03:17 --delay 2.3
    python scripts/slice_beacons.py sim_beacons.wav --start-utc 14:03:17 --out real_data/sim
    # compare real_data/sim/labels.csv with sim_beacons_truth.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import soundfile as sf

from morse.beacons import BANDS_KHZ, parse_utc_seconds_in_hour, simulate_recording


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="sim_beacons.wav")
    ap.add_argument("--start-utc", default="14:03:17")
    ap.add_argument("--minutes", type=float, default=6.0)
    ap.add_argument("--band", type=int, default=14100, choices=BANDS_KHZ)
    ap.add_argument("--delay", type=float, default=2.3, help="true audio delay vs the schedule (s)")
    ap.add_argument("--tone-hz", type=float, default=620.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    x, truth = simulate_recording(parse_utc_seconds_in_hour(args.start_utc), args.minutes * 60, args.band,
                                  delay_s=args.delay, tone_hz=args.tone_hz, seed=args.seed)
    sf.write(args.out, x, 8000)
    tpath = os.path.splitext(args.out)[0] + "_truth.csv"
    with open(tpath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["slot", "callsign", "snr_db"])
        w.writeheader()
        w.writerows(truth)
    print(f"wrote {args.out} ({len(x) / 8000:.0f} s) and {tpath}")


if __name__ == "__main__":
    main()
