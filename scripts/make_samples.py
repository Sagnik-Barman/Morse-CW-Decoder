"""Write a few synthetic clips as .wav (+ labels.csv) so you can listen to the data.

    python scripts/make_samples.py --n 20 --out samples
    python scripts/make_samples.py --n 5 --snr -6 --text "CQ CQ DE VU2ABC K"
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import soundfile as sf

from morse.alphabet import text_to_morse
from morse.synth import SynthConfig, generate_sample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", default="samples")
    ap.add_argument("--snr", type=float, default=None)
    ap.add_argument("--wpm", type=float, default=None)
    ap.add_argument("--text", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--spectrogram", action="store_true", help="also save a spectrogram PNG per clip")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    cfg = SynthConfig()
    rows = []
    for i in range(args.n):
        x, label, m = generate_sample(rng, cfg, text=args.text, snr_db=args.snr, wpm=args.wpm)
        name = f"clip_{i:03d}.wav"
        sf.write(os.path.join(args.out, name), x, cfg.sample_rate)
        rows.append({"file": name, "text": label, "morse": text_to_morse(label),
                     **{k: (round(v, 2) if isinstance(v, float) else v) for k, v in m.items()}})
        if args.spectrogram:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.specgram(x, NFFT=256, Fs=cfg.sample_rate, noverlap=192, cmap="magma")
            ax.set_ylim(0, 2000)
            ax.set_title(f"{label}  |  {m['wpm']:.0f} WPM, {m['snr_db']:.0f} dB")
            fig.tight_layout()
            fig.savefig(os.path.join(args.out, name.replace(".wav", ".png")), dpi=110)
            plt.close(fig)
    with open(os.path.join(args.out, "labels.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.n} clips to {args.out}/")


if __name__ == "__main__":
    main()
