"""Build a character n-gram language model for beam-search decoding.

    python scripts/build_lm.py --out lm/char4.json                  # from the synthetic text generator
    python scripts/build_lm.py --out lm/char4.json --text my_qsos.txt  # from your own transcripts

Only structured traffic (QSO / contest / words) is used from the generator:
random-character groups carry no language information.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from morse.alphabet import normalize_text
from morse.decode import CharNgramLM
from morse.text_gen import sample_text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="lm/char4.json")
    ap.add_argument("--order", type=int, default=4)
    ap.add_argument("--k", type=float, default=0.1, help="add-k smoothing")
    ap.add_argument("--n", type=int, default=200000, help="generated sentences")
    ap.add_argument("--text", default=None, help="optional text file, one message per line")
    args = ap.parse_args()

    rng = np.random.default_rng(7)
    texts = [normalize_text(sample_text(rng, {"qso": 0.45, "contest": 0.2, "words": 0.35})) for _ in range(args.n)]
    if args.text:
        with open(args.text, encoding="utf-8") as f:
            texts += [normalize_text(l) for l in f if l.strip()]
    lm = CharNgramLM(args.order, args.k).fit(texts)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    lm.save(args.out)
    print(f"saved {args.order}-gram LM over {len(texts)} texts, {len(lm.counts)} contexts -> {args.out}")


if __name__ == "__main__":
    main()
