"""Benchmark decoders on REAL recordings.

    # beacon clips from slice_beacons.py (have measured SNR -> CER vs SNR)
    python scripts/benchmark_real.py real_data/beacons/rec1 --checkpoint runs/crnn/best.pt --baseline

    # several sources at once, with beam search + LM, and the synthetic curve overlaid
    python scripts/benchmark_real.py real_data/beacons/rec1 real_data/beacons/rec2 real_data/w1aw ^
        --checkpoint runs/crnn/best.pt --baseline --beam --lm lm/char4.json ^
        --synthetic-results results/results_full.csv

Each data argument is a folder of <name>.wav + <name>.txt pairs, a folder containing
labels.csv, or a CSV manifest (columns file,text[,snr_db,use]).  See real_data/README.md.

Reports: CER / WER / exact-match rate per system (overall and per source), CER by
measured SNR when available, and the sim-to-real gap against evaluate.py's curve.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from morse.baseline import estimate_tone
from morse.infer import baseline_long, load_audio, shift_tone
from morse.metrics import ErrorRate
from morse.realdata import load_manifest

SR = 8000


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", nargs="+")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--beam", action="store_true")
    ap.add_argument("--beam-size", type=int, default=16)
    ap.add_argument("--lm", default=None)
    ap.add_argument("--lm-weight", type=float, default=0.3)
    ap.add_argument("--char-bonus", type=float, default=0.5)
    ap.add_argument("--auto-center", action="store_true",
                    help="shift each file's strongest tone to 700 Hz (not needed for slice_beacons clips)")
    ap.add_argument("--chunk-seconds", type=float, default=8.0)
    ap.add_argument("--no-pad", action="store_true",
                    help="don't pad short clips with matched noise to --chunk-seconds (old behaviour)")
    ap.add_argument("--snr-bins", default="-30,-15,-10,-5,0,5,10,20,60")
    ap.add_argument("--synthetic-results", default=None, help="results_*.csv from evaluate.py to overlay")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="results_real")
    return ap.parse_args()


def bin_label(v, edges):
    for lo, hi in zip(edges[:-1], edges[1:]):
        if lo <= v < hi:
            return (lo, hi)
    return None


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    items = []
    for d in args.data:
        src = os.path.basename(os.path.normpath(d if os.path.isdir(d) else os.path.dirname(os.path.abspath(d))))
        for it in load_manifest(d):
            it["source"] = src
            items.append(it)
    items = [it for it in items if it["text"]]
    if not items:
        print("no labelled audio found")
        return
    print(f"{len(items)} labelled recordings from {len(args.data)} source(s), "
          f"{sum(1 for it in items if it['snr_db'] is not None)} with measured SNR")

    audios = []
    for it in items:
        x = load_audio(it["path"], SR)
        if args.auto_center:
            x = shift_tone(x, SR, estimate_tone(x, SR, 200, 2000), 700.0)
        audios.append(x)
    total_s = sum(len(a) for a in audios) / SR
    print(f"total audio: {total_s / 60:.1f} min")

    systems = {}
    if args.checkpoint:
        from morse.infer import load_checkpoint, pick_device, transcribe_long
        device = pick_device(args.device)
        model, cfg, _ = load_checkpoint(args.checkpoint, device)
        arch = cfg["model"]["arch"]
        t = time.time()
        systems[f"{arch}_greedy"] = [transcribe_long(model, a, device, SR, args.chunk_seconds,
                                                       pad_short=not args.no_pad) for a in audios]
        print(f"{arch} greedy: {time.time() - t:.1f}s")
        lm = None
        if args.lm:
            from morse.decode import CharNgramLM
            lm = CharNgramLM.load(args.lm)
        if args.beam or lm:
            t = time.time()
            systems[f"{arch}_beam" + ("_lm" if lm else "")] = [
                transcribe_long(model, a, device, SR, args.chunk_seconds, decoder="beam", beam=args.beam_size,
                                lm=lm, lm_weight=args.lm_weight, char_bonus=args.char_bonus,
                                pad_short=not args.no_pad) for a in audios]
            print(f"{arch} beam: {time.time() - t:.1f}s")
    if args.baseline:
        t = time.time()
        systems["baseline"] = [baseline_long(a, SR) for a in audios]
        print(f"baseline: {time.time() - t:.1f}s")
    if not systems:
        print("nothing to evaluate: pass --checkpoint and/or --baseline")
        return

    names = list(systems)
    refs = [it["text"] for it in items]
    edges = [float(v) for v in args.snr_bins.split(",")]
    has_snr = any(it["snr_db"] is not None for it in items)

    summary = {}
    for nm in names:
        overall, by_src, by_snr = ErrorRate(), defaultdict(ErrorRate), defaultdict(ErrorRate)
        exact = 0
        for it, h, r in zip(items, systems[nm], refs):
            overall.update(r, h)
            by_src[it["source"]].update(r, h)
            exact += int(h == r)
            if it["snr_db"] is not None:
                b = bin_label(it["snr_db"], edges)
                if b:
                    by_snr[b].update(r, h)
        summary[nm] = {"overall": overall, "src": by_src, "snr": by_snr, "exact": exact / len(items)}

    print("\nOverall (real audio)")
    print("| system | CER | WER | exact match |")
    print("|---|---|---|---|")
    for nm in names:
        s = summary[nm]
        print(f"| {nm} | {s['overall'].cer:.3f} | {s['overall'].wer:.3f} | {s['exact']:.1%} |")

    sources = sorted({it["source"] for it in items})
    if len(sources) > 1:
        print("\nCER by source")
        print("| source | n | " + " | ".join(names) + " |")
        print("|---" * (len(names) + 2) + "|")
        for src in sources:
            n = summary[names[0]]["src"][src].n
            print(f"| {src} | {n} | " + " | ".join(f"{summary[nm]['src'][src].cer:.3f}" for nm in names) + " |")

    bins = [b for b in zip(edges[:-1], edges[1:]) if any(summary[nm]["snr"][b].n for nm in names)]
    if has_snr and bins:
        print("\nCER by measured SNR (dB, 2500 Hz ref)")
        print("| SNR bin | n | " + " | ".join(names) + " |")
        print("|---" * (len(names) + 2) + "|")
        for b in bins:
            n = summary[names[0]]["snr"][b].n
            print(f"| {b[0]:+.0f}..{b[1]:+.0f} | {n} | " + " | ".join(f"{summary[nm]['snr'][b].cer:.3f}" for nm in names) + " |")

    # ---- files
    with open(os.path.join(args.out, "real_per_file.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["source", "file", "snr_db", "ref"] + names)
        for i, it in enumerate(items):
            w.writerow([it["source"], it["name"], "" if it["snr_db"] is None else it["snr_db"], refs[i]]
                       + [systems[nm][i] for nm in names])
    with open(os.path.join(args.out, "real_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["system", "group", "value", "n", "cer", "wer"])
        for nm in names:
            s = summary[nm]
            w.writerow([nm, "overall", "", s["overall"].n, f"{s['overall'].cer:.5f}", f"{s['overall'].wer:.5f}"])
            for src, e in s["src"].items():
                w.writerow([nm, "source", src, e.n, f"{e.cer:.5f}", f"{e.wer:.5f}"])
            for b, e in s["snr"].items():
                w.writerow([nm, "snr_bin", f"{b[0]:g}..{b[1]:g}", e.n, f"{e.cer:.5f}", f"{e.wer:.5f}"])
    print(f"\nwrote {args.out}/real_per_file.csv and real_summary.csv")

    # ---- plot: real CER vs measured SNR, optionally with the synthetic curves
    if not (has_snr and bins):
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    colors = {}
    for nm in names:
        xs, ys = [], []
        for b in bins:
            e = summary[nm]["snr"][b]
            if e.n:
                vals = [it["snr_db"] for it in items if it["snr_db"] is not None and b[0] <= it["snr_db"] < b[1]]
                xs.append(float(np.median(vals)))
                ys.append(min(e.cer, 1.0))
        line, = ax.plot(xs, ys, marker="o", label=f"{nm} (real)")
        colors[nm] = line.get_color()
    if args.synthetic_results and os.path.exists(args.synthetic_results):
        syn = defaultdict(list)
        with open(args.synthetic_results) as f:
            for r in csv.DictReader(f):
                if r["group"] == "snr_db":
                    syn[r["system"]].append((float(r["value"]), min(float(r["cer"]), 1.0)))
        for nm, pts in syn.items():
            pts.sort()
            ax.plot([p[0] for p in pts], [p[1] for p in pts], linestyle="--", color=colors.get(nm),
                    alpha=0.8, label=f"{nm} (synthetic)")
    ax.set_xlabel("measured SNR (dB, 2500 Hz reference)")
    ax.set_ylabel("character error rate (clipped at 1)")
    ax.set_title("Real vs synthetic: the sim-to-real gap")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    png = os.path.join(args.out, "real_cer_vs_snr.png")
    fig.savefig(png, dpi=150)
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
