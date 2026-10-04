"""Benchmark decoders on a fixed synthetic test grid: CER vs SNR and vs WPM.

    # baseline only (no training needed)
    python scripts/evaluate.py --baseline

    # a trained model vs the baseline, greedy + beam search with a char LM
    python scripts/evaluate.py --checkpoint runs/crnn/best.pt --baseline --beam --lm lm/char4.json

    # clean-channel condition (AWGN only: no fading / QRM / QRN / hand-keying)
    python scripts/evaluate.py --checkpoint runs/crnn/best.pt --condition clean

Writes <out>/results_<condition>.csv, per_sample_<condition>.csv and cer_vs_snr_<condition>.png
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from collections import defaultdict
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from morse.baseline import decode_audio
from morse.data import FixedSet, synth_config_from_dict
from morse.metrics import ErrorRate

WPM_BUCKETS = [(10, 20), (20, 30), (30, 40), (40, 50)]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--baseline", action="store_true", help="also evaluate the classical decoder")
    ap.add_argument("--beam", action="store_true", help="also evaluate prefix beam search")
    ap.add_argument("--beam-size", type=int, default=16)
    ap.add_argument("--lm", default=None, help="char n-gram LM json (from scripts/build_lm.py)")
    ap.add_argument("--lm-weight", type=float, default=0.3)
    ap.add_argument("--char-bonus", type=float, default=0.5)
    ap.add_argument("--snrs", default="-15,-12,-9,-6,-3,0,3,6,10,20")
    ap.add_argument("--n-per-snr", type=int, default=200)
    ap.add_argument("--condition", choices=["full", "clean"], default="full")
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--synth-config", default=None,
                    help="take simulator settings from this YAML (default: the checkpoint's own)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="results")
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    snrs = [float(s) for s in args.snrs.split(",")]

    model = cfg = device = lm = None
    if args.checkpoint:
        from morse.infer import load_checkpoint, pick_device
        device = pick_device(args.device)
        model, cfg, _ = load_checkpoint(args.checkpoint, device)
    synth_src = (cfg or {}).get("synth")
    if args.synth_config:
        # evaluate under another config's simulator, e.g. the v1 test set for a v2-trained model
        import yaml
        with open(args.synth_config) as f:
            synth_src = yaml.safe_load(f).get("synth")
        print(f"simulator settings from {args.synth_config}")
    synth = synth_config_from_dict(synth_src)
    if args.condition == "clean":
        synth = replace(synth, p_hand=0.0, p_fading=0.0, p_qrm=0.0, p_qrn=0.0, p_chirp=0.0,
                        p_farnsworth=0.0, speed_drift_range=(0.0, 0.0), freq_drift_hz_range=(0.0, 0.0))
    if args.lm:
        from morse.decode import CharNgramLM
        lm = CharNgramLM.load(args.lm)

    n = args.n_per_snr * len(snrs)
    print(f"generating {n} test clips ({args.condition} condition)...")
    test = FixedSet(synth, n, seed=args.seed, snr_values=snrs, progress=True)
    metas = test.meta
    refs = test.labels

    systems = {}
    if model is not None:
        from morse.infer import transcribe_dataset
        t = time.time()
        hyps, _, _ = transcribe_dataset(model, test, device, batch_size=args.batch_size)
        systems[f"{cfg['model']['arch']}_greedy"] = hyps
        print(f"greedy decoding: {time.time() - t:.1f}s")
        if args.beam or lm is not None:
            t = time.time()
            name = f"{cfg['model']['arch']}_beam" + ("_lm" if lm else "")
            hyps, _, _ = transcribe_dataset(model, test, device, batch_size=args.batch_size, decoder="beam",
                                            beam=args.beam_size, lm=lm, lm_weight=args.lm_weight,
                                            char_bonus=args.char_bonus)
            systems[name] = hyps
            print(f"beam decoding: {time.time() - t:.1f}s")
    if args.baseline:
        t = time.time()
        systems["baseline"] = [decode_audio(test[i][0], synth.sample_rate) for i in range(len(test))]
        print(f"baseline decoding: {time.time() - t:.1f}s")
    if not systems:
        print("nothing to evaluate: pass --checkpoint and/or --baseline")
        return

    # ---- aggregate
    rows = []
    summary = {}
    for name, hyps in systems.items():
        overall = ErrorRate()
        by_snr = defaultdict(ErrorRate)
        by_wpm = defaultdict(ErrorRate)
        for h, r, m in zip(hyps, refs, metas):
            overall.update(r, h)
            by_snr[m["snr_db"]].update(r, h)
            for lo, hi in WPM_BUCKETS:
                if lo <= m["wpm"] < hi:
                    by_wpm[f"{lo}-{hi}"].update(r, h)
        summary[name] = (overall, by_snr, by_wpm)
        for s in snrs:
            rows.append({"system": name, "group": "snr_db", "value": s, "cer": by_snr[s].cer, "wer": by_snr[s].wer})
        for k, e in by_wpm.items():
            if e.n == 0:
                continue
            rows.append({"system": name, "group": "wpm", "value": k, "cer": e.cer, "wer": e.wer})
        rows.append({"system": name, "group": "overall", "value": "", "cer": overall.cer, "wer": overall.wer})

    names = list(systems)
    print(f"\nCER by SNR ({args.condition} condition, {args.n_per_snr} clips per SNR)")
    print("| SNR (dB) | " + " | ".join(names) + " |")
    print("|---" * (len(names) + 1) + "|")
    for s in snrs:
        print(f"| {s:+.0f} | " + " | ".join(f"{summary[nm][1][s].cer:.3f}" for nm in names) + " |")
    print("\nCER by speed")
    print("| WPM | " + " | ".join(names) + " |")
    print("|---" * (len(names) + 1) + "|")
    for lo, hi in WPM_BUCKETS:
        k = f"{lo}-{hi}"
        cells = [f"{summary[nm][2][k].cer:.3f}" if summary[nm][2][k].n else "-" for nm in names]
        print(f"| {k} | " + " | ".join(cells) + " |")
    print("\nOverall: " + ", ".join(f"{nm} CER {summary[nm][0].cer:.3f} / WER {summary[nm][0].wer:.3f}" for nm in names))

    res_path = os.path.join(args.out, f"results_{args.condition}.csv")
    with open(res_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["system", "group", "value", "cer", "wer"])
        w.writeheader()
        w.writerows(rows)
    per_path = os.path.join(args.out, f"per_sample_{args.condition}.csv")
    with open(per_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["idx", "snr_db", "wpm", "hand", "fading", "qrm", "qrn", "ref"] + names)
        for i, (r, m) in enumerate(zip(refs, metas)):
            w.writerow([i, m["snr_db"], f"{m['wpm']:.1f}", m["hand"], m["fading"], m["qrm"], m["qrn"], r]
                       + [systems[nm][i] for nm in names])
    print(f"\nwrote {res_path} and {per_path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4.2))
        for nm in names:
            ax.plot(snrs, [min(summary[nm][1][s].cer, 1.0) for s in snrs], marker="o", label=nm)
        ax.set_xlabel("SNR (dB, 2500 Hz reference bandwidth)")
        ax.set_ylabel("character error rate (clipped at 1)")
        ax.set_title(f"CER vs SNR - {args.condition} channel")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        png = os.path.join(args.out, f"cer_vs_snr_{args.condition}.png")
        fig.savefig(png, dpi=150)
        print(f"wrote {png}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
