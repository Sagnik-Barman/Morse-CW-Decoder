"""Is attention dropout slow because of compute, or because of GPU memory?

    python scripts/speed_dropout.py

Times one real Conformer training step (CTC loss, AdamW, batch 32 of 8 s clips, configs/conformer.yaml)
with attention dropout 0.1 and 0, in bf16 and fp16. Each setting runs in a FRESH Python process, so
memory left over from one setting cannot slow down the next. Prints time, clips/s and peak GPU memory
next to the card's physical memory: on Windows, a step that needs more than the card has keeps
running but spills into much slower system RAM.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def one(p: float, dtype_name: str, batch: int) -> None:
    import torch
    import yaml
    from morse.data import SynthStream, collate, synth_config_from_dict
    from morse.model import build_model
    from speed_test import timed

    dev = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    with open("configs/conformer.yaml") as f:
        cfg = yaml.safe_load(f)
    gen = iter(SynthStream(synth_config_from_dict(cfg.get("synth")), seed=1))
    b = collate([next(gen) for _ in range(batch)])
    torch.manual_seed(0)
    m = build_model(cfg).to(dev)
    for blk in m.encoder.blocks:
        blk.att.p = p
    torch.cuda.reset_peak_memory_stats()
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[dtype_name]
    sec = timed(m, b, dev, dtype, iters=10, warmup=3)
    print(json.dumps({"ms": sec * 1000, "cps": batch / sec,
                      "peak_gb": torch.cuda.max_memory_allocated() / 2**30,
                      "reserved_gb": torch.cuda.max_memory_reserved() / 2**30,
                      "card_gb": torch.cuda.get_device_properties(0).total_memory / 2**30}))


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--one":
        one(float(sys.argv[2]), sys.argv[3], int(sys.argv[4]))
        return
    rows = []
    for p in (0.1, 0.0):
        for dt in ("bf16", "fp16"):
            out = subprocess.run([sys.executable, os.path.abspath(__file__), "--one", str(p), dt, "32"],
                                 capture_output=True, text=True)
            line = [l for l in out.stdout.splitlines() if l.startswith("{")]
            if not line:
                print(f"attn dropout {p} {dt}: failed\n{out.stderr[-800:]}")
                continue
            r = json.loads(line[-1])
            rows.append((p, dt, r))
            print(f"done: attn dropout {p}, {dt}", flush=True)
    if not rows:
        return
    card = rows[0][2]["card_gb"]
    print(f"\nGPU physical memory: {card:.2f} GB")
    print(f"{'attn dropout':<14}{'precision':<11}{'ms/step':>9}{'clips/s':>9}{'peak GB':>9}{'reserved GB':>13}")
    for p, dt, r in rows:
        print(f"{p:<14}{dt:<11}{r['ms']:>9.0f}{r['cps']:>9.0f}{r['peak_gb']:>9.2f}{r['reserved_gb']:>13.2f}")


if __name__ == "__main__":
    main()
