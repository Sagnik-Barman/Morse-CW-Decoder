"""Find what limits training speed on this machine (run it while training is PAUSED).

    python scripts/speed_test.py

Times one forward+backward pass of the models with different settings, plus the
CPU data generator, and prints clips/s for each. Paste the output to decide fixes.
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import yaml

from morse.data import SynthStream, collate, synth_config_from_dict
from morse.model import build_model


def timed(model, batch, device, dtype, iters=8, warmup=3):
    audio = batch["audio"].to(device)
    alen = batch["audio_lengths"].to(device)
    tgt, tlen = batch["targets"].to(device), batch["target_lengths"].to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)

    def step():
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
            lp, ol = model(audio, alen)
        loss = F.ctc_loss(lp.float().transpose(0, 1), tgt, ol, tlen, zero_infinity=True)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

    for _ in range(warmup):
        step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t = time.time()
    for _ in range(iters):
        step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.time() - t) / iters


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    print(f"torch {torch.__version__}  device {device}"
          + (f"  {torch.cuda.get_device_name(0)}" if device.type == "cuda" else ""))
    if device.type == "cuda":
        print(f"SDPA backends enabled: flash={torch.backends.cuda.flash_sdp_enabled()} "
              f"mem_efficient={torch.backends.cuda.mem_efficient_sdp_enabled()} math={torch.backends.cuda.math_sdp_enabled()}")

    with open("configs/conformer.yaml") as f:
        ccfg = yaml.safe_load(f)
    with open("configs/crnn.yaml") as f:
        rcfg = yaml.safe_load(f)
    gen_it = iter(SynthStream(synth_config_from_dict(ccfg.get("synth")), seed=1))
    batches = {n: collate([next(gen_it) for _ in range(n)]) for n in (32, 64)}

    # CPU data generation speed (one worker)
    t = time.time()
    for _ in range(32):
        next(gen_it)
    per_worker = 32 / (time.time() - t)
    print(f"\ndata generation: {per_worker:.0f} clips/s per worker "
          f"(x {os.cpu_count()} logical CPUs; config uses {ccfg['train']['num_workers']} workers)")

    rows = []

    def run(name, cfg, dtype, bs, tweak=None):
        torch.manual_seed(0)
        m = build_model(cfg).to(device)
        if tweak:
            tweak(m)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        try:
            sec = timed(m, batches[bs], device, dtype)
            mem = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else float("nan")
            rows.append((name, bs, sec * 1000, bs / sec, mem))
        except RuntimeError as e:
            msg = "out of memory" if "out of memory" in str(e) else str(e).splitlines()[0][:60]
            rows.append((name, bs, float("nan"), float("nan"), float("nan")))
            print(f"  {name} (batch {bs}): {msg}")
        del m
        if device.type == "cuda":
            torch.cuda.empty_cache()

    def no_attn_dropout(m):
        for b in m.encoder.blocks:
            b.att.p = 0.0

    def no_dwconv(m):
        no_attn_dropout(m)
        for b in m.encoder.blocks:
            b.conv.dw = torch.nn.Identity()

    bf16 = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
    fp16 = torch.float16
    run("A crnn fp16 (reference)", rcfg, fp16, 64)
    run("B conformer bf16 (current run)", ccfg, bf16, 32)
    run("C conformer bf16, attn dropout 0", ccfg, bf16, 32, no_attn_dropout)
    run("D conformer fp16, attn dropout 0", ccfg, fp16, 32, no_attn_dropout)
    run("E conformer bf16 (current run)", ccfg, bf16, 64)
    run("F conformer bf16, attn dropout 0", ccfg, bf16, 64, no_attn_dropout)
    run("G conformer fp16, attn dropout 0", ccfg, fp16, 64, no_attn_dropout)
    run("H [diagnostic] C without depthwise conv", ccfg, bf16, 32, no_dwconv)

    print(f"\n{'setting':<42}{'batch':>6}{'ms/batch':>10}{'clips/s':>9}{'GPU GB':>8}")
    for name, bs, ms, cps, mem in rows:
        print(f"{name:<42}{bs:>6}{ms:>10.0f}{cps:>9.0f}{mem:>8.2f}")
    print("\n(training also does validation and data loading, so real clips/s is a bit lower)")


if __name__ == "__main__":
    main()
