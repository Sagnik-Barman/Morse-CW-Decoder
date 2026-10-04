"""Train a CTC Morse decoder on endlessly generated synthetic audio.

    python scripts/train.py --config configs/crnn.yaml
    python scripts/train.py --config configs/conformer.yaml --batch-size 128
    python scripts/train.py --resume runs/crnn/last.pt

Outputs go to runs/<run_name>/ : best.pt, last.pt, log.csv, config.yaml
(and TensorBoard events if tensorboard is installed).
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from collections import defaultdict
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from morse.data import FixedSet, SynthStream, collate, synth_config_from_dict
from morse.infer import pick_device, transcribe_dataset
from morse.metrics import ErrorRate
from morse.model import build_model, count_parameters

VAL_SNRS = [-12, -9, -6, -3, 0, 5, 10, 20]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/crnn.yaml")
    ap.add_argument("--resume", default=None, help="path to a checkpoint (last.pt) to continue from")
    ap.add_argument("--out-dir", default="runs")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--init", default=None, help="fine-tune: start from this checkpoint's weights (fresh optimiser)")
    ap.add_argument("--override", nargs="*", default=None, metavar="KEY=VALUE",
                    help="change config values, e.g. model.encoder.attn_dropout=0 train.amp_dtype=fp16")
    ap.add_argument("--accum-steps", type=int, default=None,
                    help="micro-batches per optimizer step (effective batch = batch-size x accum-steps)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--compile", action="store_true", help="torch.compile the model (needs triton; Linux)")
    ap.add_argument("--real-val", nargs="+", default=None,
                    help="real recordings to monitor during training (folders / labels.csv; see real_data/README.md)")
    return ap.parse_args()


def lr_lambda(step, warmup, total):
    if step < warmup:
        return (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return 0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * progress))


def load_real_val(paths):
    from morse.infer import load_audio
    from morse.realdata import load_manifest
    items = []
    for pth in paths:
        for it in load_manifest(pth):
            if it["text"]:
                items.append((load_audio(it["path"]), it["text"]))
    return items


def validate_real(model, real_items, device):
    from morse.infer import transcribe_long
    er = ErrorRate()
    exact = 0
    for audio, ref in real_items:
        hyp = transcribe_long(model, audio, device)
        er.update(ref, hyp)
        exact += int(hyp == ref)
    return er, exact / max(1, len(real_items))


def validate(model, val_set, device, batch_size):
    hyps, refs, metas = transcribe_dataset(model, val_set, device, batch_size=batch_size)
    total = ErrorRate()
    by_snr = defaultdict(ErrorRate)
    for h, r, m in zip(hyps, refs, metas):
        total.update(r, h)
        by_snr[round(m["snr_db"])].update(r, h)
    examples = list(zip(refs[:3], hyps[:3]))
    return total, {k: v.cer for k, v in sorted(by_snr.items())}, examples


def main():
    args = parse_args()
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        cfg = ckpt["config"]
    else:
        ckpt = None
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
    tcfg = cfg["train"]
    if args.batch_size:
        tcfg["batch_size"] = args.batch_size
    for item in args.override or []:
        key, _, val = item.partition("=")
        node = cfg
        parts = key.split(".")
        for k in parts[:-1]:
            node = node.setdefault(k, {})
        node[parts[-1]] = yaml.safe_load(val)
        print(f"override: {key} = {node[parts[-1]]!r}")
    if args.steps:
        cur = tcfg["curriculum"]
        last = cur[-1]["until_step"]
        if args.steps < last:
            # shorten the whole run: stage boundaries shrink proportionally
            scale = args.steps / last
            for st in cur:
                st["until_step"] = int(round(st["until_step"] * scale))
        cur[-1]["until_step"] = args.steps
        tcfg["steps"] = args.steps
        print("curriculum: " + ", ".join(f"<{st['until_step']}: {st['snr_db_range']} dB" for st in cur))
    if args.accum_steps is not None:
        tcfg["accum_steps"] = args.accum_steps
    if args.num_workers is not None:
        tcfg["num_workers"] = args.num_workers
    if args.no_amp:
        tcfg["amp"] = False
    run_name = args.run_name or cfg.get("run_name", "run")
    run_dir = os.path.join(args.out_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    device = pick_device(args.device)
    torch.manual_seed(tcfg.get("seed", 0))
    np.random.seed(tcfg.get("seed", 0))
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model = build_model(cfg).to(device)
    print(f"model: {cfg['model']['arch']}  params: {count_parameters(model) / 1e6:.2f}M  device: {device}")
    init_from = args.init or tcfg.get("init_from")
    if init_from and ckpt is None:
        # fine-tuning: start from another run's weights, with a fresh optimiser / schedule / curriculum
        src = torch.load(init_from, map_location="cpu", weights_only=False)
        model.load_state_dict(src["model"])
        print(f"initialised weights from {init_from} (step {src.get('step', '?')} of its run)")
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"], betas=(0.9, 0.98))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_lambda(s, tcfg["warmup_steps"], tcfg["steps"]))

    use_amp = tcfg.get("amp", True) and device.type == "cuda"
    want = str(tcfg.get("amp_dtype", "auto")).lower()
    if want == "bf16" or (want == "auto" and device.type == "cuda" and torch.cuda.is_bf16_supported()):
        amp_dtype = torch.bfloat16
    else:
        amp_dtype = torch.float16
    if use_amp:
        print(f"mixed precision: {amp_dtype}")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)

    step, best_cer = 0, float("inf")
    if ckpt is not None:
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        sched.load_state_dict(ckpt["scheduler"])
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        step, best_cer = ckpt["step"], ckpt.get("best_cer", float("inf"))
        print(f"resumed from {args.resume} at step {step} (best CER {best_cer:.4f})")

    train_model = torch.compile(model) if args.compile else model

    synth_cfg = synth_config_from_dict(cfg.get("synth"))
    val_cfg = replace(synth_cfg, snr_db_range=(-14.0, 30.0))
    print(f"generating fixed validation set ({tcfg['val_size']} clips)...")
    t0 = time.time()
    val_set = FixedSet(val_cfg, tcfg["val_size"], seed=12345, snr_values=VAL_SNRS, progress=True)
    print(f"  done in {time.time() - t0:.0f}s")

    real_paths = args.real_val or tcfg.get("real_val") or []
    if isinstance(real_paths, str):
        real_paths = [real_paths]
    real_items = load_real_val(real_paths) if real_paths else []
    if real_paths:
        print(f"real-audio monitor set: {len(real_items)} recordings "
              f"({sum(len(a) for a, _ in real_items) / 8000 / 60:.1f} min)")
    select_on = tcfg.get("select_on", "synthetic")
    if select_on == "real" and not real_items:
        select_on = "synthetic"

    tb = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        tb = SummaryWriter(run_dir)
    except Exception:
        pass
    log_path = os.path.join(run_dir, "log.csv")
    new_log = not os.path.exists(log_path) or ckpt is None
    log_f = open(log_path, "w" if new_log else "a", newline="")
    log = csv.writer(log_f)
    if new_log:
        log.writerow(["step", "loss", "lr", "clips_per_s", "val_cer", "val_wer", "real_cer", "real_wer"])

    def save(path):
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                    "scaler": scaler.state_dict(), "step": step, "best_cer": best_cer, "config": cfg}, path)

    total_steps = tcfg["steps"]
    running, n_run, t_last, clips = 0.0, 0, time.time(), 0
    accum = max(1, int(tcfg.get("accum_steps", 1)))
    micro = 0
    if accum > 1:
        print(f"gradient accumulation: {accum} x {tcfg['batch_size']} = effective batch {accum * tcfg['batch_size']}")
    try:
        for stage_idx, stage in enumerate(tcfg["curriculum"]):
            until = min(stage["until_step"], total_steps)
            if step >= until:
                continue
            stage_cfg = replace(synth_cfg, snr_db_range=tuple(stage["snr_db_range"]))
            print(f"== stage {stage_idx}: steps {step}->{until}, SNR {stage_cfg.snr_db_range} dB")
            nw = tcfg["num_workers"]
            loader = DataLoader(SynthStream(stage_cfg, seed=tcfg.get("seed", 0) * 1_000_003 + step),
                                batch_size=tcfg["batch_size"], collate_fn=collate, num_workers=nw,
                                pin_memory=device.type == "cuda", prefetch_factor=4 if nw > 0 else None)
            model.train()
            micro = 0
            for batch in loader:
                audio = batch["audio"].to(device, non_blocking=True)
                alen = batch["audio_lengths"].to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    log_probs, out_len = train_model(audio, alen)
                loss = F.ctc_loss(log_probs.float().transpose(0, 1), batch["targets"].to(device),
                                  out_len, batch["target_lengths"].to(device),
                                  blank=0, reduction="mean", zero_infinity=True)
                # gradient accumulation: `accum` micro-batches of batch_size make one optimizer step
                if micro == 0:
                    opt.zero_grad(set_to_none=True)
                scaler.scale(loss / accum).backward()
                micro += 1
                running += loss.item()
                n_run += 1
                clips += audio.shape[0]
                if micro < accum:
                    continue
                micro = 0
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
                scaler.step(opt)
                scaler.update()
                sched.step()
                step += 1

                if step % tcfg["log_every"] == 0:
                    dt = time.time() - t_last
                    cps = clips / max(dt, 1e-9)
                    lr = sched.get_last_lr()[0]
                    print(f"step {step:6d}  loss {running / n_run:.4f}  lr {lr:.2e}  {cps:.0f} clips/s")
                    log.writerow([step, f"{running / n_run:.5f}", f"{lr:.3e}", f"{cps:.1f}", "", ""])
                    log_f.flush()
                    if tb:
                        tb.add_scalar("train/loss", running / n_run, step)
                        tb.add_scalar("train/lr", lr, step)
                    running, n_run, t_last, clips = 0.0, 0, time.time(), 0

                if step % tcfg["val_every"] == 0 or step == total_steps:
                    er, by_snr, ex = validate(model, val_set, device, tcfg["batch_size"])
                    snr_str = "  ".join(f"{k:+d}dB:{v:.3f}" for k, v in by_snr.items())
                    print(f"[val] step {step}  CER {er.cer:.4f}  WER {er.wer:.4f}\n      by SNR  {snr_str}")
                    for r, h in ex:
                        print(f"      ref: {r}\n      hyp: {h}")
                    real_er = None
                    if real_items:
                        real_er, real_exact = validate_real(model, real_items, device)
                        print(f"[real] CER {real_er.cer:.4f}  WER {real_er.wer:.4f}  exact {real_exact:.1%}  "
                              f"(sim-to-real gap: {real_er.cer - er.cer:+.4f} CER)")
                    log.writerow([step, "", "", "", f"{er.cer:.5f}", f"{er.wer:.5f}",
                                  f"{real_er.cer:.5f}" if real_er else "", f"{real_er.wer:.5f}" if real_er else ""])
                    log_f.flush()
                    if tb:
                        tb.add_scalar("val/cer", er.cer, step)
                        tb.add_scalar("val/wer", er.wer, step)
                        if real_er:
                            tb.add_scalar("val_real/cer", real_er.cer, step)
                            tb.add_scalar("val_real/wer", real_er.wer, step)
                        for k, v in by_snr.items():
                            tb.add_scalar(f"val_cer_by_snr/{k:+d}dB", v, step)
                    score = real_er.cer if (select_on == "real" and real_er) else er.cer
                    if score < best_cer:
                        best_cer = score
                        save(os.path.join(run_dir, "best.pt"))
                        print(f"      new best -> {run_dir}/best.pt")
                    save(os.path.join(run_dir, "last.pt"))
                    model.train()
                if step >= until:
                    break
            del loader
    except KeyboardInterrupt:
        # Ctrl+C: save progress so `--resume runs/<name>/last.pt` continues from here
        save(os.path.join(run_dir, "last.pt"))
        log_f.close()
        print(f"\ninterrupted at step {step} -> saved {run_dir}/last.pt")
        print(f"resume with:  python scripts/train.py --resume {run_dir}/last.pt")
        return
    log_f.close()
    print(f"finished. best val CER {best_cer:.4f}")


if __name__ == "__main__":
    main()
