"""Which attention kernel does the Conformer get, and why was attention dropout slow?

    python scripts/sdpa_check.py

PyTorch's F.scaled_dot_product_attention (used in morse/model.py) picks one of several kernels at run
time: FlashAttention, memory-efficient, cuDNN, or the plain "math" fallback, which builds the full
(batch, heads, T, T) attention matrix and runs softmax and dropout as separate GPU operations.
This script (1) measures the real Conformer's training step with attention dropout 0.1 vs 0,
and (2) for the exact attention shape the model uses, forces each kernel in turn to show which
ones accept that input with and without dropout, and how fast each is.
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import yaml

from morse.model import build_model

dev = torch.device("cuda")
DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def sync_time(fn, iters=10, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t) / iters


def model_step_time(cfg, attn_p, batch=32):
    torch.manual_seed(0)
    m = build_model(cfg).to(dev).train()
    for b in m.encoder.blocks:
        b.att.p = attn_p
    shape = {}
    def grab(mod, args):
        shape.setdefault("x", tuple(args[0].shape))   # returns None, so the inputs are left unchanged
    m.encoder.blocks[0].att.register_forward_pre_hook(grab)
    audio = torch.randn(batch, 8 * 8000, device=dev) * 0.1
    alen = torch.full((batch,), 8 * 8000, device=dev, dtype=torch.long)
    alen[batch // 2:] = 5 * 8000  # some padded clips, so the padding mask is really used
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)

    def step():
        with torch.autocast("cuda", dtype=DTYPE):
            lp, ol = m(audio, alen)
        loss = lp.float().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    sec = sync_time(step)
    return sec, shape["x"], m.encoder.blocks[0].att


def backend_table(b, t, d, heads):
    dk = d // heads
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        backends = [("flash", SDPBackend.FLASH_ATTENTION), ("mem_efficient", SDPBackend.EFFICIENT_ATTENTION),
                    ("math", SDPBackend.MATH)]
        if hasattr(SDPBackend, "CUDNN_ATTENTION"):
            backends.insert(2, ("cudnn", SDPBackend.CUDNN_ATTENTION))
        ctx = lambda be: sdpa_kernel(be)
    except ImportError:  # older PyTorch
        names = {"flash": dict(enable_flash=True, enable_mem_efficient=False, enable_math=False),
                 "mem_efficient": dict(enable_flash=False, enable_mem_efficient=True, enable_math=False),
                 "math": dict(enable_flash=False, enable_mem_efficient=False, enable_math=True)}
        backends = [(n, n) for n in names]
        ctx = lambda be: torch.backends.cuda.sdp_kernel(**names[be])

    q, k, v = (torch.randn(b, heads, t, dk, device=dev, dtype=DTYPE, requires_grad=True) for _ in range(3))
    lengths = torch.full((b,), t, device=dev)
    lengths[b // 2:] = int(t * 5 / 8)
    keep = (torch.arange(t, device=dev)[None, :] < lengths[:, None])[:, None, None, :]

    print(f"\nattention shape: batch {b}, heads {heads}, T {t}, head dim {dk}, {str(DTYPE).replace('torch.', '')}, padding mask on")
    print(f"{'kernel':<16}{'dropout 0':>16}{'dropout 0.1':>16}")
    for name, be in [("default (auto)", None)] + backends:
        cells = []
        for p in (0.0, 0.1):
            def run():
                y = F.scaled_dot_product_attention(q, k, v, attn_mask=keep, dropout_p=p)
                y.float().sum().backward()
            try:
                if be is None:
                    ms = sync_time(run) * 1000
                else:
                    with ctx(be):
                        ms = sync_time(run) * 1000
                cells.append(f"{ms:9.2f} ms")
            except RuntimeError as e:
                cells.append("not usable")
        print(f"{name:<16}{cells[0]:>16}{cells[1]:>16}")


def main():
    print(f"torch {torch.__version__}, {torch.cuda.get_device_name(0)}, autocast {DTYPE}")
    with open("configs/conformer.yaml") as f:
        cfg = yaml.safe_load(f)
    s_drop, x_shape, att = model_step_time(cfg, 0.1)
    s_none, _, _ = model_step_time(cfg, 0.0)
    print(f"\nwhole Conformer training step, batch 32 of 8 s clips:")
    print(f"  attention dropout 0.1: {s_drop * 1000:7.0f} ms  ({32 / s_drop:5.0f} clips/s)")
    print(f"  attention dropout 0  : {s_none * 1000:7.0f} ms  ({32 / s_none:5.0f} clips/s)")
    print(f"  speed-up: {s_drop / s_none:.2f}x")
    b, t, d = x_shape
    backend_table(b, t, d, att.h)


if __name__ == "__main__":
    main()
