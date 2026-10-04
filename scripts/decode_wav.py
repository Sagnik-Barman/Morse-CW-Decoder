"""Decode a real recording (wav/flac/ogg) with a trained model and/or the baseline.

    python scripts/decode_wav.py recording.wav --checkpoint runs/crnn/best.pt
    python scripts/decode_wav.py recording.wav --checkpoint runs/crnn/best.pt --auto-center --beam --lm lm/char4.json
    python scripts/decode_wav.py recording.wav --baseline-only

Tips for real audio (e.g. WebSDR / KiwiSDR recordings in CW mode):
  * the model was trained on tones between 450 and 950 Hz; --auto-center moves the
    strongest tone in 200-2000 Hz to 700 Hz first
  * long files are split into chunks at the quietest point near each chunk end,
    so characters are not cut in half
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from morse.baseline import estimate_tone
from morse.infer import baseline_long, load_audio, shift_tone

SR = 8000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--baseline-only", action="store_true")
    ap.add_argument("--auto-center", action="store_true", help="shift the strongest tone to --center-hz")
    ap.add_argument("--center-hz", type=float, default=700.0)
    ap.add_argument("--chunk-seconds", type=float, default=16.0, help="baseline chunk length")
    ap.add_argument("--model-chunk-seconds", type=float, default=8.0, help="model chunk length (training clips are 8 s)")
    ap.add_argument("--beam", action="store_true")
    ap.add_argument("--beam-size", type=int, default=16)
    ap.add_argument("--lm", default=None)
    ap.add_argument("--lm-weight", type=float, default=0.3)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    x = load_audio(args.audio, SR)
    print(f"{args.audio}: {len(x) / SR:.1f} s")
    if args.auto_center:
        f0 = estimate_tone(x, SR, 200, 2000)
        print(f"strongest tone at {f0:.0f} Hz -> shifting to {args.center_hz:.0f} Hz")
        x = shift_tone(x, SR, f0, args.center_hz)

    print("\n[baseline]")
    print(baseline_long(x, SR, args.chunk_seconds))
    if args.baseline_only or not args.checkpoint:
        return

    from morse.infer import load_checkpoint, pick_device, transcribe_long
    lm = None
    if args.lm:
        from morse.decode import CharNgramLM
        lm = CharNgramLM.load(args.lm)
    device = pick_device(args.device)
    model, cfg, _ = load_checkpoint(args.checkpoint, device)
    decoder = "beam" if (args.beam or lm) else "greedy"
    text = transcribe_long(model, x, device, SR, args.model_chunk_seconds, decoder=decoder,
                           beam=args.beam_size, lm=lm, lm_weight=args.lm_weight)
    print(f"\n[{cfg['model']['arch']} / {decoder}]")
    print(text)


if __name__ == "__main__":
    main()
