# morse-cw-decoder

Deep-learning Morse code (CW) decoder: audio in, text out. It is trained entirely on
synthetic audio with realistic impairments and benchmarked against a classical DSP decoder,
on simulated signals and on real off-air radio recorded from public receivers in four countries.

## Results at a glance

| Test | Classical DSP decoder | This project (best model) |
|---|---:|---:|
| Simulated signals, 6–7% character error rate (CER) reached at | +10 dB | **-6 dB** (16 dB more sensitive) |
| **Held-out real radio** (UK + US receivers, normal band conditions, 71 clips) | 95% CER | **17% CER** |
| Held-out real radio, all conditions (132 clips) | 109% CER | **38% CER** |
| Real radio, strong signals (US receiver, ≥ 0 dB) | 8 of 9 callsigns | **9 of 9** |

- **Real-world test set:** 380+ beacon clips recorded from KiwiSDR receivers in India, Australia, the UK and the USA,
  labelled automatically from the international beacon schedule (`scripts/record_beacons.py`, `scripts/slice_beacons.py`).
- **Simulated-to-real gap:** error analysis (fast fading, a clip-length mismatch) cut real-audio CER from 31% to 17%.
- **Open problem:** crowded bands, where other strong stations sit next to the target, are still hard for every model (59–73% CER).
- Full numbers, confidence intervals and the experiment log, including what didn't work, are in [`RESULTS.md`](RESULTS.md).

```
audio (8 kHz) -> log-spectrogram (GPU) -> 2-D CNN -> BiGRU or Conformer -> CTC -> text
```

## What's inside

| Path | What it does |
|---|---|
| `morse/alphabet.py` | Morse table and the 45-symbol CTC vocabulary (blank, space, A-Z, 0-9, `. , ? / = + -`; `+` = AR, `=` = BT) |
| `morse/text_gen.py` | Label text: ham QSO templates, contest exchanges, words, random character groups |
| `morse/synth.py` | Audio generator. Randomises speed (WPM), Farnsworth spacing, hand-keyed timing (fist) with jitter, dash/dot ratio and weight, speed drift, pitch and drift, chirp, edge rise time, Rayleigh/Rician fading (QSB), an interfering signal (QRM), static crashes (QRN), AWGN at a target SNR and the receiver filter |
| `morse/baseline.py` | Classical decoder: tone detection, envelope, 2-means threshold with hysteresis, duration clustering, table lookup |
| `morse/model.py` | `MorseCTC` model: spectrogram front-end, CNN, then **CRNN** (BiGRU) or **Conformer** encoder |
| `morse/decode.py` | Greedy CTC decoding, prefix beam search, and a character n-gram language model |
| `morse/metrics.py` | CER / WER from Levenshtein edit distance |
| `scripts/train.py` | Training: on-the-fly data, SNR curriculum, AMP, warmup + cosine schedule, best/last checkpoints, resume, CSV + TensorBoard logs |
| `scripts/evaluate.py` | Fixed test grid: CER vs SNR and vs WPM for model / beam / LM / baseline, with CSV and plot output |
| `scripts/decode_wav.py` | Decode a real recording (resample, auto-centre the tone, chunk at quiet points) |
| `morse/beacons.py` | NCDXF/IARU beacon schedule, alignment of a recording to the schedule, and SNR measurement |
| `morse/realdata.py` | Loads real test sets (folders of `.wav` + `.txt` pairs, or a `labels.csv` manifest) |
| `scripts/slice_beacons.py` | Cuts an off-air beacon recording into labelled, SNR-measured callsign clips |
| `scripts/record_beacons.py` | Records beacons from a public KiwiSDR (via kiwiclient) and slices them, in one command |
| `scripts/simulate_beacons.py` | Makes a fake beacon recording with known truth, to test the slicer |
| `scripts/benchmark_real.py` | **Real-audio benchmark**: CER / WER / exact match per system, CER vs measured SNR, and the sim-to-real gap plot |
| `real_data/README.md` | How to record and label real data (beacons, W1AW, hand-labelled traffic) |
| `scripts/build_lm.py` | Build the character n-gram LM used in beam search |
| `scripts/make_samples.py` | Write example clips as .wav files (+ spectrograms) so you can listen to the data |
| `configs/*.yaml` | Model and training hyper-parameters |

**SNR convention:** SNR is the key-down carrier power against noise power in a **2500 Hz** bandwidth,
the usual reference for CW decoders. Skilled operators copy down to about -15 dB on this scale.

## Setup (Windows + CUDA)

```powershell
cd D:\Projects\morse-cw-decoder
python -m venv .venv
.venv\Scripts\activate
# 1) PyTorch with CUDA -- pick the command for your CUDA version at https://pytorch.org/get-started/locally/
pip install torch --index-url https://download.pytorch.org/whl/cu124
# 2) everything else
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -m pytest -q tests
```

Always run the scripts from the project root (`D:\Projects\morse-cw-decoder`).

## Quick start

```powershell
# 0. listen to / look at the training data
python scripts/make_samples.py --n 10 --out samples --spectrogram

# 1. baseline numbers (no training needed)
python scripts/evaluate.py --baseline --n-per-snr 100

# 2. train (on-the-fly data; nothing to download)
python scripts/train.py --config configs/crnn.yaml
#    live curves:  tensorboard --logdir runs
#    pause:        Ctrl+C (saves runs/crnn/last.pt immediately)
#    resume:       python scripts/train.py --resume runs/crnn/last.pt
#    shorten / tweak on resume:  --steps 45000 (curriculum shrinks proportionally)
#                  --override model.encoder.attn_dropout=0 train.amp_dtype=fp16
#    find the speed bottleneck (training paused):  python scripts/speed_test.py

# 3. evaluate against the baseline
python scripts/build_lm.py --out lm/char4.json
python scripts/evaluate.py --checkpoint runs/crnn/best.pt --baseline --beam --lm lm/char4.json
python scripts/evaluate.py --checkpoint runs/crnn/best.pt --baseline --condition clean

# 3b. fine-tune on simulator v2 (more realistic fading, "nothing there" examples) -- see RESULTS.md
python scripts/train.py --config configs/crnn_ft.yaml          # ~2 h on an RTX 4050 laptop
python scripts/train.py --config configs/conformer_ft.yaml     # ~3 h
#    compare on the old (v1) synthetic test set:  add  --synth-config configs/crnn.yaml  to evaluate.py
# 3c. fine-tune on simulator v3 (busy band: other stations near the target, some louder) -- see RESULTS.md
python scripts/train.py --config configs/crnn_ft3.yaml         # ~2 h; starts from runs/crnn_ft/best.pt
#    NOTE: v3 was a negative result (worse on held-out real audio) -- see RESULTS.md. Use runs/conformer_ft/best.pt.
#    v3 models expect the wanted signal at ~700 Hz: beacon clips already are; for your own audio
#    use decode_wav.py --auto-center (or shift the wanted signal there yourself)

# 4. the Conformer
python scripts/train.py --config configs/conformer.yaml

# 5. real audio
python scripts/decode_wav.py my_recording.wav --checkpoint runs/crnn/best.pt --auto-center
```

## Real-data benchmark (the numbers that matter)

Synthetic results show *why* a model fails. Results on real, off-air audio show whether it *works*.
[`real_data/README.md`](real_data/README.md) explains how to collect three kinds of labelled real audio:
**beacons** (labels and SNR come from the schedule), **W1AW code practice** (known transcripts) and
**hand-labelled** traffic (real human sending).

```powershell
# beacons, one command (records from a public KiwiSDR and slices; needs kiwiclient, see real_data/README.md)
python scripts/record_beacons.py --kiwi HOST:PORT --minutes 30

# ...or record by hand on a WebSDR/KiwiSDR, note the UTC start time, then
python scripts/slice_beacons.py rec.wav --start-utc 2026-10-02T14:03:17Z --band 14100 --out real_data/beacons/rec1

# benchmark real audio, with the synthetic curve overlaid (sim-to-real gap)
python scripts/benchmark_real.py real_data/beacons/rec1 real_data/w1aw --checkpoint runs/crnn/best.pt --baseline ^
    --synthetic-results results/results_full.csv

# watch real CER during training (a monitor set, not your final test set)
python scripts/train.py --config configs/crnn.yaml --real-val real_data/beacons/rec1
```

### Throughput tips

- Audio is **synthesised on the CPU by DataLoader workers**, and the spectrogram and model run on the GPU.
  If the logged `clips/s` is low and GPU utilisation (in `nvidia-smi`) is well below 90%, the workers are
  the bottleneck, so raise `--num-workers`. A good value is your number of physical cores minus 1 or 2.
  One worker produces roughly 60–120 clips/s, depending on the CPU.
- If you run out of GPU memory, lower `--batch-size`. If you have memory to spare, 128 also works well.
- Windows uses the `spawn` start method, so each epoch's workers take a few seconds to start. That's normal.
- The CRNN at batch 64 runs about 0.05–0.15 s/step on a mid-range RTX card, so the 60k steps in
  `configs/crnn.yaml` take roughly 1–3 hours. The first few thousand steps sit at a high, flat loss while
  CTC learns alignment; this is expected. It should drop sharply before about step 3–5k.
- `--compile` (torch.compile) needs Triton, which usually isn't available on native Windows. Leave it off.

## What training looks like

1. Loss starts around 3–5 (per target, after the `mean` reduction), plateaus while the model outputs only blanks,
   then drops quickly once alignment "clicks".
2. Every `val_every` steps you get CER on a fixed 1000-clip validation set, broken down by SNR (-12 … +20 dB),
   plus three reference/hypothesis pairs.
3. The curriculum widens the SNR range in stages: 10–30 dB, then 0–30, then -8–30, then -14–30 dB.
   Validation CER at low SNR only improves after the later stages begin.

## Experiments worth running (the research part)

1. **Main result:** CER-vs-SNR curves for the baseline, CRNN and Conformer, each with greedy decoding,
   beam search and beam + LM, under both `--condition full` and `--condition clean`.
2. **Ablations:** remove one impairment at a time from training (fading / QRM / QRN / hand-keying) and measure
   how much each one matters. Also compare with and without the curriculum, and different LM weights.
3. **Sim-to-real gap:** use `benchmark_real.py --synthetic-results` to compare real and synthetic CER at the
   same measured SNR. Listen to the real failures, add the missing effect to `morse/synth.py`, retrain,
   and measure how much the gap closes. A labelled real benchmark would be a contribution on its own.
4. **Fist adaptation:** fine-tune on (or condition the model on) a few seconds of one operator's sending.
5. **Multiple signals:** decode every CW signal in the band (the job CW Skimmer does). This would need a
   frequency-aware output head or a detect-then-decode pipeline.

## Known limitations / design choices

- Training clips always contain whole characters (clips are cut at character boundaries). For live
  streaming decoding you would also train on clips that start or end mid-character.
- The target signal is assumed to be the strongest one: QRM is 3–20 dB weaker.
- The target tone is 450–950 Hz during training. For real audio, use `--auto-center`.
- Prosigns other than AR (`+`) and BT (`=`) aren't in the vocabulary. SK, KN and others are decoded as their letters.
