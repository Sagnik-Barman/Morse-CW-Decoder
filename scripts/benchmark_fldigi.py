"""Benchmark fldigi (a widely used classical CW decoder) on the SAME held-out real clips
that the neural models were scored on, so "neural vs classical" is measured against a
real-world decoder, not only against morse/baseline.py.

fldigi cannot decode a wav file from the command line in its standard Windows build, so
this script plays each clip into fldigi in real time through a virtual audio cable and
reads back what fldigi decoded over its XML-RPC interface (port 7362 by default).

Setup (once):
  1. Install fldigi (http://www.w1hkj.org) and VB-Audio Virtual Cable (https://vb-audio.com/Cable/).
  2. In fldigi: Configure > Sound Card > Devices > PortAudio: Capture = "CABLE Output (VB-Audio
     Virtual Cable)". Op Mode > CW. Configure > Modems > CW: receive speed 22 WPM, speed tracking on.
  3. pip install sounddevice

Then:
  python scripts/benchmark_fldigi.py --list-devices          # find the CABLE Input device
  python scripts/benchmark_fldigi.py --check                 # one clip end to end
  python scripts/benchmark_fldigi.py                         # full held-out set, squelch off and on

Held-out set (default): real_data/test_wessex.csv (112 clips, has a `busy` column) plus the three
Fayetteville, US recordings, keeping clips with measured SNR >= -10 dB (20 clips, none busy).
That is the same 132-clip set as the "Pooled held-out set" table in RESULTS.md.

Each clip is padded to 8 s with matched background noise (morse.infer.pad_with_noise), exactly as
the neural models received it, and played at 48 kHz. Spaces fldigi inserts inside a callsign are
removed before scoring (the references have none).

fldigi's score depends on its settings, so a small grid is run and every cell is reported:
squelch off / on (how much noise it prints) x lead-in 0 / 3 s (matched noise played before the clip,
so fldigi's threshold and speed tracker have settled, as they would for an operator already
listening). Quote fldigi's BEST cell. That choice is made on the test set itself, which favours
fldigi, so it makes the comparison conservative for the neural models.

Outputs (in --out): fldigi_per_file.csv and fldigi_summary.csv, and a printed table in the same
groups as RESULTS.md (all / quiet / quiet -10..0 dB / busy).
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import xmlrpc.client
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from morse.alphabet import normalize_text
from morse.infer import load_audio, pad_with_noise
from morse.metrics import ErrorRate
from morse.realdata import load_manifest

SR = 8000
PLAY_SR = 48000
US_RECORDINGS = ["20261003T194729Z_14099300_usb", "20261003T200500Z_14099300_usb",
                 "20261003T200855Z_14099300_usb"]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wessex", default="real_data/test_wessex.csv")
    ap.add_argument("--us-dirs", nargs="*", default=[os.path.join("real_data", "beacons", d) for d in US_RECORDINGS])
    ap.add_argument("--min-snr", type=float, default=-10.0, help="applied to the US recordings")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7362)
    ap.add_argument("--device", default="CABLE Input", help="substring of the output device name")
    ap.add_argument("--carrier", type=int, default=700, help="audio frequency of the beacon tone in the clips")
    ap.add_argument("--squelch", choices=["off", "on", "both"], default="both")
    ap.add_argument("--pad-seconds", type=float, default=8.0)
    ap.add_argument("--lead-seconds", type=float, nargs="+", default=[0.0, 3.0],
                    help="matched noise played before each clip; each value is a separate pass")
    ap.add_argument("--gap", type=float, default=1.5, help="silence after each clip before reading fldigi")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--list-devices", action="store_true")
    ap.add_argument("--check", action="store_true",
                    help="run 4 clips (2 quiet, 2 busy) through every setting and print what fldigi decoded")
    ap.add_argument("--out", default="results_fldigi")
    return ap.parse_args()


def load_items(args) -> list[dict]:
    items = []
    if args.wessex and os.path.exists(args.wessex):
        for it in load_manifest(args.wessex):
            it["source"] = "wessex"
            it["busy"] = int(str(it.get("busy", "0")).strip() in ("1", "1.0", "True", "true"))
            items.append(it)
    for d in args.us_dirs:
        if not os.path.isdir(d):
            print(f"warning: {d} not found, skipped")
            continue
        for it in load_manifest(d):
            if it["snr_db"] is None or it["snr_db"] < args.min_snr:
                continue
            it["source"] = "us"
            it["busy"] = 0
            items.append(it)
    return [it for it in items if it["text"]]


def find_device(sd, name_part: str) -> int:
    for i, dev in enumerate(sd.query_devices()):
        if name_part.lower() in dev["name"].lower() and dev["max_output_channels"] > 0:
            return i
    raise SystemExit(f"no output device containing '{name_part}'. Run with --list-devices.")


def to_play(x: np.ndarray, pad_s: float, lead_s: float = 0.0) -> np.ndarray:
    from scipy.signal import resample_poly
    x = x.astype(np.float32)
    body = pad_with_noise(x, SR, int(pad_s * SR))            # what the neural models received
    n_lead = int(lead_s * SR)
    if n_lead > 0:                                           # same background, tone notched out
        lead = pad_with_noise(x, SR, len(x) + n_lead, seed=1)[len(x):]
        body = np.concatenate([lead, body])
    x = body
    y = resample_poly(x, PLAY_SR // SR, 1).astype(np.float32)
    peak = float(np.max(np.abs(y))) or 1.0
    return 0.5 * y / peak


def rx_text(server) -> str:
    data = server.rx.get_data()
    raw = data.data if isinstance(data, xmlrpc.client.Binary) else data
    if isinstance(raw, bytes):
        raw = raw.decode("latin-1", errors="ignore")
    return str(raw)


def clean(text: str, ref: str) -> str:
    """Normalise fldigi's output like the references. fldigi often splits a callsign with spaces
    when the keying gaps vary; the references are single callsigns with no spaces, so spaces are
    removed when the reference has none (otherwise every split would count as an error)."""
    out = normalize_text(text.upper()).strip()
    return out.replace(" ", "") if " " not in ref else out


def run_pass(server, sd, dev, items, audios, args, squelch: bool, tag: str) -> list[str]:
    server.main.set_squelch(squelch)
    hyps = []
    t0 = time.time()
    for i, (it, y) in enumerate(zip(items, audios)):
        rx_text(server)                     # flush anything decoded before this clip
        server.text.clear_rx()
        sd.play(y, PLAY_SR, device=dev, blocking=True)
        time.sleep(args.gap)
        raw = rx_text(server)
        hyps.append(clean(raw, it["text"]))
        if args.check or (i + 1) % 10 == 0:
            el = time.time() - t0
            print(f"  [{tag}] {i + 1}/{len(items)}  ref={it['text']!r} "
                  f"fldigi={raw.strip()!r}  ({el / (i + 1) * (len(items) - i - 1) / 60:.0f} min left)", flush=True)
    return hyps


def summarise(items, hyps) -> dict:
    groups = defaultdict(ErrorRate)
    for it, h in zip(items, hyps):
        r = it["text"]
        groups["all"].update(r, h)
        groups["busy" if it["busy"] else "quiet"].update(r, h)
        if not it["busy"] and it["snr_db"] is not None and -10 <= it["snr_db"] < 0:
            groups["quiet, -10..0 dB"].update(r, h)
        groups[f"source {it['source']}"].update(r, h)
    exact = defaultdict(int)
    for it, h in zip(items, hyps):
        exact["all"] += int(h == it["text"])
    return {"groups": groups, "exact_all": exact["all"]}


def main():
    args = parse_args()
    import sounddevice as sd
    if args.list_devices:
        print(sd.query_devices())
        return
    dev = find_device(sd, args.device)
    print(f"output device: {sd.query_devices(dev)['name']}")

    server = xmlrpc.client.ServerProxy(f"http://{args.host}:{args.port}")
    try:
        print(f"fldigi connected; modem was {server.modem.get_name()}")
    except (OSError, xmlrpc.client.Fault) as e:
        raise SystemExit(f"cannot reach fldigi XML-RPC at {args.host}:{args.port} ({e}). Is fldigi running?")
    server.modem.set_by_name("CW")
    server.modem.set_carrier(args.carrier)
    print(f"modem {server.modem.get_name()}, carrier {server.modem.get_carrier()} Hz")

    items = load_items(args)
    if args.check:
        quiet = [it for it in items if not it["busy"]][:2]
        busy = [it for it in items if it["busy"]][:2]
        items = quiet + busy
    elif args.limit:
        items = items[:args.limit]
    print(f"{len(items)} clips: {sum(it['source'] == 'wessex' for it in items)} Wessex, "
          f"{sum(it['source'] == 'us' for it in items)} US; {sum(it['busy'] for it in items)} busy")
    raw_audio = [load_audio(it["path"], SR) for it in items]
    audios = {lead: [to_play(x, args.pad_seconds, lead) for x in raw_audio] for lead in args.lead_seconds}
    squelches = {"off": [False], "on": [True], "both": [False, True]}[args.squelch]
    passes = [(lead, sq) for lead in args.lead_seconds for sq in squelches]
    est = sum(sum(len(y) / PLAY_SR + args.gap + 0.3 for y in audios[lead]) for lead, _ in passes) / 60
    print(f"{len(passes)} passes (lead-in {args.lead_seconds} s x squelch {args.squelch}); about {est:.0f} min in total")

    results = {}
    for lead, sq in passes:
        name = f"fldigi_lead{lead:g}s_sq{'on' if sq else 'off'}"
        print(f"\n{name}")
        results[name] = run_pass(server, sd, dev, items, audios[lead], args, sq, name)

    if args.check:
        return
    os.makedirs(args.out, exist_ok=True)
    names = list(results)
    with open(os.path.join(args.out, "fldigi_per_file.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["source", "file", "snr_db", "busy", "ref"] + names)
        for i, it in enumerate(items):
            w.writerow([it["source"], it["name"], it["snr_db"], it["busy"], it["text"]]
                       + [results[n][i] for n in names])

    order = ["all", "quiet", "quiet, -10..0 dB", "busy", "source wessex", "source us"]
    summ = {n: summarise(items, results[n]) for n in names}
    print("\nCER on the held-out real clips (compare with RESULTS.md, 'Pooled held-out set')")
    print("| group | n | " + " | ".join(names) + " |")
    print("|---" * (len(names) + 2) + "|")
    for g in order:
        n = summ[names[0]]["groups"][g].n
        if n:
            print(f"| {g} | {n} | " + " | ".join(f"{summ[nm]['groups'][g].cer:.3f}" for nm in names) + " |")
    print("| exact callsign, all | | " + " | ".join(str(summ[nm]["exact_all"]) for nm in names) + " |")
    best = min(names, key=lambda nm: summ[nm]["groups"]["all"].cer)
    print(f"\nbest fldigi setting on all clips: {best} (CER {summ[best]['groups']['all'].cer:.3f})")
    with open(os.path.join(args.out, "fldigi_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["system", "group", "n", "cer", "wer"])
        for nm in names:
            for g in order:
                e = summ[nm]["groups"][g]
                if e.n:
                    w.writerow([nm, g, e.n, f"{e.cer:.5f}", f"{e.wer:.5f}"])
    print(f"\nwrote {args.out}/fldigi_per_file.csv and fldigi_summary.csv")


if __name__ == "__main__":
    main()
