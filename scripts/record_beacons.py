"""Record NCDXF/IARU beacons from a public KiwiSDR receiver, then slice them into labelled clips.

One-time setup (next to this project):
    cd D:\\Projects
    git clone https://github.com/jks-prv/kiwiclient.git
    (no git? download the ZIP from that page and unzip it to D:\\Projects\\kiwiclient)

Then:
    python scripts/record_beacons.py --kiwi HOST:PORT --minutes 30
    python scripts/record_beacons.py --kiwi HOST:PORT --minutes 30 --band 21150

HOST:PORT is the address of any public KiwiSDR, e.g. from the map at http://rx.linkfanel.net
(open a receiver in your browser; the address bar shows something like abc.proxy.kiwisdr.com:8073).

What it does:
  1. runs kiwirecorder (USB mode, dial 0.7 kHz below the beacon -> the beacon is a 700 Hz tone,
     2.4 kHz receiver filter like the training data) for --minutes;
     the file is named with its exact UTC start time, e.g. 20261002T141317Z_14099300_usb.wav
  2. runs slice_beacons.py on it -> real_data/beacons/<file name>/ (clips + labels.csv)
"""
from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from morse.beacons import BANDS_KHZ  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kiwi", required=True, help="KiwiSDR address HOST:PORT (port is usually 8073)")
    ap.add_argument("--minutes", type=float, default=30.0)
    ap.add_argument("--band", type=int, default=14100, choices=BANDS_KHZ, help="beacon frequency in kHz")
    ap.add_argument("--kiwiclient", default=os.path.join(os.path.dirname(ROOT), "kiwiclient"),
                    help="folder of the kiwiclient download (default: next to this project)")
    ap.add_argument("--raw-dir", default=os.path.join(ROOT, "real_data", "raw"))
    ap.add_argument("--user", default="morse-research", help="name shown to the receiver's owner")
    ap.add_argument("--no-slice", action="store_true")
    ap.add_argument("--start-timeout", type=float, default=60.0,
                    help="give up if no audio has arrived after this many seconds")
    args = ap.parse_args()

    rec = os.path.join(args.kiwiclient, "kiwirecorder.py")
    if not os.path.exists(rec):
        sys.exit(f"kiwirecorder.py not found in {args.kiwiclient}\n"
                 "download it once:  git clone https://github.com/jks-prv/kiwiclient.git  (next to this project)")
    # accept "host:port", or a full browser URL like http://host:8073/?f=14100usb
    addr = args.kiwi.split("://", 1)[-1].split("/", 1)[0].split("?", 1)[0]
    host, _, port = addr.partition(":")
    port = port or "8073"
    os.makedirs(args.raw_dir, exist_ok=True)
    before = set(glob.glob(os.path.join(args.raw_dir, "*.wav")))

    dial = args.band - 0.7  # USB: a carrier at the beacon frequency becomes a 700 Hz tone
    cmd = [sys.executable, rec, "-s", host, "-p", port, "-f", f"{dial:.1f}", "-m", "usb",
           "--tlimit", str(int(args.minutes * 60)), "-d", args.raw_dir, "--user", args.user, "--ncomp",
           "--connect-retries", "3", "--connect-timeout", "10", "--busy-retries", "2", "--busy-timeout", "15"]
    print("recording:", " ".join(cmd))
    print(f"(about {args.minutes:.0f} min; Ctrl+C stops early and keeps what was recorded)")
    proc = subprocess.Popen(cmd)
    t0 = time.time()
    try:
        while proc.poll() is None:
            time.sleep(1)
            started = set(glob.glob(os.path.join(args.raw_dir, "*.wav"))) - before
            if time.time() - t0 > args.minutes * 60 + 30:
                # kiwirecorder's --tlimit restarts whenever the receiver drops us and it reconnects
                # (many receivers cut sessions at 15-60 min), so enforce the total time here.
                # Safe to stop: kiwirecorder rewrites the WAV header after every block.
                print(f"\nreached {args.minutes:.0f} min in total; stopping"
                      + (f" (the receiver disconnected us, so the audio is in {len(started)} files)"
                         if len(started) > 1 else ""))
                proc.terminate()
                proc.wait(timeout=10)
                break
            if not started and time.time() - t0 > args.start_timeout:
                proc.terminate()
                proc.wait(timeout=10)
                sys.exit(f"\nno audio from {host}:{port} after {args.start_timeout:.0f} s -- the receiver is probably "
                         "offline, full, or password-protected.\nOpen http://" + f"{host}:{port}" +
                         " in your browser to check, or try another receiver.")
    except KeyboardInterrupt:
        print("\nstopped early")
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()

    new = sorted(set(glob.glob(os.path.join(args.raw_dir, "*.wav"))) - before)
    if not new:
        sys.exit("no recording was made -- check the address, or try another receiver (it may be full)")
    for wav in new:
        mb = os.path.getsize(wav) / 1e6
        print(f"saved {wav} ({mb:.1f} MB)")
        if mb < 0.2:
            print("  very short -- the receiver may have dropped the connection; skipping")
            continue
        if not args.no_slice:
            out = os.path.join(ROOT, "real_data", "beacons", os.path.splitext(os.path.basename(wav))[0])
            subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "slice_beacons.py"), wav,
                            "--out", out], check=False)


if __name__ == "__main__":
    main()
