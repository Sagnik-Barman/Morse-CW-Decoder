"""Loading real (recorded) test data.

Two layouts are accepted:

1. a folder of pairs      real_data/w1aw/2026-10-02.wav  +  .../2026-10-02.txt
2. a manifest CSV         columns: file, text   [optional: snr_db, and any others]
   (file paths are relative to the CSV's folder). scripts/slice_beacons.py
   writes this format as labels.csv.

If a folder contains labels.csv / manifest.csv, that file is used.
"""
from __future__ import annotations

import csv
import glob
import os

from .alphabet import normalize_text

AUDIO_EXT = (".wav", ".flac", ".ogg", ".mp3")


def _read_text(path: str) -> str:
    with open(path, encoding="utf-8", errors="ignore") as f:
        return normalize_text(f.read())


def load_manifest(path: str) -> list[dict]:
    """Return [{'path': abs audio path, 'text': normalised reference, 'snr_db': float|None, ...}]"""
    if os.path.isdir(path):
        for name in ("labels.csv", "manifest.csv"):
            if os.path.exists(os.path.join(path, name)):
                return load_manifest(os.path.join(path, name))
        items = []
        for audio in sorted(glob.glob(os.path.join(path, "**", "*"), recursive=True)):
            if not audio.lower().endswith(AUDIO_EXT):
                continue
            txt = os.path.splitext(audio)[0] + ".txt"
            if os.path.exists(txt):
                items.append({"path": audio, "text": _read_text(txt), "snr_db": None,
                              "name": os.path.relpath(audio, path)})
        return items

    base = os.path.dirname(os.path.abspath(path))
    items = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if str(row.get("use", "1")).strip().lower() in ("0", "false", "no"):
                continue
            snr = row.get("snr_db")
            items.append({**row, "path": os.path.join(base, row["file"]), "text": normalize_text(row["text"]),
                          "snr_db": float(snr) if snr not in (None, "") else None, "name": row["file"]})
    return items
