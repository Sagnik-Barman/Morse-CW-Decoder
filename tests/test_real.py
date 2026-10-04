import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import soundfile as sf

from morse.baseline import estimate_tone
from morse.beacons import (align, baseband_power, beacon_for_slot, parse_utc_seconds_in_hour,
                           simulate_recording, slot_plan)
from morse.realdata import load_manifest


def test_schedule_matches_ncdxf_table():
    # rows of https://www.ncdxf.org/beacon/beaconschedule.html
    assert beacon_for_slot(0, 14100) == "4U1UN"
    assert beacon_for_slot(1, 18110) == "4U1UN"
    assert beacon_for_slot(4, 28200) == "4U1UN"
    assert beacon_for_slot(17, 14100) == "YV5B"
    assert beacon_for_slot(0, 18110) == "YV5B"
    assert beacon_for_slot(0, 28200) == "CS3B"
    assert beacon_for_slot(18, 14100) == "4U1UN"  # 3-minute cycle


def test_parse_utc():
    assert parse_utc_seconds_in_hour("2026-10-02T14:03:17Z") == 197
    assert parse_utc_seconds_in_hour("14:03:17") == 197
    assert parse_utc_seconds_in_hour("03:17") == 197


def test_alignment_recovers_delay():
    start = parse_utc_seconds_in_hour("14:03:17")
    x, truth = simulate_recording(start, 120, 14100, delay_s=2.3, tone_hz=620, seed=1)
    f0 = estimate_tone(x, 8000, 200, 2000)
    assert abs(f0 - 620) < 10
    # the user noted a start time 4 s too early -> schedule looks 4 s later -> offset 2.3 - 4
    plan = slot_plan(start - 4.0, 120, 14100, 8000)
    off, conf = align(baseband_power(x, 8000, f0), plan, 100, 8.0, 8000)
    assert abs(off - (2.3 - 4.0)) < 0.05 and conf > 1.3


def test_manifest_formats(tmp_path):
    sr = 8000
    sf.write(tmp_path / "a.wav", np.zeros(sr, dtype=np.float32), sr)
    (tmp_path / "a.txt").write_text("cq de vu2abc\n")
    items = load_manifest(str(tmp_path))
    assert len(items) == 1 and items[0]["text"] == "CQ DE VU2ABC"
    with open(tmp_path / "labels.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "text", "snr_db", "use"])
        w.writerow(["a.wav", "4U1UN", "3.5", "1"])
        w.writerow(["a.wav", "VE8AT", "", "0"])
    items = load_manifest(str(tmp_path))
    assert len(items) == 1 and items[0]["snr_db"] == 3.5
