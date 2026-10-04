# Real data: how to collect and label it

Synthetic data is for training and controlled sweeps. **The headline numbers should come from
real, off-air audio.** Here are three sources whose text is already known, ordered from least
to most labelling effort.

Recommended layout:

```
real_data/
  beacons/
    2026-10-02_1403_14100/      <- output of slice_beacons.py (clips/ + labels.csv)
    2026-10-03_0921_21150/
  w1aw/
    2026-10-02_20wpm.wav  +  2026-10-02_20wpm.txt
  handlabeled/
    contest_01.wav  +  contest_01.txt
```

**Keep a split.** Use some recording days as a *monitor* set (`train.real_val`) and never look at
others until the final *test* run. Split by day and receiver, not by clip, because clips from the
same session are strongly correlated.

---

## 1. NCDXF/IARU beacons (labels + real SNR for free)

18 beacons around the world transmit on a fixed schedule on 14.100, 18.110, 21.150, 24.930 and
28.200 MHz. In each 10-second slot a beacon sends its callsign at 22 WPM, then four 1-second
dashes at 100 W, 10 W, 1 W and 0.1 W. The schedule tells us which callsign is on air, and the
dashes let us measure the real signal-to-noise ratio.

**Recording, the easy way (recommended).** `scripts/record_beacons.py` records straight from a public
KiwiSDR receiver. No browser or clock-watching is needed: the file name carries the exact UTC start time.
It then slices the recording automatically.
```powershell
cd D:\Projects
git clone https://github.com/jks-prv/kiwiclient.git        # once (or download the ZIP and unzip it to D:\Projects\kiwiclient)
cd D:\Projects\morse-cw-decoder
python scripts/record_beacons.py --kiwi HOST:PORT --minutes 30
```
Find a receiver on the map at http://rx.linkfanel.net, open it in your browser, and copy the address
(for example `abc.proxy.kiwisdr.com:8073`). Pasting the whole browser URL also works. Use receivers on
different continents for different propagation paths. If a receiver is full or limits session time,
just pick another one.

**Recording by hand (any WebSDR / KiwiSDR in the browser)**
1. Open a public receiver: a WebSDR (list at websdr.org) or a KiwiSDR (map at kiwisdr.com).
   Receivers on different continents give different propagation paths, so use several.
2. Tune to **14100.00 kHz** (or another beacon frequency) in **CW** mode. Leave the filter at a few hundred Hz or wider.
3. Look at a UTC clock (e.g. time.is/UTC), press **record**, and write down the UTC time.
   An error of a few seconds is fine: the slicer searches ±8 s and aligns on the callsign keying.
4. Record 20–60 minutes. Each hour gives about 360 slots. 14 MHz is usually best in the daytime and evening.
5. Save as `real_data/raw/<date>_<HHMM>_<band>.wav`.

**Slicing**
```powershell
python scripts/slice_beacons.py real_data/raw/2026-10-02_1403_14100.wav --start-utc 2026-10-02T14:03:17Z --band 14100 --out real_data/beacons/2026-10-02_1403_14100
```
The script prints:
- **Alignment offset and confidence.** A confidence below about 1.3 usually means the UTC start time or
  the band is wrong, or nothing was audible.
- **One line per slot:** measured SNR, whether the beacon was detected (`use`), and the SNR of each
  power-step dash. On a clean path the 100 W / 10 W / 1 W / 0.1 W dashes should step down about 10 dB each.
  `ladder_ok = 1` marks slots where they do, which is a check that the SNR measurement is sound.

Off-air beacons, and slots where nothing is audible, get `use = 0` and are skipped by the benchmark.
Check the current beacon status at ncdxf.org, because some beacons are sometimes off the air.

Want to see it working first? Simulate a recording with known truth:
```powershell
python scripts/simulate_beacons.py --out sim_beacons.wav --start-utc 14:03:17 --delay 2.3
python scripts/slice_beacons.py sim_beacons.wav --start-utc 14:03:17 --out real_data/sim
```
In testing, the slicer recovered the 2.3 s delay exactly, and its SNR estimates matched the truth
with no bias and about 1.3 dB spread.

**Caveat:** a beacon clip is only a short callsign at a fixed 22 WPM, sent by a machine. It measures
the *channel* (propagation, fading, noise) well, but not human timing or other speeds.

## 2. W1AW code practice (long text, known transcript)

ARRL's station W1AW broadcasts code practice at several speeds on a published schedule (see the
W1AW Operating Schedule on arrl.org). The texts, and clean MP3s, are posted in ARRL's code
practice archives.

- **Off-air (preferred):** record the broadcast on a WebSDR/KiwiSDR in **North America**
  (W1AW is in Connecticut) at the scheduled time and frequency. Save it as `name.wav`.
- Download the matching text file and save it as `name.txt` next to it.
- Make the text match the audio exactly. The broadcast may start with a preamble
  (e.g. `VVV VVV ... DE W1AW`) and end with sign-offs that are not in the text file.
  Either trim the audio or add those words to the `.txt`, otherwise they count as errors.
- The archive MP3s are clean, machine-sent audio. They make a useful **clean-channel** real test.
  If an MP3 won't load, convert it: `ffmpeg -i in.mp3 -ar 8000 -ac 1 out.wav`.

Check ARRL's terms before redistributing their audio or text. Using it for your own evaluation is fine.

## 3. Hand-labelled everyday traffic (human senders)

This is the only source with real hand-keyed sending, varied speeds and crowded bands.
Record QSOs or contest activity (e.g. 14.020–14.060 MHz on a weekend) and cut 10–60 s segments. Then:

```powershell
python scripts/decode_wav.py segment.wav --checkpoint runs/crnn/best.pt --auto-center
```

Use the output as a draft, correct it while listening, and save it as `segment.txt`.
10–20 minutes is enough for a meaningful test set. Label only the station you tuned to,
and write `?` or leave out words you can't copy with confidence.

## Running the benchmark

```powershell
python scripts/benchmark_real.py real_data/beacons/2026-10-02_1403_14100 real_data/w1aw real_data/handlabeled ^
    --checkpoint runs/crnn/best.pt --baseline --beam --lm lm/char4.json ^
    --synthetic-results results/results_full.csv
```
Add `--auto-center` when the folders contain raw recordings whose tone isn't around 700 Hz.
The beacon clips are already shifted to 700 Hz by the slicer.
