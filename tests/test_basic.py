import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from morse.alphabet import MORSE, NUM_CLASSES, decode_ids, encode, text_to_morse
from morse.baseline import decode_audio
from morse.data import SynthStream, collate
from morse.decode import CharNgramLM, greedy_decode, prefix_beam_search
from morse.metrics import cer, edit_distance
from morse.model import build_model
from morse.synth import SynthConfig, generate_sample

CLEAN = SynthConfig(p_hand=0, p_fading=0, p_qrm=0, p_qrn=0, p_chirp=0, p_farnsworth=0,
                    speed_drift_range=(0, 0), freq_drift_hz_range=(0, 0))


def test_alphabet_roundtrip():
    assert decode_ids(encode("cq de vu2abc")) == "CQ DE VU2ABC"
    assert text_to_morse("SOS") == "... --- ..."
    assert len(set(MORSE.values())) == len(MORSE)
    assert NUM_CLASSES == len(MORSE) + 2


def test_metrics():
    assert edit_distance("KITTEN", "SITTING") == 3
    assert cer("ABC", "ABC") == 0


def test_synth_shape_and_label():
    rng = np.random.default_rng(0)
    for _ in range(20):
        x, label, meta = generate_sample(rng)
        assert x.shape == (64000,) and x.dtype == np.float32
        assert np.abs(x).max() <= 1.0
        assert label == label.strip()


def test_fixed_text_fits():
    rng = np.random.default_rng(0)
    x, label, _ = generate_sample(rng, CLEAN, text="CQ CQ DE VU2ABC K", wpm=30, snr_db=30)
    assert label == "CQ CQ DE VU2ABC K"
    # too long for one clip at 12 WPM -> truncated at a character boundary
    x, label, _ = generate_sample(rng, CLEAN, text="CQ CQ DE VU2ABC K", wpm=12, snr_db=30)
    assert "CQ CQ DE VU2ABC K".startswith(label) and len(label) < 17


def test_baseline_decodes_clean_signal():
    rng = np.random.default_rng(1)
    errs = []
    for _ in range(10):
        x, label, _ = generate_sample(rng, CLEAN, snr_db=20)
        errs.append(cer(label, decode_audio(x)))
    assert np.mean(errs) < 0.05


def test_model_forward_and_ctc():
    for arch in ["crnn", "conformer"]:
        model = build_model({"model": {"arch": arch, "encoder": {"layers": 2} if arch == "crnn" else {"layers": 2}}})
        it = iter(SynthStream(SynthConfig(), seed=0))
        b = collate([next(it) for _ in range(2)])
        lp, ln = model(b["audio"], b["audio_lengths"])
        assert lp.shape[-1] == NUM_CLASSES
        assert int(ln[0]) >= int(b["target_lengths"].max())
        loss = torch.nn.functional.ctc_loss(lp.transpose(0, 1), b["targets"], ln, b["target_lengths"])
        assert torch.isfinite(loss)


def test_decoders_on_perfect_posteriors():
    # build posteriors spelling "AB  A" -> "AB A" with blanks between
    from morse.alphabet import CHAR2IDX
    seq = [0, CHAR2IDX["A"], CHAR2IDX["A"], 0, CHAR2IDX["B"], 0, CHAR2IDX[" "], 0, CHAR2IDX["A"], 0]
    lp = np.full((len(seq), NUM_CLASSES), -20.0)
    for t, k in enumerate(seq):
        lp[t, k] = 0.0
    assert greedy_decode(lp[None], np.array([len(seq)]))[0] == "AB A"
    assert prefix_beam_search(lp, beam=4) == "AB A"
    lm = CharNgramLM(3).fit(["AB A", "AB AB"])
    assert prefix_beam_search(lp, beam=4, lm=lm) == "AB A"


def test_synth_v3_busy_band():
    """v3: several interferers, possibly stronger than the target; empty clips keep an empty label."""
    import numpy as np
    from morse.synth import SynthConfig, generate_sample
    cfg = SynthConfig(tone_hz_range=(650.0, 750.0), p_qrm=1.0, qrm_count_range=(1, 3),
                      qrm_rel_db_range=(-20.0, 10.0), qrm_offset_hz_range=(50.0, 500.0),
                      qrm_on_empty=True, qrm_fading=True, p_empty=0.5)
    seen = set()
    for i in range(40):
        x, label, meta = generate_sample(np.random.default_rng(i), cfg)
        assert x.shape == (int(cfg.clip_seconds * cfg.sample_rate),) and np.isfinite(x).all()
        assert meta["qrm"] and 1 <= meta["n_qrm"] <= 3 and 650 <= meta["tone_hz"] <= 750
        assert (label == "") == meta["empty"]
        seen.add(meta["n_qrm"])
    assert seen == {1, 2, 3}
