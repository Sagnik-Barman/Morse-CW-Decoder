"""PyTorch datasets: infinite on-the-fly synthetic stream + fixed eval sets."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from .alphabet import encode
from .synth import SynthConfig, generate_sample


def synth_config_from_dict(d: dict | None) -> SynthConfig:
    cfg = SynthConfig()
    if d:
        d = {k: (tuple(v) if isinstance(v, list) else v) for k, v in d.items()}
        cfg = replace(cfg, **d)
    return cfg


class SynthStream(IterableDataset):
    """Endless stream of freshly generated clips. Each worker gets its own seed."""

    def __init__(self, cfg: SynthConfig, seed: int = 0):
        super().__init__()
        self.cfg, self.seed = cfg, seed

    def __iter__(self):
        info = get_worker_info()
        wid = info.id if info is not None else 0
        rng = np.random.default_rng([self.seed, wid])
        while True:
            audio, label, meta = generate_sample(rng, self.cfg)
            if not label and not meta.get("empty"):
                continue
            yield audio, label, meta


class FixedSet(Dataset):
    """Deterministic evaluation set, generated once and kept in memory (int16)."""

    def __init__(self, cfg: SynthConfig, n: int, seed: int = 12345, snr_values=None, wpm_values=None,
                 progress: bool = False):
        rng = np.random.default_rng(seed)
        self.audio, self.labels, self.meta = [], [], []
        it = range(n)
        if progress:
            try:
                from tqdm import tqdm
                it = tqdm(it, desc="generating eval set", leave=False)
            except ImportError:
                pass
        for i in it:
            snr = None if snr_values is None else float(snr_values[i % len(snr_values)])
            wpm = None if wpm_values is None else float(wpm_values[i % len(wpm_values)])
            while True:
                a, l, m = generate_sample(rng, cfg, snr_db=snr, wpm=wpm)
                if l:
                    break
            self.audio.append((a * 32767).astype(np.int16))
            self.labels.append(l)
            self.meta.append(m)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.audio[i].astype(np.float32) / 32767.0, self.labels[i], self.meta[i]


def collate(batch):
    audios, labels, metas = zip(*batch)
    lengths = torch.tensor([len(a) for a in audios], dtype=torch.long)
    x = torch.zeros(len(audios), int(lengths.max()))
    for i, a in enumerate(audios):
        x[i, : len(a)] = torch.from_numpy(np.asarray(a))
    targets = [torch.tensor(encode(l), dtype=torch.long) for l in labels]
    target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
    return {
        "audio": x,
        "audio_lengths": lengths,
        "targets": torch.cat(targets) if targets else torch.zeros(0, dtype=torch.long),
        "target_lengths": target_lengths,
        "texts": list(labels),
        "meta": list(metas),
    }
