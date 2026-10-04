"""CTC decoding: greedy, prefix beam search, optional character n-gram LM."""
from __future__ import annotations

import json
import math
from collections import defaultdict

import numpy as np

from .alphabet import IDX2CHAR, VOCAB

NEG_INF = -float("inf")


def greedy_decode(log_probs, lengths) -> list[str]:
    """log_probs: (B, T, C) tensor/array; lengths: (B,)"""
    lp = log_probs.detach().cpu().numpy() if hasattr(log_probs, "detach") else np.asarray(log_probs)
    ln = lengths.detach().cpu().numpy() if hasattr(lengths, "detach") else np.asarray(lengths)
    out = []
    for b in range(lp.shape[0]):
        best = lp[b, : ln[b]].argmax(-1)
        chars, prev = [], -1
        for k in best:
            if k != prev and k != 0:
                chars.append(IDX2CHAR[int(k)])
            prev = k
        out.append(" ".join("".join(chars).split()))
    return out


# --------------------------------------------------------------------------
# character n-gram language model (add-k smoothing)
# --------------------------------------------------------------------------

class CharNgramLM:
    BOS = "^"

    def __init__(self, order: int = 4, k: float = 0.1):
        self.order, self.k = order, k
        self.counts = defaultdict(lambda: defaultdict(int))
        self.vocab = [c for c in VOCAB[1:]]

    def fit(self, texts):
        for t in texts:
            s = self.BOS * (self.order - 1) + t
            for i in range(self.order - 1, len(s)):
                self.counts[s[i - self.order + 1:i]][s[i]] += 1
        return self

    def logp(self, context: str, ch: str) -> float:
        ctx = (self.BOS * (self.order - 1) + context)[-(self.order - 1):]
        row = self.counts.get(ctx)
        v = len(self.vocab)
        if not row:
            return -math.log(v)
        tot = sum(row.values())
        return math.log((row.get(ch, 0) + self.k) / (tot + self.k * v))

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"order": self.order, "k": self.k, "counts": {c: dict(r) for c, r in self.counts.items()}}, f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            d = json.load(f)
        lm = cls(d["order"], d["k"])
        for c, r in d["counts"].items():
            lm.counts[c] = defaultdict(int, r)
        return lm


# --------------------------------------------------------------------------
# prefix beam search
# --------------------------------------------------------------------------

def _lse(*xs):
    m = max(xs)
    if m == NEG_INF:
        return NEG_INF
    return m + math.log(sum(math.exp(x - m) for x in xs))


def prefix_beam_search(log_probs: np.ndarray, beam: int = 16, lm: CharNgramLM | None = None,
                       lm_weight: float = 0.3, char_bonus: float = 0.5, prune: float = -8.0) -> str:
    """log_probs: (T, C) numpy array of one utterance (already trimmed to length)."""
    beams = {"": (0.0, NEG_INF)}  # prefix -> (log p ending in blank, log p ending in non-blank)
    lm_cache = {}

    def lm_score(prefix, ch):
        if lm is None:
            return 0.0
        key = (prefix[-(lm.order - 1):], ch)
        if key not in lm_cache:
            lm_cache[key] = lm.logp(prefix, ch)
        return lm_weight * lm_cache[key] + char_bonus

    for t in range(log_probs.shape[0]):
        row = log_probs[t]
        cands = np.flatnonzero(row > prune)
        nxt = defaultdict(lambda: [NEG_INF, NEG_INF])
        for prefix, (pb, pnb) in beams.items():
            ptot = _lse(pb, pnb)
            for c in cands:
                p = float(row[c])
                if c == 0:
                    e = nxt[prefix]
                    e[0] = _lse(e[0], ptot + p)
                    continue
                ch = IDX2CHAR[int(c)]
                last = prefix[-1] if prefix else None
                new = prefix + ch
                bonus = lm_score(prefix, ch)
                if ch == last:
                    # repeated char needs a blank in between; otherwise it extends the same symbol
                    e_new = nxt[new]
                    e_new[1] = _lse(e_new[1], pb + p + bonus)
                    e_same = nxt[prefix]
                    e_same[1] = _lse(e_same[1], pnb + p)
                else:
                    e_new = nxt[new]
                    e_new[1] = _lse(e_new[1], ptot + p + bonus)
        beams = dict(sorted(nxt.items(), key=lambda kv: -_lse(*kv[1]))[:beam])
    best = max(beams.items(), key=lambda kv: _lse(*kv[1]))[0]
    return " ".join(best.split())


def beam_decode(log_probs, lengths, beam=16, lm=None, lm_weight=0.3, char_bonus=0.5) -> list[str]:
    lp = log_probs.detach().cpu().numpy() if hasattr(log_probs, "detach") else np.asarray(log_probs)
    ln = lengths.detach().cpu().numpy() if hasattr(lengths, "detach") else np.asarray(lengths)
    return [prefix_beam_search(lp[b, : ln[b]], beam, lm, lm_weight, char_bonus if lm else 0.0)
            for b in range(lp.shape[0])]
