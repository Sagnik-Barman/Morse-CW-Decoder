"""Character / word error rates via Levenshtein edit distance."""
from __future__ import annotations


def edit_distance(a, b) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(ref: str, hyp: str) -> float:
    """Character error rate of one pair (can exceed 1 with many insertions)."""
    return edit_distance(ref, hyp) / max(1, len(ref))


def wer(ref: str, hyp: str) -> float:
    r, h = ref.split(), hyp.split()
    return edit_distance(r, h) / max(1, len(r))


class ErrorRate:
    """Corpus-level accumulator: sum(edits) / sum(reference lengths)."""

    def __init__(self):
        self.c_err = self.c_tot = self.w_err = self.w_tot = 0
        self.n = 0

    def update(self, ref: str, hyp: str):
        self.c_err += edit_distance(ref, hyp)
        self.c_tot += max(1, len(ref))
        r, h = ref.split(), hyp.split()
        self.w_err += edit_distance(r, h)
        self.w_tot += max(1, len(r))
        self.n += 1

    @property
    def cer(self):
        return self.c_err / max(1, self.c_tot)

    @property
    def wer(self):
        return self.w_err / max(1, self.w_tot)
