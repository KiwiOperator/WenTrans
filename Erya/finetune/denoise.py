"""BART-style span masking helpers used by :class:`MonoDenoiseDataset`.

The actual masking is implemented on the dataset itself for efficiency; this
module just exposes a function for unit-testing the corruption logic.
"""
from __future__ import annotations

from typing import List

import numpy as np


def span_corrupt(
    ids: List[int],
    mask_id: int,
    mask_ratio: float = 0.30,
    poisson_lambda: float = 3.0,
    rng: np.random.Generator | None = None,
) -> List[int]:
    """Replace random Poisson-length spans with a single mask token.

    Mirrors ``BartDenoising`` text-infilling. ``mask_ratio`` is the fraction
    of *original* tokens that should be replaced.
    """
    if rng is None:
        rng = np.random.default_rng()
    n = len(ids)
    if n == 0:
        return ids
    target = int(round(mask_ratio * n))
    out = list(ids)
    masked = 0
    guard = 0
    cur_n = n
    while masked < target and guard < 64 and cur_n > 1:
        guard += 1
        span_len = max(1, int(rng.poisson(poisson_lambda)))
        span_len = min(span_len, target - masked, cur_n)
        if span_len <= 0:
            break
        start = int(rng.integers(0, cur_n - span_len + 1))
        out = out[:start] + [mask_id] + out[start + span_len :]
        masked += span_len
        cur_n -= span_len - 1
    return out
