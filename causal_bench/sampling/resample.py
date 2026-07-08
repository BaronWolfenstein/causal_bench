"""Systematic resampling (cumsum + searchsorted — the GPU-parallel primitive)
and the adaptive-resampling trigger. Ancestor indices are the raw material for
the localization diagnostic's lineage-collapse component; callers persist them."""
from __future__ import annotations

import numpy as np

from .weights import kish_ess, normalize_log_weights


def systematic_resample(w: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Return ancestor indices. One uniform draw, N evenly-spaced positions,
    searchsorted into the CDF. O(N) and vectorized."""
    n = len(w)
    positions = (rng.random() + np.arange(n)) / n
    cdf = np.cumsum(w)
    cdf[-1] = 1.0                                    # guard fp drift at the top
    return np.searchsorted(cdf, positions).astype(np.int64)


def should_resample(log_w: np.ndarray, ess_frac: float = 0.5) -> bool:
    """Adaptive resampling: only trigger the barrier when ESS < ess_frac * N.
    Most steps then have no global sync at all."""
    return kish_ess(log_w) < ess_frac * len(log_w)
