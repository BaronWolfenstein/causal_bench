"""Systematic resampling (cumsum + searchsorted — the GPU-parallel primitive)
and the adaptive-resampling trigger. Ancestor indices are the raw material for
the localization diagnostic's lineage-collapse component; callers persist them."""
from __future__ import annotations

import numpy as np

from .backend import get_namespace
from .weights import kish_ess, normalize_log_weights


def systematic_resample(w, rng: np.random.Generator):
    """Return ancestor indices. One HOST uniform draw, N evenly-spaced
    positions, searchsorted into the CDF. O(N), vectorized. numpy or cupy
    in (matches `w`); the rng stays host so a shared seed yields byte-identical
    indices across ranks (the sharded invariant)."""
    xp = get_namespace(w)
    n = len(w)
    positions = (rng.random() + xp.arange(n)) / n    # host scalar broadcasts onto xp
    cdf = xp.cumsum(w)
    cdf[-1] = 1.0                                     # guard fp drift at the top
    return xp.searchsorted(cdf, positions).astype(xp.int64)


def should_resample(log_w: np.ndarray, ess_frac: float = 0.5) -> bool:
    """Adaptive resampling: only trigger the barrier when ESS < ess_frac * N.
    Most steps then have no global sync at all. Returns a host bool."""
    return kish_ess(log_w) < ess_frac * len(log_w)
