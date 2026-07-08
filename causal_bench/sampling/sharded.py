"""CPU simulation of the multi-GPU resample. In production each rank all-gathers
weights so every rank holds w[1..N], then computes IDENTICAL systematic indices
from identical weights + a shared seed; an all-to-all redistributes particles
whose owner changed. Here we simulate the k ranks in-process and assert they
reproduce the single-rank indices exactly — the invariant that makes the real
all-to-all correct. Only absolute communication cost needs the A100 fabric."""
from __future__ import annotations

import numpy as np

from .resample import systematic_resample


def sharded_systematic_resample(w: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Each of k simulated ranks computes the full index vector from the same
    all-gathered weights and the same seed. Identical inputs + identical seed =>
    byte-identical indices on every rank (asserted equal to the serial run)."""
    n = len(w)
    shard_bounds = np.array_split(np.arange(n), k)
    full = None
    for _rank in range(k):
        rng = np.random.default_rng(seed)               # SHARED seed across ranks
        idx = systematic_resample(w, rng)               # full index vector per rank
        if full is None:
            full = idx
        else:
            assert np.array_equal(full, idx), "ranks disagree — seed not shared"
    # each rank keeps the slice it owns; concatenation reconstructs the whole
    return np.concatenate([full[b] for b in shard_bounds])
