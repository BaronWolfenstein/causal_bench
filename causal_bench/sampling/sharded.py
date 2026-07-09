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


def island_resample(w: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Island / local resampling (spec §1d): each of k ranks resamples ONLY its
    own contiguous sub-population from its locally renormalized weights — no
    all-to-all particle exchange. Returns GLOBAL ancestor indices (each mapped
    back into the full array), so no index ever crosses an island boundary.
    Per-island RNGs are seeded independently (seed + rank) to avoid correlating
    islands. This is the low-communication default; the small bias/variance cost
    is characterized on-box against the global sharded oracle."""
    n = len(w)
    bounds = np.array_split(np.arange(n), k)
    out = []
    for rank, b in enumerate(bounds):
        local_w = w[b]
        local_w = local_w / local_w.sum()               # renormalize within island
        rng = np.random.default_rng(seed + rank)        # independent per island
        local_idx = systematic_resample(local_w, rng)   # indices into the local slice
        out.append(b[local_idx])                         # map local -> global
    return np.concatenate(out)
