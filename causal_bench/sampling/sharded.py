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


def sharded_logsumexp(shards) -> float:
    """CPU simulation of the two-scalar distributed log-sum-exp over log-weights
    sharded across k ranks. The numerically-stable cross-rank reduction is:

        global_max = MAX over ranks of local_max          # all_reduce(op=MAX)
        global_sum = SUM over ranks of sum(exp(local - global_max))  # all_reduce(op=SUM)
        lse        = global_max + log(global_sum)

    Subtracting the GLOBAL max before exponentiating is what keeps it overflow-free
    (the distributed analogue of `weights.normalize_log_weights`). Two scalar
    all-reduces suffice, so this folds into the ESS-reduction barrier the SMC step
    already has. Returns a value equal (to fp tolerance) to the serial
    `logsumexp` over the concatenation — the distributed==serial invariant the
    on-box NCCL `all_reduce(MAX)` + `all_reduce(SUM)` must reproduce.
    """
    local_maxes = [float(np.max(s)) if len(s) else -np.inf for s in shards]
    global_max = max(local_maxes)
    if not np.isfinite(global_max):
        return float(global_max)                     # all -inf collapse (or +inf leak)
    total = sum(float(np.exp(np.asarray(s, dtype=float) - global_max).sum())
                for s in shards)
    return float(global_max + np.log(total))
