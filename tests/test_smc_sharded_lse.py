"""Two-scalar distributed log-sum-exp oracle: the numerically-stable cross-rank
LSE (all_reduce MAX of the local maxima, then all_reduce SUM of the shifted
exp-sums) must reproduce the serial logsumexp over the concatenation. This is the
distributed==serial invariant the on-box NCCL reduction must match (the LSE
analogue of sharded_systematic_resample)."""
import numpy as np
from scipy.special import logsumexp

from causal_bench.sampling.sharded import sharded_logsumexp


def test_sharded_lse_matches_serial():
    full = np.random.default_rng(0).standard_normal(97) * 5.0
    shards = np.array_split(full, 4)
    assert np.isclose(sharded_logsumexp(list(shards)), logsumexp(full))


def test_sharded_lse_is_overflow_stable_for_large_values():
    full = np.array([1000.0, 1001.0, 999.0, 1000.5, 998.0])
    shards = np.array_split(full, 3)
    assert np.isclose(sharded_logsumexp(list(shards)), logsumexp(full))  # no exp overflow


def test_sharded_lse_handles_a_dead_shard():
    # one rank fully out of support (all -inf); the reduction must still match serial
    full = np.concatenate([np.full(5, -np.inf), np.array([0.0, 1.0, 2.0])])
    shards = [full[:5], full[5:]]
    assert np.isclose(sharded_logsumexp(shards), logsumexp(full))


def test_sharded_lse_total_collapse_returns_neg_inf():
    shards = [np.full(3, -np.inf), np.full(2, -np.inf)]
    assert sharded_logsumexp(shards) == -np.inf
