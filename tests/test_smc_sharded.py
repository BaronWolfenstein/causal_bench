import numpy as np
from causal_bench.sampling.resample import systematic_resample
from causal_bench.sampling.sharded import sharded_systematic_resample, island_resample


def test_sharded_indices_match_single_rank_bit_for_bit():
    rng = np.random.default_rng(3)
    w = rng.random(64); w /= w.sum()
    serial = systematic_resample(w, np.random.default_rng(123))
    for k in (2, 4, 8):
        distributed = sharded_systematic_resample(w, k=k, seed=123)
        assert np.array_equal(serial, distributed)     # the decisive invariant


def test_particle_count_conserved_across_shards():
    rng = np.random.default_rng(1)
    w = rng.random(60); w /= w.sum()
    idx = sharded_systematic_resample(w, k=3, seed=9)
    assert len(idx) == 60                               # N in == N out


def test_island_resample_conserves_count_and_stays_local():
    w = np.random.default_rng(2).random(60); w /= w.sum()
    k = 3
    idx = island_resample(w, k=k, seed=7)
    assert len(idx) == 60                                  # N in == N out
    bounds = np.array_split(np.arange(60), k)
    off = 0
    for b in bounds:
        seg = idx[off:off + len(b)]
        off += len(b)
        assert seg.min() >= b[0] and seg.max() <= b[-1]    # never left the island


def test_islands_are_independent_not_shared_seed():
    # two islands with identical local weights must NOT produce identical local
    # draws (independent per-rank RNG), else islands are correlated.
    # Use skewed weights that are sensitive to seed variation
    left_weights = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.45, 0.45, 0.01, 0.01, 0.01])
    right_weights = left_weights.copy()  # identical to left
    w = np.concatenate([left_weights, right_weights]); w /= w.sum()
    idx = island_resample(w, k=2, seed=1)
    left, right = idx[:10], idx[10:] - 10
    # With independent seeds (seed + rank), islands get different RNG states
    # rank 0 uses seed=1, rank 1 uses seed=2, producing different resampling results
    assert not np.array_equal(left, right)
