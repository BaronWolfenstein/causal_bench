import numpy as np
from causal_bench.sampling.resample import systematic_resample
from causal_bench.sampling.sharded import sharded_systematic_resample


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
