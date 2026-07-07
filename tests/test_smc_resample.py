import numpy as np
from causal_bench.sampling.resample import systematic_resample, should_resample

def test_systematic_resample_duplicates_the_dominant_particle():
    w = np.array([0.001, 0.001, 0.997, 0.001])
    idx = systematic_resample(w, np.random.default_rng(0))
    assert len(idx) == 4
    assert (idx == 2).sum() >= 3                     # dominant survivor fans out
    assert idx.dtype.kind == "i"

def test_systematic_resample_is_deterministic_under_shared_seed():
    w = np.array([0.25, 0.25, 0.25, 0.25])
    a = systematic_resample(w, np.random.default_rng(7))
    b = systematic_resample(w, np.random.default_rng(7))
    assert np.array_equal(a, b)                       # shared seed -> identical

def test_should_resample_triggers_only_on_degeneracy():
    assert should_resample(np.zeros(10), ess_frac=0.5) is False   # ESS=10 > 5
    spike = np.full(10, -1e9); spike[0] = 0.0
    assert should_resample(spike, ess_frac=0.5) is True           # ESS~1 < 5
