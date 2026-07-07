import numpy as np
from causal_bench.sampling.ipcw import ipcw_weights, positivity_floor

def test_ipcw_restores_unbiased_mean_after_informative_kill():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(10000)
    # informative filter: keep-prob depends on x (censoring on the covariate)
    G = 1.0 / (1.0 + np.exp(-(x + 1.0)))            # survival prob per sample
    kept = rng.random(len(x)) < G
    naive = x[kept].mean()                           # biased (survivors skew high)
    w = ipcw_weights(G[kept])
    corrected = np.average(x[kept], weights=w)
    assert abs(corrected) < abs(naive)               # bias reduced toward 0
    assert abs(corrected) < 0.05

def test_positivity_floor_flags_near_zero_survival():
    G = np.array([0.5, 0.4, 1e-6])
    clipped, violations = positivity_floor(G, floor=1e-3)
    assert violations.tolist() == [False, False, True]
    assert clipped[2] == 1e-3                         # clipped, and flagged
