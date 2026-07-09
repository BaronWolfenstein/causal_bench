import numpy as np
import pytest

from causal_bench.generative.tail_aware import inverse_density_weights


@pytest.mark.parametrize("seed", range(10))
def test_rare_points_get_higher_weight(seed):
    # Genuinely-separated rare vs. common clusters, freshly constructed per
    # seed, so the rare>common invariant is structural rather than luck on
    # one hardcoded seed (a diag-covariance GMM at the brief's default seed
    # was shown to fail this invariant at seed=13 without n_init + capping).
    rng = np.random.default_rng(seed)
    common = rng.standard_normal((300, 2))
    rare = rng.standard_normal((30, 2)) + 6.0
    X = np.vstack([common, rare])
    w = inverse_density_weights(X, n_components=3)
    assert np.isfinite(w).all()                     # never inf/nan
    assert w[300:].mean() > w[:300].mean()           # tail upweighted
    assert np.isclose(w.mean(), 1.0, atol=1e-6)      # normalized to mean 1
