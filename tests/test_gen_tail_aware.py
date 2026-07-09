import numpy as np
from causal_bench.generative.tail_aware import inverse_density_weights


def test_rare_points_get_higher_weight():
    rng = np.random.default_rng(0)
    common = rng.standard_normal((300, 2))
    rare = rng.standard_normal((30, 2)) + 4.0
    X = np.vstack([common, rare])
    w = inverse_density_weights(X, n_components=3)
    assert w[300:].mean() > w[:300].mean()          # tail upweighted
    assert np.isclose(w.mean(), 1.0, atol=1e-6)     # normalized to mean 1
