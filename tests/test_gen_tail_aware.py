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


def test_outlier_cap_enabled_vs_disabled():
    # Controlled case: a tight common cluster (fit with a single GMM
    # component, so it cannot "absorb" the outlier into its own dedicated
    # component) plus a single far-flung outlier whose raw 1/p(z) weight is
    # astronomically larger than the rest of the vector, so the default cap
    # must engage and truncate it.
    rng = np.random.default_rng(0)
    common = rng.standard_normal((300, 2)) * 0.5
    outlier = np.array([[15.0, 15.0]])
    X = np.vstack([common, outlier])

    cap_percentile, cap_multiplier = 99.5, 5.0
    w_capped = inverse_density_weights(
        X, n_components=1, cap_percentile=cap_percentile,
        cap_multiplier=cap_multiplier,
    )
    assert np.isfinite(w_capped).all()
    # Documented cap (in normalized-weight units): a weight vector that was
    # truncated at percentile(raw_w, cap_percentile) * cap_multiplier and then
    # divided by its own mean cannot exceed cap_multiplier * (# samples), since
    # the pre-cap percentile is itself bounded by the mean of a
    # non-negative vector times its length. We instead assert the cap holds in
    # relative terms: the capped max is bounded by cap_multiplier times the
    # capped array's own 99.5th percentile (the same relationship the
    # implementation enforces internally, up to the mean-normalization it
    # applies afterward).
    normalized_cap_bound = np.percentile(w_capped, cap_percentile) * cap_multiplier
    assert w_capped.max() <= normalized_cap_bound + 1e-6

    w_uncapped = inverse_density_weights(X, n_components=1, cap_multiplier=None)
    assert np.isfinite(w_uncapped).all()
    assert w_uncapped.max() > w_capped.max()
