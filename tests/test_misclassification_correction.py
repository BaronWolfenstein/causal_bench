"""Self-validating tests for decoded-subgroup misclassification correction (issue #149)."""
import numpy as np

from causal_bench.validation.misclassification_correction import (
    confusion_matrix, corrected_subgroup_effect, report,
)


def test_correction_recovers_true_subgroup_means():
    r = report(n=8000, K=3, err=0.30, seed=0)
    assert r["naive_bias"] > 0.3                        # decoded-subgroup estimand is badly biased
    assert r["corrected_bias"] < 0.15                   # misclassification correction recovers
    assert r["corrected_bias"] < 0.3 * r["naive_bias"]  # ... and is far better than naive


def test_no_error_needs_no_correction():
    r = report(n=6000, K=3, err=0.0, seed=1)
    assert r["naive_bias"] < 0.1                         # perfect decode -> naive already unbiased
    assert r["corrected_bias"] < 0.1                     # correction does no harm (M = identity)


def test_decoded_subgroup_effect_recovers():
    """A heterogeneous subgroup treatment effect, with subgroup membership DECODED with error, is
    recovered by correcting each arm's decoded-subgroup means and differencing."""
    rng = np.random.default_rng(0); n, K = 10000, 3
    s_true = rng.integers(0, K, n); A = rng.integers(0, 2, n)
    tau = np.array([0.0, 1.0, 2.0])                      # heterogeneous per-subgroup effect
    Y = (1.0 + 2.0 * s_true) + tau[s_true] * A + rng.normal(size=n)
    keep = rng.random(n) >= 0.25
    s_pred = np.where(keep, s_true, (s_true + rng.integers(1, K, n)) % K)  # non-differential decode
    val = rng.choice(n, 2000, replace=False)
    M = confusion_matrix(s_true[val], s_pred[val], K)
    eff = corrected_subgroup_effect(Y, A, s_pred, M)
    assert np.mean(np.abs(eff - tau)) < 0.2             # recovers the heterogeneous subgroup effect
