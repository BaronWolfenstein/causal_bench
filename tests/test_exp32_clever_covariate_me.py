"""Tests for exp32 — Σ_x propagated into the TMLE clever covariate (#66).

Pins the production story: measurement error in a confounder biases the TMLE
ATE (via the propensity → clever covariate), and regression-calibrating the
confounder before the estimator sees it recovers toward the oracle. Arms are
compared on SHARED replicates (same data, so the measurement-error effect is
low-variance) rather than against an external MC truth.
"""
import numpy as np

from experiments.exp32_clever_covariate_me import (
    arm_frame, estimate_arm_tmle, regression_calibrate_w1, simulate_me_survival)


def test_dgp_and_rc_mechanics():
    df, w1t, w1o = simulate_me_survival(sigma_x=1.0, n=8000, seed=1)
    # classical additive error variance
    assert abs((w1o - w1t).var() - 1.0) < 0.1
    # arm frames swap the W1 column the estimator reads
    assert np.allclose(arm_frame(df, "oracle", w1t, w1o, 1.0)["W1"].to_numpy(), w1t)
    assert np.allclose(arm_frame(df, "naive", w1t, w1o, 1.0)["W1"].to_numpy(), w1o)
    # regression calibration de-attenuates: E[W1_true|·] is less variable than the
    # noisy observation and correlates at least as well with the truth
    w1h = regression_calibrate_w1(df, w1o, 1.0)
    assert w1h.var() < w1o.var()
    assert np.corrcoef(w1h, w1t)[0, 1] >= np.corrcoef(w1o, w1t)[0, 1] - 1e-3


def test_residual_confounding_and_recovery_on_tmle_estimand():
    """On shared replicates: naive is systematically biased toward the null vs the
    oracle (residual confounding through the clever covariate); corrected recovers."""
    dn, dc = [], []
    for r in range(8):
        df, w1t, w1o = simulate_me_survival(1.5, n=1500, seed=300 + r)
        o = estimate_arm_tmle(arm_frame(df, "oracle", w1t, w1o, 1.5))["point"]
        n = estimate_arm_tmle(arm_frame(df, "naive", w1t, w1o, 1.5))["point"]
        c = estimate_arm_tmle(arm_frame(df, "corrected", w1t, w1o, 1.5))["point"]
        dn.append(n - o)
        dc.append(c - o)
    dn, dc = np.array(dn), np.array(dc)
    # naive residual-confounds toward the null, consistently across replicates
    assert dn.mean() < -0.005
    assert (dn < 0).mean() >= 0.75
    # corrected sits closer to the oracle than naive does (recovers >1/3 of the gap)
    assert np.abs(dc).mean() < np.abs(dn).mean()
    assert np.abs(dc).mean() < 0.67 * np.abs(dn).mean()
