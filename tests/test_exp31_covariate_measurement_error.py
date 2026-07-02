"""Tests for exp31 — covariate measurement-error (Σ_x) residual-confounding arm.

Pins the exp3-sibling story: adjusting for a noisy confounder leaves residual
confounding (naive biased, coverage collapses), regression calibration recovers
most of the oracle→naive gap, and a stationary GP conflates covariate noise with
its length-scale.
"""
import numpy as np
import pytest

from experiments.exp31_covariate_measurement_error import (
    TAU_TRUE, estimate_arm, gp_length_scale, recovery_fraction,
    run_sigma_x_sweep, simulate_covariate_me, tipping_point)


def test_dgp_error_variance_and_confounding():
    df = simulate_covariate_me(sigma_x=0.7, n=20000, seed=1)
    # classical additive error: var(X_obs) ≈ var(X_true) + σ_x²
    assert abs((df["X_obs"] - df["X_true"]).var() - 0.49) < 0.05
    # X_true confounds treatment (drives A) and outcome
    assert np.corrcoef(df["X_true"], df["A"])[0, 1] > 0.15
    assert np.corrcoef(df["X_true"], df["Y"])[0, 1] > 0.3


def test_single_replicate_arm_ordering():
    df = simulate_covariate_me(sigma_x=1.0, n=8000, seed=2)
    o = estimate_arm(df, "oracle", 1.0)["tau_hat"]
    nv = estimate_arm(df, "naive", 1.0)["tau_hat"]
    cr = estimate_arm(df, "corrected", 1.0)["tau_hat"]
    assert abs(o - TAU_TRUE) < 0.1                         # oracle ~ unbiased
    assert abs(nv - TAU_TRUE) > 0.5                        # naive residual confounding
    assert abs(cr - TAU_TRUE) < abs(nv - TAU_TRUE) / 2     # corrected recovers


@pytest.fixture(scope="module")
def sweep():
    return run_sigma_x_sweep([0.0, 0.5, 1.0], n_sims=40, n=1500, seed=31)


def test_naive_bias_grows_and_coverage_collapses(sweep):
    naive = sweep[sweep.arm == "naive"].set_index("sigma_x")
    assert naive.loc[0.0, "abs_bias"] < naive.loc[0.5, "abs_bias"] < naive.loc[1.0, "abs_bias"]
    assert naive.loc[1.0, "coverage"] < 0.2                # CI stops covering truth
    oracle = sweep[sweep.arm == "oracle"].set_index("sigma_x")
    assert oracle.loc[1.0, "coverage"] > 0.85             # oracle stays calibrated


def test_corrected_recovers_most_of_the_gap(sweep):
    rec = recovery_fraction(sweep).set_index("sigma_x")
    assert rec.loc[1.0, "recovery_fraction"] > 0.7        # >70% of oracle→naive gap
    corrected = sweep[sweep.arm == "corrected"].set_index("sigma_x")
    naive = sweep[sweep.arm == "naive"].set_index("sigma_x")
    assert corrected.loc[1.0, "abs_bias"] < naive.loc[1.0, "abs_bias"] / 2


def test_tipping_point_is_finite_and_early(sweep):
    tp = tipping_point(sweep, coverage_floor=0.90)
    assert np.isfinite(tp) and tp <= 0.5                  # naive fails by σ_x=0.5


def test_gp_length_scale_conflates_with_covariate_noise():
    df = simulate_covariate_me(sigma_x=1.0, n=2000, seed=31)
    ls_true = gp_length_scale(df, "X_true")
    ls_obs = gp_length_scale(df, "X_obs")
    assert ls_obs > ls_true                                # noise inflates length-scale
