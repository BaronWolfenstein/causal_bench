"""Tests for the shared regression-calibration helper."""
import numpy as np

from causal_bench.measurement_error import regression_calibrate


def test_identity_at_zero_error():
    rng = np.random.default_rng(0)
    w = rng.normal(0, 1, 2000)
    Z = rng.normal(0, 1, (2000, 2))
    w_hat = regression_calibrate(w, Z, sigma_x=0.0)
    assert np.allclose(w_hat, w, atol=1e-6)          # no noise → no correction


def test_deattenuates_and_reports_residual_variance():
    rng = np.random.default_rng(1)
    w_true = rng.normal(0, 1, 20000)
    a = (rng.random(20000) < 1 / (1 + np.exp(-w_true))).astype(float)  # A depends on w_true
    sigma_x = 1.0
    w_obs = w_true + rng.normal(0, sigma_x, 20000)
    w_hat, tau2 = regression_calibrate(w_obs, a, sigma_x, return_residual_variance=True)
    # E[w_true|·] is less variable than the noisy observation
    assert w_hat.var() < w_obs.var()
    # residual variance is the conditional variance: 0 < τ² < var(w_true)≈1, ~O(σ²)
    assert 0 < tau2 < 1.0
    assert abs(tau2 - w_true.var() * (1 - 1 / (1 + sigma_x**2))) < 0.15


def test_accepts_1d_and_2d_conditioning():
    rng = np.random.default_rng(2)
    w = rng.normal(0, 1, 500)
    assert regression_calibrate(w, rng.normal(0, 1, 500), 0.5).shape == (500,)
    assert regression_calibrate(w, rng.normal(0, 1, (500, 3)), 0.5).shape == (500,)
