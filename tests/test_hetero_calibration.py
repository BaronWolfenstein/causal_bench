"""Heteroskedastic regression calibration (#182 step 4).

Scalar-sigma_x calibration gives every unit the SAME posterior variance, which makes the
projected clever covariate a near-uniform rescaling — and TMLE targeting is invariant to
exactly that (step 2). These pin the per-unit behaviour that makes the correction
non-uniform, which is the precondition for the projection to matter at all.
"""
import numpy as np
import pytest

from causal_bench.measurement_error import regression_calibrate, regression_calibrate_hetero


def _sim(n=4000, seed=0, sigmas=(0.2, 1.5)):
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, 2))
    w_true = 0.7 * Z[:, 0] - 0.4 * Z[:, 1] + rng.normal(size=n)
    sx = np.where(rng.random(n) < 0.5, sigmas[0], sigmas[1])     # two "sites"
    w_obs = w_true + sx * rng.normal(size=n)
    return w_obs, Z, sx, w_true


def test_posterior_variance_varies_across_units_unlike_the_scalar_version():
    w_obs, Z, sx, _ = _sim()
    _, tau2 = regression_calibrate_hetero(w_obs, Z, sx)
    assert np.std(tau2) > 0.05                       # genuinely unit-specific
    assert tau2[sx == 1.5].mean() > 3 * tau2[sx == 0.2].mean()   # noisier ⇒ more posterior var
    # the scalar routine, by construction, yields ONE residual variance for everyone
    _, tau2_scalar = regression_calibrate(w_obs, Z, float(np.mean(sx)),
                                          return_residual_variance=True)
    assert np.isscalar(tau2_scalar) or np.ndim(tau2_scalar) == 0


def test_precise_units_shrink_less_than_noisy_ones():
    w_obs, Z, sx, _ = _sim()
    m, _ = regression_calibrate_hetero(w_obs, Z, sx)
    # shrinkage is toward the regression line: |m - w_obs| should be larger where sigma_x is
    pull = np.abs(m - w_obs)
    assert pull[sx == 1.5].mean() > 3 * pull[sx == 0.2].mean()


def test_calibration_reduces_error_against_the_latent_truth():
    w_obs, Z, sx, w_true = _sim()
    m, _ = regression_calibrate_hetero(w_obs, Z, sx)
    assert np.mean((m - w_true) ** 2) < np.mean((w_obs - w_true) ** 2)


def test_degenerates_to_a_constant_posterior_when_error_is_homoskedastic():
    rng = np.random.default_rng(1)
    n = 2000
    Z = rng.normal(size=(n, 2))
    w_true = Z[:, 0] + rng.normal(size=n)
    w_obs = w_true + 0.8 * rng.normal(size=n)
    _, tau2 = regression_calibrate_hetero(w_obs, Z, np.full(n, 0.8))
    assert np.std(tau2) == pytest.approx(0.0, abs=1e-12)   # uniform ⇒ nothing for the
    assert tau2[0] > 0                                     # projection to exploit
