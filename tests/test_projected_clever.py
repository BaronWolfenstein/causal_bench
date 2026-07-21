"""Projected clever covariate (#182) — the EIF-projection successor to exp32's
first-order regression calibration.

The clever covariate H(A,W) = A/g(W) − (1−A)/(1−g(W)) is NONLINEAR in W, so with a
mismeasured confounder

    E[H(A, W_true) | W_obs, A]   ≠   H(A, E[W_true | W_obs, A])
         the projection                    exp32's plug-in

These pin the projection: it collapses to the plug-in when the calibration posterior is
degenerate, it matches a Monte-Carlo evaluation of the same conditional expectation, and
— because 1/g and 1/(1−g) are convex under a logistic g — the plug-in systematically
UNDERSTATES the clever covariate's magnitude, by a gap that grows with σ_x.
"""
import numpy as np
import pytest

from causal_bench.estimators.projected_clever import (
    projected_clever_covariate, plugin_clever_covariate, jensen_gap_report,
)


def _logistic_g(intercept=-0.3, slope=0.8):
    """g(w1) = sigma(intercept + slope*w1); 1/g and 1/(1-g) are then convex in w1."""
    def g(w1_grid):
        return 1.0 / (1.0 + np.exp(-(intercept + slope * np.asarray(w1_grid, float))))
    return g


def test_projection_collapses_to_plugin_with_a_degenerate_posterior():
    # sd = 0 -> W_true is known -> the projection IS the plug-in.
    rng = np.random.default_rng(0)
    n = 200
    A = rng.integers(0, 2, n)
    m = rng.normal(size=n)
    g = _logistic_g()
    proj = projected_clever_covariate(A, g, m, np.zeros(n))
    plug = plugin_clever_covariate(A, g, m)
    assert np.allclose(proj, plug, atol=1e-10)


def test_projection_matches_monte_carlo_of_the_same_conditional_expectation():
    # Gauss-Hermite quadrature must agree with brute-force MC over the posterior.
    rng = np.random.default_rng(1)
    n = 60
    A = rng.integers(0, 2, n)
    m = rng.normal(size=n)
    sd = np.full(n, 0.7)
    g = _logistic_g()
    proj = projected_clever_covariate(A, g, m, sd, n_quad=48)

    draws = m[:, None] + sd[:, None] * rng.normal(size=(n, 400_000))
    gv = np.clip(g(draws), 1e-6, 1 - 1e-6)
    mc = np.where(A[:, None] == 1, 1.0 / gv, -1.0 / (1.0 - gv)).mean(axis=1)
    assert np.max(np.abs(proj - mc)) < 0.02


def test_plugin_understates_magnitude_and_the_gap_grows_with_error():
    # Jensen: 1/g and 1/(1-g) are convex in w1 for logistic g, so E[H(W)] exceeds
    # H(E[W]) in magnitude. exp32's RC arm therefore under-weights systematically.
    rng = np.random.default_rng(2)
    n = 400
    A = rng.integers(0, 2, n)
    m = rng.normal(size=n)
    g = _logistic_g()
    plug = plugin_clever_covariate(A, g, m)

    prev = 0.0
    for sd_val in (0.3, 0.6, 1.0):
        proj = projected_clever_covariate(A, g, m, np.full(n, sd_val))
        assert np.all(np.abs(proj) >= np.abs(plug) - 1e-9)      # never smaller
        gap = float(np.mean(np.abs(proj) - np.abs(plug)))
        assert gap > prev                                        # and monotone in sigma_x
        prev = gap


def test_jensen_gap_report_is_zero_without_error_and_grows_with_it():
    rep0 = jensen_gap_report(sigma_x=0.0, n=1500, seed=3)
    assert abs(rep0["mean_abs_gap"]) < 1e-8
    assert abs(rep0["relative_gap"]) < 1e-8
    reps = [jensen_gap_report(sigma_x=s, n=1500, seed=3) for s in (0.25, 0.5, 1.0)]
    gaps = [r["mean_abs_gap"] for r in reps]
    assert gaps == sorted(gaps) and gaps[0] > 0
    # the projection should sit closer to the oracle-posterior target than the plug-in
    assert reps[-1]["plugin_error"] > reps[-1]["projected_error"]
