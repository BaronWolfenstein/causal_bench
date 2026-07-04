"""Smoke tests for exp38 — frozen vs refit AIPW under train-vs-deploy
covariate shift (LENS-faithful, causal_bench #83).

Pins the plumbing (draw_shifted's consistency, true_tau_shifted's Monte
Carlo estimate, the frozen/refit contrast) rather than asserting a specific
degradation magnitude (n_reps too small here for that).
"""
import numpy as np

from experiments.exp38_frozen_model_shift import (
    draw_shifted, run_shift_grid, summarize, true_tau_shifted,
)


def test_draw_shifted_at_zero_matches_unshifted_moments():
    df = draw_shifted(50_000, seed=0, mean_shift=0.0)
    assert abs(df["W1"].mean()) < 0.02  # unshifted: W1 ~ N(0,1)


def test_draw_shifted_mean_shift_moves_w1_only():
    df = draw_shifted(50_000, seed=1, mean_shift=2.0)
    assert abs(df["W1"].mean() - 2.0) < 0.05
    assert abs(df["W2"].mean()) < 0.05   # other covariates untouched


def test_true_tau_shifted_at_zero_is_cached_and_stable():
    t1 = true_tau_shifted(0.0)
    t2 = true_tau_shifted(0.0)
    assert t1 == t2  # lru_cache -> identical, not just close
    assert np.isfinite(t1)


def test_true_tau_shifted_differs_from_unshifted_at_nonzero_shift():
    from causal_bench.dgp.point_treatment import true_tau
    t_shifted = true_tau_shifted(2.0)
    t_baseline = true_tau(exp38_surface())
    assert t_shifted != t_baseline


def exp38_surface():
    from experiments.exp38_frozen_model_shift import SURFACE
    return SURFACE


def test_run_shift_grid_shape_and_arms():
    df = run_shift_grid(n_reps=2, seed=0)
    assert set(df["arm"].unique()) == {"frozen", "refit"}
    for col in ["covariate_shift", "rep", "arm", "point", "true_tau", "bias"]:
        assert col in df.columns
    from experiments.exp38_frozen_model_shift import SHIFT_GRID
    assert len(df) == len(SHIFT_GRID) * 2 * 2   # shifts x reps x {frozen,refit}


def test_summarize_produces_degradation_gap_column():
    df = run_shift_grid(n_reps=2, seed=1)
    summary = summarize(df)
    for col in ["covariate_shift", "frozen_bias", "frozen_rmse", "refit_bias",
               "refit_rmse", "degradation_gap"]:
        assert col in summary.columns
    assert np.allclose(summary["degradation_gap"],
                       summary["frozen_rmse"] - summary["refit_rmse"])


def test_no_shift_frozen_and_refit_are_comparable():
    # At shift=0.0 (deploy == train distribution), frozen and refit should be
    # in the same ballpark -- no genuine shift to expose a gap.
    # NOTE: the column is named covariate_shift (not "shift") specifically to
    # avoid colliding with DataFrame.shift() -- dot-access on a "shift" column
    # would silently return the bound method instead, an earlier version of
    # this test hit exactly that ("KeyError: False") before the rename.
    df = run_shift_grid(n_reps=8, seed=2)
    summary = summarize(df)
    row0 = summary[summary["covariate_shift"] == 0.0].iloc[0]
    assert abs(row0["frozen_rmse"] - row0["refit_rmse"]) < 0.3  # loose, small n_reps
