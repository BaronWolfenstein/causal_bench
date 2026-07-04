"""Smoke tests for exp37 — compounding covariate shift (unmeasured
confounding x enrollment drift, causal_bench #82).

Not a statistical validation (n_sims too small for that) — pins the plumbing:
the grid runs, bias is computed, the compounding-table decomposition is
internally consistent, and the naive estimator (most exposed to unmeasured
confounding) shows nonzero bias once U>0.
"""
import numpy as np

from experiments.exp37_compounding_shift import (
    compounding_table, run_grid,
)


def test_run_grid_shape_and_columns():
    df = run_grid(n_sims=3, seed=0)
    # 2x2 grid (small, for speed) would need overriding UC_GRID/DRIFT_GRID —
    # instead just check the full module-level grid produced valid rows.
    for col in ["estimator", "U", "drift", "bias", "true_value", "n_sim"]:
        assert col in df.columns
    assert len(df) > 0
    assert df["n_sim"].eq(3).all()


def test_naive_estimator_biased_at_high_unmeasured_confounding():
    df = run_grid(n_sims=5, seed=1)
    naive_high_u = df[(df.estimator == "naive") & (df.U == 0.8) & (df.drift == 0.0)]
    assert len(naive_high_u) == 1
    assert abs(naive_high_u["bias"].iloc[0]) > 0.01


def test_compounding_table_decomposition_is_internally_consistent():
    df = run_grid(n_sims=3, seed=2)
    comp = compounding_table(df)
    assert set(comp.columns) >= {
        "estimator", "U", "drift", "bias_observed",
        "additive_prediction", "excess_bias", "compounding"}
    # excess_bias must equal bias_observed - additive_prediction, exactly
    assert np.allclose(
        comp["excess_bias"], comp["bias_observed"] - comp["additive_prediction"])
    # compounding flag matches the sign of excess_bias
    assert (comp["compounding"] == (comp["excess_bias"] > 0)).all()
    # every non-(0,*)/(*,0) cell of the grid is present
    n_nonzero_u = sum(1 for u in [0.0, 0.27, 0.53, 0.8] if u != 0.0)
    n_nonzero_d = sum(1 for d in [0.0, 0.17, 0.33, 0.5] if d != 0.0)
    assert len(comp) == len(comp["estimator"].unique()) * n_nonzero_u * n_nonzero_d
