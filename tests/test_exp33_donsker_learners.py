import numpy as np

from causal_bench.dgp.point_treatment import draw_point_treatment, true_tau
from causal_bench.estimators.point import oracle_nuisances
from experiments.exp33_donsker_learners import (
    ep_and_remainder, run_cell, summarize)

W_COLS = ["W1", "W2", "W3", "W4"]


def test_oracle_ep_and_remainder_are_zero():
    # With f_hat == f_0, eif0(f_hat) - eif0(f_0) == 0 pointwise, so both
    # the EP term and the remainder must vanish (remainder up to MC error).
    df = draw_point_treatment(n=700, surface="smooth", seed=0)
    nf = oracle_nuisances(df[W_COLS].values, "smooth")
    ep, rem = ep_and_remainder(nf, df, "smooth")
    assert ep == 0.0
    assert abs(rem) < 0.01   # MC error of the fixed 1e5 evaluation sample


def test_run_cell_logistic_shape_and_columns():
    out = run_cell("logistic", crossfit=False, surface="smooth",
                   n=300, n_sims=3, base_seed=0)
    assert len(out) == 6        # 3 sims x {aipw, tmle}
    for col in ["learner", "crossfit", "surface", "estimator", "sim", "point",
                "se", "covered", "g_rmse", "q_rmse", "sqrtn_ep", "remainder"]:
        assert col in out.columns
    assert set(out["estimator"]) == {"aipw", "tmle"}
    assert out["point"].between(-1, 1).all()


def test_summarize_aggregates():
    out = run_cell("logistic", crossfit=True, surface="jumpy",
                   n=300, n_sims=3, base_seed=1)
    summ = summarize(out)
    assert len(summ) == 2       # one row per estimator within the cell
    for col in ["bias", "rmse", "coverage", "se_ratio", "g_rmse",
                "sqrtn_ep_mean", "sqrtn_ep_sd"]:
        assert col in summ.columns
    tau0 = true_tau("jumpy")
    assert np.isfinite(summ["bias"]).all()
    assert (summ["rmse"] >= abs(summ["bias"]) - 1e-12).all()
    assert summ["coverage"].between(0, 1).all()
    assert np.isfinite(tau0)


def test_oracle_cell_has_zero_nuisance_rmse():
    out = run_cell("oracle", crossfit=False, surface="smooth",
                   n=200, n_sims=1, base_seed=3)
    assert (out["g_rmse"] == 0).all()
    assert (out["q_rmse"] == 0).all()
