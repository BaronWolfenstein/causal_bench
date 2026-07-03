"""Plumbing smoke for exp33b (Donsker-nuisance TMLE comparison).

Only the fast `default` arm is exercised — the LTB/HAR arms are correctness-
tested in test_tmle_ipcw_learners.py; here we just confirm the experiment's
run()/summarize() wiring produces the expected frame shape.
"""
import numpy as np

from experiments.exp33b_donsker_nuisance_tmle import run, summarize


def test_run_and_summarize_default_and_cv_arms():
    # default (tmle_ipcw) and cv (tmle_ipcw_cv) are the fast arms; the LTB arms
    # are correctness-tested in test_tmle_ipcw_learners.py.
    raw, tau0 = run(n_sims=2, seed=1, arms=["default", "cv"])
    assert np.isfinite(tau0)
    assert set(raw["arm"]) == {"default", "cv"}
    assert len(raw) == 4
    summ = summarize(raw, tau0)
    assert len(summ) == 2
    for col in ["arm", "n_ok", "bias", "rmse", "coverage", "mean_se", "emp_sd"]:
        assert col in summ.columns
    assert (summ["coverage"].between(0.0, 1.0)).all()
