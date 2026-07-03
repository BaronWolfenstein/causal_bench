"""Plumbing smoke for exp33b (Donsker-nuisance TMLE comparison).

Only the fast `default` arm is exercised — the LTB/HAR arms are correctness-
tested in test_tmle_ipcw_learners.py; here we just confirm the experiment's
run()/summarize() wiring produces the expected frame shape.
"""
import numpy as np

from experiments.exp33b_donsker_nuisance_tmle import run, summarize


def test_run_and_summarize_default_arm():
    raw, tau0 = run(n_sims=2, seed=1, arms=["default"])
    assert np.isfinite(tau0)
    assert set(raw["arm"]) == {"default"}
    assert len(raw) == 2
    summ = summarize(raw, tau0)
    assert len(summ) == 1
    for col in ["arm", "n_ok", "bias", "rmse", "coverage", "mean_se", "emp_sd"]:
        assert col in summ.columns
    assert 0.0 <= summ["coverage"].iloc[0] <= 1.0
