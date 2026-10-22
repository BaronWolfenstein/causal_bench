"""Known-truth regression tests for the 2026-10-21 AIPW / pointwise-RMST audit fixes (see estimators/_ipcw_rd.py)."""
import numpy as np
from causal_bench.dgp.config import DGPConfig
from causal_bench.dgp.survival import generate_data, compute_true_effects, compute_true_rmst
from causal_bench.estimators.aipw import AIPWEstimator
from causal_bench.estimators.pointwise_rmst import PointwiseRMSTEstimator
from causal_bench.estimators._ipcw_rd import censoring_survival_fn


def test_no_censoring_gives_unit_G():
    df = generate_data(DGPConfig(n=400, seed=1, censoring_rate=0.0))
    G = censoring_survival_fn(df, df[["W1", "W2", "W3", "W4"]].values, 1.0)
    assert np.allclose(G(df["T_obs"].values), 1.0)


def test_aipw_unbiased_without_censoring():
    # Single Q in plug-in and residual terms: with no censoring AIPW must hit the true risk difference.
    cfg = DGPConfig(n=2000, seed=3, censoring_rate=0.0)
    r = AIPWEstimator(n_folds=3).estimate(generate_data(cfg), horizon=cfg.horizon)[0]
    assert abs(r.point_estimate - compute_true_effects(cfg)["ATE"]) < 4 * r.standard_error


def test_pointwise_rmst_unbiased_under_censoring():
    # IPCW: 40% censoring must not bias the K=20 RMST difference (the old version counted censored patients as
    # event-free, a bias that does not shrink with K).
    cfg = DGPConfig(n=2000, seed=4, censoring_rate=0.4)
    r = PointwiseRMSTEstimator(n_grid=20, n_folds=3).estimate(generate_data(cfg), horizon=cfg.horizon)[0]
    assert abs(r.point_estimate - compute_true_rmst(cfg)["ATE"]) < 4 * r.standard_error


def test_aipw_ipcw_option_runs():
    cfg = DGPConfig(n=400, seed=5, censoring_rate=0.3)
    r = AIPWEstimator(n_folds=3, ipcw=True).estimate(generate_data(cfg), horizon=cfg.horizon)[0]
    assert np.isfinite(r.point_estimate) and r.standard_error > 0


def test_l1_effects_default_off_and_only_after_tL1():
    from causal_bench.dgp.survival import _post_l1
    base = generate_data(DGPConfig(n=300, seed=7))
    same = generate_data(DGPConfig(n=300, seed=7, l1_hazard_effect=0.0, l1_censor_effect=0.0))
    assert base.equals(same)
    T = np.array([0.2, 0.5, 0.9]); L1 = np.array([1.0, 1.0, 1.0])
    out = _post_l1(T, L1, 0.5, 1.0)
    assert out[0] == 0.2 and out[1] == 0.5 and out[2] < 0.9        # only residual time after t_L1 is scaled


def test_ltmle_runs_with_time_varying_confounding():
    from causal_bench.estimators.ltmle import LTMLEEstimator
    cfg = DGPConfig(n=600, seed=8, censoring_rate=0.3, l1_hazard_effect=1.0, l1_censor_effect=2.0)
    r = LTMLEEstimator(n_folds=3).estimate(generate_data(cfg), horizon=cfg.horizon)[0]
    assert np.isfinite(r.point_estimate) and r.standard_error > 0
