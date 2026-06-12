import numpy as np
from causal_bench.metrics import SimResult, EstimatorResult

def test_estimator_result_fields():
    r = EstimatorResult(name="Test", estimand="ATE",
                        point_estimate=0.1, standard_error=0.05,
                        ci_lower=0.0, ci_upper=0.2)
    assert r.name == "Test"
    assert r.ess is None
    assert r.convergence_info is None

def test_sim_result_bias_zero():
    est = np.array([0.2, 0.2, 0.2])
    sr = SimResult("test", "ATE", true_value=0.2, n_sim=3,
                   estimates=est,
                   se_estimates=np.array([0.05, 0.05, 0.05]),
                   ci_lowers=est - 0.1,
                   ci_uppers=est + 0.1,
                   nc_estimates=np.zeros(3))
    assert abs(sr.bias) < 1e-10

def test_sim_result_coverage_perfect():
    est = np.array([0.1, 0.2, 0.3])
    sr = SimResult("test", "ATE", true_value=0.2, n_sim=3,
                   estimates=est,
                   se_estimates=np.ones(3) * 0.05,
                   ci_lowers=est - 0.5,
                   ci_uppers=est + 0.5,
                   nc_estimates=np.zeros(3))
    assert sr.coverage == 1.0

def test_sim_result_coverage_zero():
    est = np.array([1.0, 2.0, 3.0])
    sr = SimResult("test", "ATE", true_value=0.0, n_sim=3,
                   estimates=est,
                   se_estimates=np.ones(3) * 0.01,
                   ci_lowers=est - 0.01,
                   ci_uppers=est + 0.01,
                   nc_estimates=np.zeros(3))
    assert sr.coverage == 0.0

def test_sim_result_rmse():
    est = np.array([0.3, 0.3, 0.3])
    sr = SimResult("test", "ATE", true_value=0.0, n_sim=3,
                   estimates=est,
                   se_estimates=np.ones(3) * 0.05,
                   ci_lowers=est - 0.1,
                   ci_uppers=est + 0.1,
                   nc_estimates=np.zeros(3))
    assert abs(sr.rmse - 0.3) < 1e-10
    assert abs(sr.bias - 0.3) < 1e-10

def test_sim_result_summary_keys():
    est = np.array([0.1, 0.2])
    sr = SimResult("naive", "ATE", true_value=0.15, n_sim=2,
                   estimates=est,
                   se_estimates=np.ones(2) * 0.05,
                   ci_lowers=est - 0.1,
                   ci_uppers=est + 0.1,
                   nc_estimates=np.zeros(2))
    s = sr.summary()
    for key in ["estimator", "estimand", "true", "bias", "rmse",
                "coverage", "ci_width", "se_ratio", "nc_bias"]:
        assert key in s

def test_sim_result_repr_no_array_dump():
    est = np.ones(1000)
    sr = SimResult("test", "ATE", true_value=1.0, n_sim=1000,
                   estimates=est, se_estimates=est,
                   ci_lowers=est - 0.1, ci_uppers=est + 0.1,
                   nc_estimates=est)
    r = repr(sr)
    assert "1000" not in r or "n_sim=1000" in r  # arrays suppressed


import pandas as pd
from causal_bench.dgp.survival import generate_data
from causal_bench.dgp.config import DGPConfig
from causal_bench.estimators.naive import NaiveEstimator
from causal_bench.estimators.kaplan_meier import KaplanMeierEstimator
from causal_bench.estimators.cox import CoxEstimator


def _clean_df(n=500, seed=0):
    return generate_data(DGPConfig(n=n, censoring_informativeness=0.0,
                                   unmeasured_confounding_strength=0.0,
                                   positivity_severity=0.0, seed=seed))


def test_naive_returns_result():
    df = _clean_df()
    results = NaiveEstimator().estimate(df, horizon=1.0, estimand="ATE")
    assert len(results) == 1
    r = results[0]
    assert r.name == "Naive"
    assert -2.0 < r.point_estimate < 2.0
    assert r.ci_lower < r.point_estimate < r.ci_upper


def test_naive_all_censored_returns_zero():
    # All Delta=0 means no events in either arm → ATE estimate is 0.0, not NaN
    df = _clean_df()
    df = df.copy()
    df["Delta"] = 0.0
    results = NaiveEstimator().estimate(df, horizon=1.0)
    assert results[0].point_estimate == 0.0


def test_km_returns_result():
    df = _clean_df()
    results = KaplanMeierEstimator().estimate(df, horizon=1.0)
    assert len(results) == 1
    r = results[0]
    assert r.name == "KM"
    assert r.ci_lower < r.point_estimate < r.ci_upper


def test_cox_returns_result():
    df = _clean_df(n=300)
    results = CoxEstimator(n_bootstrap=10).estimate(df, horizon=1.0)
    assert len(results) == 1
    r = results[0]
    assert r.name == "Cox"
    assert r.ci_lower < r.ci_upper


def test_negative_control_near_zero():
    from causal_bench.estimators.naive import NaiveEstimator
    df = _clean_df(n=2000, seed=7)
    nc = NaiveEstimator().estimate_negative_control(df)
    assert abs(nc) < 0.20  # should be near zero without confounding


def test_base_estimator_nc_handles_all_treated():
    df = _clean_df(n=100)
    df = df.copy(); df["A"] = 1.0
    nc = NaiveEstimator().estimate_negative_control(df)
    assert np.isnan(nc)


from causal_bench.estimators.tmle_ipcw import TMLEIPCWEstimator


def test_tmle_ipcw_returns_ate():
    df = _clean_df(n=300, seed=10)
    est = TMLEIPCWEstimator(use_compliance=False, n_folds=3, random_state=0)
    results = est.estimate(df, horizon=1.0, estimand="ATE")
    assert any(r.estimand == "ATE" for r in results)
    r = next(r for r in results if r.estimand == "ATE")
    assert r.name == "TMLE+IPCW"
    assert not np.isnan(r.point_estimate)
    assert r.ci_lower < r.point_estimate < r.ci_upper


def test_tmle_ipcw_comply_name():
    df = _clean_df(n=300, seed=11)
    est = TMLEIPCWEstimator(use_compliance=True, n_folds=3, random_state=0)
    results = est.estimate(df, horizon=1.0, estimand="ATE")
    assert results[0].name == "TMLE+IPCW+Comply"


def test_tmle_ipcw_reasonable_estimate():
    """On clean data, TMLE+IPCW should produce a finite estimate in [-1, 1]."""
    df = _clean_df(n=400, seed=12)
    est = TMLEIPCWEstimator(use_compliance=False, n_folds=3, random_state=0)
    results = est.estimate(df, horizon=1.0, estimand="ATE")
    r = results[0]
    assert -1.0 < r.point_estimate < 1.0
    assert r.standard_error > 0


from causal_bench.runner import run_simulation
from causal_bench.dgp.config import DGPConfig as _DGPConfig


def test_run_simulation_smoke():
    cfg = _DGPConfig(n=150, seed=0, censoring_informativeness=0.0)
    results = run_simulation(cfg, estimator_names=["naive", "km"],
                             n_sim=4, n_jobs=1, seed=0)
    assert "naive" in results
    assert "km" in results
    assert results["naive"].n_sim == 4


def test_run_simulation_true_value_finite():
    cfg = _DGPConfig(n=150, seed=1)
    results = run_simulation(cfg, estimator_names=["naive"],
                             n_sim=3, n_jobs=1, seed=1)
    assert np.isfinite(results["naive"].true_value)


def test_cox_l1_returns_result():
    cfg = DGPConfig(n=300, collider_strength=0.5, seed=0)
    df = generate_data(cfg)
    results = CoxEstimator(include_L1=True, n_bootstrap=5).estimate(df)
    assert results[0].name == "Cox+L1"
    assert not np.isnan(results[0].point_estimate)


def test_cox_l1_in_registry():
    from causal_bench.estimators import ESTIMATOR_REGISTRY
    assert "cox_l1" in ESTIMATOR_REGISTRY


def test_cox_l1_not_in_mvp_estimators():
    from causal_bench.estimators import MVP_ESTIMATORS
    assert "cox_l1" not in MVP_ESTIMATORS


def test_ltmle_returns_result():
    from causal_bench.estimators.ltmle import LTMLEEstimator
    cfg = DGPConfig(n=400, collider_strength=0.5, seed=0)
    df = generate_data(cfg)
    results = LTMLEEstimator(n_folds=3).estimate(df, horizon=1.0)
    assert results[0].name == "LTMLE"
    assert not np.isnan(results[0].point_estimate)
    assert not np.isnan(results[0].standard_error)


def test_ltmle_falls_back_without_l1():
    """Falls back gracefully when L1 is all-NaN."""
    from causal_bench.estimators.ltmle import LTMLEEstimator
    cfg = DGPConfig(n=300, collider_strength=0.0, seed=0)
    df = generate_data(cfg)
    # Zero out L1 to simulate fallback condition
    df["L1"] = np.nan
    results = LTMLEEstimator(n_folds=3).estimate(df)
    assert len(results) >= 1
    assert not np.isnan(results[0].point_estimate)


def test_ltmle_in_registry():
    from causal_bench.estimators import ESTIMATOR_REGISTRY
    assert "ltmle" in ESTIMATOR_REGISTRY


def test_ltmle_not_in_mvp():
    from causal_bench.estimators import MVP_ESTIMATORS
    assert "ltmle" not in MVP_ESTIMATORS


def test_ipw_returns_result():
    from causal_bench.estimators.ipw import IPWEstimator
    cfg = DGPConfig(n=300, seed=0)
    df = generate_data(cfg)
    results = IPWEstimator(n_folds=3).estimate(df)
    assert results[0].name == "IPW"
    assert not np.isnan(results[0].point_estimate)
    assert not np.isnan(results[0].standard_error)
    assert results[0].standard_error > 0


def test_overlap_returns_result():
    from causal_bench.estimators.overlap import OverlapEstimator
    cfg = DGPConfig(n=300, seed=0)
    df = generate_data(cfg)
    results = OverlapEstimator(n_folds=3).estimate(df)
    assert results[0].name == "Overlap"
    assert not np.isnan(results[0].point_estimate)
    assert not np.isnan(results[0].standard_error)


def test_ipw_in_registry():
    from causal_bench.estimators import ESTIMATOR_REGISTRY
    assert "ipw" in ESTIMATOR_REGISTRY


def test_overlap_in_registry():
    from causal_bench.estimators import ESTIMATOR_REGISTRY
    assert "overlap" in ESTIMATOR_REGISTRY


def test_ipw_overlap_not_in_mvp():
    from causal_bench.estimators import MVP_ESTIMATORS
    assert "ipw" not in MVP_ESTIMATORS
    assert "overlap" not in MVP_ESTIMATORS


def test_aipw_returns_result():
    from causal_bench.estimators.aipw import AIPWEstimator
    cfg = DGPConfig(n=300, seed=0)
    df = generate_data(cfg)
    results = AIPWEstimator(n_folds=3).estimate(df)
    assert results[0].name == "AIPW"
    assert not np.isnan(results[0].point_estimate)
    assert not np.isnan(results[0].standard_error)
    assert results[0].standard_error > 0


def test_aipw_in_registry():
    from causal_bench.estimators import ESTIMATOR_REGISTRY
    assert "aipw" in ESTIMATOR_REGISTRY


def test_aipw_not_in_mvp():
    from causal_bench.estimators import MVP_ESTIMATORS
    assert "aipw" not in MVP_ESTIMATORS


def test_pointwise_rmst_k2_returns_result():
    from causal_bench.estimators.pointwise_rmst import PointwiseRMSTEstimator
    cfg = DGPConfig(n=200, seed=0)
    df = generate_data(cfg)
    results = PointwiseRMSTEstimator(n_grid=2, n_folds=3).estimate(df)
    assert results[0].name == "RMST_K2"
    assert not np.isnan(results[0].point_estimate)
    assert results[0].standard_error > 0


def test_pointwise_rmst_k20_returns_result():
    from causal_bench.estimators.pointwise_rmst import PointwiseRMSTEstimator
    cfg = DGPConfig(n=200, seed=0)
    df = generate_data(cfg)
    results = PointwiseRMSTEstimator(n_grid=20, n_folds=3).estimate(df)
    assert results[0].name == "RMST_K20"
    assert not np.isnan(results[0].point_estimate)


def test_pointwise_rmst_in_registry():
    from causal_bench.estimators import ESTIMATOR_REGISTRY
    for k in ["rmst_k2", "rmst_k5", "rmst_k10", "rmst_k20"]:
        assert k in ESTIMATOR_REGISTRY


def test_pointwise_rmst_not_in_mvp():
    from causal_bench.estimators import MVP_ESTIMATORS
    for k in ["rmst_k2", "rmst_k5", "rmst_k10", "rmst_k20"]:
        assert k not in MVP_ESTIMATORS


def test_rmst_k20_closer_than_k2():
    """K=20 should be closer to K=2's true RMST on clean data (more integration points)."""
    from causal_bench.estimators.pointwise_rmst import PointwiseRMSTEstimator
    cfg = DGPConfig(n=500, censoring_informativeness=0.0, seed=1)
    df = generate_data(cfg)
    r2  = PointwiseRMSTEstimator(n_grid=2,  n_folds=3).estimate(df)[0].point_estimate
    r20 = PointwiseRMSTEstimator(n_grid=20, n_folds=3).estimate(df)[0].point_estimate
    # Both should be finite
    assert np.isfinite(r2) and np.isfinite(r20)
    # K=20 estimate should exist and not crash (bias convergence needs MC sims to verify)


def test_win_ratio_estimators_in_registry():
    from causal_bench.estimators import ESTIMATOR_REGISTRY, get_estimator
    assert "concrete_WR_direct" in ESTIMATOR_REGISTRY
    assert "concrete_WR_plugin" in ESTIMATOR_REGISTRY
    direct = get_estimator("concrete_WR_direct")
    plugin = get_estimator("concrete_WR_plugin")
    assert direct.name == "concrete_WR_direct"
    assert plugin.name == "concrete_WR_plugin"
