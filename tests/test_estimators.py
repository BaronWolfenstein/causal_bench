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
