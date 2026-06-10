"""Tests for causal_bench.bootstrap IC bootstrap CI utilities."""
import numpy as np
import pytest
from causal_bench.bootstrap import ic_bootstrap_ci
from causal_bench.metrics import EstimatorResult


def _make_result(ic, point=0.1):
    se = float(np.std(ic, ddof=1) / np.sqrt(len(ic)))
    return EstimatorResult(
        name="Test", estimand="ATE",
        point_estimate=point, standard_error=se,
        ci_lower=point - 1.96 * se, ci_upper=point + 1.96 * se,
        ic=ic,
    )


def test_raises_without_ic():
    result = EstimatorResult(
        name="X", estimand="ATE",
        point_estimate=0.1, standard_error=0.05,
        ci_lower=0.0, ci_upper=0.2,
    )
    with pytest.raises(ValueError, match="no ic array"):
        ic_bootstrap_ci(result)


@pytest.mark.parametrize("method", ["percentile", "t", "bca"])
def test_ci_returns_floats_and_ordered(method):
    rng = np.random.default_rng(0)
    ic = rng.normal(0, 0.05, 500)
    result = _make_result(ic, point=0.1)
    lo, hi = ic_bootstrap_ci(result, B=500, method=method, rng=np.random.default_rng(1))
    assert isinstance(lo, float)
    assert isinstance(hi, float)
    assert lo < hi


@pytest.mark.parametrize("method", ["percentile", "t", "bca"])
def test_ci_contains_true_value_normal_ic(method):
    # With Gaussian IC and large n, all methods should cover truth ~95% of the time.
    # We use a single fixed draw — just check the CI brackets the estimate.
    rng = np.random.default_rng(42)
    ic = rng.normal(0, 0.05, 1000)
    true_val = 0.0
    result = _make_result(ic, point=float(np.mean(ic)))
    lo, hi = ic_bootstrap_ci(result, B=1000, method=method, rng=np.random.default_rng(0))
    assert lo <= true_val <= hi, f"{method}: [{lo:.4f}, {hi:.4f}] does not contain {true_val}"


def test_bca_bias_correction_shifts_ci():
    # When most bootstrap samples are below theta (negative z0), BCa should shift
    # the CI leftward relative to a plain percentile CI.
    rng = np.random.default_rng(7)
    # Skewed IC: mostly negative values
    ic = rng.exponential(0.05, 500) - 0.07
    result = _make_result(ic, point=0.1)
    lo_pct, hi_pct = ic_bootstrap_ci(result, B=2000, method="percentile",
                                     rng=np.random.default_rng(0))
    lo_bca, hi_bca = ic_bootstrap_ci(result, B=2000, method="bca",
                                     rng=np.random.default_rng(0))
    # BCa and percentile should differ when IC is skewed
    assert lo_pct != lo_bca or hi_pct != hi_bca


def test_unknown_method_raises():
    rng = np.random.default_rng(0)
    ic = rng.normal(0, 0.05, 200)
    result = _make_result(ic)
    with pytest.raises(ValueError, match="Unknown method"):
        ic_bootstrap_ci(result, method="bogus")


def test_ic_bootstrap_reproducible():
    rng_seed = np.random.default_rng(99)
    ic = rng_seed.normal(0, 0.05, 300)
    result = _make_result(ic)
    lo1, hi1 = ic_bootstrap_ci(result, B=500, rng=np.random.default_rng(42))
    lo2, hi2 = ic_bootstrap_ci(result, B=500, rng=np.random.default_rng(42))
    assert lo1 == lo2 and hi1 == hi2


def test_estimators_store_ic():
    """TMLE+IPCW, LTMLE, AIPW should all return results with ic arrays."""
    import pandas as pd
    from causal_bench.dgp.survival import DGPConfig, generate_data
    from causal_bench.estimators.tmle_ipcw import TMLEIPCWEstimator
    from causal_bench.estimators.ltmle import LTMLEEstimator
    from causal_bench.estimators.aipw import AIPWEstimator

    df = generate_data(DGPConfig(n=200), rng=np.random.default_rng(0))
    for cls in (TMLEIPCWEstimator, LTMLEEstimator, AIPWEstimator):
        results = cls().estimate(df)
        r = results[0]
        assert r.ic is not None, f"{cls.__name__} did not store ic"
        assert len(r.ic) == len(df)
        assert np.isfinite(r.ic).all(), f"{cls.__name__} has non-finite ic values"


def test_ic_bootstrap_end_to_end():
    """BCa CI from a real estimator should be finite and ordered."""
    from causal_bench.dgp.survival import DGPConfig, generate_data
    from causal_bench.estimators.tmle_ipcw import TMLEIPCWEstimator

    df = generate_data(DGPConfig(n=300), rng=np.random.default_rng(5))
    result = TMLEIPCWEstimator().estimate(df)[0]
    lo, hi = ic_bootstrap_ci(result, B=500, method="bca", rng=np.random.default_rng(0))
    assert np.isfinite(lo) and np.isfinite(hi)
    assert lo < hi
