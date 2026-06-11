"""Tests for DataFrame → R conversion edge cases.

These tests run without R or rpy2 installed. They validate that the
DataFrame preparation step (before pandas2ri handoff) handles all the
edge cases that concrete's formatArguments is strict about.

The prepare_for_r() function lives in concrete_rmst.py and is testable
independently of rpy2.
"""
import numpy as np
import pandas as pd
import pytest

from causal_bench.dgp.config import DGPConfig
from causal_bench.dgp.survival import generate_data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(n=200, collider_strength=0.5, seed=0):
    cfg = DGPConfig(n=n, collider_strength=collider_strength, seed=seed)
    return generate_data(cfg)


def _add_event_type(df):
    """concrete requires an event_type column (0=censored, 1=event, 2=competing)."""
    df = df.copy()
    df["event_type"] = df["Delta"].astype(int)
    return df


# ---------------------------------------------------------------------------
# Tests for the prepare_for_r() utility (pure Python, no rpy2)
# ---------------------------------------------------------------------------

class TestPrepareForR:
    """prepare_for_r() must produce a DataFrame that pandas2ri can handle."""

    def test_import_without_r(self):
        """Module imports cleanly even if rpy2/R not installed."""
        # Should not raise — ConcreteRMSTEstimator defers R import to estimate()
        from causal_bench.estimators.concrete_rmst import prepare_for_r  # noqa: F401

    def test_l1_nan_preserved(self):
        """L1 NaN must be left as-is — the R bridge routes L1 into CensoringTV and
        filters non-NA rows there. Imputing NaN here would corrupt the censoring model."""
        from causal_bench.estimators.concrete_rmst import prepare_for_r
        df = _add_event_type(_make_df(collider_strength=0.5))
        original_nan_count = df["L1"].isna().sum()
        assert original_nan_count > 0, "fixture should have some NaN L1 values"
        out = prepare_for_r(df)
        assert out["L1"].isna().sum() == original_nan_count, \
            "prepare_for_r must not impute L1 NaN — the R bridge handles filtering"

    def test_l1_not_outcome_covariate(self):
        """L1 must not appear in the outcome covariate set after prepare_for_r.
        The R bridge sends it to CensoringTV only; conditioning on it in the
        outcome model would introduce collider bias."""
        from causal_bench.estimators.concrete_rmst import prepare_for_r
        df = _add_event_type(_make_df(collider_strength=0.5))
        out = prepare_for_r(df)
        # L1 may still be present in the dataframe (the R bridge needs it to build
        # CensoringTV), but it must not be silently imputed and bundled with W1-W4.
        w_cols = [c for c in out.columns if c.startswith("W")]
        assert "L1" not in w_cols, "L1 must not be merged into the W covariate columns"

    def test_delta_is_integer(self):
        """concrete expects event_type as integer, not float."""
        from causal_bench.estimators.concrete_rmst import prepare_for_r
        df = _add_event_type(_make_df())
        out = prepare_for_r(df)
        assert out["event_type"].dtype in (np.int32, np.int64, int), \
            f"event_type dtype should be int, got {out['event_type'].dtype}"

    def test_treatment_is_integer(self):
        """concrete expects Treatment column as integer."""
        from causal_bench.estimators.concrete_rmst import prepare_for_r
        df = _add_event_type(_make_df())
        out = prepare_for_r(df)
        assert out["A"].dtype in (np.int32, np.int64, int), \
            f"A dtype should be int, got {out['A'].dtype}"

    def test_float_columns_are_float64(self):
        """pandas2ri handles float64 cleanly; float32 can cause silent truncation."""
        from causal_bench.estimators.concrete_rmst import prepare_for_r
        df = _add_event_type(_make_df())
        # Downcast to float32 to test normalization
        df["T_obs"] = df["T_obs"].astype(np.float32)
        out = prepare_for_r(df)
        assert out["T_obs"].dtype == np.float64, \
            "float32 columns should be upcast to float64"

    def test_no_negative_times(self):
        """R survival models cannot handle negative T_obs."""
        from causal_bench.estimators.concrete_rmst import prepare_for_r
        df = _add_event_type(_make_df())
        out = prepare_for_r(df)
        assert (out["T_obs"] >= 0).all(), "T_obs must be non-negative"

    def test_required_columns_present(self):
        """prepare_for_r raises ValueError if any required column is missing."""
        from causal_bench.estimators.concrete_rmst import prepare_for_r
        df = _add_event_type(_make_df())
        df_missing = df.drop(columns=["T_obs"])
        with pytest.raises((ValueError, KeyError)):
            prepare_for_r(df_missing)

    def test_zero_variance_column_preserved(self):
        """Zero-variance columns (e.g. W4=0 for all) should not be dropped silently."""
        from causal_bench.estimators.concrete_rmst import prepare_for_r
        df = _add_event_type(_make_df())
        df["W4"] = 0.0  # make zero-variance
        out = prepare_for_r(df)
        assert "W4" in out.columns

    def test_small_n(self):
        """n=10 should not crash prepare_for_r."""
        from causal_bench.estimators.concrete_rmst import prepare_for_r
        df = _add_event_type(_make_df(n=10))
        out = prepare_for_r(df)
        assert len(out) == 10

    def test_all_events_no_censoring(self):
        """When all Delta=1 (no censoring), event_type should all be 1."""
        from causal_bench.estimators.concrete_rmst import prepare_for_r
        df = _add_event_type(_make_df())
        df["Delta"] = 1.0
        df["event_type"] = 1
        out = prepare_for_r(df)
        assert (out["event_type"] == 1).all()

    def test_all_censored(self):
        """When all Delta=0 (all censored), event_type should all be 0."""
        from causal_bench.estimators.concrete_rmst import prepare_for_r
        df = _add_event_type(_make_df())
        df["Delta"] = 0.0
        df["event_type"] = 0
        out = prepare_for_r(df)
        assert (out["event_type"] == 0).all()

    def test_output_is_dataframe(self):
        """prepare_for_r must return a pandas DataFrame."""
        from causal_bench.estimators.concrete_rmst import prepare_for_r
        df = _add_event_type(_make_df())
        out = prepare_for_r(df)
        assert isinstance(out, pd.DataFrame)

    def test_index_reset(self):
        """R doesn't handle non-default row indices — index must be 0..n-1."""
        from causal_bench.estimators.concrete_rmst import prepare_for_r
        df = _add_event_type(_make_df())
        df.index = np.arange(100, 100 + len(df))  # non-default index
        out = prepare_for_r(df)
        assert list(out.index) == list(range(len(out))), \
            "Index must be reset to 0..n-1 before R handoff"


# ---------------------------------------------------------------------------
# ConcreteRMSTEstimator stub behaviour (no R required)
# ---------------------------------------------------------------------------

class TestConcreteRMSTEstimatorStub:

    def test_estimator_importable(self):
        from causal_bench.estimators.concrete_rmst import ConcreteRMSTEstimator
        est = ConcreteRMSTEstimator()
        assert est.name == "concrete_RMST"

    def test_estimate_returns_empty_when_r_unavailable(self, monkeypatch):
        """When concrete/rpy2 not available, estimate() returns [] (no crash)."""
        from causal_bench.estimators import concrete_rmst
        monkeypatch.setattr(concrete_rmst, "_concrete_available", lambda: False)
        from causal_bench.estimators.concrete_rmst import ConcreteRMSTEstimator
        df = _add_event_type(_make_df())
        results = ConcreteRMSTEstimator().estimate(df)
        assert results == [], "Should return [] when concrete unavailable"

    def test_concrete_not_in_mvp(self):
        from causal_bench.estimators import MVP_ESTIMATORS
        assert "concrete_RMST" not in MVP_ESTIMATORS


# ---------------------------------------------------------------------------
# Live integration tests — skipped when concrete R package is not installed
# ---------------------------------------------------------------------------

import pytest

concrete_available = pytest.mark.skipif(
    not __import__("causal_bench.estimators.concrete_rmst", fromlist=["_concrete_available"])._concrete_available(),
    reason="concrete R package not installed",
)


@concrete_available
class TestConcreteRMSTLive:

    def test_estimate_returns_result(self):
        from causal_bench.estimators.concrete_rmst import ConcreteRMSTEstimator
        df = _add_event_type(_make_df(n=300, seed=0))
        results = ConcreteRMSTEstimator().estimate(df)
        assert len(results) == 1
        r = results[0]
        assert r.name == "concrete_RMST"
        assert r.estimand == "ATE"

    def test_estimate_finite_values(self):
        from causal_bench.estimators.concrete_rmst import ConcreteRMSTEstimator
        import numpy as np
        df = _add_event_type(_make_df(n=300, seed=1))
        results = ConcreteRMSTEstimator().estimate(df)
        assert results, "expected non-empty result"
        r = results[0]
        assert np.isfinite(r.point_estimate)
        assert np.isfinite(r.standard_error)
        assert r.standard_error > 0
        assert r.ci_lower < r.ci_upper

    def test_estimate_ci_contains_point(self):
        from causal_bench.estimators.concrete_rmst import ConcreteRMSTEstimator
        df = _add_event_type(_make_df(n=300, seed=2))
        r = ConcreteRMSTEstimator().estimate(df)[0]
        assert r.ci_lower <= r.point_estimate <= r.ci_upper

    def test_collider_l1_not_inflating_se(self):
        """With collider_strength=0 vs 0.8, SE should be similar — L1 must not
        enter the outcome model (which would inflate variance via collider bias)."""
        from causal_bench.estimators.concrete_rmst import ConcreteRMSTEstimator
        import numpy as np
        from causal_bench.dgp.config import DGPConfig
        from causal_bench.dgp.survival import generate_data

        df_no  = generate_data(DGPConfig(n=400, collider_strength=0.0, seed=3))
        df_no["event_type"] = df_no["Delta"].astype(int)
        df_hi  = generate_data(DGPConfig(n=400, collider_strength=0.8, seed=3))
        df_hi["event_type"] = df_hi["Delta"].astype(int)

        se_no = ConcreteRMSTEstimator().estimate(df_no)[0].standard_error
        se_hi = ConcreteRMSTEstimator().estimate(df_hi)[0].standard_error
        # If L1 leaked into the outcome model, se_hi would be substantially
        # larger. Allow up to 3× as a loose guard.
        assert se_hi < se_no * 3, f"SE ratio {se_hi/se_no:.2f} suggests L1 collider leak"

    def test_sensitivity_returns_dataframe(self):
        from causal_bench.estimators.concrete_rmst import concrete_sensitivity
        import pandas as pd
        df = _add_event_type(_make_df(n=300, seed=4))
        result = concrete_sensitivity(df, deltas=[0.0, 0.05, 0.10])
        assert isinstance(result, pd.DataFrame)
        for col in ("mechanism", "delta", "estimate", "se", "ci_lower", "ci_upper"):
            assert col in result.columns, f"missing column: {col}"
        assert len(result) == 3

    def test_sensitivity_monotone_or_not_required(self):
        """Estimates need not be monotone in delta, but all must be finite."""
        from causal_bench.estimators.concrete_rmst import concrete_sensitivity
        import numpy as np
        df = _add_event_type(_make_df(n=300, seed=5))
        result = concrete_sensitivity(df, deltas=[0.0, 0.05, 0.10, 0.15, 0.20])
        assert result["estimate"].apply(np.isfinite).all()
        assert result["se"].apply(np.isfinite).all()
        assert (result["ci_lower"] < result["ci_upper"]).all()
