"""Tests for causal_bench.diagnostics."""
import numpy as np
import pandas as pd
import pytest

from causal_bench.dgp.config import DGPConfig
from causal_bench.dgp.survival import generate_data
from causal_bench.metrics import SimResult


def _make_df(n=300, seed=0):
    return generate_data(DGPConfig(n=n, seed=seed))


def _make_sim_results(n_sim=30, seed=0):
    rng = np.random.default_rng(seed)
    results = {}
    for name in ["naive", "tmle_ipcw"]:
        results[name] = SimResult(
            estimator_name=name, estimand="ATE", true_value=0.1, n_sim=n_sim,
            estimates=rng.normal(0.1, 0.05, n_sim),
            se_estimates=np.abs(rng.normal(0.05, 0.005, n_sim)),
            ci_lowers=rng.normal(0.0, 0.05, n_sim),
            ci_uppers=rng.normal(0.2, 0.05, n_sim),
            nc_estimates=rng.normal(0, 0.02, n_sim),
        )
    return results


class TestPositivitySummary:
    def test_returns_dict_with_expected_keys(self):
        from causal_bench.diagnostics import positivity_summary
        df = _make_df()
        d = positivity_summary(df)
        for k in ["g_mean", "g_min", "g_max", "g_std", "pct_extreme",
                  "effective_sample_size", "overlap_ratio"]:
            assert k in d, f"missing key: {k}"

    def test_g_in_unit_interval(self):
        from causal_bench.diagnostics import positivity_summary
        d = positivity_summary(_make_df())
        assert 0 < d["g_min"] <= d["g_mean"] <= d["g_max"] < 1

    def test_pct_extreme_in_0_100(self):
        from causal_bench.diagnostics import positivity_summary
        d = positivity_summary(_make_df())
        assert 0 <= d["pct_extreme"] <= 100

    def test_ess_positive(self):
        from causal_bench.diagnostics import positivity_summary
        d = positivity_summary(_make_df())
        assert d["effective_sample_size"] > 0

    def test_overlap_ratio_in_0_1(self):
        from causal_bench.diagnostics import positivity_summary
        d = positivity_summary(_make_df())
        assert 0 < d["overlap_ratio"] <= 1


class TestPlotOverlap:
    def test_returns_figure(self):
        from causal_bench.diagnostics import plot_overlap
        import matplotlib.pyplot as plt
        fig = plot_overlap(_make_df())
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_saves_file(self, tmp_path):
        from causal_bench.diagnostics import plot_overlap
        import matplotlib.pyplot as plt
        path = str(tmp_path / "overlap.png")
        fig = plot_overlap(_make_df(), save_path=path)
        assert (tmp_path / "overlap.png").exists()
        plt.close(fig)


class TestSMDTable:
    def test_returns_dataframe(self):
        from causal_bench.diagnostics import smd_table
        tbl = smd_table(_make_df())
        assert isinstance(tbl, pd.DataFrame)
        assert "smd_raw" in tbl.columns
        assert "smd_ipw" in tbl.columns

    def test_covers_all_w_cols(self):
        from causal_bench.diagnostics import smd_table
        tbl = smd_table(_make_df())
        for col in ["W1", "W2", "W3", "W4"]:
            assert col in tbl.index

    def test_ipw_reduces_smd_on_average(self):
        """IPW adjustment should reduce |SMD| on average."""
        from causal_bench.diagnostics import smd_table
        # Use positivity-stressed scenario for visible imbalance
        cfg = DGPConfig(n=600, positivity_severity=2.0, seed=5)
        df = generate_data(cfg)
        tbl = smd_table(df)
        mean_raw = tbl["smd_raw"].abs().mean()
        mean_ipw = tbl["smd_ipw"].abs().mean()
        assert mean_ipw < mean_raw, \
            f"IPW should reduce |SMD| on average: raw={mean_raw:.3f}, ipw={mean_ipw:.3f}"


class TestPlotLove:
    def test_returns_figure(self):
        from causal_bench.diagnostics import plot_love
        import matplotlib.pyplot as plt
        fig = plot_love(_make_df())
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_saves_file(self, tmp_path):
        from causal_bench.diagnostics import plot_love
        import matplotlib.pyplot as plt
        path = str(tmp_path / "love.png")
        plot_love(_make_df(), save_path=path)
        assert (tmp_path / "love.png").exists()
        plt.close("all")


class TestSECalibration:
    def test_table_returns_dataframe(self):
        from causal_bench.diagnostics import se_calibration_table
        tbl = se_calibration_table(_make_sim_results())
        assert isinstance(tbl, pd.DataFrame)
        assert "se_ratio" in tbl.columns
        assert "empirical_se" in tbl.columns

    def test_table_has_row_per_estimator(self):
        from causal_bench.diagnostics import se_calibration_table
        results = _make_sim_results()
        tbl = se_calibration_table(results)
        assert len(tbl) == len(results)

    def test_plot_returns_figure(self):
        from causal_bench.diagnostics import plot_se_calibration
        import matplotlib.pyplot as plt
        fig = plot_se_calibration(_make_sim_results())
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_handles_empty(self):
        from causal_bench.diagnostics import plot_se_calibration
        import matplotlib.pyplot as plt
        fig = plot_se_calibration({})
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_none_results_skipped(self):
        from causal_bench.diagnostics import se_calibration_table
        results = _make_sim_results()
        results["missing"] = None
        tbl = se_calibration_table(results)
        assert "missing" not in tbl.index


class TestTippingPoint:
    def test_table_returns_dataframe(self):
        from causal_bench.diagnostics import tipping_point_table
        tbl = tipping_point_table(_make_sim_results())
        assert isinstance(tbl, pd.DataFrame)
        assert "tipping_bias" in tbl.columns
        assert "tipping_se_units" in tbl.columns

    def test_tipping_bias_equals_abs_mean_estimate(self):
        from causal_bench.diagnostics import tipping_point_table
        results = _make_sim_results()
        tbl = tipping_point_table(results)
        for name, sr in results.items():
            expected = abs(float(np.mean(sr.estimates)))
            assert abs(tbl.loc[name, "tipping_bias"] - expected) < 1e-6

    def test_tipping_se_units_positive(self):
        from causal_bench.diagnostics import tipping_point_table
        tbl = tipping_point_table(_make_sim_results())
        assert (tbl["tipping_se_units"] > 0).all()

    def test_none_results_skipped(self):
        from causal_bench.diagnostics import tipping_point_table
        results = _make_sim_results()
        results["missing"] = None
        tbl = tipping_point_table(results)
        assert "missing" not in tbl.index

    def test_empty_results(self):
        from causal_bench.diagnostics import tipping_point_table
        tbl = tipping_point_table({})
        assert isinstance(tbl, pd.DataFrame)
        assert len(tbl) == 0

    def test_plot_returns_figure(self):
        from causal_bench.diagnostics import plot_tipping_point
        import matplotlib.pyplot as plt
        fig = plot_tipping_point(_make_sim_results())
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_saves_file(self, tmp_path):
        from causal_bench.diagnostics import plot_tipping_point
        import matplotlib.pyplot as plt
        path = str(tmp_path / "tipping.png")
        plot_tipping_point(_make_sim_results(), save_path=path)
        assert (tmp_path / "tipping.png").exists()
        plt.close("all")

    def test_plot_handles_empty(self):
        from causal_bench.diagnostics import plot_tipping_point
        import matplotlib.pyplot as plt
        fig = plot_tipping_point({})
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestESSAcrossSims:
    def test_returns_dict_with_expected_keys(self):
        from causal_bench.diagnostics import ess_across_sims
        cfg = DGPConfig(n=200, seed=0)
        result = ess_across_sims(cfg, n_draws=5, n_folds=2)
        for k in ["ess_values", "mean_ess", "median_ess", "min_ess", "max_ess", "ess_pct"]:
            assert k in result, f"missing key: {k}"

    def test_ess_values_length(self):
        from causal_bench.diagnostics import ess_across_sims
        cfg = DGPConfig(n=200, seed=0)
        result = ess_across_sims(cfg, n_draws=5, n_folds=2)
        assert len(result["ess_values"]) == 5

    def test_ess_positive(self):
        from causal_bench.diagnostics import ess_across_sims
        cfg = DGPConfig(n=200, seed=0)
        result = ess_across_sims(cfg, n_draws=5, n_folds=2)
        assert result["min_ess"] > 0

    def test_ess_not_exceed_n(self):
        from causal_bench.diagnostics import ess_across_sims
        cfg = DGPConfig(n=200, seed=0)
        result = ess_across_sims(cfg, n_draws=5, n_folds=2)
        # ESS ≤ n for IPW weights (can occasionally be slightly above due to stabilisation)
        assert result["median_ess"] <= cfg.n * 1.1

    def test_ess_pct_in_range(self):
        from causal_bench.diagnostics import ess_across_sims
        cfg = DGPConfig(n=200, seed=0)
        result = ess_across_sims(cfg, n_draws=5, n_folds=2)
        assert 0 < result["ess_pct"] <= 110  # small slack for stabilised weights

    def test_positivity_stress_lowers_ess(self):
        """High positivity severity should give lower median ESS than clean."""
        from causal_bench.diagnostics import ess_across_sims
        cfg_clean  = DGPConfig(n=400, positivity_severity=0.0, seed=1)
        cfg_stress = DGPConfig(n=400, positivity_severity=2.5, seed=1)
        r_clean  = ess_across_sims(cfg_clean,  n_draws=10, n_folds=2, seed=1)
        r_stress = ess_across_sims(cfg_stress, n_draws=10, n_folds=2, seed=1)
        assert r_stress["median_ess"] < r_clean["median_ess"], (
            f"Positivity stress should lower ESS: "
            f"clean={r_clean['median_ess']:.1f}, stress={r_stress['median_ess']:.1f}"
        )

    def test_plot_returns_figure(self):
        from causal_bench.diagnostics import plot_ess_distribution
        import matplotlib.pyplot as plt
        cfg = DGPConfig(n=200, seed=0)
        fig = plot_ess_distribution(cfg, n_draws=5, n_folds=2)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_saves_file(self, tmp_path):
        from causal_bench.diagnostics import plot_ess_distribution
        import matplotlib.pyplot as plt
        cfg = DGPConfig(n=200, seed=0)
        path = str(tmp_path / "ess.png")
        plot_ess_distribution(cfg, n_draws=5, n_folds=2, save_path=path)
        assert (tmp_path / "ess.png").exists()
        plt.close("all")


class TestTippingPointMNAR:
    def test_returns_dataframe(self):
        from causal_bench.diagnostics import tipping_point_mnar
        from causal_bench.estimators import ESTIMATOR_REGISTRY
        df = _make_df(n=200)
        cfg = DGPConfig(n=200, seed=0)
        result = tipping_point_mnar(df, ESTIMATOR_REGISTRY["km"],
                                    horizon=cfg.horizon, n_grid=3)
        assert isinstance(result, pd.DataFrame)

    def test_grid_size(self):
        from causal_bench.diagnostics import tipping_point_mnar
        from causal_bench.estimators import ESTIMATOR_REGISTRY
        df = _make_df(n=200)
        cfg = DGPConfig(n=200, seed=0)
        result = tipping_point_mnar(df, ESTIMATOR_REGISTRY["km"],
                                    horizon=cfg.horizon, n_grid=4)
        assert len(result) == 16  # 4 x 4

    def test_expected_columns(self):
        from causal_bench.diagnostics import tipping_point_mnar
        from causal_bench.estimators import ESTIMATOR_REGISTRY
        df = _make_df(n=200)
        cfg = DGPConfig(n=200, seed=0)
        result = tipping_point_mnar(df, ESTIMATOR_REGISTRY["km"],
                                    horizon=cfg.horizon, n_grid=3)
        for col in ["p_treated", "p_control", "estimate", "se",
                    "ci_lower", "ci_upper", "significant",
                    "n_censored_treated", "n_censored_control"]:
            assert col in result.columns, f"missing column: {col}"

    def test_p_range(self):
        from causal_bench.diagnostics import tipping_point_mnar
        from causal_bench.estimators import ESTIMATOR_REGISTRY
        df = _make_df(n=200)
        cfg = DGPConfig(n=200, seed=0)
        result = tipping_point_mnar(df, ESTIMATOR_REGISTRY["km"],
                                    horizon=cfg.horizon, n_grid=3)
        assert result["p_treated"].min() >= 0
        assert result["p_treated"].max() <= 1
        assert result["p_control"].min() >= 0
        assert result["p_control"].max() <= 1

    def test_accepts_string_estimator(self):
        from causal_bench.diagnostics import tipping_point_mnar
        df = _make_df(n=200)
        cfg = DGPConfig(n=200, seed=0)
        result = tipping_point_mnar(df, "km", horizon=cfg.horizon, n_grid=3)
        assert isinstance(result, pd.DataFrame)

    def test_mar_attrs_present(self):
        from causal_bench.diagnostics import tipping_point_mnar
        df = _make_df(n=200)
        cfg = DGPConfig(n=200, seed=0)
        result = tipping_point_mnar(df, "km", horizon=cfg.horizon, n_grid=3)
        assert "mar_p_treated" in result.attrs
        assert "mar_p_control" in result.attrs

    def test_administrative_censoring_not_imputed(self):
        """Patients censored at exactly horizon should not be imputed."""
        from causal_bench.diagnostics import tipping_point_mnar, _impute_censored
        df = _make_df(n=200)
        cfg = DGPConfig(n=200, seed=0)
        horizon = cfg.horizon
        rng = np.random.default_rng(0)
        # Manually mark some patients as administratively censored
        admin_idx = df.index[(df["Delta"] == 0) & (df["T_obs"] < horizon - 1e-9)][:5]
        # set T_obs = horizon for these (administrative)
        df2 = df.copy()
        df2.loc[admin_idx, "T_obs"] = horizon
        df3 = _impute_censored(df2, p_treated=1.0, p_control=1.0, horizon=horizon,
                                t_impute=horizon, rng=rng)
        # admin censored rows should still have Delta=0
        assert (df3.loc[admin_idx, "Delta"] == 0).all()

    def test_plot_returns_figure(self):
        from causal_bench.diagnostics import tipping_point_mnar, plot_tipping_point_mnar
        import matplotlib.pyplot as plt
        df = _make_df(n=200)
        cfg = DGPConfig(n=200, seed=0)
        result = tipping_point_mnar(df, "km", horizon=cfg.horizon, n_grid=3)
        fig = plot_tipping_point_mnar(result)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_saves_file(self, tmp_path):
        from causal_bench.diagnostics import tipping_point_mnar, plot_tipping_point_mnar
        import matplotlib.pyplot as plt
        df = _make_df(n=200)
        cfg = DGPConfig(n=200, seed=0)
        result = tipping_point_mnar(df, "km", horizon=cfg.horizon, n_grid=3)
        path = str(tmp_path / "mnar.png")
        plot_tipping_point_mnar(result, save_path=path)
        assert (tmp_path / "mnar.png").exists()
        plt.close("all")
