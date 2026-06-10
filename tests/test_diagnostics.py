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
