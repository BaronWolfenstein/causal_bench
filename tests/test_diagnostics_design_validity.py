"""Tests for causal_bench.diagnostics.design_validity."""
import os
import tempfile

import numpy as np
import pytest

from causal_bench.diagnostics.design_validity import (
    DesignValidityResult,
    rdd_placebo_test,
    plot_running_var_density,
    plot_rdd_scatter,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def flat_data():
    """Running variable uniform on [-2, 2], outcomes pure noise — no discontinuity."""
    rng = np.random.default_rng(0)
    n   = 400
    x   = rng.uniform(-2.0, 2.0, size=n)
    y   = rng.normal(0.0, 1.0, size=n)
    return x, y


@pytest.fixture
def step_data():
    """Sharp step of +3 at cutoff=0 — should always reject the placebo test."""
    rng = np.random.default_rng(1)
    n   = 400
    x   = rng.uniform(-2.0, 2.0, size=n)
    y   = np.where(x > 0, 3.0, 0.0) + rng.normal(0, 0.1, size=n)
    return x, y


@pytest.fixture
def rare_mask_far(flat_data):
    """Rare patients clustered far from the cutoff (x < -1.5)."""
    x, _ = flat_data
    return x < -1.5


@pytest.fixture
def rare_mask_near(flat_data):
    """Rare patients clustered near the cutoff (|x| < 0.3)."""
    x, _ = flat_data
    return np.abs(x) < 0.3


# ─── TestRDDPlaceboTest ────────────────────────────────────────────────────────

class TestRDDPlaceboTest:

    def test_returns_design_validity_result(self, flat_data):
        x, y = flat_data
        result = rdd_placebo_test(y, x, cutoff=0.0, bandwidth=1.0)
        assert isinstance(result, DesignValidityResult)

    def test_no_discontinuity_passes(self, flat_data):
        x, y = flat_data
        result = rdd_placebo_test(y, x, cutoff=0.0, bandwidth=1.0)
        assert result.passed

    def test_sharp_discontinuity_fails(self, step_data):
        x, y = step_data
        result = rdd_placebo_test(y, x, cutoff=0.0, bandwidth=1.0)
        assert not result.passed

    def test_metrics_keys_present(self, flat_data):
        x, y = flat_data
        result = rdd_placebo_test(y, x, cutoff=0.0, bandwidth=1.0)
        for key in ("gap", "se_gap", "z_stat", "p_value", "bandwidth_used",
                    "n_left", "n_right"):
            assert key in result.metrics, f"missing key: {key}"

    def test_p_value_in_unit_interval(self, flat_data):
        x, y = flat_data
        result = rdd_placebo_test(y, x, cutoff=0.0, bandwidth=1.0)
        p = result.metrics["p_value"]
        assert 0.0 <= p <= 1.0

    def test_bandwidth_used_positive(self, flat_data):
        x, y = flat_data
        result = rdd_placebo_test(y, x, cutoff=0.0, bandwidth=0.5)
        assert result.metrics["bandwidth_used"] > 0.0

    def test_n_left_n_right_positive(self, flat_data):
        x, y = flat_data
        result = rdd_placebo_test(y, x, cutoff=0.0, bandwidth=1.0)
        assert result.metrics["n_left"] > 0
        assert result.metrics["n_right"] > 0

    def test_auto_bandwidth_when_none(self, flat_data):
        x, y = flat_data
        result = rdd_placebo_test(y, x, cutoff=0.0, bandwidth=None)
        assert result.metrics["bandwidth_used"] > 0.0

    def test_explicit_bandwidth_respected(self, flat_data):
        x, y = flat_data
        bw = 0.7
        result = rdd_placebo_test(y, x, cutoff=0.0, bandwidth=bw)
        assert abs(result.metrics["bandwidth_used"] - bw) < 1e-9

    def test_narrow_bandwidth_fewer_observations(self, flat_data):
        x, y = flat_data
        wide   = rdd_placebo_test(y, x, cutoff=0.0, bandwidth=1.5)
        narrow = rdd_placebo_test(y, x, cutoff=0.0, bandwidth=0.3)
        n_wide   = wide.metrics["n_left"]   + wide.metrics["n_right"]
        n_narrow = narrow.metrics["n_left"] + narrow.metrics["n_right"]
        assert n_narrow <= n_wide

    def test_rare_mask_adds_metrics(self, flat_data, rare_mask_far):
        x, y = flat_data
        result = rdd_placebo_test(y, x, cutoff=0.0, bandwidth=1.0,
                                  rare_mask=rare_mask_far)
        assert "rare_n_near_cutoff"   in result.metrics
        assert "rare_pct_near_cutoff" in result.metrics

    def test_no_rare_mask_omits_rare_metrics(self, flat_data):
        x, y = flat_data
        result = rdd_placebo_test(y, x, cutoff=0.0, bandwidth=1.0)
        assert "rare_n_near_cutoff" not in result.metrics

    def test_rare_pct_in_range(self, flat_data, rare_mask_near):
        x, y = flat_data
        result = rdd_placebo_test(y, x, cutoff=0.0, bandwidth=1.0,
                                  rare_mask=rare_mask_near)
        pct = result.metrics["rare_pct_near_cutoff"]
        assert 0.0 <= pct <= 100.0

    def test_rare_far_yields_low_n_near(self, flat_data, rare_mask_far):
        """Rare patients far from cutoff → rare_n_near_cutoff should be small."""
        x, y = flat_data
        result = rdd_placebo_test(y, x, cutoff=0.0, bandwidth=0.5,
                                  rare_mask=rare_mask_far)
        assert result.metrics["rare_n_near_cutoff"] < 10

    def test_rare_near_yields_higher_n_near(self, flat_data, rare_mask_near):
        """Rare patients near cutoff → rare_n_near_cutoff should be non-trivial."""
        x, y = flat_data
        result = rdd_placebo_test(y, x, cutoff=0.0, bandwidth=1.0,
                                  rare_mask=rare_mask_near)
        assert result.metrics["rare_n_near_cutoff"] > 0

    def test_uniform_kernel_also_passes_null(self, flat_data):
        x, y = flat_data
        result = rdd_placebo_test(y, x, cutoff=0.0, bandwidth=1.0, kernel="uniform")
        assert result.passed

    def test_notes_nonempty(self, flat_data):
        x, y = flat_data
        result = rdd_placebo_test(y, x, cutoff=0.0, bandwidth=1.0)
        assert len(result.notes) > 0

    def test_notes_mention_passed_on_null(self, flat_data):
        x, y = flat_data
        result = rdd_placebo_test(y, x, cutoff=0.0, bandwidth=1.0)
        assert "PASSED" in result.notes

    def test_notes_mention_failed_on_step(self, step_data):
        x, y = step_data
        result = rdd_placebo_test(y, x, cutoff=0.0, bandwidth=1.0)
        assert "FAILED" in result.notes

    def test_rare_note_warns_when_too_few(self, flat_data, rare_mask_far):
        """Notes should warn about low rare density when < 10 near cutoff."""
        x, y = flat_data
        result = rdd_placebo_test(y, x, cutoff=0.0, bandwidth=0.5,
                                  rare_mask=rare_mask_far)
        if result.metrics["rare_n_near_cutoff"] < 10:
            assert "NOTE" in result.notes

    def test_custom_alpha(self, flat_data):
        """With alpha=1.0 any p_value < 1 will fail; nearly anything should fail."""
        x, y = flat_data
        result = rdd_placebo_test(y, x, cutoff=0.0, bandwidth=1.0, alpha=1.0)
        # p_value < 1.0 always for finite data → passed = False
        assert not result.passed


# ─── TestPlotRunningVarDensity ────────────────────────────────────────────────

class TestPlotRunningVarDensity:

    def test_returns_figure(self, flat_data):
        import matplotlib.pyplot as plt
        x, _ = flat_data
        fig = plot_running_var_density(x, cutoff=0.0, bandwidth=1.0)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_rare_mask(self, flat_data, rare_mask_near):
        import matplotlib.pyplot as plt
        x, _ = flat_data
        fig = plot_running_var_density(x, cutoff=0.0, bandwidth=1.0,
                                       rare_mask=rare_mask_near)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_without_rare_mask(self, flat_data):
        import matplotlib.pyplot as plt
        x, _ = flat_data
        fig = plot_running_var_density(x, cutoff=0.0, bandwidth=1.0)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_title_contains_rare_count(self, flat_data, rare_mask_near):
        import matplotlib.pyplot as plt
        x, _ = flat_data
        fig = plot_running_var_density(x, cutoff=0.0, bandwidth=1.0,
                                       rare_mask=rare_mask_near)
        title = fig.axes[0].get_title()
        assert "Rare patients within bandwidth" in title
        plt.close(fig)

    def test_save_path_creates_file(self, flat_data, tmp_path):
        import matplotlib.pyplot as plt
        x, _ = flat_data
        out = str(tmp_path / "density.png")
        fig = plot_running_var_density(x, cutoff=0.0, bandwidth=1.0, save_path=out)
        assert os.path.isfile(out)
        plt.close(fig)


# ─── TestRDDScatterPlot ───────────────────────────────────────────────────────

class TestRDDScatterPlot:

    def test_returns_figure(self, flat_data):
        import matplotlib.pyplot as plt
        x, y = flat_data
        fig = plot_rdd_scatter(y, x, cutoff=0.0, bandwidth=1.0)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_rare_mask(self, flat_data, rare_mask_near):
        import matplotlib.pyplot as plt
        x, y = flat_data
        fig = plot_rdd_scatter(y, x, cutoff=0.0, bandwidth=1.0,
                               rare_mask=rare_mask_near)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_without_rare_mask(self, flat_data):
        import matplotlib.pyplot as plt
        x, y = flat_data
        fig = plot_rdd_scatter(y, x, cutoff=0.0, bandwidth=1.0)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_uniform_kernel(self, flat_data):
        import matplotlib.pyplot as plt
        x, y = flat_data
        fig = plot_rdd_scatter(y, x, cutoff=0.0, bandwidth=1.0, kernel="uniform")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_save_path_creates_file(self, flat_data, tmp_path):
        import matplotlib.pyplot as plt
        x, y = flat_data
        out = str(tmp_path / "scatter.png")
        fig = plot_rdd_scatter(y, x, cutoff=0.0, bandwidth=1.0, save_path=out)
        assert os.path.isfile(out)
        plt.close(fig)

    def test_step_shows_in_scatter_without_error(self, step_data):
        import matplotlib.pyplot as plt
        x, y = step_data
        fig = plot_rdd_scatter(y, x, cutoff=0.0, bandwidth=1.0)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
