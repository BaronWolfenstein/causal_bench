"""t_rare* (#122 probe) wired into the localization diagnostic (items 4/5)."""
import numpy as np

from causal_bench.diagnostics.localization import (
    rare_recoverability_threshold, run_diagnostic,
)
from causal_bench.diagnostics.localization import test_a as _test_a   # alias: not a pytest test


def _rare_common(sep, seed=0, n_rare=60, n_common=200, dim=8):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_rare, dim)) + sep, rng.standard_normal((n_common, dim))


def test_faint_cohort_erodes_strong_survives():
    rare_faint, common = _rare_common(sep=0.5, seed=1)
    rare_strong, common2 = _rare_common(sep=4.0, seed=1)
    t_faint = rare_recoverability_threshold(rare_faint, common,
                                            rng=np.random.default_rng(2))["t_rare_star"]
    t_strong = rare_recoverability_threshold(rare_strong, common2,
                                             rng=np.random.default_rng(2))["t_rare_star"]
    assert t_strong >= 0.99                       # survives the whole schedule
    assert t_faint < t_strong                     # faint rare detail dies earlier


def test_schedule_probe_adds_metric_to_test_a_and_report():
    rare, common = _rare_common(sep=3.0, seed=2)
    ra = _test_a(rare, common, schedule_probe=True)
    assert "t_rare_star" in ra.metrics and 0.0 < ra.metrics["t_rare_star"] <= 1.0
    assert "t_rare_star" not in _test_a(rare, common).metrics       # default off
    rep = run_diagnostic(rare, common, schedule_probe=True)         # flows through
    a = [t for t in rep.tests_run if t.test == "A"][0]
    assert "t_rare_star" in a.metrics
