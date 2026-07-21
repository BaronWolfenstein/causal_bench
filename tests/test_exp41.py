"""Exp 41 driver — report formatting (numpy). The MCMC grid is exercised via the
joint_fidelity engine's pymc test; here we only check the pure reporting path."""
from experiments.exp41_borrowing_calibration import report, SCENARIOS, POLICIES


def test_report_renders_a_markdown_row_per_cell():
    rows = [{"level": "member", "theta0": 0.6, "scenario": "alt", "policy": "canonical",
             "reject_rate": 0.0, "coverage": 1.0, "mean_tau_sd": 0.914, "tau_true": 0.3,
             "n_used": 5}]
    out = report(rows)
    assert out.startswith("| level")
    assert "| member | 0.60 | alt | canonical |" in out
    assert "0.91" in out                                       # mean_tau_sd rendered


def test_scenario_and_policy_grid_is_complete():
    assert set(SCENARIOS) == {"global_null", "hetero_null", "alt"}
    assert POLICIES == ["flat", "oracle", "canonical", "empirical"]
