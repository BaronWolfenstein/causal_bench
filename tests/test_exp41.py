"""Exp 41 driver — grid enumeration + report formatting (numpy). The MCMC grid is
exercised via the joint_fidelity engine's pymc test; here we only check the pure
enumeration/reporting paths."""
from experiments.exp41_borrowing_calibration import (
    report, SCENARIOS, POLICIES, KS, iter_cells, dims_for_K,
)


def test_report_renders_a_markdown_row_per_cell():
    rows = [{"level": "member", "theta0": 0.6, "K": 8, "scenario": "alt",
             "policy": "canonical", "reject_rate": 0.0, "coverage": 1.0,
             "mean_ci_width": 0.482, "mean_tau_sd": 0.914, "mean_decode_acc": 0.771,
             "tau_true": 0.3, "n_used": 5}]
    out = report(rows)
    assert out.startswith("| level")
    assert "| member | 0.60 | 8 | alt | canonical |" in out
    assert "0.91" in out                                       # mean_tau_sd rendered
    assert "0.48" in out                                       # CI width (headline OC)
    assert "0.77" in out                                       # decode accuracy


def test_scenario_and_policy_grid_is_complete():
    assert set(SCENARIOS) == {"global_null", "hetero_null", "alt"}
    assert POLICIES == ["flat", "oracle", "canonical", "empirical"]


def test_K_grid_escapes_the_small_K_regime():
    # #144: coverage and size are BOTH degenerate at g=4/b_size=3, so K must be on the
    # grid for the experiment to discriminate at all. The sweep has to reach well past
    # the ~4 subgroups where the credible interval is unconditionally conservative.
    assert KS[0] <= 4 and max(KS) >= 16
    assert KS == sorted(KS)


def test_iter_cells_sweeps_K():
    cells = list(iter_cells(["group", "member"], [0.5, 0.9], [4, 16]))
    assert len(cells) == 2 * 2 * 2 * len(SCENARIOS) * len(POLICIES)
    level, theta0, K, scen, policy = cells[0]                  # cells are 5-tuples now
    assert level in ("group", "member") and K in (4, 16)
    assert scen in SCENARIOS and policy in POLICIES
    assert {c[2] for c in cells} == {4, 16}


def test_dims_for_K_maps_K_onto_the_tested_level():
    # K is the number of subgroups the meta-analysis pools over, so it must set the
    # cardinality of the level under test; the other level keeps its default.
    assert dims_for_K("group", 16, g=4, b_size=3) == (16, 3)
    assert dims_for_K("member", 16, g=4, b_size=3) == (4, 16)
    assert dims_for_K("group", 4, g=4, b_size=3) == (4, 3)     # default round-trips
