"""Tests for exp29 — balance-table + region-R overlap diagnostics.

Pins the failure mode the experiment exists to expose: edge-fill augmentation
inflates the headline R-ESS while leaving the deep interior of R unsupported —
visible only in the region-resolved ESS map, never in the global SMD column.
"""
import numpy as np

from experiments.exp29_balance_diagnostics import (
    love_frame, plot_love_regions, run_panels)


def test_three_panel_balance_and_ess_map():
    balance, ess = run_panels(seed=20260702)
    ess = ess.set_index("panel")
    bal = balance.set_index(["panel", "covariate"])

    # canonical false pass: unaugmented panel clears every GLOBAL post-SMD
    # threshold while region R still fails on the severity covariate
    none_global = balance[balance.panel == "none"]["smd_post"].abs()
    assert (none_global < 0.1).all()
    assert abs(bal.loc[("none", "X5"), "smd_post_R"]) > 0.25

    # interior fill genuinely restores overlap: deep-R ESS multiplies and the
    # severity covariate balances within R
    assert ess.loc["interior", "ess_deepR"] > 3 * ess.loc["none", "ess_deepR"]
    assert abs(bal.loc[("interior", "X5"), "smd_post_R"]) < 0.1

    # edge fill is the trap: headline R-ESS looks as repaired as interior...
    assert ess.loc["edge", "ess_R"] > 3 * ess.loc["none", "ess_R"]
    # ...but the deep interior of R is untouched (≈ no augmentation at all)
    assert ess.loc["edge", "ess_deepR"] < 1.5 * ess.loc["none", "ess_deepR"]
    # and within-R imbalance on the severity covariate is WORSE than doing nothing
    assert abs(bal.loc[("edge", "X5"), "smd_post_R"]) > abs(
        bal.loc[("none", "X5"), "smd_post_R"])


def test_region_split_love_plot():
    balance, _ = run_panels(seed=20260702)
    lf = love_frame(balance, "edge")
    # two series (global + region R), both present for the severity covariate
    assert set(lf["series"]) == {"global", "region R"}
    x5 = lf[lf.covariate == "X5"].set_index("series")["abs_smd"]
    # the region-R series exposes what the global series hides
    assert x5["region R"] > x5["global"]
    fig = plot_love_regions(balance, "edge")
    # one y-tick per covariate
    assert len(fig.axes[0].get_yticks()) == balance[balance.panel == "edge"].shape[0]
