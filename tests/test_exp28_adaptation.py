"""Tests for exp28 — the Q2 three-arm adaptation contrast (#46).

Headline is the marginal-capture ratio (naive − NC-flag)/(naive − oracle), and
its predicted degradation as the negative control weakens (spec §4). At
coupling 1.0 the detector is near-perfect, so nc_flag sits statistically AT the
oracle ceiling (capture ≈ 1); strict oracle < nc_flag < naive betweenness is
only a meaningful prediction where the detector is imperfect (low coupling)."""
import numpy as np

from experiments.exp28_q2_adaptation import run_three_arm, run_capture_vs_observability


def test_three_arm_near_ceiling_at_high_observability():
    tbl = run_three_arm(shock_delta=2.0, nc_coupling=1.0, n_trajectories=400,
                        n_turns=12, seed=11)
    assert set(tbl["arm"]) == {"naive", "nc_flag", "oracle"}
    err = tbl.set_index("arm")["post_shock_err"]
    assert err["oracle"] < err["naive"]          # an achievable gap exists
    assert err["nc_flag"] < err["naive"]         # the flag captures some of it
    cap = tbl["capture"].iloc[0]
    assert (tbl["capture"] == cap).all()
    assert cap > 0.7                # near-perfect detector ≈ at the ceiling
    rec = tbl.set_index("arm")["time_to_recover"]
    assert rec["oracle"] <= rec["naive"]


def test_three_arm_strict_ordering_at_imperfect_detection():
    tbl = run_three_arm(shock_delta=2.0, nc_coupling=0.3, n_trajectories=400,
                        n_turns=12, seed=13)
    err = tbl.set_index("arm")["post_shock_err"]
    # imperfect detector: strictly between the ceiling and naive (spec §4)
    assert err["oracle"] < err["nc_flag"] < err["naive"]
    cap = tbl["capture"].iloc[0]
    assert 0.1 < cap < 0.9


def test_capture_degrades_as_control_weakens():
    tbl = run_capture_vs_observability(couplings=[1.0, 0.05], shock_delta=2.0,
                                       n_trajectories=400, n_turns=12, seed=11)
    caps = tbl.groupby("nc_coupling")["capture"].first()
    # as the control degrades the NC-flag arm slides from oracle toward naive
    assert caps[0.05] < caps[1.0]
