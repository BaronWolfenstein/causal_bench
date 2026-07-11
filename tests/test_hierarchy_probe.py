"""Tests for the diffusion phase-transition hierarchy probe (#122).

Wyart / Sclocchi-Favero-Wyart: on hierarchically-structured data, forward
diffusion destroys FINE (sub-class) identity at a lower noise level than the
COARSE (class) identity — two separated transitions. Flat (single-scale) data
has one transition. This probe measures exactly that separation via the analytic
Bayes posterior over class at each noise level (the inference / BP view of
denoising). numpy + our VP-SDE, CPU.
"""
import numpy as np

from causal_bench.generative.vpsde import Schedule
from causal_bench.diagnostics.hierarchy_probe import (
    sample_hierarchical_gaussian, phase_transition_scan, estimate_transition_t,
)


def test_hierarchical_coarse_survives_longer_than_fine():
    d = sample_hierarchical_gaussian(n_coarse=4, n_fine=3, per_leaf=60, dim=8,
                                     coarse_sep=2.5, fine_sep=0.7, seed=0)
    sch = Schedule(n_steps=200)
    res = phase_transition_scan(d["X"], d["coarse"], d["fine"], d["coarse_means"],
                                d["fine_means"], sch=sch, n_grid=25,
                                rng=np.random.default_rng(0))
    # the signature: fine identity lost at LOWER noise than the coarse class
    assert res["t_fine_star"] < res["t_coarse_star"] - 0.05
    # both accurate at t≈0; fine erodes to ~chance; coarse erodes too, but later
    assert res["acc_coarse"][0] > 0.9 and res["acc_fine"][0] > 0.9
    assert res["acc_fine"][-1] < 0.5
    assert res["acc_coarse"][-1] < res["acc_coarse"][0] - 0.15


def test_flat_data_single_transition():
    # n_fine=1 → fine structure == coarse structure → transitions coincide.
    d = sample_hierarchical_gaussian(n_coarse=4, n_fine=1, per_leaf=60, dim=8,
                                     coarse_sep=2.5, fine_sep=2.5, seed=1)
    sch = Schedule(n_steps=200)
    res = phase_transition_scan(d["X"], d["coarse"], d["fine"], d["coarse_means"],
                                d["fine_means"], sch=sch, n_grid=25,
                                rng=np.random.default_rng(1))
    assert abs(res["t_fine_star"] - res["t_coarse_star"]) < 0.05   # one scale


def test_estimate_transition_is_monotone_crossing():
    t = np.linspace(0, 1, 21)
    acc = np.linspace(1.0, 0.25, 21)          # decreasing accuracy
    ts = estimate_transition_t(t, acc, chance=0.25)
    assert 0.4 < ts < 0.6                      # crosses halfway (~0.625) near the middle
    # a curve that never erodes has no transition → returns 1.0 (survives to the end)
    assert estimate_transition_t(t, np.ones_like(t), chance=0.25) == 1.0


def test_deeper_hierarchy_widens_the_gap():
    # larger coarse/fine separation ratio → wider transition gap.
    sch = Schedule(n_steps=200)
    def gap(fine_sep):
        d = sample_hierarchical_gaussian(n_coarse=4, n_fine=3, per_leaf=60, dim=8,
                                         coarse_sep=2.5, fine_sep=fine_sep, seed=2)
        r = phase_transition_scan(d["X"], d["coarse"], d["fine"], d["coarse_means"],
                                  d["fine_means"], sch=sch, n_grid=25,
                                  rng=np.random.default_rng(2))
        return r["t_coarse_star"] - r["t_fine_star"]
    assert gap(0.5) > gap(1.2)                 # smaller fine_sep → wider scale gap


# ---- (2) multi-level depth counting ----
from causal_bench.diagnostics.hierarchy_probe import (
    sample_multi_level_gaussian, depth_scan, multi_level_transition_scan,
    bp_predict_transition,
)


def test_labeled_multi_level_transitions_are_monotone():
    # rigorous depth tool: given the tree levels, t* is monotone — coarser levels
    # survive to higher noise.
    sch = Schedule(n_steps=200)
    d = sample_multi_level_gaussian(depth=3, branching=2, per_leaf=40, dim=8, seed=1)
    r = multi_level_transition_scan(d["X"], d["level_labels"], d["level_means"],
                                    sch=sch, rng=np.random.default_rng(1))
    non_root = r["t_star_per_level"][1:]              # level 0 is the trivial root
    assert np.all(np.diff(non_root) <= 1e-9)          # non-increasing coarse→fine


def test_depth_scan_curve_is_monotone_non_increasing():
    # t*(k) always declines with k (finer clusters ⇒ closer centroids); the
    # unlabeled level COUNT is a heuristic (false-positive-prone on flat data —
    # the labeled scan above is the rigorous tool), but the curve is robust.
    d = sample_multi_level_gaussian(depth=2, branching=3, per_leaf=40, dim=8, seed=4)
    r = depth_scan(d["X"], rng=np.random.default_rng(4))
    t = r["t_star_of_k"]
    assert len(t) == len(r["k_grid"]) and np.all(np.diff(t) <= 1e-9)
    assert r["estimated_levels"] >= 1


def test_strong_hierarchy_is_detected():
    # a strongly-separated tree is reliably flagged as multi-level.
    d = sample_multi_level_gaussian(depth=2, branching=4, per_leaf=50, dim=8,
                                    base_sep=8.0, decay=0.2, seed=0)
    r = depth_scan(d["X"], rng=np.random.default_rng(0))
    assert r["estimated_levels"] >= 2


# ---- (7) BP / reconstruction-threshold prediction of t* ----
def test_bp_prediction_matches_empirical_transition():
    sch = Schedule(n_steps=200)
    d = sample_hierarchical_gaussian(n_coarse=4, n_fine=3, per_leaf=80, dim=8,
                                     coarse_sep=2.5, fine_sep=0.7, sigma_within=0.3,
                                     seed=5)
    emp = phase_transition_scan(d["X"], d["coarse"], d["fine"], d["coarse_means"],
                                d["fine_means"], sch=sch, n_grid=30,
                                rng=np.random.default_rng(5))
    # confusers at a level are the SIBLINGS (branching), not all leaves:
    pred_fine = bp_predict_transition(0.7, n_classes=3, sigma_within=0.3, sch=sch)
    pred_coarse = bp_predict_transition(2.5, n_classes=4, sigma_within=0.3, sch=sch)
    assert abs(pred_fine["t_star"] - emp["t_fine_star"]) < 0.25      # analytic ≈ empirical
    assert pred_fine["t_star"] < pred_coarse["t_star"]               # ordering preserved


# ---- (3) S5 wired into STRUCT-S ----
def test_struct_s_s5_flags_hierarchy():
    from causal_bench.diagnostics.struct_s import struct_s_screen, hierarchical_levels
    d = sample_multi_level_gaussian(depth=2, branching=4, per_leaf=50, dim=8,
                                    base_sep=8.0, decay=0.2, seed=0)
    s5 = hierarchical_levels(d["X"])
    assert s5["estimated_levels"] >= 2 and s5["hierarchical"] is True
    out = struct_s_screen(d["X"], run_s5=True)
    assert "S5_levels" in out and "S5_hierarchical" in out and out["S5_hierarchical"] is True
