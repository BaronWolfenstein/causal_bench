"""Per-level identifiability report for manual hierarchical borrowing (#137 follow-up).
SFW as a diagnostic: coarse levels survive VP-SDE noise more than fine ones, so t_star
orders coarse→fine and flags which levels are trustworthy to pool across."""
import numpy as np

from causal_bench.diagnostics.hierarchy_probe import sample_multi_level_gaussian
from causal_bench.diagnostics.borrowing_informativeness import (
    level_identifiability, borrowing_report,
)


def _levels(seed=0, rng_seed=1):
    d = sample_multi_level_gaussian(depth=3, branching=2, per_leaf=60, dim=8, seed=seed)
    return level_identifiability(d["X"], d["level_labels"], n_reps=6,
                                 rng=np.random.default_rng(rng_seed))


def test_skips_trivial_level_and_reports_class_counts():
    levels = _levels()
    # depth-3 branching-2 tree: root (1 class, skipped) then 2, 4, 8 classes
    assert [L["n_classes"] for L in levels] == [2, 4, 8]
    assert all(0.0 <= L["t_star"] <= 1.0 for L in levels)


def test_coarse_levels_survive_more_noise_than_fine():
    # THE SFW ordering: t_star decreases coarse→fine (coarse identity is more robust).
    ts = [L["t_star"] for L in _levels()]
    assert ts[0] > ts[-1] + 0.15                                # coarse clearly > finest
    assert all(ts[i] >= ts[i + 1] - 1e-9 for i in range(len(ts) - 1))  # non-increasing


# ─── borrowing_report logic (deterministic — hand-built levels, no scan noise) ──
def _mk(name, t_star, n=2):
    return {"name": name, "n_classes": n, "t_star": t_star}


def test_report_recommends_less_borrowing_for_robust_levels():
    rep = borrowing_report([_mk("coarse", 0.90), _mk("mid", 0.55), _mk("fine", 0.20)])
    recs = [L["recommendation"] for L in rep["levels"]]
    assert "borrow little" in recs[0]                           # high t_star → robust
    assert "partial pooling" in recs[1]                         # moderate SNR
    assert "borrow heavily" in recs[2]                          # low t_star → starved


def test_report_separation_gaps_and_well_separated():
    rep = borrowing_report([_mk("A", 0.90), _mk("B", 0.55), _mk("C", 0.20)])
    assert np.allclose(rep["separation_gaps"], [0.35, 0.35])    # coarse − fine, both real
    assert rep["well_separated"] is True
    assert rep["unresolved_splits"] == []


def test_report_flags_unresolved_split_when_levels_have_equal_t_star():
    rep = borrowing_report([_mk("A", 0.80), _mk("B", 0.79)], sep_tol=0.05)
    assert abs(rep["separation_gaps"][0]) < 0.05
    assert rep["well_separated"] is False
    assert ("A", "B") in rep["unresolved_splits"]


# ─── wiring to the hierarchical fit's borrowing knob (tau_sd) ─────────────────
def test_suggest_tau_prior_is_monotone_and_bounded():
    from causal_bench.diagnostics.borrowing_informativeness import suggest_tau_prior
    assert suggest_tau_prior(0.0, tau_sd_min=0.05, tau_sd_max=1.0) == 0.05   # min at t*=0
    assert suggest_tau_prior(1.0, tau_sd_min=0.05, tau_sd_max=1.0) == 1.0    # max at t*=1
    assert suggest_tau_prior(-5) == suggest_tau_prior(0.0)                   # clipped
    assert suggest_tau_prior(0.8) > suggest_tau_prior(0.4)                   # monotone ↑


def test_recommend_tau_priors_maps_robust_levels_to_weaker_pooling():
    from causal_bench.diagnostics.borrowing_informativeness import recommend_tau_priors
    # robust coarse (high t*) → larger tau_sd (weak pooling); starved fine → smaller.
    levels = [_mk("coarse", 0.90), _mk("mid", 0.55), _mk("fine", 0.20)]
    rec = recommend_tau_priors(levels)
    taus = [rec["per_level"][n]["tau_sd"] for n in ("coarse", "mid", "fine")]
    assert taus[0] > taus[1] > taus[2]                                      # weaker→stronger pooling
    assert all(t > 0 for t in taus)                                         # valid HalfNormal scales
    assert rec["well_separated"] is True


def test_recommend_tau_priors_carries_unresolved_split_warning():
    from causal_bench.diagnostics.borrowing_informativeness import recommend_tau_priors
    rec = recommend_tau_priors([_mk("A", 0.80), _mk("B", 0.79)], sep_tol=0.05)
    assert ("A", "B") in rec["unresolved_splits"]                           # don't fit as distinct
    assert rec["well_separated"] is False
