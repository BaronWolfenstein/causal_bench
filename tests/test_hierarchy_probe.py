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
