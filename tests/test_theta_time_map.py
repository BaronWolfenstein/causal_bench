"""theta <-> VP-SDE time mapping — #137 non-gated half. Token channel is exact
(closed form); embedding channel runs on synthetic labeled hierarchical Gaussians
(known classes), not real MEDS embeddings (that calibration stays gated)."""
import numpy as np

from causal_bench.generative.vpsde import Schedule
from causal_bench.diagnostics.hierarchy_probe import sample_hierarchical_gaussian
from causal_bench.diagnostics.theta_time_map import (
    theta_to_vpsde_time, flip_rate, embedding_channel_overlap,
    embedding_transition_scan, transition_report,
)


def test_theta_to_vpsde_time_round_trips_through_alpha_bar():
    sch = Schedule(n_steps=200)
    for t in (10, 50, 100, 150, 190):
        theta = sch.alphas_bar[t]
        assert theta_to_vpsde_time(theta, sch) == t


def test_theta_to_vpsde_time_monotone_in_theta():
    sch = Schedule(n_steps=200)
    t_hi = theta_to_vpsde_time(0.9, sch)      # high retention -> low t (little noise)
    t_lo = theta_to_vpsde_time(0.1, sch)      # low retention -> high t (lots of noise)
    assert t_hi < t_lo


def test_flip_rate_zero_at_theta_one_and_positive_below():
    assert flip_rate(1.0, 8) == 0.0
    assert flip_rate(0.5, 8) > 0.0
    assert abs(flip_rate(0.0, 8) - 7 / 8) < 1e-9


def test_embedding_channel_overlap_perfect_and_chance():
    means = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
    labels = np.array([0, 1, 2])
    Xt_perfect = means.copy()
    assert embedding_channel_overlap(Xt_perfect, means, labels) > 0.99
    rng = np.random.default_rng(0)
    Xt_random = rng.normal(scale=50, size=(300, 2))
    labels_random = rng.integers(0, 3, 300)
    ov = embedding_channel_overlap(Xt_random, means, labels_random)
    assert -0.3 < ov < 0.3                                    # ~0 at chance


def test_embedding_transition_scan_is_monotone_and_locates_transition():
    d = sample_hierarchical_gaussian(n_coarse=4, n_fine=1, per_leaf=80, seed=0)
    r = embedding_transition_scan(d["X"], d["coarse"], d["coarse_means"],
                                  sch=Schedule(n_steps=200), n_grid=25,
                                  rng=np.random.default_rng(1))
    assert r["overlap"][0] > 0.9                              # near-perfect at t~0
    assert r["overlap"][-1] < r["overlap"][0] - 0.4           # substantially eroded
    assert np.all(np.diff(r["overlap"]) < 0.05)               # non-increasing (small tol)
    assert 0.3 < r["t_star"] < 1.0


def test_transition_report_gap_is_nonnegative_and_bounded():
    d = sample_hierarchical_gaussian(n_coarse=4, n_fine=1, per_leaf=80, seed=0)
    sch = Schedule(n_steps=200)
    r = embedding_transition_scan(d["X"], d["coarse"], d["coarse_means"], sch=sch,
                                  n_grid=25, rng=np.random.default_rng(1))
    rep = transition_report(0.45, 8, sch, r)                  # theta_c from #136
    assert 0.0 <= rep["token_t_frac"] <= 1.0
    assert 0.0 <= rep["embedding_t_star"] <= 1.0
    assert rep["gap"] == abs(rep["token_t_frac"] - rep["embedding_t_star"])
