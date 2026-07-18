"""Stratified ESS + Euclidean positivity R. The load-bearing test is
`test_global_ess_false_passes_tail_collapse`: it constructs a cloud the global
Kish rule calls healthy while a single particle owns the tail, and asserts the
stratified rule fires where the global one does not. That is the entire reason
the unit exists (the §7 exp29-analog seam)."""
import numpy as np

from causal_bench.sampling import (stratified_ess, stratified_resample_needed,
                                    positivity_overlap, strata_from_region, kish_ess)


def _make_false_pass(n_bulk=400, n_tail=40, seed=0):
    """Bulk: near-uniform weights (healthy). Tail: one particle owns ~all the
    tail mass (collapsed). Sized so global ESS stays > N/2 — the trap."""
    rng = np.random.default_rng(seed)
    lw_bulk = rng.normal(0.0, 0.05, n_bulk)          # near-uniform -> ESS ~ n_bulk
    lw_tail = np.full(n_tail, -8.0)
    lw_tail[0] = 0.0                                  # one dominant tail particle
    log_w = np.concatenate([lw_bulk, lw_tail])
    strata = np.concatenate([np.zeros(n_bulk, int), np.ones(n_tail, int)])
    return log_w, strata


def test_global_ess_false_passes_tail_collapse():
    log_w, strata = _make_false_pass()
    n = log_w.size

    # global ESS looks healthy: the tail's total mass is small, so the global
    # Kish number is dominated by the near-uniform bulk.
    g = kish_ess(log_w)
    assert g > 0.5 * n, f"setup invalid: global ESS {g:.1f} should exceed N/2={n/2}"
    fires_global = g < 0.5 * n
    assert not fires_global  # the global rule waves it through

    rep = stratified_ess(log_w, strata)
    tail = rep.stratum(1)
    assert tail["ess"] < 2.0            # one particle owns the tail
    assert tail["ess_ratio"] < 0.1      # << bulk

    fire, reason = stratified_resample_needed(rep, tail_label=1, tail_frac=0.5)
    assert fire, "stratified rule must catch the tail collapse global ESS misses"
    assert "tail ESS_ratio" in reason


def test_healthy_cloud_does_not_fire():
    rng = np.random.default_rng(1)
    log_w = rng.normal(0.0, 0.05, 440)
    strata = np.concatenate([np.zeros(400, int), np.ones(40, int)])
    rep = stratified_ess(log_w, strata)
    assert rep.stratum(1)["ess_ratio"] > 0.8
    fire, reason = stratified_resample_needed(rep, tail_label=1, tail_frac=0.5)
    assert not fire and reason == "healthy"


def test_bool_mask_is_two_strata():
    log_w = np.zeros(10)
    mask = np.array([False] * 7 + [True] * 3)
    rep = stratified_ess(log_w, mask)
    assert set(rep.labels.tolist()) == {0, 1}
    assert rep.stratum(0)["count"] == 7 and rep.stratum(1)["count"] == 3


def test_empty_and_singleton_strata():
    log_w = np.array([0.0, -1.0, 0.5])
    strata = np.array([2, 2, 2])          # nothing in the tail label 1
    rep = stratified_ess(log_w, strata)
    # tail label absent -> trigger only considers the global rule
    fire, reason = stratified_resample_needed(rep, tail_label=1, tail_frac=0.5,
                                              global_frac=0.0)
    assert not fire
    # a genuinely empty stratum reports 0 ESS, not a crash
    from causal_bench.sampling.stratified import _within_ess
    assert _within_ess(np.array([])) == 0.0


def test_positivity_coverage_and_gap():
    rng = np.random.default_rng(2)
    # two target clusters; particles cover only the first
    targets = np.concatenate([rng.normal([0, 0], 0.1, (20, 2)),
                              rng.normal([10, 10], 0.1, (20, 2))])
    particles = rng.normal([0, 0], 0.3, (200, 2))     # all near cluster 1
    r = positivity_overlap(particles, targets, radius=0.5)
    assert 0.4 < r.coverage < 0.6                      # ~half the targets reached
    assert r.uncovered[20:].all()                      # cluster-2 targets unreached
    assert not r.uncovered[:20].any()


def test_positivity_reaches_R_but_negligible_mass():
    """Coverage is fine yet weight in R is ~0 — tail-mass collapse, a distinct
    failure from a support gap. Ties positivity R to the stratified ESS story."""
    rng = np.random.default_rng(3)
    targets = rng.normal([5, 5], 0.1, (10, 2))
    bulk = rng.normal([0, 0], 0.3, (190, 2))
    tail = rng.normal([5, 5], 0.1, (10, 2))            # 10 particles DO reach R
    particles = np.concatenate([bulk, tail])
    log_w = np.concatenate([np.zeros(190), np.full(10, -10.0)])  # but ~no weight
    r = positivity_overlap(particles, targets, radius=0.5, log_w=log_w)
    assert r.coverage > 0.9                            # R is reached
    assert r.mass_in_region < 1e-3                     # yet carries no weight

    # and the region-defined stratification catches it as tail collapse
    strata = strata_from_region(particles, targets, radius=0.5)
    rep = stratified_ess(log_w, strata)
    fire, _ = stratified_resample_needed(rep, tail_label=1, tail_frac=0.5,
                                         tail_mass_floor=1e-2)
    assert fire


def test_within_ess_matches_global_when_single_stratum():
    rng = np.random.default_rng(4)
    log_w = rng.normal(0, 1.0, 100)
    rep = stratified_ess(log_w, np.zeros(100, int))
    assert np.isclose(rep.stratum(0)["ess"], kish_ess(log_w), rtol=1e-9)
    assert np.isclose(rep.global_ess, kish_ess(log_w), rtol=1e-9)
