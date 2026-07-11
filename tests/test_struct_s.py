"""STRUCT-S stratification detector (S2-S4) — CPU, needs only the embedding Z.
S2 spectral component count (eigengap), S3 local-intrinsic-dimension heterogeneity,
S4 density-gap. S1 (event-aligned displacement bimodality) is on-box-gated and not
here. These are corroborating / rule-out signals; S1 is the decisive test."""
import numpy as np

from causal_bench.diagnostics.struct_s import (
    spectral_component_count, local_id_heterogeneity, density_gap, struct_s_screen,
)


# ---- S2: spectral component count (eigengap heuristic) ----

def test_s2_two_separated_blobs_gives_two_strata():
    rng = np.random.default_rng(0)
    Z = np.vstack([rng.standard_normal((80, 3)),
                   rng.standard_normal((80, 3)) + 20.0])
    assert spectral_component_count(Z, n_neighbors=8)["n_strata"] == 2


def test_s2_single_blob_gives_one_stratum():
    rng = np.random.default_rng(1)
    Z = rng.standard_normal((150, 3))
    assert spectral_component_count(Z, n_neighbors=10)["n_strata"] == 1


# ---- S3: local intrinsic dimension heterogeneity ----

def test_s3_local_id_finite_and_near_true_dim():
    rng = np.random.default_rng(2)
    Z = rng.standard_normal((300, 4))           # intrinsic dim ~4
    r = local_id_heterogeneity(Z, k=15)
    assert np.isfinite(r["local_id"]).all() and (r["local_id"] > 0).all()
    assert 2.0 < np.median(r["local_id"]) < 6.0  # MLE is biased low — loose band


def test_s3_mixed_dimension_more_heterogeneous_than_homogeneous():
    rng = np.random.default_rng(3)
    line = np.zeros((150, 6)); line[:, 0] = rng.standard_normal(150) * 5.0  # ~1D
    blob = rng.standard_normal((150, 6))                                    # ~6D
    hetero = local_id_heterogeneity(np.vstack([line, blob]), k=15)["cv"]
    homo = local_id_heterogeneity(rng.standard_normal((300, 6)), k=15)["cv"]
    assert hetero > homo


# ---- S4: density gap / support connectedness ----

def test_s4_two_blobs_has_density_gap():
    rng = np.random.default_rng(4)
    Z = np.vstack([rng.standard_normal((80, 3)),
                   rng.standard_normal((80, 3)) + 15.0])
    r = density_gap(Z)
    assert r["has_gap"] and r["gap_ratio"] > 3.0


def test_s4_single_blob_no_density_gap():
    rng = np.random.default_rng(5)
    assert not density_gap(rng.standard_normal((160, 3)))["has_gap"]


# ---- combined screen ----

def test_struct_s_screen_flags_stratification_candidate():
    rng = np.random.default_rng(6)
    strat = np.vstack([rng.standard_normal((80, 3)),
                       rng.standard_normal((80, 3)) + 18.0])
    flat = rng.standard_normal((160, 3))
    assert struct_s_screen(strat)["candidate_stratified"] is True
    assert struct_s_screen(flat)["candidate_stratified"] is False
    # honest: S2-S4 corroborate; S1 is decisive and not run here
    assert struct_s_screen(flat)["needs_S1_to_confirm"] is True


def test_s4_lone_outlier_is_not_a_density_gap():
    # a single far outlier makes ONE long MST edge, but cutting it isolates just that
    # point (unbalanced) — must NOT be flagged as a density gap.
    rng = np.random.default_rng(7)
    Z = np.vstack([rng.standard_normal((150, 3)), np.array([[40.0, 40.0, 40.0]])])
    r = density_gap(Z)
    assert r["gap_ratio"] > 3.0          # the outlier does make a long edge
    assert not r["has_gap"]              # ...but the split is unbalanced -> no gap
    assert r["min_side_frac"] < 0.05
