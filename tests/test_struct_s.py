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


# ---- S1: event-aligned displacement bimodality (synthetic-validated; real MEDS gated) ----
from causal_bench.diagnostics.struct_s import (
    standardized_bimodality, event_aligned_bimodality,
)


def _two_mode_displacements(n, sep, rng, align=True, mislabel=0.0):
    """Half the samples at ~0, half at ~sep. If align, ice_flags mark the high mode
    (with `mislabel` fraction flipped); if not, ice is random wrt mode."""
    hi = rng.random(n) < 0.5
    x = np.where(hi, rng.normal(sep, 1.0, n), rng.normal(0.0, 1.0, n))
    if align:
        ice = hi.copy()
        flip = rng.random(n) < mislabel
        ice = np.where(flip, ~ice, ice)
    else:
        ice = rng.random(n) < 0.5
    return x, ice.astype(int)


def test_standardized_bimodality_separates_uni_from_bimodal():
    rng = np.random.default_rng(0)
    uni = rng.normal(0.0, 1.0, 400)
    bi, _ = _two_mode_displacements(400, sep=6.0, rng=rng)
    z_uni = standardized_bimodality(uni, n_null=30, seed=1)["z_bimodal"]
    z_bi = standardized_bimodality(bi, n_null=30, seed=1)["z_bimodal"]
    assert z_bi > 3.0                                  # clearly bimodal
    assert z_uni < 2.0                                 # unimodal stays low
    assert z_bi > z_uni + 3.0


def test_standardized_bimodality_is_size_invariant():
    # the Z-Dip property: the standardized score for the same effect is comparable
    # across sample sizes (unlike the raw dip/BIC gap, which drifts with n).
    z = {}
    for n in (200, 800):
        rng = np.random.default_rng(2)
        x, _ = _two_mode_displacements(n, sep=5.0, rng=rng)
        z[n] = standardized_bimodality(x, n_null=30, seed=3)["z_bimodal"]
    assert z[200] > 3.0 and z[800] > 3.0               # both fire
    assert abs(z[200] - z[800]) < 0.75 * max(z.values())   # comparable, not runaway


def test_s1_confirms_when_bimodal_and_event_aligned():
    rng = np.random.default_rng(4)
    x, ice = _two_mode_displacements(400, sep=6.0, rng=rng, align=True, mislabel=0.05)
    r = event_aligned_bimodality(x, ice, n_null=30, seed=5)
    assert r["bimodal"] and r["ice_aligned"] and r["s1_confirms"]
    assert r["ice_alignment"] > 0.8


def test_s1_rejects_bimodal_but_event_unaligned():
    # a real regime split, but NOT driven by the intercurrent event → not licensed.
    rng = np.random.default_rng(6)
    x, ice = _two_mode_displacements(400, sep=6.0, rng=rng, align=False)
    r = event_aligned_bimodality(x, ice, n_null=30, seed=7)
    assert r["bimodal"] and not r["ice_aligned"] and not r["s1_confirms"]


def test_s1_rejects_unimodal_displacements():
    rng = np.random.default_rng(8)
    x = rng.normal(0.0, 1.0, 400)
    ice = (rng.random(400) < 0.5).astype(int)
    r = event_aligned_bimodality(x, ice, n_null=30, seed=9)
    assert not r["bimodal"] and not r["s1_confirms"]


def test_s1_accepts_vector_displacements():
    # displacement vectors are reduced to magnitude before the 1-D bimodality test.
    rng = np.random.default_rng(10)
    hi = rng.random(300) < 0.5
    base = rng.normal(0, 1, (300, 4))
    shift = np.where(hi[:, None], 6.0, 0.0)
    X = base + shift
    ice = hi.astype(int)
    r = event_aligned_bimodality(X, ice, n_null=30, seed=11)
    assert r["s1_confirms"]


def test_screen_runs_s1_when_displacements_provided():
    rng = np.random.default_rng(12)
    Z = np.vstack([rng.normal(0, 0.3, (120, 3)), rng.normal(8, 0.3, (120, 3))])
    x, ice = _two_mode_displacements(240, sep=6.0, rng=rng, align=True, mislabel=0.05)
    out = struct_s_screen(Z, displacements=x, ice_flags=ice, n_null=30)
    assert out["needs_S1_to_confirm"] is False         # S1 was actually run
    assert out["S1_confirms"] is True
