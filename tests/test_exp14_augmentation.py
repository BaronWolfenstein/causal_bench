import numpy as np
import pytest
from pydantic import ValidationError

from causal_bench.crossfit import make_folds
from causal_bench.dgp.augmentation import AugmentationConfig, generate_augmented_data
from causal_bench.dgp.config import DGPConfig
from causal_bench.dgp.survival import compute_true_effects
from causal_bench.estimators.tmle_ipcw import TMLEIPCWEstimator
from causal_bench.metrics import SimResult


def _cfg(**overrides):
    return DGPConfig(n=500, censoring_rate=0.3, horizon=0.7, seed=0).with_overrides(**overrides)


def _aug(n_real, n_synth_per_real, leakage_strength, fold_mode="group"):
    return AugmentationConfig(
        n_real=n_real, n_synth_per_real=n_synth_per_real,
        leakage_strength=leakage_strength, fold_mode=fold_mode,
    )


# ── generate_augmented_data correctness ──

def test_provenance_group_shape_and_density():
    n_real, n_synth_per_real = 30, 4
    df = generate_augmented_data(_cfg(), _aug(n_real, n_synth_per_real, 0.5),
                                  rng=np.random.default_rng(0))
    assert len(df) == n_real * (1 + n_synth_per_real)
    assert set(df["provenance_group"].unique()) == set(range(n_real))


def test_synthetic_rows_carry_parent_group_and_flag():
    n_real, n_synth_per_real = 10, 3
    df = generate_augmented_data(_cfg(), _aug(n_real, n_synth_per_real, 0.5),
                                  rng=np.random.default_rng(1))
    real = df[df["is_synthetic"] == 0]
    synth = df[df["is_synthetic"] == 1]
    assert len(real) == n_real
    assert len(synth) == n_real * n_synth_per_real
    # every real unit has a unique provenance_group...
    assert sorted(real["provenance_group"]) == list(range(n_real))
    # ...and every synthetic group id refers to an existing real parent
    assert set(synth["provenance_group"]).issubset(set(real["provenance_group"]))
    # each parent has exactly n_synth_per_real children
    counts = synth["provenance_group"].value_counts()
    assert (counts == n_synth_per_real).all()


def test_zero_synth_per_real_returns_real_only():
    df = generate_augmented_data(_cfg(), _aug(15, 0, 0.5),
                                  rng=np.random.default_rng(2))
    assert len(df) == 15
    assert (df["is_synthetic"] == 0).all()


def test_leakage_one_synthetic_covariates_equal_parent():
    n_real, n_synth_per_real = 10, 2
    df = generate_augmented_data(_cfg(), _aug(n_real, n_synth_per_real, 1.0),
                                  rng=np.random.default_rng(3))
    real = df[df["is_synthetic"] == 0].set_index("provenance_group")
    synth = df[df["is_synthetic"] == 1]
    for col in ("W1", "W2", "W3", "W4"):
        parent_vals = real.loc[synth["provenance_group"], col].values
        assert np.allclose(synth[col].values, parent_vals)


def test_leakage_zero_synthetic_independent_of_parent():
    """At leakage_strength=0, synthetic continuous covariates should not be
    pinned to the parent's value (no Var(X_synth - X_parent) -> 0 collapse)."""
    n_real, n_synth_per_real = 200, 1
    df = generate_augmented_data(_cfg(), _aug(n_real, n_synth_per_real, 0.0),
                                  rng=np.random.default_rng(4))
    real = df[df["is_synthetic"] == 0].set_index("provenance_group")
    synth = df[df["is_synthetic"] == 1]
    parent_vals = real.loc[synth["provenance_group"], "W1"].values
    corr = np.corrcoef(synth["W1"].values, parent_vals)[0, 1]
    assert abs(corr) < 0.25  # should be ~0, not 1


def test_invalid_leakage_strength_raises():
    with pytest.raises(ValidationError):
        AugmentationConfig(n_real=10, n_synth_per_real=1, leakage_strength=1.5)


def test_augmentation_does_not_change_true_effect():
    """Augmentation is a sampling/cross-fitting concern, not a DGP change —
    compute_true_effects on the same (unmodified-n) cfg must be unaffected."""
    cfg = _cfg()
    ate_before = compute_true_effects(cfg, n_ref=5_000)["ATE"]
    generate_augmented_data(cfg, _aug(20, 2, 0.8), rng=np.random.default_rng(5))
    ate_after = compute_true_effects(cfg, n_ref=5_000)["ATE"]
    assert ate_before == ate_after


# ── fold-mode group integrity ──

def test_group_mode_never_splits_a_provenance_group():
    n_real, n_synth_per_real = 20, 3
    df = generate_augmented_data(_cfg(), _aug(n_real, n_synth_per_real, 1.0),
                                  rng=np.random.default_rng(6))
    groups = df["provenance_group"].values
    X = df[["W1", "W2", "W3", "W4"]].values
    folds = make_folds(X, n_folds=5, mode="group", groups=groups, random_state=0)
    for _, val_idx in folds:
        val_groups = groups[val_idx]
        for g in np.unique(val_groups):
            # every row sharing group g must be entirely inside this val fold
            all_rows_with_g = np.where(groups == g)[0]
            assert set(all_rows_with_g).issubset(set(val_idx))


def test_iid_mode_generally_splits_provenance_groups():
    """Sanity check that the two modes actually differ: with many small
    groups and few folds, iid mode should split at least one group across
    folds (otherwise this test wouldn't distinguish the two modes at all)."""
    n_real, n_synth_per_real = 30, 3
    df = generate_augmented_data(_cfg(), _aug(n_real, n_synth_per_real, 1.0, fold_mode="iid"),
                                  rng=np.random.default_rng(7))
    groups = df["provenance_group"].values
    X = df[["W1", "W2", "W3", "W4"]].values
    folds = make_folds(X, n_folds=5, mode="iid", groups=groups, random_state=0)
    split_any = False
    for _, val_idx in folds:
        val_groups = set(groups[val_idx])
        for g in val_groups:
            all_rows_with_g = set(np.where(groups == g)[0])
            if not all_rows_with_g.issubset(set(val_idx)):
                split_any = True
    assert split_any


def test_group_mode_requires_groups_argument():
    X = np.zeros((10, 2))
    with pytest.raises(ValueError):
        make_folds(X, n_folds=3, mode="group", groups=None)


# ── fast smoke test: iid SE ratio understated relative to group at high leakage ──

def test_iid_se_ratio_below_group_se_ratio_at_full_leakage():
    """At leakage_strength=1.0, fold_mode='iid' should understate the EIC SE
    relative to the empirical SE more than fold_mode='group' does (i.e. the
    iid se_ratio is the smaller of the two) — the cross-fitting independence
    violation this whole module exists to detect.

    n_folds=2 (vs. TMLEIPCWEstimator's production default of 5) is a
    deliberate choice, not an oversight: with only 2 folds, any provenance
    group that gets split lands roughly half-in/half-out of the training
    fold, maximizing the chance iid mode actually exploits the leak. At the
    production default of n_folds=5 this same effect was confirmed to exist
    (see test_group_mode_never_splits_a_provenance_group) but was too weak
    and noisy to assert reliably within a fast smoke test's sim budget —
    that weaker-at-default-settings result is itself reported honestly in
    experiments/exp14_synthetic_augmentation.py's "Key finding" block rather
    than papered over here. n_real=10 / n_synth_per_real=15 / n_sims=60 was
    empirically verified (5 independent seeds) to confirm this inequality
    reliably in ~20-25s.
    """
    n_real, n_synth_per_real, n_folds, n_sims = 10, 15, 2, 60
    cfg = _cfg(n=500)
    true_ate = compute_true_effects(cfg, n_ref=5_000)["ATE"]

    results = {"iid": {"pts": [], "ses": []}, "group": {"pts": [], "ses": []}}
    rng = np.random.default_rng(123)
    for _ in range(n_sims):
        for mode in ("iid", "group"):
            aug = _aug(n_real, n_synth_per_real, 1.0, fold_mode=mode)
            df = generate_augmented_data(cfg, aug, rng=rng)
            est = TMLEIPCWEstimator(fold_mode=mode, n_folds=n_folds)
            res = est.estimate(df, horizon=cfg.horizon, estimand="ATE")
            match = next((r for r in res if r.estimand == "ATE"), None)
            if match is not None and np.isfinite(match.point_estimate) and np.isfinite(match.standard_error):
                results[mode]["pts"].append(match.point_estimate)
                results[mode]["ses"].append(match.standard_error)

    se_ratios = {}
    for mode in ("iid", "group"):
        pts = np.array(results[mode]["pts"])
        ses = np.array(results[mode]["ses"])
        assert len(pts) >= n_sims // 2, f"too many non-converged replicates for mode={mode}"
        sr = SimResult(
            estimator_name=f"tmle_ipcw[{mode}]", estimand="ATE", true_value=true_ate,
            n_sim=len(pts), estimates=pts, se_estimates=ses,
            ci_lowers=pts - 1.96 * ses, ci_uppers=pts + 1.96 * ses,
            nc_estimates=np.zeros(len(pts)),
        )
        se_ratios[mode] = sr.se_ratio

    assert se_ratios["iid"] < se_ratios["group"], (
        f"expected iid se_ratio ({se_ratios['iid']:.3f}) < group se_ratio "
        f"({se_ratios['group']:.3f}) at full leakage — variance should be more "
        f"understated without GroupKFold"
    )
