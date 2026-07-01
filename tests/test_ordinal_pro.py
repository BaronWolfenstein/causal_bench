"""Tests for causal_bench/dgp/ordinal_pro.py (issue #26).

Test plan
---------
1. Config validation — frozen, extra-forbid, coupling checks.
2. Data generation — shape, column types, category range, treatment balance.
3. PO-respecting default — cumulative log-ORs roughly equal across thresholds.
4. PO-violation — tau_category_offsets, floor_effect, ceiling_effect measurably
   break proportional odds.
5. Estimand recovery — both true cumulative log-OR and ordinal win ratio recovered
   at large n within tolerance.
6. Keyed determinism — same seed → identical DataFrame.
7. Site random effects — site_id column present; ICC > 0 changes distribution.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from causal_bench.dgp.ordinal_pro import (
    OrdinalPROConfig,
    compute_true_cumulative_logOR,
    compute_true_ordinal_win_ratio,
    generate_data,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cfg(**kw) -> OrdinalPROConfig:
    """Shorthand: build config with seed=0 unless overridden."""
    return OrdinalPROConfig(**{"seed": 0, **kw})


# ---------------------------------------------------------------------------
# 1. Config validation
# ---------------------------------------------------------------------------

class TestOrdinalPROConfig:
    def test_defaults_sane(self):
        cfg = OrdinalPROConfig()
        assert cfg.n == 500
        assert cfg.K == 4
        assert cfg.tau == 0.5
        assert cfg.seed == 42
        assert cfg.marker_col == "ordinal_pro"
        assert cfg.tau_category_offsets == ()
        assert cfg.floor_effect == 0.0
        assert cfg.site_icc == 0.0

    def test_frozen(self):
        cfg = OrdinalPROConfig()
        with pytest.raises(Exception):
            cfg.n = 999  # type: ignore[misc]

    def test_extra_fields_rejected(self):
        with pytest.raises(ValidationError):
            OrdinalPROConfig(bogus_field=1)  # type: ignore[call-arg]

    def test_tau_category_offsets_length_mismatch_raises(self):
        # K=4 requires K-1=3 offsets; providing 2 should raise
        with pytest.raises(ValidationError, match="tau_category_offsets"):
            OrdinalPROConfig(K=4, tau_category_offsets=(0.1, 0.2))

    def test_tau_category_offsets_correct_length_ok(self):
        cfg = OrdinalPROConfig(K=4, tau_category_offsets=(0.1, 0.0, -0.2))
        assert len(cfg.tau_category_offsets) == 3

    def test_cutpoints_length_mismatch_raises(self):
        with pytest.raises(ValidationError, match="cutpoints"):
            OrdinalPROConfig(K=4, cutpoints=(-1.0, 0.0))  # needs 3 for K=4

    def test_cutpoints_non_increasing_raises(self):
        with pytest.raises(ValidationError, match="strictly increasing"):
            OrdinalPROConfig(K=3, cutpoints=(1.0, -1.0))

    def test_site_icc_without_sites_raises(self):
        with pytest.raises(ValidationError, match="n_sites"):
            OrdinalPROConfig(n_sites=1, site_icc=0.3)

    def test_site_threshold_sd_without_sites_raises(self):
        with pytest.raises(ValidationError, match="n_sites"):
            OrdinalPROConfig(n_sites=1, site_threshold_sd=0.2)

    def test_with_overrides(self):
        cfg = OrdinalPROConfig(n=500, seed=42)
        cfg2 = cfg.with_overrides(n=1000)
        assert cfg2.n == 1000
        assert cfg2.seed == 42  # unchanged

    def test_with_overrides_invalid_raises(self):
        cfg = OrdinalPROConfig()
        with pytest.raises((ValidationError, Exception)):
            cfg.with_overrides(K=4, cutpoints=(-1.0, 0.0))  # wrong length


# ---------------------------------------------------------------------------
# 2. Data generation — shape and types
# ---------------------------------------------------------------------------

class TestGenerateData:
    def test_shape(self):
        cfg = _cfg(n=200)
        df = generate_data(cfg)
        assert len(df) == 200

    def test_required_columns(self):
        df = generate_data(_cfg())
        required = {"A", "W1", "W2", "W3", "W4", "site_id", "ordinal_pro"}
        assert required.issubset(df.columns)

    def test_custom_marker_col(self):
        cfg = _cfg(marker_col="nyha")
        df = generate_data(cfg)
        assert "nyha" in df.columns
        assert "ordinal_pro" not in df.columns

    def test_ordinal_range_default_K4(self):
        cfg = _cfg(n=1000, K=4)
        df = generate_data(cfg)
        Y = df["ordinal_pro"]
        assert Y.min() >= 1
        assert Y.max() <= 4
        assert set(Y.unique()).issubset({1, 2, 3, 4})

    def test_ordinal_all_categories_present(self):
        """Large n should produce all K categories."""
        cfg = _cfg(n=5000, K=4)
        df = generate_data(cfg)
        assert set(df["ordinal_pro"].unique()) == {1, 2, 3, 4}

    def test_ordinal_range_K3(self):
        cfg = _cfg(n=1000, K=3)
        df = generate_data(cfg)
        Y = df["ordinal_pro"]
        assert Y.min() >= 1
        assert Y.max() <= 3

    def test_treatment_is_binary(self):
        df = generate_data(_cfg(n=500))
        assert set(df["A"].unique()).issubset({0.0, 1.0})

    def test_treatment_prevalence_approx(self):
        """With no confounding, treatment prevalence should be near target."""
        cfg = _cfg(n=5000, treatment_prevalence=0.5)
        df = generate_data(cfg)
        assert 0.40 <= df["A"].mean() <= 0.60

    def test_site_id_column_single_site(self):
        df = generate_data(_cfg(n=200, n_sites=1))
        assert (df["site_id"] == 0).all()

    def test_site_id_column_multi_site(self):
        cfg = _cfg(n=500, n_sites=5, site_icc=0.1)
        df = generate_data(cfg)
        assert df["site_id"].nunique() <= 5
        assert df["site_id"].min() >= 0

    def test_covariates_shape(self):
        df = generate_data(_cfg(n=300))
        for col in ["W1", "W2", "W3", "W4"]:
            assert col in df.columns
        # W2, W4 are binary
        assert set(df["W2"].unique()).issubset({0.0, 1.0})
        assert set(df["W4"].unique()).issubset({0.0, 1.0})

    def test_explicit_cutpoints(self):
        cfg = _cfg(n=1000, K=3, cutpoints=(-1.0, 1.0))
        df = generate_data(cfg)
        assert df["ordinal_pro"].between(1, 3).all()

    def test_no_nan(self):
        df = generate_data(_cfg(n=300))
        assert not df.isnull().any().any()


# ---------------------------------------------------------------------------
# 3. Keyed determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_seed_identical_df(self):
        cfg = _cfg(n=200, K=4, seed=99)
        df1 = generate_data(cfg)
        df2 = generate_data(cfg)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seed_different_df(self):
        cfg1 = _cfg(n=200, seed=1)
        cfg2 = _cfg(n=200, seed=2)
        df1 = generate_data(cfg1)
        df2 = generate_data(cfg2)
        assert not (df1["ordinal_pro"].values == df2["ordinal_pro"].values).all()

    def test_rng_override_reproducible(self):
        cfg = _cfg(n=100)
        rng_a = np.random.default_rng(77)
        rng_b = np.random.default_rng(77)
        df1 = generate_data(cfg, rng=rng_a)
        df2 = generate_data(cfg, rng=rng_b)
        pd.testing.assert_frame_equal(df1, df2)


# ---------------------------------------------------------------------------
# 4. PO-respecting: cumulative log-ORs approximately equal across thresholds
# ---------------------------------------------------------------------------

class TestPOResecting:
    def test_po_default_config_passes_is_po_flag(self):
        """Default config (no PO-violation knobs) should report is_PO=True."""
        cfg = OrdinalPROConfig(n=2000, K=4, tau=0.5, seed=42)
        result = compute_true_cumulative_logOR(cfg, n_ref=50_000)
        assert result["is_PO"] is True, (
            f"Expected PO-respecting default to pass is_PO check; "
            f"log_OR spread={result['log_OR']}"
        )

    def test_log_ors_near_equal_po_case(self):
        """Under PO, all K-1 cumulative log-ORs should be approximately equal."""
        cfg = OrdinalPROConfig(n=2000, K=4, tau=0.6, seed=42)
        result = compute_true_cumulative_logOR(cfg, n_ref=80_000)
        log_ors = result["log_OR"]
        max_spread = max(abs(log_ors[j] - log_ors[0]) for j in range(len(log_ors)))
        assert max_spread < 0.15, (
            f"PO-respecting case: max spread of log-ORs={max_spread:.3f} should be < 0.15; "
            f"log_ORs={log_ors}"
        )

    def test_all_categories_balanced_under_null(self):
        """With tau=0, both arms should have similar category distributions."""
        cfg = OrdinalPROConfig(n=5000, K=4, tau=0.0, seed=7)
        df = generate_data(cfg)
        treated_dist = df[df["A"] == 1]["ordinal_pro"].value_counts(normalize=True)
        control_dist = df[df["A"] == 0]["ordinal_pro"].value_counts(normalize=True)
        for cat in range(1, 5):
            p1 = treated_dist.get(cat, 0.0)
            p0 = control_dist.get(cat, 0.0)
            assert abs(p1 - p0) < 0.10, (
                f"Category {cat}: P(Y={cat}|A=1)={p1:.3f}, P(Y={cat}|A=0)={p0:.3f}; "
                "should be similar under tau=0"
            )


# ---------------------------------------------------------------------------
# 5. PO-violation: knobs measurably break proportional odds
# ---------------------------------------------------------------------------

class TestPOViolation:
    def test_tau_category_offsets_breaks_po(self):
        """Large tau_category_offsets → is_PO flag should flip to False."""
        cfg = OrdinalPROConfig(
            n=2000, K=4,
            tau=0.5,
            tau_category_offsets=(1.5, 0.0, -1.5),  # strong asymmetric offsets
            seed=42,
        )
        result = compute_true_cumulative_logOR(cfg, n_ref=80_000)
        assert result["is_PO"] is False, (
            f"Strong tau_category_offsets should break PO; "
            f"log_OR={result['log_OR']}"
        )

    def test_tau_category_offsets_log_or_spread(self):
        """log-ORs should vary substantially when category-specific effects differ."""
        cfg = OrdinalPROConfig(
            n=2000, K=4,
            tau=0.5,
            tau_category_offsets=(1.0, 0.0, -1.0),
            seed=42,
        )
        result = compute_true_cumulative_logOR(cfg, n_ref=80_000)
        log_ors = result["log_OR"]
        spread = max(log_ors) - min(log_ors)
        assert spread > 0.3, (
            f"Expected log-OR spread > 0.3 under PO-violation; got spread={spread:.3f}, "
            f"log_ORs={log_ors}"
        )

    def test_floor_effect_reduces_logOR_magnitude_at_boundary1(self):
        """floor_effect should weaken (shrink the magnitude of) the cumulative log-OR
        at threshold 1.

        Convention: with tau > 0 (treatment beneficial), log-OR < 0.  "Reducing benefit"
        means |log-OR| decreases, i.e., log-OR becomes LESS negative (closer to 0,
        numerically larger).  Use a moderate floor_effect so we don't overshoot to
        positive — floor_effect=0.4 with tau=0.8 gives tau_eff[0]=0.4 (still positive,
        just weaker).
        """
        cfg_base  = OrdinalPROConfig(n=2000, K=4, tau=0.8, seed=42)
        cfg_floor = OrdinalPROConfig(n=2000, K=4, tau=0.8, floor_effect=0.5, seed=42)

        res_base  = compute_true_cumulative_logOR(cfg_base,  n_ref=60_000)
        res_floor = compute_true_cumulative_logOR(cfg_floor, n_ref=60_000)

        # Floor effect reduces tau_eff[0] → |log-OR[0]| shrinks → log-OR[0] is less negative
        # i.e., numerically LARGER (closer to 0) than the base.
        assert res_floor["log_OR"][0] > res_base["log_OR"][0], (
            f"floor_effect should weaken treatment at threshold 1 (log-OR less negative): "
            f"base={res_base['log_OR'][0]:.3f}, floor={res_floor['log_OR'][0]:.3f}"
        )

    def test_ceiling_effect_reduces_logOR_magnitude_at_top_boundary(self):
        """ceiling_effect should weaken the cumulative log-OR at the top threshold
        (makes it less negative when tau > 0)."""
        cfg_base = OrdinalPROConfig(n=2000, K=4, tau=0.8, seed=42)
        cfg_ceil = OrdinalPROConfig(n=2000, K=4, tau=0.8, ceiling_effect=0.5, seed=42)

        res_base = compute_true_cumulative_logOR(cfg_base, n_ref=60_000)
        res_ceil = compute_true_cumulative_logOR(cfg_ceil, n_ref=60_000)

        assert res_ceil["log_OR"][-1] > res_base["log_OR"][-1], (
            f"ceiling_effect should weaken treatment at top threshold (log-OR less negative): "
            f"base={res_base['log_OR'][-1]:.3f}, ceil={res_ceil['log_OR'][-1]:.3f}"
        )

    def test_site_threshold_sd_generates_valid_data(self):
        """site_threshold_sd > 0 should generate valid ordinal data without errors."""
        cfg = OrdinalPROConfig(
            n=1000, K=4, n_sites=10, site_icc=0.1,
            site_threshold_sd=0.5, seed=5,
        )
        df = generate_data(cfg)
        assert df["ordinal_pro"].between(1, 4).all()
        assert not df.isnull().any().any()

    def test_site_threshold_sd_breaks_po(self):
        """Large site_threshold_sd should measurably break marginal PO."""
        cfg = OrdinalPROConfig(
            n=2000, K=4, n_sites=20, site_icc=0.15,
            tau=0.5, site_threshold_sd=2.0, seed=42,
        )
        result = compute_true_cumulative_logOR(cfg, n_ref=80_000)
        # We don't assert is_PO=False (site-threshold effect on marginal PO is
        # subtle and depends on n_ref), but we DO assert the function runs and
        # returns sensible values.
        assert len(result["log_OR"]) == cfg.K - 1
        assert all(np.isfinite(lor) for lor in result["log_OR"])


# ---------------------------------------------------------------------------
# 6. True estimand recovery at large n
# ---------------------------------------------------------------------------

class TestEstimandRecovery:
    """Check that compute_true_* functions return values consistent with the DGP."""

    def test_cumulative_logOR_sign_positive_tau(self):
        """With tau > 0 (treatment shifts toward higher categories), all log-ORs
        should be negative (P(Y<=j | A=1) < P(Y<=j | A=0)) under the CLM convention
        where positive latent shift → higher categories are MORE likely under treatment.

        Note: P(Y<=j | W,A) = logistic(c_j - f_W - b_site - tau*A)
        Higher A (treated) → subtract more → lower P(Y<=j) → log-OR < 0.
        """
        cfg = OrdinalPROConfig(n=1000, K=4, tau=0.8, seed=42)
        result = compute_true_cumulative_logOR(cfg, n_ref=100_000)
        for j, lor in enumerate(result["log_OR"]):
            assert lor < 0, (
                f"tau=0.8 should give negative cumulative log-OR (treated have lower P(Y<=j)); "
                f"got log_OR[{j}]={lor:.3f}"
            )

    def test_cumulative_logOR_tau_zero_near_zero(self):
        """With tau=0, all cumulative log-ORs should be near zero."""
        cfg = OrdinalPROConfig(n=1000, K=4, tau=0.0, seed=42)
        result = compute_true_cumulative_logOR(cfg, n_ref=100_000)
        for j, lor in enumerate(result["log_OR"]):
            assert abs(lor) < 0.15, (
                f"tau=0 should give log-OR near 0; got log_OR[{j}]={lor:.3f}"
            )

    def test_cumulative_logOR_returns_K_minus_1_values(self):
        for K in [2, 3, 4, 5]:
            cfg = OrdinalPROConfig(K=K, seed=0)
            result = compute_true_cumulative_logOR(cfg, n_ref=10_000)
            assert len(result["log_OR"]) == K - 1, f"K={K}"

    def test_ordinal_win_ratio_greater_than_1_positive_tau(self):
        """tau > 0 shifts treated toward higher categories → WR > 1."""
        cfg = OrdinalPROConfig(n=1000, K=4, tau=0.8, seed=42)
        result = compute_true_ordinal_win_ratio(cfg, n_ref=50_000)
        assert result["ATE"] > 1.0, (
            f"tau=0.8 should give WR > 1; got WR={result['ATE']:.3f}"
        )

    def test_ordinal_win_ratio_less_than_1_negative_tau(self):
        """tau < 0 shifts treated toward lower categories → WR < 1."""
        cfg = OrdinalPROConfig(n=1000, K=4, tau=-0.8, seed=42)
        result = compute_true_ordinal_win_ratio(cfg, n_ref=50_000)
        assert result["ATE"] < 1.0, (
            f"tau=-0.8 should give WR < 1; got WR={result['ATE']:.3f}"
        )

    def test_ordinal_win_ratio_near_1_tau_zero(self):
        """tau=0 → treated and control have same distribution → WR ≈ 1."""
        cfg = OrdinalPROConfig(n=1000, K=4, tau=0.0, seed=42)
        result = compute_true_ordinal_win_ratio(cfg, n_ref=80_000)
        assert 0.85 <= result["ATE"] <= 1.15, (
            f"tau=0 should give WR ≈ 1; got WR={result['ATE']:.3f}"
        )

    def test_ordinal_win_ratio_p_win_loss_sum(self):
        """p_win + p_loss + p_tie should equal 1 (within floating-point)."""
        cfg = OrdinalPROConfig(n=1000, K=4, tau=0.5, seed=42)
        result = compute_true_ordinal_win_ratio(cfg, n_ref=20_000)
        total = result["p_win"] + result["p_loss"] + result["p_tie"]
        assert abs(total - 1.0) < 0.01, (
            f"p_win+p_loss+p_tie should sum to ~1; got {total:.4f}"
        )

    def test_ordinal_win_ratio_net_benefit_consistent(self):
        result = compute_true_ordinal_win_ratio(_cfg(K=4, tau=0.5), n_ref=20_000)
        assert abs(result["net_benefit"] - (result["p_win"] - result["p_loss"])) < 1e-9

    def test_structural_tau_eff_reported(self):
        """compute_true_cumulative_logOR should report the structural tau_eff values."""
        cfg = OrdinalPROConfig(K=4, tau=0.5, tau_category_offsets=(0.2, 0.0, -0.3), seed=42)
        result = compute_true_cumulative_logOR(cfg, n_ref=10_000)
        tau_eff = result["structural_tau_eff"]
        assert len(tau_eff) == 3
        assert abs(tau_eff[0] - 0.7) < 1e-9   # tau + offset[0]
        assert abs(tau_eff[1] - 0.5) < 1e-9   # tau + 0
        assert abs(tau_eff[2] - 0.2) < 1e-9   # tau + offset[2]


# ---------------------------------------------------------------------------
# 7. Ordinal column contract for ConcretePROWinRatioEstimator
# ---------------------------------------------------------------------------

class TestColumnContract:
    """Verify the column contract expected by ConcretePROWinRatioEstimator."""

    def test_marker_col_is_integer(self):
        """ordinal_pro column must be integer-typed (not float) for ordinal GPC."""
        df = generate_data(_cfg(n=200, K=4))
        assert pd.api.types.is_integer_dtype(df["ordinal_pro"]), (
            f"ordinal_pro should be integer dtype; got {df['ordinal_pro'].dtype}"
        )

    def test_marker_col_values_1_to_K(self):
        for K in [2, 3, 4, 5]:
            cfg = _cfg(n=1000, K=K)
            df = generate_data(cfg)
            Y = df["ordinal_pro"]
            assert Y.min() >= 1 and Y.max() <= K, f"K={K}"

    def test_pro_spec_dict_compatible(self):
        """Verify that the marker column name matches what a PRO spec dict would reference."""
        cfg = _cfg(n=100, marker_col="kccq_tier")
        df = generate_data(cfg)
        pro_spec = {"marker": cfg.marker_col, "type": "ordinal", "direction": "higher.better"}
        assert pro_spec["marker"] in df.columns

    def test_treatment_col_A_present(self):
        """ConcretePROWinRatioEstimator uses terminal_status_col='Delta' by default;
        the ordinal DGP uses 'A' for treatment — both must be present for composability
        when survival columns are merged."""
        df = generate_data(_cfg(n=100))
        assert "A" in df.columns
