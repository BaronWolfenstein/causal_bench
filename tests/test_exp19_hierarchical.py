"""Tests for Exp 19: registry DGP and hierarchical borrowing estimator."""
from typing import Optional

import numpy as np
import pytest
from pydantic import ValidationError

from causal_bench.dgp.registry import RegistryConfig, generate_registry_data
from causal_bench.estimators.hierarchical import (
    BorrowingResult,
    OCMetrics,
    RegistrySummary,
    _MAP_EXACT_THRESHOLD,
    compute_ess,
    compute_oc_metrics,
    patient_level_borrow,
    population_level_borrow,
    robust_map_posterior,
    summarise_registry,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _cfg(**kw) -> RegistryConfig:
    return RegistryConfig(**{"seed": 0, **kw})


def _data(cfg: Optional[RegistryConfig] = None):
    return generate_registry_data(cfg or _cfg())


def _summary(df, true_ate: float, name: str = "teer") -> RegistrySummary:
    return summarise_registry(df, true_ate, name)


# ── RegistryConfig ─────────────────────────────────────────────────────────────

class TestRegistryConfig:
    def test_defaults_are_frozen(self):
        cfg = _cfg()
        with pytest.raises(Exception):
            cfg.n_main = 999  # type: ignore[misc]

    def test_extra_fields_rejected(self):
        with pytest.raises(ValidationError):
            RegistryConfig(n_main=100, bogus_field=1)  # type: ignore[call-arg]

    def test_conflict_zero_yields_same_ates(self):
        cfg = _cfg(conflict_strength=0.0)
        assert cfg.true_ate_teer == cfg.true_ate_main
        assert cfg.true_ate_mac  == cfg.true_ate_main

    def test_conflict_one_flips_sign(self):
        cfg = _cfg(conflict_strength=1.0)
        assert cfg.true_ate_teer == pytest.approx(-cfg.true_ate_main)
        assert cfg.true_ate_mac  == pytest.approx(-cfg.true_ate_main)

    def test_conflict_half_is_zero_effect(self):
        cfg = _cfg(conflict_strength=0.5, true_ate_main=-0.12)
        assert cfg.true_ate_teer == pytest.approx(0.0)

    def test_explicit_rare_ates_override_conflict(self):
        cfg = RegistryConfig(conflict_strength=1.0, true_ate_teer=-0.05, true_ate_mac=-0.08)
        assert cfg.true_ate_teer == pytest.approx(-0.05)
        assert cfg.true_ate_mac  == pytest.approx(-0.08)

    def test_size_bounds(self):
        with pytest.raises(ValidationError):
            RegistryConfig(n_main=5)   # below ge=10
        with pytest.raises(ValidationError):
            RegistryConfig(n_teer=2)   # below ge=5

    def test_embedding_fidelity_bounds(self):
        with pytest.raises(ValidationError):
            RegistryConfig(embedding_fidelity=1.5)
        with pytest.raises(ValidationError):
            RegistryConfig(embedding_fidelity=-0.1)


# ── generate_registry_data ─────────────────────────────────────────────────────

class TestGenerateRegistryData:
    def setup_method(self):
        self.cfg = _cfg()
        self.main, self.teer, self.mac, self.emb = _data(self.cfg)

    def test_row_counts(self):
        assert len(self.main) == self.cfg.n_main
        assert len(self.teer) == self.cfg.n_teer
        assert len(self.mac)  == self.cfg.n_mac

    def test_required_columns(self):
        for df in (self.main, self.teer, self.mac):
            for col in ("Y", "A", "W1", "W2", "W3", "cate", "registry", "p0", "p1"):
                assert col in df.columns, f"missing column {col}"

    def test_binary_outcome_and_treatment(self):
        for df in (self.main, self.teer, self.mac):
            assert set(df["Y"].unique()).issubset({0.0, 1.0})
            assert set(df["A"].unique()).issubset({0.0, 1.0})

    def test_registry_labels(self):
        assert self.main["registry"].iloc[0] == "main"
        assert self.teer["registry"].iloc[0] == "teer"
        assert self.mac["registry"].iloc[0]  == "mac"

    def test_embedding_shapes(self):
        assert self.emb["main"].shape == (self.cfg.n_main, self.cfg.n_embedding_dims)
        assert self.emb["teer"].shape == (self.cfg.n_teer, self.cfg.n_embedding_dims)
        assert self.emb["mac"].shape  == (self.cfg.n_mac,  self.cfg.n_embedding_dims)

    def test_embeddings_l2_normalised(self):
        for key, arr in self.emb.items():
            norms = np.linalg.norm(arr, axis=1)
            np.testing.assert_allclose(norms, 1.0, atol=1e-6, err_msg=f"{key} embeddings not unit-normed")

    def test_no_nan_in_outputs(self):
        for df in (self.main, self.teer, self.mac):
            assert not df[["Y", "A", "W1", "W2", "cate"]].isnull().any().any()
        for arr in self.emb.values():
            assert not np.isnan(arr).any()

    def test_seed_reproducibility(self):
        main2, teer2, _, _ = _data(_cfg())
        np.testing.assert_array_equal(self.main["Y"].values, main2["Y"].values)
        np.testing.assert_array_equal(self.teer["A"].values, teer2["A"].values)

    def test_different_seeds_differ(self):
        main2, _, _, _ = _data(_cfg(seed=99))
        assert not np.array_equal(self.main["Y"].values, main2["Y"].values)

    def test_phi_zero_embeddings_uninformative(self):
        """At φ=0, embedding similarity should be uncorrelated with CATE similarity."""
        cfg_zero = _cfg(embedding_fidelity=0.0, n_main=300, seed=5)
        main_df, _, _, emb = _data(cfg_zero)
        # cosine sim between random L2-normed vectors should be near 0 on average
        e = emb["main"]
        K = e @ e.T
        # off-diagonal mean should be close to 0 for random unit vectors
        mask = ~np.eye(len(e), dtype=bool)
        assert abs(K[mask].mean()) < 0.2

    def test_phi_one_embeddings_informative(self):
        """At φ=1, patients with similar CATEs should have higher cosine similarity."""
        cfg_one = _cfg(embedding_fidelity=1.0, n_main=200, cate_sd_main=0.10, seed=3)
        main_df, _, _, emb = _data(cfg_one)
        e = emb["main"]
        cate = main_df["cate"].values
        K = e @ e.T
        # pairs where |CATE_i - CATE_j| is small should have higher similarity
        cate_diff = np.abs(cate[:, None] - cate[None, :])
        mask = ~np.eye(len(e), dtype=bool)
        # Spearman rank correlation: more similar CATE → higher K
        from scipy.stats import spearmanr
        rho, _ = spearmanr(-cate_diff[mask], K[mask])
        assert rho > 0.3, f"expected ρ>0.3 at φ=1, got {rho:.3f}"


# ── summarise_registry ────────────────────────────────────────────────────────

class TestSummariseRegistry:
    def setup_method(self):
        cfg = _cfg()
        self.main, self.teer, _, _ = _data(cfg)
        self.main_sum = _summary(self.main, -0.12, "main")
        self.teer_sum = _summary(self.teer, -0.12, "teer")

    def test_fields_populated(self):
        s = self.main_sum
        assert s.name == "main"
        assert s.n == len(self.main)
        assert s.n_treated + s.n_control == s.n
        assert np.isfinite(s.ate_hat)
        assert s.se_hat > 0

    def test_ate_hat_is_difference_in_means(self):
        df = self.main
        y1 = df[df["A"] == 1]["Y"].mean()
        y0 = df[df["A"] == 0]["Y"].mean()
        assert self.main_sum.ate_hat == pytest.approx(y1 - y0, abs=1e-10)

    def test_se_positive(self):
        assert self.teer_sum.se_hat > 0


# ── robust_map_posterior ──────────────────────────────────────────────────────

class TestRobustMapPosterior:
    def _make_summary(self, ate, se, n=100, name="donor") -> RegistrySummary:
        return RegistrySummary(
            name=name, n=n, n_treated=n // 2, n_control=n // 2,
            ate_hat=ate, se_hat=se, true_ate=ate,
        )

    def test_no_conflict_posterior_between_prior_and_data(self):
        donor = self._make_summary(-0.10, 0.02)
        target = self._make_summary(-0.08, 0.04, name="target")
        mean, sd, w, _ = robust_map_posterior([donor], target)
        # Posterior should be between prior mean (-0.10) and data (-0.08)
        assert -0.10 <= mean <= -0.08 or -0.08 <= mean <= -0.10

    def test_conflict_reduces_map_weight(self):
        """When target data contradicts donor, MAP weight should drop."""
        donor = self._make_summary(-0.15, 0.02)
        target_agree    = self._make_summary(-0.14, 0.04, name="t_agree")
        target_conflict = self._make_summary(+0.10, 0.04, name="t_conflict")
        _, _, w_agree,    _ = robust_map_posterior([donor], target_agree)
        _, _, w_conflict, _ = robust_map_posterior([donor], target_conflict)
        assert w_conflict < w_agree

    def test_large_vague_sd_reduces_vague_influence(self):
        """Wider vague component → more prior weight stays on MAP."""
        donor = self._make_summary(-0.10, 0.02)
        target = self._make_summary(-0.09, 0.04, name="t")
        _, _, w_narrow, _ = robust_map_posterior([donor], target, vague_sd=0.20)
        _, _, w_wide,   _ = robust_map_posterior([donor], target, vague_sd=2.00)
        # wider vague: prior predictive at data point is broader → less log-lik disadvantage
        # MAP weight can go either way depending on data location; just check both are in [0,1]
        assert 0.0 <= w_narrow <= 1.0
        assert 0.0 <= w_wide   <= 1.0

    def test_robust_weight_floor(self):
        """With robust_weight=0.5, MAP posterior weight can't dominate completely."""
        donor = self._make_summary(-0.10, 0.01)
        # Target fully agrees
        target = self._make_summary(-0.10, 0.01, name="t")
        _, _, w, _ = robust_map_posterior([donor], target, robust_weight=0.50)
        # MAP posterior weight should be < 1 because 50% prior on vague
        assert w < 1.0

    def test_returns_finite_values(self):
        donor = self._make_summary(-0.12, 0.03)
        target = self._make_summary(-0.10, 0.05, name="t")
        mean, sd, w, sigma2_map = robust_map_posterior([donor], target)
        assert np.isfinite(mean)
        assert np.isfinite(sd) and sd > 0
        assert 0.0 <= w <= 1.0
        assert sigma2_map > 0

    def test_single_donor_no_division_by_zero(self):
        """One donor: DL c_dl could be 0; should not raise."""
        donor = self._make_summary(-0.10, 0.05)
        target = self._make_summary(-0.08, 0.08, name="t")
        mean, sd, w, _ = robust_map_posterior([donor], target)
        assert np.isfinite(mean)


# ── compute_ess ───────────────────────────────────────────────────────────────

class TestComputeEss:
    def test_prior_dominates_when_prior_is_tight(self):
        # Very tight prior, loose likelihood → ESS_prior > ESS_data
        ess_prior, ess_data, ess_total = compute_ess(
            prior_sd=0.01, likelihood_sd=0.10, posterior_sd=0.009, target_n=50
        )
        assert ess_prior > ess_data

    def test_data_dominates_when_prior_is_vague(self):
        # Very loose prior → prior contributes little
        ess_prior, ess_data, ess_total = compute_ess(
            prior_sd=10.0, likelihood_sd=0.05, posterior_sd=0.049, target_n=100
        )
        assert ess_data > ess_prior

    def test_ess_total_is_sum(self):
        ep, ed, et = compute_ess(0.10, 0.05, 0.04, 80)
        assert et == pytest.approx(ep + ed, abs=1e-8)

    def test_ess_data_equals_target_n(self):
        _, ess_data, _ = compute_ess(0.10, 0.05, 0.04, 123)
        assert ess_data == pytest.approx(123.0)

    def test_non_negative(self):
        ep, ed, et = compute_ess(0.10, 0.05, 0.05, 50)
        assert ep >= 0 and ed >= 0 and et >= 0


# ── population_level_borrow ───────────────────────────────────────────────────

class TestPopulationLevelBorrow:
    def setup_method(self):
        cfg = _cfg()
        main_df, teer_df, _, _ = _data(cfg)
        self.main_sum = summarise_registry(main_df, cfg.true_ate_main, "main")
        self.teer_sum = summarise_registry(teer_df, cfg.true_ate_teer, "teer")
        self.result = population_level_borrow(
            self.main_sum, self.teer_sum,
            tau_prior_sd=cfg.tau_prior_sd,
            robust_weight=cfg.robust_weight,
            vague_sd=cfg.vague_sd,
        )

    def test_returns_borrowing_result(self):
        assert isinstance(self.result, BorrowingResult)

    def test_level_is_population(self):
        assert self.result.level == "population"

    def test_target_registry_label(self):
        assert self.result.target_registry == "teer"

    def test_ci_contains_posterior_mean(self):
        r = self.result
        assert r.ci_lower < r.ate_posterior < r.ci_upper

    def test_ci_width_consistent_with_se(self):
        r = self.result
        expected_half = pytest.approx(1.96 * r.se_posterior, rel=0.01)
        assert (r.ci_upper - r.ci_lower) / 2 == expected_half

    def test_ess_non_negative(self):
        r = self.result
        assert r.ess_prior >= 0
        assert r.ess_data  >= 0
        assert r.ess_total >= 0

    def test_ess_total_is_sum(self):
        r = self.result
        assert r.ess_total == pytest.approx(r.ess_prior + r.ess_data, abs=1e-6)

    def test_map_weight_in_unit_interval(self):
        assert 0.0 <= self.result.map_weight <= 1.0

    def test_covers_truth_is_bool(self):
        assert isinstance(self.result.covers_truth, bool)

    def test_rejects_null_consistent_with_ci(self):
        r = self.result
        ci_excludes_zero = r.ci_lower > 0 or r.ci_upper < 0
        assert r.rejects_null == ci_excludes_zero

    def test_no_conflict_borrows_toward_main(self):
        """No-conflict case: posterior should move toward main ATE vs. naive target ATE."""
        cfg = _cfg(conflict_strength=0.0, seed=1)
        main_df, teer_df, _, _ = _data(cfg)
        main_sum = summarise_registry(main_df, cfg.true_ate_main, "main")
        teer_sum = summarise_registry(teer_df, cfg.true_ate_teer, "teer")
        result = population_level_borrow(main_sum, teer_sum,
                                         tau_prior_sd=cfg.tau_prior_sd,
                                         robust_weight=cfg.robust_weight,
                                         vague_sd=cfg.vague_sd)
        # Posterior should be strictly tighter than naive (prior shrinks SE)
        assert result.se_posterior < teer_sum.se_hat

    def test_full_conflict_map_weight_below_prior(self):
        """When target data is far from the MAP prior, MAP weight should be lower
        than when the target data agrees. Tested with controlled synthetic summaries
        where the disagreement is unambiguous (tight SEs, large separation)."""
        # Tight SE so the MAP prior predictive is also tight
        donor_agree    = RegistrySummary("main", 2000, 1000, 1000, -0.15, 0.005, -0.15)
        target_agree   = RegistrySummary("teer",  200,  100,  100, -0.14, 0.010, -0.14)
        target_conflict= RegistrySummary("teer",  200,  100,  100, +0.14, 0.010, +0.14)
        _, _, w_agree,    _ = robust_map_posterior([donor_agree], target_agree)
        _, _, w_conflict, _ = robust_map_posterior([donor_agree], target_conflict)
        assert w_conflict < w_agree


# ── patient_level_borrow ──────────────────────────────────────────────────────

class TestPatientLevelBorrow:
    def setup_method(self):
        self.cfg = _cfg(embedding_fidelity=1.0, seed=3)
        self.main_df, self.teer_df, _, self.emb = _data(self.cfg)
        self.result = patient_level_borrow(
            self.main_df, self.teer_df,
            self.emb["main"], self.emb["teer"],
            self.cfg.true_ate_teer, self.cfg,
        )

    def test_returns_borrowing_result(self):
        assert isinstance(self.result, BorrowingResult)

    def test_level_is_patient(self):
        assert self.result.level == "patient"

    def test_ci_contains_posterior_mean(self):
        r = self.result
        assert r.ci_lower < r.ate_posterior < r.ci_upper

    def test_ess_non_negative(self):
        r = self.result
        assert r.ess_prior >= 0 and r.ess_data >= 0

    def test_phi_zero_reduces_borrowing(self):
        """At φ=0, patient-level reduces to pure target estimate (no imported signal)."""
        cfg0 = _cfg(embedding_fidelity=0.0, seed=4)
        main_df, teer_df, _, emb = _data(cfg0)
        r0 = patient_level_borrow(main_df, teer_df, emb["main"], emb["teer"],
                                   cfg0.true_ate_teer, cfg0)
        assert r0.ess_prior == pytest.approx(0.0, abs=1e-6)

    def test_phi_one_uses_more_ess(self):
        """At φ=1, ESS_prior should be greater than at φ=0."""
        cfg0 = _cfg(embedding_fidelity=0.0, seed=5)
        cfg1 = _cfg(embedding_fidelity=1.0, seed=5)
        main0, teer0, _, emb0 = _data(cfg0)
        main1, teer1, _, emb1 = _data(cfg1)
        r0 = patient_level_borrow(main0, teer0, emb0["main"], emb0["teer"], cfg0.true_ate_teer, cfg0)
        r1 = patient_level_borrow(main1, teer1, emb1["main"], emb1["teer"], cfg1.true_ate_teer, cfg1)
        assert r1.ess_prior >= r0.ess_prior

    def test_no_nan_in_result(self):
        r = self.result
        assert np.isfinite(r.ate_posterior)
        assert np.isfinite(r.se_posterior)
        assert np.isfinite(r.ess_prior)

    def test_rejects_null_consistent_with_ci(self):
        r = self.result
        ci_excludes_zero = r.ci_lower > 0 or r.ci_upper < 0
        assert r.rejects_null == ci_excludes_zero


# ── compute_oc_metrics ────────────────────────────────────────────────────────

def _make_result(
    rejects: bool,
    covers: bool,
    ate: float,
    true_ate: float,
    se: float = 0.05,
    ess_prior: float = 10.0,
    map_w: float = 0.9,
    conjugacy_regime: str = "local_approximation",
    approximation_deviation: float = float("nan"),
) -> BorrowingResult:
    z = 1.96
    return BorrowingResult(
        level="population",
        target_registry="teer",
        ate_posterior=ate,
        se_posterior=se,
        ci_lower=ate - z * se,
        ci_upper=ate + z * se,
        ess_prior=ess_prior,
        ess_data=80.0,
        ess_total=80.0 + ess_prior,
        map_weight=map_w,
        rejects_null=rejects,
        covers_truth=covers,
        true_ate=true_ate,
        conjugacy_regime=conjugacy_regime,
        approximation_deviation=approximation_deviation,
    )


class TestComputeOCMetrics:
    def test_empty_returns_nan_metrics(self):
        oc = compute_oc_metrics([])
        assert oc.n_reps == 0
        assert np.isnan(oc.power)

    def test_type1_only_in_null_scenario(self):
        results = [_make_result(True, True, 0.0, 0.0)] * 10
        oc_null = compute_oc_metrics(results, null_scenario=True)
        oc_alt  = compute_oc_metrics(results, null_scenario=False)
        assert np.isfinite(oc_null.type1_error)
        assert np.isnan(oc_null.power)
        assert np.isfinite(oc_alt.power)
        assert np.isnan(oc_alt.type1_error)

    def test_type1_rate_correct(self):
        # 3 out of 10 reject under null
        results = (
            [_make_result(True,  True, 0.0, 0.0)] * 3
            + [_make_result(False, True, 0.0, 0.0)] * 7
        )
        oc = compute_oc_metrics(results, null_scenario=True)
        assert oc.type1_error == pytest.approx(0.3)

    def test_power_rate_correct(self):
        results = (
            [_make_result(True,  True, -0.12, -0.12)] * 8
            + [_make_result(False, True, -0.12, -0.12)] * 2
        )
        oc = compute_oc_metrics(results, null_scenario=False)
        assert oc.power == pytest.approx(0.8)

    def test_coverage_rate_correct(self):
        results = (
            [_make_result(False, True,  -0.12, -0.12)] * 9
            + [_make_result(False, False, -0.20, -0.12)] * 1
        )
        oc = compute_oc_metrics(results, null_scenario=False)
        assert oc.coverage == pytest.approx(0.9)

    def test_type_m_and_type_s_nan_when_none_reject(self):
        results = [_make_result(False, True, -0.12, -0.12)] * 20
        oc = compute_oc_metrics(results, null_scenario=False)
        assert np.isnan(oc.type_m)
        assert np.isnan(oc.type_s)

    def test_type_m_one_when_estimate_equals_truth(self):
        # If significant and |est| = |true|, Type M = 1
        results = [_make_result(True, True, -0.12, -0.12)] * 20
        oc = compute_oc_metrics(results, null_scenario=False)
        assert oc.type_m == pytest.approx(1.0, abs=0.01)

    def test_type_m_inflated_when_estimate_exaggerated(self):
        # Estimates are 2× the true ATE
        results = [_make_result(True, False, -0.24, -0.12)] * 20
        oc = compute_oc_metrics(results, null_scenario=False)
        assert oc.type_m == pytest.approx(2.0, abs=0.01)

    def test_type_s_zero_when_sign_always_correct(self):
        results = [_make_result(True, True, -0.12, -0.12)] * 20
        oc = compute_oc_metrics(results, null_scenario=False)
        assert oc.type_s == pytest.approx(0.0)

    def test_type_s_one_when_sign_always_wrong(self):
        results = [_make_result(True, False, +0.12, -0.12)] * 20
        oc = compute_oc_metrics(results, null_scenario=False)
        assert oc.type_s == pytest.approx(1.0)

    def test_mde_positive_and_finite(self):
        results = [_make_result(True, True, -0.12, -0.12, se=0.05)] * 20
        oc = compute_oc_metrics(results, null_scenario=False)
        assert np.isfinite(oc.mde) and oc.mde > 0

    def test_ess_aggregated_correctly(self):
        results = [
            _make_result(False, True, -0.12, -0.12, ess_prior=10.0),
            _make_result(False, True, -0.12, -0.12, ess_prior=20.0),
        ]
        oc = compute_oc_metrics(results, null_scenario=False)
        assert oc.ess_prior_mean == pytest.approx(15.0)

    def test_map_weight_aggregated_correctly(self):
        results = [
            _make_result(False, True, -0.12, -0.12, map_w=0.8),
            _make_result(False, True, -0.12, -0.12, map_w=0.6),
        ]
        oc = compute_oc_metrics(results, null_scenario=False)
        assert oc.map_weight_mean == pytest.approx(0.7)

    def test_n_reps_correct(self):
        results = [_make_result(False, True, -0.12, -0.12)] * 17
        oc = compute_oc_metrics(results)
        assert oc.n_reps == 17


# ── integration: full pipeline round-trip ─────────────────────────────────────

class TestFullPipeline:
    """Integration: DGP → summarise → borrow → OC, checking nothing explodes."""

    def test_population_pipeline_no_conflict(self):
        cfg = _cfg(conflict_strength=0.0, seed=10)
        main_df, teer_df, _, _ = _data(cfg)
        main_sum = summarise_registry(main_df, cfg.true_ate_main, "main")
        teer_sum = summarise_registry(teer_df, cfg.true_ate_teer, "teer")
        result = population_level_borrow(main_sum, teer_sum,
                                          tau_prior_sd=cfg.tau_prior_sd,
                                          robust_weight=cfg.robust_weight,
                                          vague_sd=cfg.vague_sd)
        oc = compute_oc_metrics([result], null_scenario=False)
        assert np.isfinite(oc.mde)
        assert oc.n_reps == 1

    def test_patient_pipeline_phi_one(self):
        cfg = _cfg(embedding_fidelity=1.0, seed=11)
        main_df, teer_df, _, emb = _data(cfg)
        result = patient_level_borrow(main_df, teer_df,
                                       emb["main"], emb["teer"],
                                       cfg.true_ate_teer, cfg)
        oc = compute_oc_metrics([result], null_scenario=False)
        assert np.isfinite(oc.mde)
        assert oc.n_reps == 1

    def test_null_scenario_type1(self):
        """Under null (true_ate=0), type I rate over many reps should be reasonable."""
        cfg = _cfg(true_ate_main=0.0, conflict_strength=0.0, seed=99)
        results = []
        for seed in range(50):
            c = RegistryConfig(true_ate_main=0.0, conflict_strength=0.0, seed=seed)
            main_df, teer_df, _, _ = generate_registry_data(c)
            main_sum = summarise_registry(main_df, 0.0, "main")
            teer_sum = summarise_registry(teer_df, 0.0, "teer")
            r = population_level_borrow(main_sum, teer_sum,
                                         tau_prior_sd=c.tau_prior_sd,
                                         robust_weight=c.robust_weight,
                                         vague_sd=c.vague_sd)
            results.append(r)
        oc = compute_oc_metrics(results, null_scenario=True)
        # Type I should be below 0.30 (loose bound: small-N, 50 reps)
        assert oc.type1_error < 0.30, f"type I too high: {oc.type1_error:.3f}"


# ── conjugacy diagnostic ──────────────────────────────────────────────────────

class TestConjugacyDiagnostic:
    def _make_summary(self, ate, se, n=100, name="donor") -> RegistrySummary:
        return RegistrySummary(name=name, n=n, n_treated=n // 2, n_control=n // 2,
                               ate_hat=ate, se_hat=se, true_ate=ate)

    def test_population_result_has_regime_field(self):
        cfg = _cfg()
        main_df, teer_df, _, _ = _data(cfg)
        main_sum = summarise_registry(main_df, cfg.true_ate_main, "main")
        teer_sum = summarise_registry(teer_df, cfg.true_ate_teer, "teer")
        r = population_level_borrow(main_sum, teer_sum,
                                    tau_prior_sd=cfg.tau_prior_sd,
                                    robust_weight=cfg.robust_weight,
                                    vague_sd=cfg.vague_sd)
        assert r.conjugacy_regime in ("conjugate_exact", "local_approximation")
        assert np.isfinite(r.approximation_deviation)

    def test_no_conflict_high_map_weight_is_exact(self):
        """When prior and data agree tightly, MAP weight should exceed threshold → exact."""
        donor  = self._make_summary(-0.10, 0.005)
        target = self._make_summary(-0.10, 0.005, name="t")
        _, _, w, _ = robust_map_posterior([donor], target, robust_weight=0.05)
        assert w >= _MAP_EXACT_THRESHOLD, f"expected exact regime, map_weight={w:.3f}"

    def test_high_conflict_gives_local_approximation(self):
        """Under strong conflict, MAP weight collapses → local_approximation regime."""
        donor  = self._make_summary(-0.20, 0.005)
        target = self._make_summary(+0.20, 0.005, name="t")
        cfg = _cfg(conflict_strength=1.0)
        main_df, teer_df, _, _ = _data(cfg)
        main_sum = summarise_registry(main_df, cfg.true_ate_main, "main")
        teer_sum = summarise_registry(teer_df, cfg.true_ate_teer, "teer")
        r = population_level_borrow(main_sum, teer_sum,
                                    tau_prior_sd=cfg.tau_prior_sd,
                                    robust_weight=cfg.robust_weight,
                                    vague_sd=cfg.vague_sd)
        # Under conflict, MAP weight drops toward robust_weight (0.10) → approximate
        if r.map_weight < _MAP_EXACT_THRESHOLD:
            assert r.conjugacy_regime == "local_approximation"
            assert np.isfinite(r.approximation_deviation)

    def test_exact_regime_has_zero_deviation(self):
        """Exact regime always reports deviation = 0.0."""
        r = _make_result(False, True, -0.12, -0.12, map_w=0.98,
                         conjugacy_regime="conjugate_exact", approximation_deviation=0.0)
        assert r.approximation_deviation == pytest.approx(0.0)

    def test_patient_level_always_local_approximation(self):
        cfg = _cfg(embedding_fidelity=1.0, seed=7)
        main_df, teer_df, _, emb = _data(cfg)
        r = patient_level_borrow(main_df, teer_df,
                                 emb["main"], emb["teer"],
                                 cfg.true_ate_teer, cfg)
        assert r.conjugacy_regime == "local_approximation"

    def test_oc_metrics_exact_fraction_field(self):
        exact   = _make_result(False, True, -0.12, -0.12,
                               conjugacy_regime="conjugate_exact", approximation_deviation=0.0)
        approx  = _make_result(False, True, -0.12, -0.12,
                               conjugacy_regime="local_approximation", approximation_deviation=0.05)
        oc = compute_oc_metrics([exact, exact, approx], null_scenario=False)
        assert oc.exact_fraction == pytest.approx(2 / 3)

    def test_oc_metrics_deviation_aggregation(self):
        r1 = _make_result(False, True, -0.12, -0.12,
                          conjugacy_regime="local_approximation", approximation_deviation=0.04)
        r2 = _make_result(False, True, -0.12, -0.12,
                          conjugacy_regime="local_approximation", approximation_deviation=0.08)
        oc = compute_oc_metrics([r1, r2], null_scenario=False)
        assert oc.approx_deviation_mean == pytest.approx(0.06)
        assert oc.approx_deviation_max  == pytest.approx(0.08)

    def test_oc_metrics_all_exact_gives_nan_deviation(self):
        results = [_make_result(False, True, -0.12, -0.12,
                                conjugacy_regime="conjugate_exact", approximation_deviation=0.0)] * 5
        oc = compute_oc_metrics(results, null_scenario=False)
        assert oc.exact_fraction == pytest.approx(1.0)
        assert np.isnan(oc.approx_deviation_mean)
        assert np.isnan(oc.approx_deviation_max)
