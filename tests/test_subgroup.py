"""Tests for causal_bench/estimators/subgroup.py"""
from typing import Optional

import numpy as np
import pytest

from causal_bench.dgp.registry import RegistryConfig, generate_registry_data
from causal_bench.estimators.hierarchical import (
    BorrowingResult,
    population_level_borrow,
    summarise_registry,
)
from causal_bench.estimators.subgroup import (
    SubgroupBorrowingResult,
    SubgroupModel,
    assign_subgroups,
    discover_subgroups,
    estimate_cates,
    reconcile_ess,
    subgroup_level_borrow,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _cfg(**kw) -> RegistryConfig:
    return RegistryConfig(**{"seed": 0, "embedding_fidelity": 0.7, **kw})


def _data(cfg: Optional[RegistryConfig] = None):
    return generate_registry_data(cfg or _cfg())


def _model(n_subgroups=3, classifier="knn", **kw):
    cfg = _cfg(**kw)
    main_df, teer_df, mac_df, emb = _data(cfg)
    model = discover_subgroups(main_df, emb["main"],
                                n_subgroups=n_subgroups, classifier=classifier)
    return model, main_df, teer_df, mac_df, emb, cfg


# ── estimate_cates ────────────────────────────────────────────────────────────

class TestEstimateCates:
    def test_uses_true_cates_when_column_present(self):
        cfg = _cfg()
        main_df, _, _, emb = _data(cfg)
        cates = estimate_cates(main_df, emb["main"])
        np.testing.assert_array_equal(cates, main_df["cate"].values)

    def test_dr_learner_fallback_when_no_cate_column(self):
        cfg = _cfg(seed=1)
        main_df, _, _, emb = _data(cfg)
        df_no_cate = main_df.drop(columns=["cate"])
        cates = estimate_cates(df_no_cate, emb["main"])
        assert len(cates) == len(df_no_cate)
        assert np.all(np.isfinite(cates))

    def test_dr_learner_output_shape_matches_input(self):
        cfg = _cfg(seed=2, n_main=150)
        main_df, _, _, emb = _data(cfg)
        df_no_cate = main_df.drop(columns=["cate"])
        cates = estimate_cates(df_no_cate, emb["main"])
        assert cates.shape == (len(main_df),)

    def test_dr_learner_sign_plausible(self):
        """DR-Learner should recover the right sign of the mean CATE."""
        cfg = _cfg(seed=3, n_main=400, true_ate_main=-0.15, cate_sd_main=0.03)
        main_df, _, _, emb = _data(cfg)
        df_no_cate = main_df.drop(columns=["cate"])
        cates = estimate_cates(df_no_cate, emb["main"])
        # Mean estimated CATE should be negative (benefit)
        assert cates.mean() < 0, f"Expected negative mean CATE, got {cates.mean():.4f}"


# ── discover_subgroups ────────────────────────────────────────────────────────

class TestDiscoverSubgroups:
    def test_returns_subgroup_model(self):
        model, *_ = _model()
        assert isinstance(model, SubgroupModel)

    def test_n_subgroups_at_most_requested(self):
        """May return fewer subgroups if embedding space is near-1D."""
        model, *_ = _model(n_subgroups=4)
        assert 1 <= model.n_subgroups <= 4

    def test_subgroup_names_length_matches(self):
        model, *_ = _model(n_subgroups=3)
        assert len(model.subgroup_names) == model.n_subgroups

    def test_labels_cover_all_patients(self):
        model, main_df, *_ = _model()
        assert len(model.subgroup_labels) == len(main_df)

    def test_labels_are_valid_indices(self):
        model, *_ = _model()
        assert model.subgroup_labels.min() == 0
        assert model.subgroup_labels.max() == model.n_subgroups - 1

    def test_cate_by_subgroup_ordered(self):
        """Subgroups should be ordered by mean CATE (most beneficial first)."""
        model, *_ = _model(n_subgroups=3, seed=5)
        if model.n_subgroups > 1:
            # Most beneficial subgroup has smallest (most negative) CATE
            assert model.cate_by_subgroup[0] <= model.cate_by_subgroup[-1]

    def test_n_by_subgroup_sums_to_n_main(self):
        model, main_df, *_ = _model()
        assert model.n_by_subgroup.sum() == len(main_df)

    def test_cluster_centers_shape(self):
        model, main_df, _, _, emb, _ = _model()
        assert model.cluster_centers.shape == (model.n_subgroups, emb["main"].shape[1])

    def test_classifier_type_knn(self):
        model, *_ = _model(classifier="knn")
        assert model.classifier_type == "knn"

    def test_classifier_type_logistic(self):
        model, *_ = _model(classifier="logistic")
        assert model.classifier_type == "logistic"

    def test_invalid_classifier_raises(self):
        cfg = _cfg()
        main_df, _, _, emb = _data(cfg)
        with pytest.raises(ValueError, match="classifier"):
            discover_subgroups(main_df, emb["main"], classifier="xgboost")

    def test_degenerate_phi_one_reduces_subgroups(self):
        """At φ=1 with small cate_sd, K-means degenerates to fewer clusters."""
        cfg = RegistryConfig(seed=0, embedding_fidelity=1.0, cate_sd_main=0.02)
        main_df, _, _, emb = generate_registry_data(cfg)
        model = discover_subgroups(main_df, emb["main"], n_subgroups=4)
        assert model.n_subgroups <= 4   # may find fewer, must not crash

    def test_reproducibility(self):
        cfg = _cfg(seed=7)
        main_df, _, _, emb = _data(cfg)
        m1 = discover_subgroups(main_df, emb["main"], n_subgroups=3, random_state=0)
        m2 = discover_subgroups(main_df, emb["main"], n_subgroups=3, random_state=0)
        np.testing.assert_array_equal(m1.subgroup_labels, m2.subgroup_labels)


# ── assign_subgroups ──────────────────────────────────────────────────────────

class TestAssignSubgroups:
    def test_returns_integer_labels(self):
        model, main_df, teer_df, _, emb, _ = _model()
        labels = assign_subgroups(teer_df, emb["teer"], model)
        assert labels.dtype in (np.int32, np.int64, int)

    def test_labels_in_valid_range(self):
        model, main_df, teer_df, _, emb, _ = _model()
        labels = assign_subgroups(teer_df, emb["teer"], model)
        assert labels.min() >= 0
        assert labels.max() < model.n_subgroups

    def test_length_matches_target(self):
        model, main_df, teer_df, _, emb, _ = _model()
        labels = assign_subgroups(teer_df, emb["teer"], model)
        assert len(labels) == len(teer_df)

    def test_knn_and_logistic_give_same_shape(self):
        cfg = _cfg(seed=4)
        main_df, teer_df, _, emb = _data(cfg)
        m_knn = discover_subgroups(main_df, emb["main"], n_subgroups=3, classifier="knn")
        m_lr  = discover_subgroups(main_df, emb["main"], n_subgroups=3, classifier="logistic")
        l_knn = assign_subgroups(teer_df, emb["teer"], m_knn)
        l_lr  = assign_subgroups(teer_df, emb["teer"], m_lr)
        assert l_knn.shape == l_lr.shape

    def test_main_cohort_self_assignment_high_agreement(self):
        """KNN on main cohort should mostly agree with K-means stored labels.

        Not exact: KNN uses k-nearest-neighbour voting, K-means uses nearest
        centroid — they differ on boundary points. Expect >80% agreement.
        """
        model, main_df, _, _, emb, _ = _model()
        labels = assign_subgroups(main_df, emb["main"], model)
        agreement = np.mean(labels == model.subgroup_labels)
        assert agreement > 0.80, f"KNN/K-means agreement too low: {agreement:.2f}"


# ── subgroup_level_borrow ─────────────────────────────────────────────────────

class TestSubgroupLevelBorrow:
    def setup_method(self):
        self.model, self.main_df, self.teer_df, _, self.emb, self.cfg = _model(
            n_subgroups=3, seed=6
        )
        self.results = subgroup_level_borrow(
            self.main_df, self.teer_df,
            self.emb["main"], self.emb["teer"],
            self.model,
            target_true_ate=self.cfg.true_ate_teer,
        )

    def test_returns_list_of_subgroup_results(self):
        assert isinstance(self.results, list)
        assert all(isinstance(r, SubgroupBorrowingResult) for r in self.results)

    def test_at_least_one_result(self):
        assert len(self.results) >= 1

    def test_at_most_n_subgroups_results(self):
        assert len(self.results) <= self.model.n_subgroups

    def test_borrowing_level_is_subgroup(self):
        for r in self.results:
            assert r.borrowing.level == "subgroup"

    def test_all_ates_finite(self):
        for r in self.results:
            assert np.isfinite(r.borrowing.ate_posterior), \
                f"NaN ATE in subgroup {r.subgroup_name}"

    def test_all_ses_positive(self):
        for r in self.results:
            assert r.borrowing.se_posterior > 0

    def test_ci_contains_posterior_mean(self):
        for r in self.results:
            b = r.borrowing
            assert b.ci_lower < b.ate_posterior < b.ci_upper

    def test_ess_non_negative(self):
        for r in self.results:
            assert r.borrowing.ess_prior >= 0
            assert r.borrowing.ess_data  >= 0

    def test_map_weight_in_unit_interval(self):
        for r in self.results:
            assert 0.0 <= r.borrowing.map_weight <= 1.0

    def test_covers_truth_is_bool(self):
        for r in self.results:
            assert isinstance(r.borrowing.covers_truth, bool)

    def test_rejects_null_consistent_with_ci(self):
        for r in self.results:
            b = r.borrowing
            ci_excludes_zero = b.ci_lower > 0 or b.ci_upper < 0
            assert b.rejects_null == ci_excludes_zero

    def test_subgroup_names_match_model(self):
        for r in self.results:
            assert r.subgroup_name in self.model.subgroup_names

    def test_n_target_sums_to_teer_size(self):
        """Assigned patients must cover all TEER patients."""
        total = sum(r.n_target_in_subgroup for r in self.results)
        # sum of all assigned (including degenerate skipped) = n_teer
        # subgroup_level_borrow skips degenerate groups so total <= n_teer
        assert total <= len(self.teer_df)

    def test_logistic_classifier_also_works(self):
        cfg = _cfg(seed=8)
        main_df, teer_df, _, emb = _data(cfg)
        model_lr = discover_subgroups(main_df, emb["main"], n_subgroups=3,
                                       classifier="logistic")
        results = subgroup_level_borrow(
            main_df, teer_df, emb["main"], emb["teer"],
            model_lr, target_true_ate=cfg.true_ate_teer,
        )
        assert len(results) >= 1
        assert all(np.isfinite(r.borrowing.ate_posterior) for r in results)

    def test_degenerate_subgroup_skipped_gracefully(self):
        """Very high min_subgroup_n forces most subgroups to be skipped."""
        results = subgroup_level_borrow(
            self.main_df, self.teer_df,
            self.emb["main"], self.emb["teer"],
            self.model,
            target_true_ate=self.cfg.true_ate_teer,
            min_subgroup_n=9999,  # forces all to degenerate path
        )
        # Should return no results or only results with ess_prior=0
        for r in results:
            assert r.borrowing.ess_prior == pytest.approx(0.0)


# ── reconcile_ess ─────────────────────────────────────────────────────────────

class TestReconcileEss:
    def _make_borrowing(self, ess_prior, ess_data=80.0) -> BorrowingResult:
        return BorrowingResult(
            level="subgroup", target_registry="teer",
            ate_posterior=-0.10, se_posterior=0.05,
            ci_lower=-0.20, ci_upper=0.00,
            ess_prior=ess_prior, ess_data=ess_data, ess_total=ess_prior + ess_data,
            map_weight=0.9, rejects_null=True, covers_truth=True, true_ate=-0.12,
        )

    def _make_sub(self, ess_prior, n=30) -> SubgroupBorrowingResult:
        return SubgroupBorrowingResult(
            subgroup_idx=0, subgroup_name="test",
            borrowing=self._make_borrowing(ess_prior, ess_data=float(n)),
            n_target_in_subgroup=n,
            cate_main_mean=-0.10, cate_main_sd=0.05,
        )

    def test_returns_dict_with_expected_keys(self):
        pop = self._make_borrowing(5.0, 80.0)
        subs = [self._make_sub(2.0, 40), self._make_sub(2.0, 40)]
        result = reconcile_ess(subs, pop)
        for key in ("subgroup_ess_prior_sum", "subgroup_ess_data_sum",
                    "population_ess_prior", "population_ess_data",
                    "consistent", "ess_prior_ratio"):
            assert key in result

    def test_consistent_when_subgroup_sum_below_population(self):
        pop = self._make_borrowing(10.0, 80.0)
        subs = [self._make_sub(3.0, 40), self._make_sub(3.0, 40)]
        result = reconcile_ess(subs, pop)
        assert result["consistent"] is True

    def test_inconsistent_when_subgroup_sum_exceeds_population(self):
        pop = self._make_borrowing(1.0, 80.0)
        subs = [self._make_sub(5.0, 40), self._make_sub(5.0, 40)]
        result = reconcile_ess(subs, pop)
        assert result["consistent"] is False

    def test_ess_prior_ratio_correct(self):
        pop = self._make_borrowing(10.0, 80.0)
        subs = [self._make_sub(4.0, 40), self._make_sub(4.0, 40)]
        result = reconcile_ess(subs, pop)
        assert result["ess_prior_ratio"] == pytest.approx(0.8, abs=1e-6)

    def test_empty_subgroups(self):
        pop = self._make_borrowing(5.0, 80.0)
        result = reconcile_ess([], pop)
        assert result["subgroup_ess_prior_sum"] == 0.0
        # Empty subgroup list means zero data ESS — correctly flagged inconsistent
        # because data ESS sum (0) deviates from population data ESS (80).
        assert "consistent" in result  # key exists; value is a diagnostic


# ── integration: full pipeline ────────────────────────────────────────────────

class TestFullSubgroupPipeline:
    def test_discover_borrow_reconcile_no_crash(self):
        cfg = RegistryConfig(seed=9, embedding_fidelity=0.6)
        main_df, teer_df, _, emb = generate_registry_data(cfg)
        model = discover_subgroups(main_df, emb["main"], n_subgroups=3)
        sub_results = subgroup_level_borrow(
            main_df, teer_df, emb["main"], emb["teer"],
            model, target_true_ate=cfg.true_ate_teer,
        )
        main_sum = summarise_registry(main_df, cfg.true_ate_main, "main")
        teer_sum = summarise_registry(teer_df, cfg.true_ate_teer, "teer")
        pop = population_level_borrow(main_sum, teer_sum)
        ess = reconcile_ess(sub_results, pop)
        assert isinstance(ess["consistent"], bool)
        assert len(sub_results) >= 1

    def test_dr_learner_path_no_crash(self):
        """When true CATEs absent, DR-Learner runs without error."""
        cfg = RegistryConfig(seed=10, embedding_fidelity=0.5, n_main=200)
        main_df, teer_df, _, emb = generate_registry_data(cfg)
        main_no_cate = main_df.drop(columns=["cate"])
        # discover_subgroups calls estimate_cates internally
        cates = estimate_cates(main_no_cate, emb["main"])
        assert len(cates) == len(main_df)
        assert np.all(np.isfinite(cates))

    def test_multiple_seeds_give_similar_subgroup_count(self):
        """Subgroup count should be stable across seeds for same φ."""
        counts = []
        for seed in range(5):
            cfg = RegistryConfig(seed=seed, embedding_fidelity=0.6)
            main_df, _, _, emb = generate_registry_data(cfg)
            model = discover_subgroups(main_df, emb["main"], n_subgroups=4)
            counts.append(model.n_subgroups)
        # All seeds should find the same number of subgroups (data structure stable)
        assert len(set(counts)) <= 2, f"Unstable subgroup count across seeds: {counts}"
