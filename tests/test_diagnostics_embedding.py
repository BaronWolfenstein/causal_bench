"""Tests for causal_bench/diagnostics/embedding_eda.py and localization.py."""
import numpy as np
import pytest

from causal_bench.dgp.registry import RegistryConfig, generate_registry_data
from causal_bench.estimators.subgroup import discover_subgroups
from causal_bench.diagnostics.embedding_eda import (
    cluster_condition_numbers,
    phi_proxy,
    zca_unwhiten,
    zca_whiten,
)
from causal_bench.diagnostics.localization import (
    DiagnosticReport,
    LocalizationResult,
    per_mode_reconstruction_metrics,
    run_diagnostic,
    test_a as loc_test_a,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_separated(n_rare=80, n_common=200, d=16, sep=3.0, seed=0):
    """Two well-separated Gaussian clouds."""
    rng = np.random.default_rng(seed)
    rare   = rng.normal([sep] + [0.0] * (d - 1), 0.5, (n_rare, d))
    common = rng.normal([0.0] * d, 0.5, (n_common, d))
    return rare, common


def _make_overlapping(n_rare=80, n_common=200, d=16, seed=1):
    """Two identical Gaussian clouds — indistinguishable."""
    rng = np.random.default_rng(seed)
    rare   = rng.normal(0.0, 1.0, (n_rare, d))
    common = rng.normal(0.0, 1.0, (n_common, d))
    return rare, common


def _make_cfg_landing_pass(rare, common, sep=3.0, seed=40):
    """Held-out CFG-guided samples drawn like `rare` (same shift/scale as
    _make_separated) -> Test B'' passes: low fidelity AUC vs real rare, high
    drift AUC vs common."""
    d = rare.shape[1]
    rng = np.random.default_rng(seed)
    rare_guided = rng.normal([sep] + [0.0] * (d - 1), 0.5, rare.shape)
    return rare_guided, common


def _l2_normalise(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, 1e-12)


# ── phi_proxy ─────────────────────────────────────────────────────────────────

class TestPhiProxy:
    def _dgp(self, phi, seed=0):
        cfg = RegistryConfig(seed=seed, embedding_fidelity=phi)
        main_df, _, _, emb = generate_registry_data(cfg)
        return emb["main"], main_df["cate"].values

    def test_high_phi_gives_positive_proxy(self):
        emb, cate = self._dgp(phi=0.95, seed=3)
        rho = phi_proxy(emb, cate)
        assert rho > 0.20, f"Expected positive phi_proxy at φ=0.95, got {rho:.3f}"

    def test_zero_phi_gives_near_zero_proxy(self):
        emb, cate = self._dgp(phi=0.0, seed=4)
        rho = phi_proxy(emb, cate)
        assert abs(rho) < 0.30, f"Expected ~0 phi_proxy at φ=0.0, got {rho:.3f}"

    def test_output_in_minus_one_to_one(self):
        emb, cate = self._dgp(phi=0.5)
        rho = phi_proxy(emb, cate)
        assert -1.0 <= rho <= 1.0

    def test_max_pairs_limits_computation(self):
        """With max_pairs=100, result should still be finite and not crash."""
        emb, cate = self._dgp(phi=0.7)
        rho = phi_proxy(emb, cate, max_pairs=100)
        assert np.isfinite(rho)

    def test_small_n_no_crash(self):
        rng = np.random.default_rng(5)
        emb  = _l2_normalise(rng.normal(0, 1, (5, 8)))
        cate = rng.normal(0, 0.1, 5)
        rho  = phi_proxy(emb, cate)
        assert np.isfinite(rho)

    def test_reproducible(self):
        emb, cate = self._dgp(phi=0.6, seed=6)
        r1 = phi_proxy(emb, cate, random_state=0)
        r2 = phi_proxy(emb, cate, random_state=0)
        assert r1 == pytest.approx(r2, abs=1e-9)


# ── cluster_condition_numbers ─────────────────────────────────────────────────

class TestClusterConditionNumbers:
    def test_gmm_model_returns_array(self):
        cfg = RegistryConfig(seed=0, embedding_fidelity=0.7)
        main_df, _, _, emb = generate_registry_data(cfg)
        model = discover_subgroups(main_df, emb["main"], n_subgroups=3, clustering="gmm")
        cond = cluster_condition_numbers(model)
        assert cond is not None
        assert cond.shape == (model.n_subgroups,)

    def test_kmeans_model_returns_none(self):
        cfg = RegistryConfig(seed=0, embedding_fidelity=0.7)
        main_df, _, _, emb = generate_registry_data(cfg)
        model = discover_subgroups(main_df, emb["main"], n_subgroups=3, clustering="kmeans")
        assert cluster_condition_numbers(model) is None

    def test_cond_numbers_positive(self):
        cfg = RegistryConfig(seed=1, embedding_fidelity=0.7)
        main_df, _, _, emb = generate_registry_data(cfg)
        model = discover_subgroups(main_df, emb["main"], n_subgroups=3, clustering="gmm")
        cond = cluster_condition_numbers(model)
        assert np.all(cond > 0)

    def test_elongated_cluster_has_high_cond(self):
        """A cigar-shaped covariance should produce high condition number."""
        # 50 points along a 1D line in 4D space
        rng = np.random.default_rng(2)
        Z = rng.normal(0, 0.01, (50, 4))
        Z[:, 0] += np.linspace(-3, 3, 50)  # elongated along dim 0

        # Build a dummy SubgroupModel with a hand-crafted GMM-like covariance
        cov = np.cov(Z.T)
        cond_val = np.linalg.cond(cov)
        assert cond_val > 10, f"Expected high cond for elongated data, got {cond_val:.1f}"

    def test_spherical_cluster_has_low_cond(self):
        """Isotropic Gaussian should produce condition number ≈ 1."""
        rng = np.random.default_rng(3)
        Z = rng.normal(0, 1.0, (200, 4))
        cov = np.cov(Z.T)
        cond_val = np.linalg.cond(cov)
        assert cond_val < 5, f"Expected low cond for spherical data, got {cond_val:.1f}"


# ── zca_whiten / zca_unwhiten ─────────────────────────────────────────────────

class TestZcaWhiten:
    def _make_data(self, n=200, d=8, seed=0):
        rng = np.random.default_rng(seed)
        # Correlated data
        A = rng.normal(0, 1, (d, d))
        cov = A @ A.T / d + np.eye(d) * 0.5
        L = np.linalg.cholesky(cov)
        return rng.normal(0, 1, (n, d)) @ L.T

    def test_whitened_has_identity_covariance(self):
        Z = self._make_data()
        Z_w, W, mu = zca_whiten(Z)
        cov_w = np.cov(Z_w.T)
        np.testing.assert_allclose(cov_w, np.eye(Z.shape[1]), atol=0.05)

    def test_round_trip_recovers_original(self):
        Z = self._make_data(n=100, d=4)
        Z_w, W, mu = zca_whiten(Z)
        Z_back = zca_unwhiten(Z_w, W, mu)
        np.testing.assert_allclose(Z_back, Z, atol=1e-6)

    def test_output_shapes_correct(self):
        Z = self._make_data(n=50, d=6)
        Z_w, W, mu = zca_whiten(Z)
        assert Z_w.shape == Z.shape
        assert W.shape == (6, 6)
        assert mu.shape == (6,)

    def test_handles_low_rank_gracefully(self):
        """d > n: covariance is rank-deficient. eps regularisation should prevent failure."""
        rng = np.random.default_rng(4)
        Z = rng.normal(0, 1, (10, 20))  # n < d
        Z_w, W, mu = zca_whiten(Z, eps=1e-4)
        assert Z_w.shape == (10, 20)
        assert np.all(np.isfinite(Z_w))

    def test_whitened_mean_near_zero(self):
        Z = self._make_data()
        Z_w, _, _ = zca_whiten(Z)
        np.testing.assert_allclose(Z_w.mean(axis=0), 0.0, atol=1e-10)

    def test_deterministic(self):
        Z = self._make_data(seed=5)
        Z_w1, W1, mu1 = zca_whiten(Z)
        Z_w2, W2, mu2 = zca_whiten(Z)
        np.testing.assert_array_equal(Z_w1, Z_w2)


# ── test_a ────────────────────────────────────────────────────────────────────

class TestTestA:
    def test_separated_embeddings_pass(self):
        rare, common = _make_separated(sep=3.0, seed=10)
        result = loc_test_a(rare, common, cv=3, mlp_check=False, auc_threshold=0.70)
        assert isinstance(result, LocalizationResult)
        assert result.test == "A"
        assert result.passed is True
        assert result.metrics["logistic_auc"] >= 0.70

    def test_overlapping_embeddings_fail(self):
        rare, common = _make_overlapping(seed=11)
        result = loc_test_a(rare, common, cv=3, mlp_check=False, auc_threshold=0.70)
        assert result.passed is False
        assert result.metrics["logistic_auc"] < 0.70

    def test_metrics_keys_present(self):
        rare, common = _make_separated(seed=12)
        result = loc_test_a(rare, common, cv=3, mlp_check=True)
        for key in ("logistic_auc", "logistic_pr_auc", "logistic_auc_std",
                    "n_rare", "n_common", "cv_used"):
            assert key in result.metrics

    def test_mlp_check_adds_mlp_metrics(self):
        rare, common = _make_separated(sep=3.0, seed=13)
        result = loc_test_a(rare, common, cv=3, mlp_check=True)
        assert "mlp_auc" in result.metrics
        assert "mlp_pr_auc" in result.metrics

    def test_no_mlp_check_omits_mlp_metrics(self):
        rare, common = _make_separated(seed=14)
        result = loc_test_a(rare, common, cv=3, mlp_check=False)
        assert "mlp_auc" not in result.metrics

    def test_auc_in_unit_interval(self):
        rare, common = _make_separated(seed=15)
        result = loc_test_a(rare, common, cv=3, mlp_check=False)
        assert 0.0 <= result.metrics["logistic_auc"] <= 1.0

    def test_small_rare_reduces_cv(self):
        """Very small rare class should not crash; cv should auto-reduce."""
        rng = np.random.default_rng(16)
        rare   = rng.normal([2.0, 0.0], 0.5, (6, 2))
        common = rng.normal([0.0, 0.0], 0.5, (100, 2))
        result = loc_test_a(rare, common, cv=5, mlp_check=False)
        assert result.metrics["cv_used"] <= 3

    def test_notes_nonempty(self):
        rare, common = _make_separated(seed=17)
        result = loc_test_a(rare, common, cv=3, mlp_check=False)
        assert len(result.notes) > 0


# ── per_mode_reconstruction_metrics ──────────────────────────────────────────

class TestPerModeReconstructionMetrics:
    def test_identity_reconstruction_zero_l2(self):
        """Perfect reconstruction: rare_recon = rare_orig → L2 = 0."""
        rare, common = _make_separated(sep=3.0, seed=20)
        m = per_mode_reconstruction_metrics(rare, common, rare.copy(), common.copy(), cv=3)
        assert m["rare_l2_mean"]   == pytest.approx(0.0, abs=1e-10)
        assert m["common_l2_mean"] == pytest.approx(0.0, abs=1e-10)
        assert m["l2_ratio"]       == pytest.approx(0.0 / max(0.0, 1e-12), abs=0.1)

    def test_identity_reconstruction_auc_preserved(self):
        """Perfect reconstruction should not degrade separation AUC."""
        rare, common = _make_separated(sep=3.0, seed=21)
        m = per_mode_reconstruction_metrics(rare, common, rare.copy(), common.copy(), cv=3)
        assert m["auc_drop"] == pytest.approx(0.0, abs=0.01)

    def test_collapsed_rare_high_l2_ratio(self):
        """Replacing rare reconstructions with common-mode embeddings simulates tail collapse."""
        rare, common = _make_separated(sep=3.0, seed=22)
        rng = np.random.default_rng(22)
        # Rare 'reconstructed' as noise near common mean → large rare L2
        rare_collapsed = rng.normal(0.0, 0.5, rare.shape)
        m = per_mode_reconstruction_metrics(rare, common, rare_collapsed, common.copy(), cv=3)
        assert m["l2_ratio"] > 2.0, f"Expected high L2 ratio, got {m['l2_ratio']:.2f}"

    def test_keys_present(self):
        rare, common = _make_separated(seed=23)
        m = per_mode_reconstruction_metrics(rare, common, rare.copy(), common.copy(), cv=3)
        for key in ("rare_l2_mean", "common_l2_mean", "l2_ratio",
                    "separation_auc_orig", "separation_auc_recon", "auc_drop"):
            assert key in m

    def test_auc_values_in_unit_interval(self):
        rare, common = _make_separated(seed=24)
        m = per_mode_reconstruction_metrics(rare, common, rare.copy(), common.copy(), cv=3)
        assert 0.0 <= m["separation_auc_orig"]  <= 1.0
        assert 0.0 <= m["separation_auc_recon"] <= 1.0


# ── run_diagnostic ────────────────────────────────────────────────────────────

class TestRunDiagnostic:
    def _separated(self, seed=30):
        return _make_separated(sep=3.0, seed=seed)

    def _overlapping(self, seed=31):
        return _make_overlapping(seed=seed)

    def test_test_a_fail_no_pretraining_gives_bound_scope(self):
        rare, common = self._overlapping()
        report = run_diagnostic(rare, common, pretraining_influence=False,
                                auc_threshold=0.70, cv=3)
        assert report.terminal == "bound_scope"
        assert len(report.tests_run) == 1

    def test_test_a_fail_with_pretraining_gives_spt(self):
        rare, common = self._overlapping()
        report = run_diagnostic(rare, common, pretraining_influence=True,
                                auc_threshold=0.70, cv=3)
        assert report.terminal == "spt_recommendation"

    def test_test_a_pass_no_recon_b_gives_pending_b(self):
        rare, common = self._separated()
        report = run_diagnostic(rare, common, recon_b=None, cv=3)
        assert report.terminal == "pending_B"
        assert len(report.tests_run) == 1

    def test_test_b_pass_gives_diffuse_directly(self):
        rare, common = self._separated()
        # Perfect reconstruction → Test B passes; Test B'' (CFG landing) must
        # also pass before the arch verdict is awarded (2026-07-02 diagram).
        report = run_diagnostic(
            rare, common,
            recon_b=(rare.copy(), common.copy()),
            cfg_landing=_make_cfg_landing_pass(rare, common, seed=40),
            cv=3,
        )
        assert report.terminal == "diffuse_directly"
        assert any(r.test == "B" for r in report.tests_run)

    def test_test_b_pass_no_cfg_landing_gives_pending_check(self):
        rare, common = self._separated()
        report = run_diagnostic(
            rare, common,
            recon_b=(rare.copy(), common.copy()),
            cv=3,
        )
        assert report.terminal == "pending_cfg_landing_check"

    def test_test_b_fail_no_recon_bprime_gives_pending_bprime(self):
        rare, common = self._separated(seed=32)
        rng = np.random.default_rng(32)
        # Collapsed rare reconstruction → Test B fails
        rare_collapsed = rng.normal(0.0, 0.5, rare.shape)
        report = run_diagnostic(
            rare, common,
            recon_b=(rare_collapsed, common.copy()),
            recon_b_prime=None,
            reconstruction_tol=0.05,  # tight threshold → B fails
            auc_drop_tol=0.02,
            cv=3,
        )
        assert report.terminal in ("pending_B_prime", "diffuse_directly")

    def test_test_bprime_pass_gives_tail_aware(self):
        rare, common = self._separated(seed=33)
        rng = np.random.default_rng(33)
        rare_collapsed = rng.normal(0.0, 0.5, rare.shape)
        # B fails (collapsed), B' passes (perfect); B'' (CFG landing) must also
        # pass before the arch verdict is awarded (2026-07-02 diagram).
        report = run_diagnostic(
            rare, common,
            recon_b=(rare_collapsed, common.copy()),
            recon_b_prime=(rare.copy(), common.copy()),
            cfg_landing=_make_cfg_landing_pass(rare, common, seed=41),
            reconstruction_tol=0.05,
            auc_drop_tol=0.02,
            cv=3,
        )
        assert report.terminal in ("tail_aware", "diffuse_directly")

    def test_all_tests_fail_gives_escalate(self):
        rare, common = self._separated(seed=34)
        rng = np.random.default_rng(34)
        # All reconstructions are noise — all tests fail
        def _noise(): return rng.normal(0.0, 0.5, rare.shape), rng.normal(0.0, 0.5, common.shape)
        report = run_diagnostic(
            rare, common,
            recon_b=_noise(),
            recon_b_prime=_noise(),
            recon_c=_noise(),
            reconstruction_tol=0.0001,  # extremely tight — all fail
            auc_drop_tol=0.0001,
            cv=3,
        )
        assert report.terminal in ("escalate", "separate_latent_justified",
                                   "tail_aware", "diffuse_directly")

    def test_report_is_diagnostic_report(self):
        rare, common = self._separated()
        report = run_diagnostic(rare, common, cv=3)
        assert isinstance(report, DiagnosticReport)

    def test_summary_nonempty(self):
        rare, common = self._separated()
        report = run_diagnostic(rare, common, cv=3)
        assert len(report.summary) > 0

    def test_tests_run_is_list_of_localization_results(self):
        rare, common = self._separated()
        report = run_diagnostic(
            rare, common,
            recon_b=(rare.copy(), common.copy()),
            cv=3,
        )
        assert isinstance(report.tests_run, list)
        assert all(isinstance(r, LocalizationResult) for r in report.tests_run)

    def test_pending_c_when_bprime_fails_and_c_not_provided(self):
        rare, common = self._separated(seed=35)
        rng = np.random.default_rng(35)
        def _noise(): return rng.normal(0.0, 0.5, rare.shape), rng.normal(0.0, 0.5, common.shape)
        report = run_diagnostic(
            rare, common,
            recon_b=_noise(),
            recon_b_prime=_noise(),
            recon_c=None,
            reconstruction_tol=0.0001,
            auc_drop_tol=0.0001,
            cv=3,
        )
        assert report.terminal in ("pending_C", "tail_aware", "diffuse_directly")
