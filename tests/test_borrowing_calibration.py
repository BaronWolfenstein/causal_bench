"""Tests for size-calibrated decision cutoff and influence factor (issue #22)."""
import numpy as np
import pytest
from scipy.stats import norm

from causal_bench.estimators.hierarchical import influence_factor, size_calibrated_z


class TestSizeCalibratedZ:
    """Tests for the size_calibrated_z helper (Normal-Normal model, issue #22)."""

    def test_vague_prior_limit_approaches_standard_z(self):
        """r → ∞ (very large τ relative to s) → calibrated_z → z_{1-α/2}."""
        cal_z, r = size_calibrated_z(tau_prior_sd=1000.0, likelihood_sd=1.0)
        standard_z = norm.ppf(0.975)
        assert abs(cal_z - standard_z) < 0.001, (
            f"Expected c* ≈ {standard_z:.4f} (vague-prior limit), got {cal_z:.4f} (r={r:.1f})"
        )

    def test_informative_prior_gives_lower_cutoff(self):
        """Informative prior (r ≈ 1) → c* < z_{1-α/2} (stricter size control)."""
        cal_z, r = size_calibrated_z(tau_prior_sd=0.1, likelihood_sd=0.1)
        standard_z = norm.ppf(0.975)
        assert cal_z < standard_z, (
            f"Expected c* < {standard_z:.4f} under informative prior, got {cal_z:.4f} (r={r:.2f})"
        )

    def test_r_ratio_returned_correctly(self):
        """r should equal tau / s."""
        tau, s = 0.5, 0.25
        _, r = size_calibrated_z(tau_prior_sd=tau, likelihood_sd=s)
        assert r == pytest.approx(tau / s, rel=1e-6)

    def test_calibrated_z_decreases_with_more_informative_prior(self):
        """Larger τ/s (stronger prior) → lower c*."""
        cal_z_weak,  _ = size_calibrated_z(tau_prior_sd=0.1,  likelihood_sd=1.0)  # r=0.1
        cal_z_strong, _ = size_calibrated_z(tau_prior_sd=10.0, likelihood_sd=1.0)  # r=10
        assert cal_z_weak < cal_z_strong  # weaker prior → lower c* (more correction needed)

    def test_default_alpha_is_0_05(self):
        """Default alpha=0.05 → standard z target is norm.ppf(0.975)."""
        cal_z_default, _ = size_calibrated_z(tau_prior_sd=1000.0, likelihood_sd=1.0)
        cal_z_explicit, _ = size_calibrated_z(tau_prior_sd=1000.0, likelihood_sd=1.0, alpha=0.05)
        assert cal_z_default == pytest.approx(cal_z_explicit)

    def test_calibrated_z_is_positive(self):
        """c* should always be positive."""
        for tau in [0.01, 0.1, 1.0, 10.0]:
            for s in [0.01, 0.1, 1.0]:
                cal_z, _ = size_calibrated_z(tau, s)
                assert cal_z > 0, f"c* should be positive for tau={tau}, s={s}"

    def test_calibrated_z_bounded_by_standard_z(self):
        """c* ≤ z_{1-α/2} always (calibrated cutoff can only be lower or equal)."""
        standard_z = norm.ppf(0.975)
        for tau in [0.01, 0.1, 1.0, 10.0, 1000.0]:
            for s in [0.05, 0.1, 0.5]:
                cal_z, _ = size_calibrated_z(tau, s)
                assert cal_z <= standard_z + 1e-9, (
                    f"c* should not exceed z_{{1-α/2}}={standard_z:.4f}, "
                    f"got {cal_z:.4f} for tau={tau}, s={s}"
                )


class TestInfluenceFactor:
    """Tests for influence_factor helper (issue #22 item 3)."""

    def _cal_z(self):
        return norm.ppf(0.975)

    def test_full_conflict_gives_negative_log_if(self):
        """Conflict scenario: MAP posterior is pulled toward null by a conflicting prior.
        MAP-only posterior near 0 → less likely to reject than vague posterior far from 0.
        log IF = log Pr_M - log Pr_V < 0."""
        cal_z = self._cal_z()
        # MAP-only posterior: pulled to near-null by conflicting prior (map_mean ≈ 0)
        # Vague-only posterior: stays with data far from null (vague_mean = -0.20)
        log_if = influence_factor(
            map_mean=-0.02, map_sd=0.04,
            vague_mean=-0.20, vague_sd=0.06,
            calibrated_z=cal_z,
        )
        assert log_if < 0.0, f"Expected log IF < 0 (MAP pulls toward null), got {log_if:.4f}"

    def test_concordant_prior_gives_positive_log_if(self):
        """Concordance: MAP posterior far from null, vague near null.
        MAP component more likely to reject → positive log IF."""
        cal_z = self._cal_z()
        log_if = influence_factor(
            map_mean=-0.30, map_sd=0.04,
            vague_mean=-0.02, vague_sd=0.10,
            calibrated_z=cal_z,
        )
        assert log_if > 0.0, f"Expected log IF > 0 under concordance, got {log_if:.4f}"

    def test_returns_finite_float(self):
        cal_z = self._cal_z()
        log_if = influence_factor(0.0, 0.10, 0.0, 0.50, cal_z)
        assert np.isfinite(log_if)

    def test_influence_factor_wired_into_borrowing_result(self):
        """population_level_borrow should return a BorrowingResult with finite influence_factor."""
        from causal_bench.estimators.hierarchical import (
            RegistrySummary, population_level_borrow,
        )
        main = RegistrySummary("main", 2000, 1000, 1000, -0.15, 0.02, -0.15)
        target = RegistrySummary("teer", 200, 100, 100, -0.14, 0.04, -0.14)
        result = population_level_borrow(main, target)
        assert np.isfinite(result.influence_factor), (
            f"influence_factor should be finite, got {result.influence_factor}"
        )
