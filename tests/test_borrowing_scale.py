"""Tests for cloglog borrowing scale — estimand→borrowing handoff (issue #43).

The KM 1-yr composite is a cumulative incidence F ∈ [0,1], not naturally Normal.
Borrowing is performed on the complementary log-log scale θ = cloglog(F) =
log(-log(1-F)), where the Normal-Normal model and the size-calibrated cutoff are
well-founded. `s` in the cutoff is the delta-method SE of θ on that scale.
"""
import numpy as np
import pytest

from causal_bench.estimators.hierarchical import (
    to_cloglog,
    from_cloglog,
    robust_map_posterior,
    pg_test_borrow,
    RegistrySummary,
)


def _summary(ate_hat: float, se_hat: float, n: int = 100) -> RegistrySummary:
    return RegistrySummary(
        name="s", n=n, n_treated=n // 2, n_control=n // 2,
        ate_hat=ate_hat, se_hat=se_hat, true_ate=0.0,
    )


class TestToCloglog:
    def test_transform_value_and_delta_se(self):
        """cloglog(0.35) and its delta-method SE from a rate-scale SE of 0.05.

        θ = log(-log(1-F)); dθ/dF = 1 / [(1-F)·(-log(1-F))].
        For F=0.35: θ ≈ -0.84215, dθ/dF ≈ 3.57131, so SE_θ ≈ 0.05·3.57131.
        """
        theta, se_theta = to_cloglog(0.35, 0.05)
        assert theta == pytest.approx(-0.84215, abs=1e-4)
        assert se_theta == pytest.approx(0.178566, abs=1e-4)


class TestFromCloglog:
    def test_inverts_forward_transform(self):
        """from_cloglog is the inverse of cloglog: F = 1 - exp(-exp(θ))."""
        assert from_cloglog(-0.84215) == pytest.approx(0.35, abs=1e-4)

    def test_round_trip(self):
        """to_cloglog → from_cloglog recovers the rate, so a pooled θ maps back
        to the native rate scale for the 45% PG decision."""
        for f in (0.10, 0.35, 0.45, 0.72):
            theta, _ = to_cloglog(f, 0.05)
            assert from_cloglog(theta) == pytest.approx(f, abs=1e-9)


class TestVagueMeanGeneralization:
    """robust_map_posterior gains a vague_mean param so the escape component can
    be centered off zero (needed on the cloglog scale, where 0 ≠ neutral). The
    ATE default (0.0) must preserve existing behavior byte-for-byte."""

    def test_default_preserves_ate_behavior(self):
        donor, target = _summary(-0.84, 0.10), _summary(-0.80, 0.10)
        assert robust_map_posterior([donor], target) == \
            robust_map_posterior([donor], target, vague_mean=0.0)

    def test_vague_mean_shifts_escape_center(self):
        """Higher vague center → higher escape (vague-only) posterior mean.
        post_mean_vague is index 6 of the returned tuple."""
        donor, target = _summary(-0.84, 0.10), _summary(-0.80, 0.10)
        res_lo = robust_map_posterior([donor], target, vague_sd=1.0, vague_mean=-1.0)
        res_hi = robust_map_posterior([donor], target, vague_sd=1.0, vague_mean=0.5)
        assert res_hi[6] > res_lo[6]


class TestPgTestBorrow:
    """Single-arm KM cumulative-incidence borrow vs the 45% PG, on the cloglog
    scale, donor = TVT registry rate (issue #43)."""

    def test_concordant_below_pg_concludes(self):
        """Registry and trial concordant, both well below 45% with tight SE →
        concludes below PG, reports a rate-scale posterior near the data."""
        res = pg_test_borrow(
            target_rate=0.34, target_se=0.03, target_n=299,
            donor_rate=0.32, donor_se=0.03,
            performance_goal=0.45,
        )
        assert res.concludes_below_pg is True
        assert 0.30 < res.rate_posterior < 0.36
        assert 0.0 < res.ci_lower < res.ci_upper < 1.0

    def test_conflict_escapes_to_data_not_registry(self):
        """Registry optimistic (0.28, below PG) but trial data at 0.50 (above PG),
        tight SEs → prior-data conflict. The unit-information escape must revert the
        posterior to the DATA (~0.50), not drag it to the registry (~0.28), so an
        optimistic registry cannot manufacture a false 'below PG' conclusion.
        This is the Type-I-protective behavior the vague-center decision secures.

        Donor is precise+optimistic (0.20, se 0.01): confirmed that WITHOUT the
        escape (robust_weight=0) these inputs drag the posterior to 0.409 and
        falsely conclude below PG — so this assertion set genuinely tests the escape.
        """
        res = pg_test_borrow(
            target_rate=0.50, target_se=0.03, target_n=299,
            donor_rate=0.20, donor_se=0.01,
            performance_goal=0.45,
        )
        assert res.map_weight < 0.5, "mixture should escape the conflicting registry prior"
        assert res.rate_posterior > 0.45, "posterior should revert to the data, not the registry"
        assert res.concludes_below_pg is False
