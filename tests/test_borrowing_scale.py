"""Tests for cloglog borrowing scale — estimand→borrowing handoff (issue #43).

The KM 1-yr composite is a cumulative incidence F ∈ [0,1], not naturally Normal.
Borrowing is performed on the complementary log-log scale θ = cloglog(F) =
log(-log(1-F)), where the Normal-Normal model and the size-calibrated cutoff are
well-founded. `s` in the cutoff is the delta-method SE of θ on that scale.
"""
import numpy as np
import pytest

from causal_bench.estimators.hierarchical import to_cloglog, from_cloglog


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
