"""Charter-validation tests for ClinicalPSNBEstimator (no R/concrete needed — these
exercise __init__ only). Estimand semantics follow McCoy et al., arXiv:2607.22950:
PSNB = Σ_k α_k Δ_k with a charter α that is non-negative and sums to 1 (Def 1), on the
illness-death (2-tier) mapping."""
import pytest

from causal_bench.estimators.concrete_psnb import ClinicalPSNBEstimator


def test_valid_two_tier_charter_accepted():
    est = ClinicalPSNBEstimator(charter=(0.3, 0.7))
    assert est._charter == (0.3, 0.7)
    assert ClinicalPSNBEstimator().name == "clinical_PSNB"   # default (0.5, 0.5)


def test_charter_must_sum_to_one():
    with pytest.raises(ValueError, match="sum to 1"):
        ClinicalPSNBEstimator(charter=(0.5, 0.6))


def test_charter_must_be_two_tiers():
    # The illness-death mapping is 2-tier; a length-3 charter (even summing to 1) must
    # fail fast here, not deep inside the R bridge which hardcodes n_tiers=2.
    with pytest.raises(ValueError, match="2 weights"):
        ClinicalPSNBEstimator(charter=(0.3, 0.3, 0.4))


def test_charter_must_be_nonnegative():
    with pytest.raises(ValueError, match="non-negative"):
        ClinicalPSNBEstimator(charter=(-0.1, 1.1))
