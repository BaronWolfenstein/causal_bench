"""Self-validating checks for exp49 (QBA for unmeasured confounding).

Pins the three claims: (1) the oracle recovers tau; (2) correct priors recover the
effect and the total-error interval covers; (3) misspecified priors undercover.
"""
from causal_bench.validation.qba_confounding import (
    naive_oracle, simulate, calibrate, compare_v1_v2)


def test_oracle_recovers_naive_biased():
    no = naive_oracle(simulate(3000, conf=1.2, seed=0))
    assert abs(no["oracle"] - 1.0) < 0.10        # oracle (U measured) recovers the effect
    assert no["naive"] - 1.0 > 0.15               # naive (U omitted) is biased


def test_correct_priors_recover_and_cover():
    r = calibrate(n=1200, n_reps=80, mode="correct", seed=0)
    assert abs(r["adjusted_bias"]) < 0.08         # QBA with correct priors recovers tau
    assert r["coverage_total"] > 0.90             # total-error interval covers


def test_misspecification_undercovers():
    r = calibrate(n=1200, n_reps=80, mode="misspec_null", seed=0)
    assert r["coverage_total"] < 0.25             # assuming U harmless -> poor coverage
    assert r["adjusted_bias"] > 0.15              # stays near the naive bias


def test_v2_flex_recovers_on_nonlinear():
    # v2: on a nonlinear-in-X outcome, flexible base + IF-one-step OVB recovers
    r = compare_v1_v2(n=1000, n_reps=4, seed=0)
    assert r["flex_naive_bias"] > 0.20            # naive (U omitted) is biased
    assert abs(r["flex_v2adjusted_bias"]) < 0.20  # v2-adjusted recovers
