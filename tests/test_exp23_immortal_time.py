"""Immortal-time bias (#21/exp23). Load-bearing: the effect is a true null, the
naive time-zero-at-eligibility contrast shows a large spurious protective effect,
covariate adjustment does NOT remove it (estimator-proof), and only the landmark
design fix recovers the null."""
import numpy as np

from causal_bench.dgp.immortal_time import (
    ImmortalTimeConfig, draw_immortal_time, naive_risk_difference,
    adjusted_effect, landmark_risk_difference)
from experiments.exp23_immortal_time import run


def test_naive_shows_spurious_protective_effect():
    r = run(n=4000, reps=200)
    assert r["true_effect"] == 0.0
    assert r["naive_immortal"] < -0.15          # large spurious 'device protective'


def test_adjustment_does_not_fix_it_estimator_proof():
    """The bias is mis-aligned time-zero, not confounding — adjusting for the
    real confounder X leaves it essentially intact."""
    r = run(n=4000, reps=200)
    assert r["adjusted_for_X"] < -0.15          # still badly biased
    # adjustment removes < half the bias (it is NOT a confounding problem)
    assert abs(r["adjusted_for_X"]) > 0.5 * abs(r["naive_immortal"])


def test_landmark_design_fix_recovers_null():
    r = run(n=4000, reps=200)
    assert abs(r["landmark_design_fix"]) < 0.03           # ~ true null
    assert abs(r["landmark_design_fix"]) < 0.3 * abs(r["naive_immortal"])


def test_true_effect_stays_null_with_landmark_across_configs():
    """Even with a stronger confounder / different implant timing, the landmark
    recovers ~null (the device truly does nothing)."""
    cfg = ImmortalTimeConfig(beta_x=1.0, implant_rate=1.2)
    lm = [landmark_risk_difference(draw_immortal_time(4000, s, cfg), cfg) for s in range(150)]
    assert abs(np.mean(lm)) < 0.04
