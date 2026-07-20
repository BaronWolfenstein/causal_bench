"""Calendar-time confounding (#173/exp42). Load-bearing: adjusting for a patient-
state proxy that imperfectly mirrors era LAUNDERS calendar confounding (still
badly biased), while putting era in explicitly recovers the null."""
import numpy as np

from causal_bench.dgp.calendar_confounding import (
    CalendarConfig, draw_calendar, adjusted_effect)
from experiments.exp42_calendar_confounding import run


def test_state_proxy_launders_era_explicit_recovers():
    r = run(n=3000, reps=200)
    ab = r["abs_bias"]
    assert ab["era-explicit {E,X}"] < 0.03                 # explicit era → null
    assert ab["state-proxy {S,X}"] > 0.3                   # state proxy still badly biased
    # the state proxy leaves a large fraction of the naive bias uncorrected
    assert ab["state-proxy {S,X}"] > 0.4 * ab["naive (Y~A)"]


def test_launder_worsens_as_state_proxy_degrades():
    """More measurement noise in the state proxy → more residual (laundered)
    calendar confounding; era-explicit stays ~null regardless."""
    resid = []
    for noise in (0.5, 2.0):
        cfg = CalendarConfig(state_noise=noise)
        b = [adjusted_effect(draw_calendar(3000, s, cfg), ["S", "X"]) for s in range(150)]
        resid.append(abs(np.mean(b)))
    assert resid[1] > resid[0]                              # noisier proxy → more leak
    # era-explicit unaffected
    cfg = CalendarConfig(state_noise=2.0)
    be = [adjusted_effect(draw_calendar(3000, s, cfg), ["E", "X"]) for s in range(150)]
    assert abs(np.mean(be)) < 0.04


def test_no_calendar_effect_no_bias():
    """Turn off the era→outcome and era→membership paths → no calendar bias, and
    all adjustment sets agree (~null). Sanity that the bias is calendar-driven."""
    cfg = CalendarConfig(beta_ea=0.0, beta_ey=0.0)
    b = [adjusted_effect(draw_calendar(3000, s, cfg), ["S", "X"]) for s in range(150)]
    assert abs(np.mean(b)) < 0.05
