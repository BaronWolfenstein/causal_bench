"""Propensity guards (#173/#174 → #99): instrument screen, era-contamination
check + residualize, and their wiring into sca_weighting.propensity_scores."""
import numpy as np
import pandas as pd

from causal_bench.propensity_guards import (
    outcome_adaptive_screen, era_contamination, residualize_era)
from causal_bench.sca_weighting import propensity_scores


def _instrument_frame(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    conf = rng.standard_normal(n)          # confounder: affects membership AND outcome
    instr = rng.standard_normal(n)         # instrument: membership only, NOT outcome
    Y = 1.5 * conf + rng.standard_normal(n)
    return conf, instr, Y, rng


def test_screen_drops_instrument_keeps_confounder():
    conf, instr, Y, _ = _instrument_frame()
    X = np.column_stack([conf, instr])
    keep = outcome_adaptive_screen(X, Y, ["conf", "instr"])
    assert keep == ["conf"]


def test_era_contamination_detected_and_residualized():
    rng = np.random.default_rng(1)
    n = 2000
    era = rng.standard_normal(n)
    state = rng.standard_normal((n, 5))
    contam = np.column_stack([state, era + 0.3 * rng.standard_normal(n)])
    clean = np.column_stack([state, rng.standard_normal(n)])
    assert era_contamination(contam, era)["contaminated"]
    assert not era_contamination(clean, era)["contaminated"]
    # remedy: residualizing removes the leakage
    assert not era_contamination(residualize_era(contam, era), era)["contaminated"]


def test_propensity_screen_instruments_flag():
    """sca_weighting.propensity_scores with screen_instruments drops the instrument
    covariate (fit on fewer covs) and still returns valid probabilities."""
    conf, instr, Y, rng = _instrument_frame(n=1500)
    # membership driven by BOTH conf and instr (instr is a strong membership predictor)
    member = (rng.standard_normal(1500) + 1.2 * instr + 0.8 * conf > 0)
    df = pd.DataFrame({"conf": conf, "instr": instr, "Y": Y, "member": member})
    tgt, base = df[df.member], df[~df.member]
    outcome = np.r_[tgt["Y"].to_numpy(), base["Y"].to_numpy()]
    et, eb = propensity_scores(tgt, base, ["conf", "instr"], method="logistic",
                               outcome=outcome, screen_instruments=True)
    assert et.shape[0] == len(tgt) and eb.shape[0] == len(base)
    assert np.all((et > 0) & (et < 1))


def test_propensity_era_explicit_column():
    conf, instr, Y, rng = _instrument_frame(n=1500)
    member = (rng.standard_normal(1500) + conf > 0)
    era = rng.standard_normal(1500)
    df = pd.DataFrame({"conf": conf, "member": member})
    tgt, base = df[df.member], df[~df.member]
    era_pooled = np.r_[era[df.member.to_numpy()], era[~df.member.to_numpy()]]
    et, eb = propensity_scores(tgt, base, ["conf"], method="logistic", era=era_pooled)
    assert et.shape[0] == len(tgt) and np.all((eb > 0) & (eb < 1))


def test_screen_requires_outcome():
    df = pd.DataFrame({"conf": [0.0, 1.0], "member": [True, False]})
    try:
        propensity_scores(df[df.member], df[~df.member], ["conf"],
                          method="logistic", screen_instruments=True)
        assert False, "should have raised without outcome"
    except ValueError:
        pass
