"""Tests for SCA propensity weighting + region-R as an output of the positivity map."""
import numpy as np
import pytest

from causal_bench.dgp.sca_registry import REGISTRY_COVS, synthetic_registry
from causal_bench.hal import _hal9001_available
from causal_bench.sca_weighting import (
    odds_weights, propensity_scores, region_r_from_positivity)

COVS = list(REGISTRY_COVS)


def _split(seed=1):
    df = synthetic_registry(seed=seed)
    return df[df.group == "target"], df[df.group == "baseline"]


def test_propensity_separates_target_from_baseline():
    t, b = _split()
    et, eb = propensity_scores(t, b, COVS, method="logistic")
    assert len(et) == len(t) and len(eb) == len(b)
    assert (et >= 0).all() and (et <= 1).all()
    # Target records score higher P(target) than Baseline on average
    assert et.mean() > eb.mean()


def test_odds_weights_normalized_and_positive():
    t, b = _split()
    _, eb = propensity_scores(t, b, COVS, method="logistic")
    w = odds_weights(eb)
    assert (w > 0).all()
    assert abs(w.mean() - 1.0) < 1e-9


def test_region_r_recovered_from_positivity_map_not_hardcoded():
    # R must be an OUTPUT of the fitted propensity/positivity map: the sparse
    # region where Baseline support is thin — which for this DGP coincides with
    # the severity upper tail, but is discovered, not assumed.
    t, b = _split()
    et, eb = propensity_scores(t, b, COVS, method="logistic")
    r = region_r_from_positivity(t, b, et, eb, ess_floor=40.0)
    assert r["in_R_target"].dtype == bool and r["in_R_baseline"].dtype == bool
    # R is enriched for high severity (discovered, not cut on `sev`)
    assert t.loc[r["in_R_target"], "sev"].mean() > t["sev"].mean() + 0.5
    # and it captures most of the genuinely sparse tail
    true_tail = (t["sev"] >= 2.0).to_numpy()
    captured = r["in_R_target"][true_tail].mean()
    assert captured > 0.6
    # Baseline ESS inside R is thin (that's what makes it R)
    assert r["ess_baseline_R"] < r["ess_baseline_global"]


@pytest.mark.skipif(not _hal9001_available(), reason="hal9001 not installed")
def test_hal_propensity_path_runs_and_separates():
    # small cohort — this is a wiring/separation check on the production HAL
    # learner, not a scaling benchmark (HAL cost grows fast in n)
    df = synthetic_registry(seed=7, n_target=80, n_baseline=300)
    t, b = df[df.group == "target"], df[df.group == "baseline"]
    et, eb = propensity_scores(t, b, COVS, method="hal")
    assert et.mean() > eb.mean()
    w = odds_weights(eb)
    assert abs(w.mean() - 1.0) < 1e-9
