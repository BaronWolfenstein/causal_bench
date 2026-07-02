"""Tests for the cobalt balance cross-check bridge (package-time, guarded).

rpy2 / the cobalt R package are optional. When absent, the bridge must skip
gracefully (return None with a warning) rather than raise — matching the repo's
concrete-bridge convention. The live-R path is exercised only where rpy2+cobalt
are installed.
"""
import numpy as np
import pandas as pd
import pytest

from causal_bench.diagnostics.cobalt import _cobalt_available, cobalt_baltab
from experiments.exp29_balance_diagnostics import draw_cohorts, fit_odds_weights


def _inputs():
    target, baseline = draw_cohorts(seed=20260702)
    w_b = fit_odds_weights(target, baseline)
    covs = ["X1", "X2", "X3", "X4", "X5"]
    weights = np.r_[np.ones(len(target)), w_b]
    return target, baseline, weights, covs


def test_skips_gracefully_without_cobalt():
    if _cobalt_available():
        pytest.skip("cobalt available — exercised by the live-R test instead")
    target, baseline, weights, covs = _inputs()
    with pytest.warns(UserWarning, match="cobalt"):
        assert cobalt_baltab(target, baseline, weights, covs) is None


@pytest.mark.skipif(not _cobalt_available(), reason="rpy2 + cobalt not installed")
def test_baltab_runs_when_cobalt_present():
    target, baseline, weights, covs = _inputs()
    out = cobalt_baltab(target, baseline, weights, covs)
    assert out is not None
    assert "covariate" in out.columns
    assert len(out) >= len(covs)
