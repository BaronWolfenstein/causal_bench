"""Tests for exp30 — registry-path balance with HAL propensity and R-as-output."""
import numpy as np

from causal_bench.dgp.sca_registry import synthetic_registry
from experiments.exp30_registry_balance import run_registry_balance


def test_registry_path_balances_globally_and_exposes_region_R():
    tbl, meta, region = run_registry_balance(method="logistic", seed=20260702)
    # global balance is achievable...
    assert (tbl["smd_post"].abs() < 0.1).all()
    # ...but region R is discovered from the positivity map and is a real subset
    assert 0.0 < region["q_star"] < 1.0
    assert meta["n_target_R"] > 0 and meta["n_baseline_R"] > 0
    # Baseline support is thin inside the discovered R (that is what defines it)
    assert meta["ess_R"] < meta["ess_global"]
    # the discovered R is enriched for severity — recovered, not cut on `sev`
    df = synthetic_registry(20260702)
    t = df[df.group == "target"].reset_index(drop=True)
    assert t.loc[region["in_R_target"], "sev"].mean() > t["sev"].mean()


def test_load_path_accepts_external_frame_unchanged():
    # a caller-supplied conforming frame flows through the same pipeline
    df = synthetic_registry(seed=11)
    tbl, meta, region = run_registry_balance(source=df, method="logistic", seed=11)
    assert set(tbl["covariate"]) == {"x_a", "x_b", "x_skew", "x_bin", "sev"}
    assert np.isfinite(meta["ess_global"])
