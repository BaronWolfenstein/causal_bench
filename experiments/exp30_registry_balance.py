"""Exp 30: registry-path balance table with HAL propensity and R-as-output.

The real-data-path counterpart to exp29. Where exp29 uses a logistic stand-in
and a hardcoded region R (sev >= 2.0), exp30 assembles the production seam:

    load_registry  ->  propensity_scores (HAL primary)  ->  region_r_from_positivity
                   ->  odds_weights       ->  balance_panel (reused from exp29)

- **Loader:** `synthetic_registry` is the benchmark path; swap in `load_registry`
  on a real registry export (same schema) and nothing downstream changes.
- **Propensity:** HAL is the primary nuisance learner (its rate licenses the
  downstream doubly-robust inference); method='logistic' is the fast fallback.
- **Region R is an OUTPUT** of the positivity map (the propensity tail where
  Baseline ESS drops below a floor), not a covariate cutoff — so it transfers to
  real covariates where no single threshold defines the sparse region.

The balance/ESS diagnostics (global vs region-R SMDs, deep-R ESS map, region
Love plots) are reused verbatim from exp29 — exp30 only changes how the cohort,
weights, and R are produced, so the same edge-fill guards apply.

Sigma_x seam (NOT built here — issue #58): the anatomical covariates entering
`propensity_scores` are measured with error. The covariate measurement-error
arm (oracle / naive / corrected via regression calibration or an EIV-GP, with a
cutoff-aware kernel at an eligibility boundary) plugs in exactly at the
`propensity_scores` call: it re-fits the propensity on error-corrected inputs
and re-runs the same balance_panel. That arm is a distinct, decision-relevant
experiment; this module provides the seam it attaches to.
"""
from pathlib import Path

import pandas as pd

from causal_bench.dgp.sca_registry import REGISTRY_COVS, load_registry, synthetic_registry
from causal_bench.sca_weighting import (
    odds_weights, propensity_scores, region_r_from_positivity)
from experiments.exp29_balance_diagnostics import balance_panel

OUT_DIR = Path("results/exp30_registry_balance")
COVS = list(REGISTRY_COVS)


def run_registry_balance(source=None, method: str = "logistic", seed: int = 20260702,
                         ess_floor: float = 40.0):
    """Balance table + ESS map + region-R map for the registry path.

    source=None uses the synthetic benchmark registry; pass a DataFrame or path
    to run on a real registry export (validated through `load_registry`).
    method='hal' uses the production propensity; 'logistic' is the fast path.
    """
    df = load_registry(synthetic_registry(seed) if source is None else source)
    target = df[df.group == "target"].reset_index(drop=True)
    baseline = df[df.group == "baseline"].reset_index(drop=True)

    e_t, e_b = propensity_scores(target, baseline, COVS, method=method, seed=seed)
    w = odds_weights(e_b)
    region = region_r_from_positivity(target, baseline, e_t, e_b, ess_floor=ess_floor)

    # feed the discovered R into the shared balance panel via a `sev`-independent
    # region column, then reuse exp29's balance_panel by aliasing R on X5-free cols
    tbl, meta = _balance_with_region(target, baseline, w, COVS, region)
    return tbl, meta, region


def _balance_with_region(target, baseline, w, covs, region):
    """Per-covariate SMD / variance-ratio globally and within the discovered R,
    plus the region-resolved ESS map. Mirrors exp29.balance_panel but keys the
    region split on the positivity-map output rather than a `sev` cutoff."""
    import numpy as np

    from experiments.exp29_balance_diagnostics import kish_ess, smd, variance_ratio

    in_r_t = region["in_R_target"]
    in_r_b = region["in_R_baseline"]
    rows = []
    for c in covs:
        rows.append({
            "covariate": c,
            "smd_pre": smd(target[c], baseline[c]),
            "smd_post": smd(target[c], baseline[c], w),
            "vr_post": variance_ratio(target[c], baseline[c], w),
            "smd_post_R": smd(target.loc[in_r_t, c], baseline.loc[in_r_b, c], w[in_r_b])
                          if in_r_b.sum() >= 5 and in_r_t.sum() >= 5 else float("nan"),
        })
    meta = {
        "q_star": region["q_star"],
        "n_baseline_R": region["n_baseline_R"], "n_target_R": region["n_target_R"],
        "ess_global": region["ess_baseline_global"], "ess_R": region["ess_baseline_R"],
        "max_w": float(w.max()), "pct_w_gt10": float((w > 10).mean() * 100),
    }
    return pd.DataFrame(rows), meta


def run(method: str = "logistic", seed: int = 20260702):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tbl, meta, region = run_registry_balance(method=method, seed=seed)
    tbl.to_parquet(OUT_DIR / "balance_table.parquet", index=False)
    pd.set_option("display.float_format", lambda v: f"{v:0.3f}")
    print(tbl.to_string(index=False))
    print(meta)
    return tbl, meta


if __name__ == "__main__":
    run()
