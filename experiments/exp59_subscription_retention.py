"""Exp59: monthly retention hazards, aggregation equivalence and future cohorts.

Run: python -m experiments.exp59_subscription_retention --bootstrap-reps 100
CPU only; numpy/scipy/pandas. No R, concrete, GPU or private subscription data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import scipy

from causal_bench.validation.subscription_retention import run


def report(result):
    selected = result["selected_using_validation_only"]
    lines = [
        "# Exp59 — monthly subscription retention",
        "",
        f"Synthetic data; n={result['configuration']['n']}, seed={result['configuration']['seed']}, "
        f"horizon={result['configuration']['horizon']} months.",
        "",
        "Acquisition cohorts: train 0–23, validation 36–41, test 54–59; as-of cutoffs 36/54/72. "
        "Maturation gaps ensure train labels are available before validation entry and validation labels before test entry. "
        "Subscribers are split before expansion.",
        f"Specification selected on validation person-month log loss only: **{selected}**. "
        "Test metrics do not choose the model or calibration method; no post-hoc calibrator is fitted.",
        "",
        "| Model | Validation log loss | Test curve RMSE vs oracle | Test restricted months | Oracle months | Test IPCW Brier at horizon |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, partitions in result["metrics"].items():
        m = partitions["test"]
        lines.append(f"| {name} | {result['validation_selection_log_loss'][name]:.6f} | "
                     f"{m['survival_rmse_vs_oracle']:.6f} | {m['restricted_active_months']:.4f} | "
                     f"{m['oracle_restricted_active_months']:.4f} | {m['ipcw_brier_by_month_known_G'][-1]:.6f} |")
    e = result["aggregation_equivalence"]
    lines += ["", "## Numerical checks", "",
              f"TRAIN: {e['person_months']} person-months → {e['aggregated_cells']} cells for the selected specification.",
              f"Maximum row/cell coefficient error: {e['max_coefficient_error']:.3g}; "
              f"covariance error: {e['max_covariance_error']:.3g}; "
              f"log-likelihood error per observation: {e['log_likelihood_error_per_observation']:.3g}.",
              "",
              "Cell aggregation preserves the complete design vector, including age, covariates, "
              "calendar seasonality and the optional age interaction. No binning of continuous covariates, "
              "regularization or class balancing is involved. Likelihoods omit combinatorial constants.",
              "", "## Uncertainty and limits", "",
              "Reported coefficient SEs are model-based inverse-information SEs, not subscriber/account-clustered robust SEs."]
    b = result["subscriber_bootstrap"]
    lines.append(f"Training-subscriber bootstrap: {b['successful']}/{b['requested']} successful; "
                 f"{b['failed']} failed. Selected specification and test covariates are fixed; selection uncertainty is excluded.")
    if b["restricted_active_months_percentile_95"] is not None:
        lo, hi = b["restricted_active_months_percentile_95"]
        lines.append(f"Selected model restricted-month forecast percentile 95% interval: [{lo:.4f}, {hi:.4f}]. "
                     "This is not a Monte Carlo coverage study.")
    else:
        lines.append("No bootstrap interval reported (requires at least 20 successful resamples).")
    lines += ["",
              "IPCW Brier/calibration use the simulation's KNOWN censoring survival; real data requires an estimated "
              "and validated censoring model. Complete outcomes and expected survival are separate oracle evaluation data, "
              "never fitting inputs. Loss to observation is independent of cancellation conditional on baseline covariates.",
              "",
              "All acquisition cohorts can reach the 12-month administrative horizon. This benchmark does not establish "
              "identification beyond available follow-up, performance under latent informative censoring, or transport "
              "through a new pricing/campaign regime. Calendar effects are periodic, not unrestricted age/cohort/period effects.",
              "",
              "Cancellation is at month ends: restricted active months = sum of S(k), k=0..H−1. "
              "This is restricted lifetime, not extrapolated lifetime or LTV. Covariates are acquisition-time values; "
              "no future engagement is used. Churn prediction is not the causal benefit of a retention offer. "
              "Exp51/54 remain the separate intervention extensions.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=24000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--bootstrap-reps", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("results/exp59_subscription_retention"))
    args = parser.parse_args()
    result = run(n=args.n, seed=args.seed, horizon=args.horizon, bootstrap_reps=args.bootstrap_reps)
    root = Path(__file__).resolve().parents[1]
    sources = [Path(__file__).resolve(), root / "causal_bench/estimators/monthly_hazard.py",
               root / "causal_bench/validation/subscription_retention.py"]
    result["provenance"] = {
        "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
        "pandas": pd.__version__,
        "source_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    (args.output / "REPORT.md").write_text(report(result))
    print(report(result))
    print(f"Executed monthly retention benchmark: {args.output / 'REPORT.md'}")


if __name__ == "__main__":
    main()
