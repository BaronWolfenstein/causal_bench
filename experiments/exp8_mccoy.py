"""Exp 8: McCoy experiment — RMST vs pointwise risk difference.

Reproduces the core finding from McCoy (2026): pointwise estimators at
coarse time grids accumulate bias that is eliminated by direct RMST targeting.

Compares:
  - TMLE+IPCW (pointwise at t=horizon, standard approach)
  - AIPW       (doubly-robust, no censoring model)
  - LTMLE      (correct for time-varying confounders)
  - concrete_RMST (direct RMST targeting — requires R + concrete package)

Uses the competing_risks_base scenario so event_type ∈ {0, 1, 2},
matching the data format concrete expects.

If concrete is not available, the script runs the Python estimators only
and leaves the concrete_RMST column as N/A.
"""
from pathlib import Path
import warnings

import numpy as np

from causal_bench.dgp.scenarios import get_scenario
from causal_bench.estimators import ESTIMATOR_REGISTRY
from causal_bench.runner import run_simulation
from causal_bench.viz import generate_summary_table, plot_forest

ESTIMATORS = ["tmle_ipcw", "aipw", "ltmle", "concrete_RMST"]
OUT_DIR = Path("results/exp8_mccoy")
N_SIMS = 200  # increase to 500 for publication


def run(n_sims: int = N_SIMS, n_jobs: int = -1, seed: int = 42):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cfg = get_scenario("competing_risks_base")

    # concrete_RMST is in ESTIMATOR_REGISTRY but returns [] if R unavailable;
    # the runner handles empty results gracefully (SimResult is None for that estimator)
    available = [e for e in ESTIMATORS if e in ESTIMATOR_REGISTRY]
    missing   = set(ESTIMATORS) - set(available)
    if missing:
        warnings.warn(f"Exp 8: estimators not in registry, skipping: {missing}")

    print(f"Exp 8: McCoy RMST experiment | scenario=competing_risks_base "
          f"| n={cfg.n} | n_sims={n_sims}")
    print(f"  estimators: {available}")

    results = run_simulation(
        dgp_config=cfg,
        estimator_names=available,
        n_sim=n_sims,
        n_jobs=n_jobs,
        seed=seed,
        horizon=cfg.horizon,
    )

    # Drop None results (concrete unavailable)
    results = {k: v for k, v in results.items() if v is not None}

    if not results:
        print("No results — all estimators failed or unavailable.")
        return {}

    # Summary table
    tbl = generate_summary_table(results)
    (OUT_DIR / "summary.md").write_text(tbl)
    print(f"\nSaved summary → {OUT_DIR}/summary.md")

    # Forest plot
    forest_path = str(OUT_DIR / "forest.png")
    plot_forest(results, save_path=forest_path)
    print(f"Saved forest → {forest_path}")

    # Save each SimResult as Parquet for cross-session comparison
    parquet_dir = OUT_DIR / "parquet"
    parquet_dir.mkdir(exist_ok=True)
    for name, sr in results.items():
        sr.to_parquet(parquet_dir / f"{name}.parquet")
    print(f"Saved Parquet files → {parquet_dir}/")

    print("\n── Results ──────────────────────────────────────────────")
    print(tbl)

    return results


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Exp 8: McCoy RMST experiment")
    p.add_argument("--n-sims",  type=int, default=N_SIMS)
    p.add_argument("--n-jobs",  type=int, default=-1)
    p.add_argument("--seed",    type=int, default=42)
    args = p.parse_args()
    run(n_sims=args.n_sims, n_jobs=args.n_jobs, seed=args.seed)
