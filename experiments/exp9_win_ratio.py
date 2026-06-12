"""Exp 9: Win ratio benchmark — direct TMLE vs plug-in via concrete PR #30.

Reproduces the core finding from McCoy PR #30: targetWinRatio() (direct TMLE)
cuts WR bias ~5x vs getWinRatio() (plug-in) by solving the win/loss EIF
estimating equations jointly rather than plugging targeted risk curves into the
win functional.

Estimand: win ratio = P(T_treated > T_control) / P(T_treated < T_control)
True value: computed via U-statistic on 50k potential-outcome pairs.

Uses competing_risks_base scenario (event_type in {0, 1, 2}), which is the
data format targetWinRatio() expects.

If concrete is unavailable, the script exits with a clear message.
"""
from pathlib import Path
import warnings

import numpy as np

from causal_bench.dgp.scenarios import get_scenario
from causal_bench.dgp.survival import compute_true_win_ratio
from causal_bench.estimators import ESTIMATOR_REGISTRY
from causal_bench.runner import run_simulation
from causal_bench.viz import generate_summary_table, plot_forest

ESTIMATORS = ["concrete_WR_direct", "concrete_WR_plugin"]
OUT_DIR = Path("results/exp9_win_ratio")
N_SIMS = 200  # increase to 500 for publication


def run(n_sims: int = N_SIMS, n_jobs: int = -1, seed: int = 42):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cfg = get_scenario("competing_risks_base")

    available = [e for e in ESTIMATORS if e in ESTIMATOR_REGISTRY]
    missing   = set(ESTIMATORS) - set(available)
    if missing:
        warnings.warn(f"Exp 9: estimators not in registry, skipping: {missing}")
    if not available:
        print("No estimators available — is concrete installed?")
        return {}

    print("Computing true win ratio...", flush=True)
    wr_true_dict = compute_true_win_ratio(cfg)
    wr_true = wr_true_dict["ATE"]
    print(f"  True WR (ATE)    = {wr_true:.4f}")
    print(f"  P(win)           = {wr_true_dict['p_win']:.4f}")
    print(f"  P(loss)          = {wr_true_dict['p_loss']:.4f}")
    print(f"  Net benefit      = {wr_true_dict['net_benefit']:.4f}")

    print(f"\nExp 9: Win ratio | scenario=competing_risks_base "
          f"| n={cfg.n} | n_sims={n_sims}")
    print(f"  estimators: {available}")

    results = run_simulation(
        dgp_config=cfg,
        estimator_names=available,
        n_sim=n_sims,
        n_jobs=n_jobs,
        seed=seed,
        horizon=cfg.horizon,
        estimand="WR",
        true_value=wr_true,
    )

    results = {k: v for k, v in results.items() if v is not None}
    if not results:
        print("No results — all estimators failed or unavailable.")
        return {}

    tbl = generate_summary_table(results)
    (OUT_DIR / "summary.md").write_text(tbl)
    print(f"\nSaved summary → {OUT_DIR}/summary.md")

    forest_path = str(OUT_DIR / "forest.png")
    plot_forest(results, save_path=forest_path)
    print(f"Saved forest → {forest_path}")

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
    p = argparse.ArgumentParser(description="Exp 9: Win ratio benchmark")
    p.add_argument("--n-sims", type=int, default=N_SIMS)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--seed",   type=int, default=42)
    args = p.parse_args()
    run(n_sims=args.n_sims, n_jobs=args.n_jobs, seed=args.seed)
