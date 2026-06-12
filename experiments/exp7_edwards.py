"""Exp 7: Edwards combined — full estimator benchmark across 3 scenarios.

The "money experiment": demonstrates the full estimator hierarchy
under realistic, optimistic, and pessimistic clinical trial conditions.

Two panels are run per scenario:

  Panel A — pointwise risk-difference estimators evaluated against the
  true risk difference at the horizon (compute_true_effects).

  Panel B — RMST estimators (concrete_RMST, rmst_k*) evaluated against
  the true RMST difference (compute_true_rmst).  Mixing these two
  estimands produces spurious large negative biases and is incorrect;
  see GitHub issue #1 for background.
"""
from pathlib import Path
import numpy as np
from causal_bench.dgp.scenarios import get_scenario
from causal_bench.dgp.survival import compute_true_rmst
from causal_bench.estimators import ESTIMATOR_REGISTRY
from causal_bench.runner import run_simulation
from causal_bench.metrics import SimResult
from causal_bench.viz import generate_summary_table, plot_forest

SCENARIOS = ["edwards_optimistic", "edwards_realistic", "edwards_pessimistic"]
_RMST_KEYS = {"concrete_RMST"} | {f"rmst_k{k}" for k in (2, 5, 10, 20)}
_WR_KEYS        = {k for k in ESTIMATOR_REGISTRY if k.startswith("concrete_WR")}
# Panel A: pointwise RD estimators only
RD_ESTIMATORS   = [k for k in ESTIMATOR_REGISTRY if k not in _RMST_KEYS | _WR_KEYS | {"cox_l1"}]
# Panel B: RMST estimators only (concrete_RMST skipped if R unavailable)
RMST_ESTIMATORS = [k for k in ESTIMATOR_REGISTRY if k in _RMST_KEYS]
OUT_DIR = Path("results/exp7_edwards")
N_SIMS = 200  # increase to 1000 for publication


def run(n_sims: int = N_SIMS, n_jobs: int = -1, seed: int = 42):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_rd_results   = {}   # {scenario_name: {estimator_name: SimResult}}
    all_rmst_results = {}

    for scenario_name in SCENARIOS:
        cfg = get_scenario(scenario_name)
        print(f"\nRunning {scenario_name} | n={cfg.n} | n_sims={n_sims}")

        # ── Panel A: risk-difference estimators ──────────────────────────
        rd_results = run_simulation(
            dgp_config=cfg,
            estimator_names=RD_ESTIMATORS,
            n_sim=n_sims,
            n_jobs=n_jobs,
            seed=seed,
            horizon=cfg.horizon,
        )
        all_rd_results[scenario_name] = rd_results
        tbl = generate_summary_table(rd_results)
        (OUT_DIR / f"summary_{scenario_name}.md").write_text(tbl)
        print(f"  Saved summary_{scenario_name}.md")

        # ── Panel B: RMST estimators against true RMST benchmark ─────────
        if RMST_ESTIMATORS:
            print(f"  Computing true RMST for {scenario_name}...")
            true_rmst = compute_true_rmst(cfg)["ATE"]
            rmst_results = run_simulation(
                dgp_config=cfg,
                estimator_names=RMST_ESTIMATORS,
                n_sim=n_sims,
                n_jobs=n_jobs,
                seed=seed,
                horizon=cfg.horizon,
                true_value=true_rmst,
            )
            all_rmst_results[scenario_name] = rmst_results
            tbl_rmst = generate_summary_table(rmst_results)
            (OUT_DIR / f"summary_rmst_{scenario_name}.md").write_text(tbl_rmst)
            print(f"  Saved summary_rmst_{scenario_name}.md")

    # Forest plot on the realistic RD results (primary panel)
    realistic_results = all_rd_results["edwards_realistic"]
    forest_path = str(OUT_DIR / "forest_realistic.png")
    plot_forest(realistic_results, save_path=forest_path)
    print(f"\nSaved forest plot → {forest_path}")

    # Cross-scenario bias table — Panel A
    print("\n" + "="*60)
    print("CROSS-SCENARIO BIAS SUMMARY — Panel A (risk difference)")
    print("="*60)
    header = f"{'Estimator':<20} {'Optimistic':>12} {'Realistic':>12} {'Pessimistic':>12}"
    print(header)
    print("-"*60)
    for est in RD_ESTIMATORS:
        biases = []
        for sc in SCENARIOS:
            r = all_rd_results[sc].get(est)
            biases.append(f"{r.bias:+.3f}" if r is not None else "  N/A  ")
        print(f"{est:<20} {biases[0]:>12} {biases[1]:>12} {biases[2]:>12}")

    # Cross-scenario bias table — Panel B
    if all_rmst_results:
        print("\n" + "="*60)
        print("CROSS-SCENARIO BIAS SUMMARY — Panel B (RMST difference)")
        print("="*60)
        print(header)
        print("-"*60)
        for est in RMST_ESTIMATORS:
            biases = []
            for sc in SCENARIOS:
                r = all_rmst_results.get(sc, {}).get(est)
                biases.append(f"{r.bias:+.3f}" if r is not None else "  N/A  ")
            print(f"{est:<20} {biases[0]:>12} {biases[1]:>12} {biases[2]:>12}")

    return {"rd": all_rd_results, "rmst": all_rmst_results}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Exp 7: Edwards combined benchmark")
    p.add_argument("--n-sims", type=int, default=N_SIMS)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    run(n_sims=args.n_sims, n_jobs=args.n_jobs, seed=args.seed)
