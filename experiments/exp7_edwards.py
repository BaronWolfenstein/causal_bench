"""Exp 7: Edwards combined — all estimators across 3 scenarios.

The "money experiment": demonstrates the full estimator hierarchy
under realistic, optimistic, and pessimistic clinical trial conditions.
Uses all available estimators so the audience sees the complete picture.
"""
from pathlib import Path
import numpy as np
from causal_bench.dgp.scenarios import get_scenario
from causal_bench.estimators import ESTIMATOR_REGISTRY
from causal_bench.runner import run_simulation
from causal_bench.metrics import SimResult
from causal_bench.viz import generate_summary_table, plot_forest

SCENARIOS = ["edwards_optimistic", "edwards_realistic", "edwards_pessimistic"]
# All estimators except cox_l1 (collider-biased by design, misleading in this context)
ESTIMATORS = [k for k in ESTIMATOR_REGISTRY if k != "cox_l1"]
OUT_DIR = Path("results/exp7_edwards")
N_SIMS = 200  # increase to 1000 for publication


def run(n_sims: int = N_SIMS, n_jobs: int = -1, seed: int = 42):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {}   # {scenario_name: {estimator_name: SimResult}}

    for scenario_name in SCENARIOS:
        cfg = get_scenario(scenario_name)
        print(f"\nRunning {scenario_name} | n={cfg.n} | n_sims={n_sims}")
        results = run_simulation(
            dgp_config=cfg,
            estimator_names=ESTIMATORS,
            n_sim=n_sims,
            n_jobs=n_jobs,
            seed=seed,
            horizon=cfg.horizon,
        )
        all_results[scenario_name] = results

        # Summary table per scenario
        tbl = generate_summary_table(results)
        (OUT_DIR / f"summary_{scenario_name}.md").write_text(tbl)
        print(f"  Saved summary_{scenario_name}.md")

    # Forest plot: bias across scenarios for each estimator
    # For the forest plot we pick the "realistic" scenario as primary
    realistic_results = all_results["edwards_realistic"]
    forest_path = str(OUT_DIR / "forest_realistic.png")
    plot_forest(realistic_results, save_path=forest_path)
    print(f"\nSaved forest plot → {forest_path}")

    # Print comparison table across all 3 scenarios
    print("\n" + "="*60)
    print("CROSS-SCENARIO BIAS SUMMARY")
    print("="*60)
    header = f"{'Estimator':<20} {'Optimistic':>12} {'Realistic':>12} {'Pessimistic':>12}"
    print(header)
    print("-"*60)
    for est in ESTIMATORS:
        biases = []
        for sc in SCENARIOS:
            r = all_results[sc].get(est)
            biases.append(f"{r.bias:+.3f}" if r is not None else "  N/A  ")
        print(f"{est:<20} {biases[0]:>12} {biases[1]:>12} {biases[2]:>12}")

    return all_results


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Exp 7: Edwards combined benchmark")
    p.add_argument("--n-sims", type=int, default=N_SIMS)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    run(n_sims=args.n_sims, n_jobs=args.n_jobs, seed=args.seed)
