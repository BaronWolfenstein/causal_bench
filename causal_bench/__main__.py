import argparse
import sys
from pathlib import Path

from causal_bench.dgp.scenarios import get_scenario, list_scenarios
from causal_bench.estimators import MVP_ESTIMATORS
from causal_bench.runner import run_simulation
from causal_bench.viz import plot_forest, generate_summary_table


def main():
    parser = argparse.ArgumentParser(
        prog="python -m causal_bench",
        description="Monte Carlo benchmarking of causal estimators for clinical trials",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available scenarios: {', '.join(list_scenarios())}",
    )
    parser.add_argument("--scenario", default="edwards_realistic",
                        help="Named DGP scenario (default: edwards_realistic)")
    parser.add_argument("--n-sims", type=int, default=100,
                        help="Number of Monte Carlo replicates (default: 100)")
    parser.add_argument("--n-jobs", type=int, default=-1,
                        help="Parallel workers, -1 = all CPUs (default: -1)")
    parser.add_argument("--estimand", default="ATE", choices=["ATE", "ATT"],
                        help="Target estimand (default: ATE)")
    parser.add_argument("--estimators", nargs="+", default=MVP_ESTIMATORS,
                        help="Estimator keys to run")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="results",
                        help="Output directory (default: results/)")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip plot generation")
    args = parser.parse_args()

    print(f"\ncausal_bench")
    print(f"  scenario  : {args.scenario}")
    print(f"  n_sims    : {args.n_sims}")
    print(f"  estimand  : {args.estimand}")
    print(f"  estimators: {args.estimators}")
    print()

    try:
        config = get_scenario(args.scenario)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    results = run_simulation(
        config,
        estimator_names=args.estimators,
        n_sim=args.n_sims,
        n_jobs=args.n_jobs,
        seed=args.seed,
        estimand=args.estimand,
    )

    if not results:
        print("No results produced. Check estimator names.", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out_dir) / args.scenario
    out_dir.mkdir(parents=True, exist_ok=True)

    table = generate_summary_table(results)
    print("\n── Results ──────────────────────────────────────────")
    print(table)
    (out_dir / "summary.md").write_text(table)
    print(f"\nSaved: {out_dir}/summary.md")

    if not args.no_plots:
        forest_path = str(out_dir / "forest.png")
        plot_forest(results, title=f"{args.scenario} | {args.estimand} | n={args.n_sims}",
                    save_path=forest_path)
        print(f"Saved: {forest_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
