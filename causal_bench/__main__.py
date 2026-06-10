import argparse
import sys
from pathlib import Path

from causal_bench.dgp.scenarios import get_scenario, list_scenarios
from causal_bench.dgp.survival import generate_data
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
    parser.add_argument("--diagnostics", action="store_true",
                        help="Generate overlap, Love plot, and SE calibration diagnostics")
    parser.add_argument("--tipping-point", action="store_true",
                        help="Print tipping-point sensitivity table and save tipping_point.png")
    parser.add_argument("--ess", action="store_true",
                        help="Compute ESS distribution across 50 sim draws and save ess_distribution.png")
    parser.add_argument("--mnar-tipping-point", action="store_true",
                        help="MNAR sensitivity grid on a single dataset (no-op when censoring_informativeness=0)")
    parser.add_argument("--mnar-estimator", default="km",
                        help="Estimator for MNAR grid sweep (default: km; use tmle_ipcw for rigour)")
    parser.add_argument("--mnar-grid", type=int, default=10,
                        help="Grid points per axis for MNAR sweep (default: 10, total runs = n^2)")
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

    if args.diagnostics:
        from causal_bench.diagnostics import plot_overlap, plot_love, plot_se_calibration, se_calibration_table
        sample_df = generate_data(config)
        plot_overlap(sample_df, save_path=str(out_dir / "overlap.png"))
        plot_love(sample_df, save_path=str(out_dir / "love.png"))
        plot_se_calibration(results, save_path=str(out_dir / "se_calibration.png"))
        print("\n── SE calibration ──────────────────────────────────")
        print(se_calibration_table(results).to_string())

    if args.tipping_point:
        from causal_bench.diagnostics import tipping_point_table, plot_tipping_point
        plot_tipping_point(results, save_path=str(out_dir / "tipping_point.png"))
        print("\n── Tipping-point sensitivity ────────────────────────")
        print(tipping_point_table(results).to_string())

    if args.ess:
        from causal_bench.diagnostics import plot_ess_distribution, ess_across_sims
        ess_summary = ess_across_sims(config, n_draws=50, seed=args.seed)
        print(f"\n── ESS summary (50 draws) ────────────────────────────")
        print(f"  median ESS : {ess_summary['median_ess']:.1f}  ({ess_summary['ess_pct']:.1f}% of n={config.n})")
        print(f"  min / max  : {ess_summary['min_ess']:.1f} / {ess_summary['max_ess']:.1f}")
        plot_ess_distribution(config, n_draws=50, seed=args.seed,
                              save_path=str(out_dir / "ess_distribution.png"))
        print(f"  Saved ESS distribution → {out_dir}/ess_distribution.png")

    if args.mnar_tipping_point:
        from causal_bench.diagnostics import tipping_point_mnar, plot_tipping_point_mnar
        if config.censoring_informativeness == 0:
            print("\n── MNAR tipping-point: skipped (censoring_informativeness=0, all censoring is MCAR) ──")
        else:
            sample_df = generate_data(config)
            n_ct = int(((sample_df["A"] == 1) & (sample_df["Delta"] == 0) & (sample_df["T_obs"] < config.horizon - 1e-9)).sum())
            n_cc = int(((sample_df["A"] == 0) & (sample_df["Delta"] == 0) & (sample_df["T_obs"] < config.horizon - 1e-9)).sum())
            print(f"\n── MNAR tipping-point ({args.mnar_estimator}, grid={args.mnar_grid}×{args.mnar_grid}) ──")
            print(f"  informatively censored: {n_ct} treated, {n_cc} control")
            r_mnar = tipping_point_mnar(
                sample_df, args.mnar_estimator,
                horizon=config.horizon,
                n_grid=args.mnar_grid,
                seed=args.seed,
            )
            plot_tipping_point_mnar(r_mnar, save_path=str(out_dir / "mnar_tipping_point.png"))
            n_sig = r_mnar["significant"].sum()
            n_total = len(r_mnar)
            print(f"  significant in {n_sig}/{n_total} grid cells ({100*n_sig/n_total:.0f}%)")
            print(f"  MAR reference: p_treated={r_mnar.attrs['mar_p_treated']:.2f}, p_control={r_mnar.attrs['mar_p_control']:.2f}")
            print(f"  Saved → {out_dir}/mnar_tipping_point.png")

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
