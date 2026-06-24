"""Exp 1: Censoring informativeness gradient.

Sweeps censoring_informativeness 0→1 with all 5 MVP estimators.
Expected: Naive/KM degrade monotonically. TMLE+IPCW stays flat.
TMLE+IPCW+Comply is best at high informativeness.
"""
from pathlib import Path

from causal_bench.dgp.config import CovariateDependentCensoringConfig
from causal_bench.dgp.scenarios import get_scenario
from causal_bench.estimators import MVP_ESTIMATORS
from causal_bench.runner import run_parameter_sweep
from causal_bench.viz import plot_panel, generate_summary_table

PARAM_VALUES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
N_SIMS = 200        # increase to 500 for publication
OUT_DIR = Path("results/exp1_censoring")


def run(n_sims: int = N_SIMS, n_jobs: int = -1, seed: int = 42):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = get_scenario("censor_moderate")   # censoring_rate=0.30 baseline

    print(f"Exp 1: censoring gradient | n_sims={n_sims} | estimators={MVP_ESTIMATORS}")
    results = run_parameter_sweep(
        base_config=base,
        param_name="censoring",
        param_values=[CovariateDependentCensoringConfig(informativeness=v) for v in PARAM_VALUES],
        estimator_names=MVP_ESTIMATORS,
        n_sim=n_sims,
        n_jobs=n_jobs,
        seed=seed,
    )

    fig = plot_panel(
        results,
        param_values=PARAM_VALUES,
        param_name="Censoring informativeness",
        title="Exp 1: Estimator performance vs censoring informativeness",
        save_path=str(OUT_DIR / "panel.png"),
    )
    print(f"Saved panel → {OUT_DIR}/panel.png")

    # Summary table for each param value
    for i, val in enumerate(PARAM_VALUES):
        slice_r = {name: lst[i] for name, lst in results.items() if lst[i] is not None}
        if slice_r:
            tbl = generate_summary_table(slice_r)
            (OUT_DIR / f"summary_inf{val:.1f}.md").write_text(tbl)

    # Print final value summary to stdout
    last = {name: lst[-1] for name, lst in results.items() if lst[-1] is not None}
    if last:
        print("\n── Results at informativeness=1.0 ──────────────────")
        print(generate_summary_table(last))

    return results


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n-sims", type=int, default=N_SIMS)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--seed",   type=int, default=42)
    args = p.parse_args()
    run(n_sims=args.n_sims, n_jobs=args.n_jobs, seed=args.seed)
