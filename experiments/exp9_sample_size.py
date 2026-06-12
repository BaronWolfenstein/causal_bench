"""Exp 9: Sample size sensitivity (asymptotic regime analysis).

Sweeps n from 100 to 2000 using edwards_realistic parameters — the hardest
realistic scenario. Key story:
- Naive/KM: bias is invariant to n (no asymptotic rescue for misspecification)
- Cox/IPW: bias stable, variance shrinks
- TMLE+IPCW: approaches near-unbiasedness only at n ≥ 700; at n=100 the
  Super Learner has too little data to fit nuisance models well
- TMLE+IPCW+Comply: needs the most data; gains only visible at n ≥ 700

Shows where asymptotic semiparametric guarantees kick in for ENCIRCLE-scale
trials (n≈700), and that smaller pilot studies cannot rely on TMLE efficiency.
"""
from pathlib import Path

from causal_bench.dgp.scenarios import get_scenario
from causal_bench.estimators import MVP_ESTIMATORS
from causal_bench.runner import run_parameter_sweep
from causal_bench.viz import plot_panel, generate_summary_table

PARAM_VALUES = [100, 400, 700, 1000, 1500, 2000]
OUT_DIR = Path("results/exp9_sample_size")
N_SIMS = 200  # increase to 500 for publication


def run(n_sims: int = N_SIMS, n_jobs: int = -1, seed: int = 42):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = get_scenario("edwards_realistic")

    print(f"Exp 9: sample size sensitivity | n_sims={n_sims} | estimators={MVP_ESTIMATORS}")
    results = run_parameter_sweep(
        base_config=base,
        param_name="n",
        param_values=PARAM_VALUES,
        estimator_names=MVP_ESTIMATORS,
        n_sim=n_sims,
        n_jobs=n_jobs,
        seed=seed,
    )

    plot_panel(
        results,
        param_values=PARAM_VALUES,
        param_name="Sample size (n)",
        title="Exp 9: Sample size sensitivity — asymptotic regime for ENCIRCLE (n=700)",
        save_path=str(OUT_DIR / "panel.png"),
    )
    print(f"Saved panel → {OUT_DIR}/panel.png")

    for i, val in enumerate(PARAM_VALUES):
        slice_r = {name: lst[i] for name, lst in results.items() if lst[i] is not None}
        if slice_r:
            tbl = generate_summary_table(slice_r)
            (OUT_DIR / f"summary_n{val}.md").write_text(tbl)

    last = {name: lst[-1] for name, lst in results.items() if lst[-1] is not None}
    if last:
        print("\n── Results at n=2000 ──────────────────")
        print(generate_summary_table(last))

    parquet_dir = OUT_DIR / "parquet"
    parquet_dir.mkdir(exist_ok=True)
    for name, sr_list in results.items():
        for i, val in enumerate(PARAM_VALUES):
            sr = sr_list[i]
            if sr is not None:
                sr.to_parquet(parquet_dir / f"{name}_{val}.parquet")
    print(f"Saved Parquet files → {parquet_dir}/")

    return results


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Exp 9: Sample size sensitivity")
    p.add_argument("--n-sims", type=int, default=N_SIMS)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--seed",   type=int, default=42)
    args = p.parse_args()
    run(n_sims=args.n_sims, n_jobs=args.n_jobs, seed=args.seed)
