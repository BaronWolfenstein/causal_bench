"""Exp 4: Crossover (treatment-switching) gradient.

Sweeps crossover_rate 0→0.3 with moderate informative crossover
(crossover_informativeness=0.5: sicker controls cross to treatment).

Key story:
- Naive/KM: as-treated analysis — crossover dilutes apparent treatment effect
  (ITT attenuation), bias grows with crossover rate
- Cox: similar attenuation under informative crossover
- TMLE+IPCW: censors at crossover time, partially recovers the per-protocol
  effect by down-weighting censored crossovers
- TMLE+IPCW+Comply: compliance score predicts crossover, yields further gain

Demonstrates the IPCW advantage when treatment switching is informative.
"""
from pathlib import Path

from causal_bench.dgp.config import DGPConfig
from causal_bench.runner import run_parameter_sweep
from causal_bench.viz import plot_panel, generate_summary_table

PARAM_VALUES = [0.0, 0.06, 0.12, 0.18, 0.24, 0.30]
ESTIMATORS = ["naive", "km", "cox", "tmle_ipcw", "tmle_ipcw_comply"]
OUT_DIR = Path("results/exp4_crossover")
N_SIMS = 200  # increase to 500 for publication


def run(n_sims: int = N_SIMS, n_jobs: int = -1, seed: int = 42):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Informative crossover: sicker patients (higher hazard) cross over
    base = DGPConfig(n=500, censoring_informativeness=0.3, censoring_rate=0.25,
                     crossover_rate=0.0, crossover_informativeness=0.5, true_tau=-0.5)

    print(f"Exp 4: crossover gradient | n_sims={n_sims} | estimators={ESTIMATORS}")
    results = run_parameter_sweep(
        base_config=base,
        param_name="crossover_rate",
        param_values=PARAM_VALUES,
        estimator_names=ESTIMATORS,
        n_sim=n_sims,
        n_jobs=n_jobs,
        seed=seed,
    )

    plot_panel(
        results,
        param_values=PARAM_VALUES,
        param_name="Crossover rate",
        title="Exp 4: Estimator performance vs treatment crossover rate",
        save_path=str(OUT_DIR / "panel.png"),
    )
    print(f"Saved panel → {OUT_DIR}/panel.png")

    for i, val in enumerate(PARAM_VALUES):
        slice_r = {name: lst[i] for name, lst in results.items() if lst[i] is not None}
        if slice_r:
            tbl = generate_summary_table(slice_r)
            (OUT_DIR / f"summary_cr{val:.2f}.md").write_text(tbl)

    last = {name: lst[-1] for name, lst in results.items() if lst[-1] is not None}
    if last:
        print("\n── Results at crossover_rate=0.30 ──────────────────")
        print(generate_summary_table(last))

    parquet_dir = OUT_DIR / "parquet"
    parquet_dir.mkdir(exist_ok=True)
    for name, sr_list in results.items():
        for i, val in enumerate(PARAM_VALUES):
            sr = sr_list[i]
            if sr is not None:
                sr.to_parquet(parquet_dir / f"{name}_{val:g}.parquet")
    print(f"Saved Parquet files → {parquet_dir}/")

    return results


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Exp 4: Crossover gradient")
    p.add_argument("--n-sims", type=int, default=N_SIMS)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--seed",   type=int, default=42)
    args = p.parse_args()
    run(n_sims=args.n_sims, n_jobs=args.n_jobs, seed=args.seed)
