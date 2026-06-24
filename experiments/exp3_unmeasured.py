"""Exp 3: Unmeasured confounding gradient — THE HONESTY EXPERIMENT.

Sweeps unmeasured_confounding_strength 0→0.8. Key story:
- ALL estimators show increasing bias as U grows
- No estimator is immune — this is a fundamental identification failure
- TMLE/AIPW remain doubly-robust against measured confounders but cannot
  correct for what they cannot observe
- Negative control outcome bias tracks primary outcome bias, confirming U as
  the source (residual confounding signal)

Demonstrates that causal methods cannot rescue unmeasured confounding.
"""
from pathlib import Path

from causal_bench.dgp.config import DGPConfig
from causal_bench.runner import run_parameter_sweep
from causal_bench.viz import plot_panel, generate_summary_table

PARAM_VALUES = [0.0, 0.16, 0.32, 0.48, 0.64, 0.8]
ESTIMATORS = ["naive", "km", "cox", "ipw", "overlap", "aipw", "tmle_ipcw", "tmle_ipcw_comply"]
OUT_DIR = Path("results/exp3_unmeasured")
N_SIMS = 200  # increase to 500 for publication


def run(n_sims: int = N_SIMS, n_jobs: int = -1, seed: int = 42):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = DGPConfig(n=500,
                     unmeasured_confounding_strength=0.0, true_tau=-0.5)

    print(f"Exp 3: unmeasured confounding gradient | n_sims={n_sims} | estimators={ESTIMATORS}")
    results = run_parameter_sweep(
        base_config=base,
        param_name="unmeasured_confounding_strength",
        param_values=PARAM_VALUES,
        estimator_names=ESTIMATORS,
        n_sim=n_sims,
        n_jobs=n_jobs,
        seed=seed,
    )

    plot_panel(
        results,
        param_values=PARAM_VALUES,
        param_name="Unmeasured confounding strength",
        title="Exp 3: All estimators fail under unmeasured confounding",
        save_path=str(OUT_DIR / "panel.png"),
    )
    print(f"Saved panel → {OUT_DIR}/panel.png")

    for i, val in enumerate(PARAM_VALUES):
        slice_r = {name: lst[i] for name, lst in results.items() if lst[i] is not None}
        if slice_r:
            tbl = generate_summary_table(slice_r)
            (OUT_DIR / f"summary_uc{val:.2f}.md").write_text(tbl)

    last = {name: lst[-1] for name, lst in results.items() if lst[-1] is not None}
    if last:
        print("\n── Results at unmeasured_confounding_strength=0.8 ──────────────────")
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
    p = argparse.ArgumentParser(description="Exp 3: Unmeasured confounding gradient")
    p.add_argument("--n-sims", type=int, default=N_SIMS)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--seed",   type=int, default=42)
    args = p.parse_args()
    run(n_sims=args.n_sims, n_jobs=args.n_jobs, seed=args.seed)
