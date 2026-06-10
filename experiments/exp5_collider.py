"""Exp 5: Collider trap — time-varying confounder gradient.

Sweeps collider_strength 0→1. Key story:
- Cox (no L1) and Cox+L1 show opposite-direction biases at high collider_strength
- LTMLE stays near zero (correct marginalization)
- TMLE+IPCW shows partial bias from L1 omission
"""
from pathlib import Path
from causal_bench.dgp.config import DGPConfig
from causal_bench.runner import run_parameter_sweep
from causal_bench.viz import plot_panel, plot_collider_panel, generate_summary_table

PARAM_VALUES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
ESTIMATORS = ["cox", "cox_l1", "ltmle", "tmle_ipcw"]
OUT_DIR = Path("results/exp5_collider")
N_SIMS = 200


def run(n_sims: int = N_SIMS, n_jobs: int = -1, seed: int = 42):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = DGPConfig(n=500, censoring_informativeness=0.3, true_tau=-0.5,
                     collider_strength=0.0)  # collider_strength swept below

    print(f"Exp 5: collider trap | n_sims={n_sims} | estimators={ESTIMATORS}")
    results = run_parameter_sweep(
        base_config=base,
        param_name="collider_strength",
        param_values=PARAM_VALUES,
        estimator_names=ESTIMATORS,
        n_sim=n_sims,
        n_jobs=n_jobs,
        seed=seed,
    )

    plot_panel(
        results,
        param_values=PARAM_VALUES,
        param_name="Collider strength",
        title="Exp 5: Collider trap — Cox vs Cox+L1 vs LTMLE",
        save_path=str(OUT_DIR / "panel.png"),
    )
    print(f"Saved panel → {OUT_DIR}/panel.png")

    plot_collider_panel(
        results,
        param_values=PARAM_VALUES,
        save_path=str(OUT_DIR / "collider_panel.png"),
    )
    print(f"Saved collider panel → {OUT_DIR}/collider_panel.png")

    # Summary tables per param value
    for i, val in enumerate(PARAM_VALUES):
        slice_r = {name: lst[i] for name, lst in results.items() if lst[i] is not None}
        if slice_r:
            tbl = generate_summary_table(slice_r)
            (OUT_DIR / f"summary_cs{val:.1f}.md").write_text(tbl)

    # Print final value summary
    last = {name: lst[-1] for name, lst in results.items() if lst[-1] is not None}
    if last:
        print("\n── Results at collider_strength=1.0 ──────────────────")
        print(generate_summary_table(last))

    return results


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n-sims", type=int, default=N_SIMS)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    run(n_sims=args.n_sims, n_jobs=args.n_jobs, seed=args.seed)
