"""Exp 2: Positivity violation gradient.

Sweeps positivity_severity 0→3. Key story:
- IPW weight variance explodes → coverage collapses
- Overlap weighting stays stable (targets ATO, different estimand)
- TMLE degrades but less dramatically than IPW
- Naive/KM unaffected (don't use propensity)

Demonstrates why positivity matters and why overlap weighting is robust.
"""
from pathlib import Path

from causal_bench.dgp.config import DGPConfig
from causal_bench.runner import run_parameter_sweep
from causal_bench.viz import plot_panel, generate_summary_table

PARAM_VALUES = [0.0, 0.6, 1.2, 1.8, 2.4, 3.0]
ESTIMATORS = ["naive", "km", "cox", "ipw", "overlap", "aipw", "tmle_ipcw", "tmle_ipcw_comply"]
OUT_DIR = Path("results/exp2_positivity")
N_SIMS = 200  # increase to 500 for publication


def run(n_sims: int = N_SIMS, n_jobs: int = -1, seed: int = 42):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = DGPConfig(n=500, censoring_informativeness=0.0, positivity_severity=0.0,
                     true_tau=-0.5)

    print(f"Exp 2: positivity gradient | n_sims={n_sims} | estimators={ESTIMATORS}")
    results = run_parameter_sweep(
        base_config=base,
        param_name="positivity_severity",
        param_values=PARAM_VALUES,
        estimator_names=ESTIMATORS,
        n_sim=n_sims,
        n_jobs=n_jobs,
        seed=seed,
    )

    plot_panel(
        results,
        param_values=PARAM_VALUES,
        param_name="Positivity severity",
        title="Exp 2: Estimator performance vs positivity violations",
        save_path=str(OUT_DIR / "panel.png"),
    )
    print(f"Saved panel → {OUT_DIR}/panel.png")

    for i, val in enumerate(PARAM_VALUES):
        slice_r = {name: lst[i] for name, lst in results.items() if lst[i] is not None}
        if slice_r:
            tbl = generate_summary_table(slice_r)
            (OUT_DIR / f"summary_pos{val:.1f}.md").write_text(tbl)

    last = {name: lst[-1] for name, lst in results.items() if lst[-1] is not None}
    if last:
        print("\n── Results at positivity_severity=3.0 ──────────────────")
        print(generate_summary_table(last))

    return results


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Exp 2: Positivity violation gradient")
    p.add_argument("--n-sims", type=int, default=N_SIMS)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--seed",   type=int, default=42)
    args = p.parse_args()
    run(n_sims=args.n_sims, n_jobs=args.n_jobs, seed=args.seed)
