"""Exp 11: Stratified randomization SE correction (concrete PR #29 benchmark).

Demonstrates McCoy's Bugni-Canay-Shaikh / Ye-Shao variance correction for
covariate-adaptive (stratified block) randomization.

DGP: stratified_base — 4 strata (W2 × W4), block size 4. Treatment is
assigned by permuted blocks within strata, not Bernoulli.

Estimators compared:
  - concrete_RMST           iid SE (ignores strata → conservative)
  - concrete_RMST_strata    BCS-corrected SE (concrete PR #29, Strata arg)
  - tmle_ipcw               no strata correction available (baseline)

Expected findings:
  - Bias: similar across all three (stratification improves balance, not bias)
  - SE ratio (median_SE / empirical_SE):
      iid SE > 1.0       (over-covers, wastes power)
      BCS SE ≈ 1.0       (correct coverage)
      tmle_ipcw ≈ some   (EIF-based, unrelated to strata)
  - Coverage:
      All ≥ 0.95, but BCS SE gives tighter CIs
  - Practical width:
      BCS CIs narrower than iid — quantifies the power gain from
      stratification acknowledgment

If R/concrete is unavailable, only tmle_ipcw runs and the script prints
a skip notice instead of failing.
"""
from pathlib import Path
import warnings

from causal_bench.dgp.scenarios import get_scenario
from causal_bench.estimators import ESTIMATOR_REGISTRY
from causal_bench.runner import run_simulation
from causal_bench.viz import generate_summary_table, plot_forest

ESTIMATORS = ["concrete_RMST", "concrete_RMST_strata", "tmle_ipcw"]
OUT_DIR = Path("results/exp11_strata")
N_SIMS = 200  # increase to 500 for publication


def run(n_sims: int = N_SIMS, n_jobs: int = -1, seed: int = 42):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cfg = get_scenario("stratified_base")

    available = [e for e in ESTIMATORS if e in ESTIMATOR_REGISTRY]
    missing = set(ESTIMATORS) - set(available)
    if missing:
        warnings.warn(f"Exp 11: estimators not in registry, skipping: {missing}")

    concrete_present = any(e.startswith("concrete_RMST") for e in available)
    if not concrete_present:
        print("Exp 11: concrete_RMST not available — R/concrete not installed. "
              "Running tmle_ipcw only; SE correction comparison will be incomplete.")

    print(f"Exp 11: stratified SE correction | scenario=stratified_base "
          f"| n={cfg.n} | n_sims={n_sims}")
    print(f"  estimators: {available}")

    results = run_simulation(
        dgp_config=cfg,
        estimator_names=available,
        n_sim=n_sims,
        n_jobs=n_jobs,
        seed=seed,
        horizon=cfg.horizon,
    )

    results = {k: v for k, v in results.items() if v is not None}

    if not results:
        print("No results — all estimators failed or unavailable.")
        return {}

    tbl = generate_summary_table(results)
    (OUT_DIR / "summary.md").write_text(tbl)
    print(f"\nSaved summary → {OUT_DIR}/summary.md")

    forest_path = str(OUT_DIR / "forest.png")
    plot_forest(results, title="Exp 11: Stratified SE correction | iid vs BCS",
                save_path=forest_path)
    print(f"Saved forest → {forest_path}")

    # SE ratio comparison table
    if len(results) >= 2:
        print("\n── SE ratio (median_SE / empirical_SE) — key metric ────────────────")
        for name, sr in results.items():
            if sr is not None:
                se_ratio = float(sr.se_ratio) if hasattr(sr, "se_ratio") else float("nan")
                print(f"  {name:<30} se_ratio={se_ratio:.3f}  "
                      f"coverage={sr.coverage:.3f}  bias={sr.bias:.4f}")

    parquet_dir = OUT_DIR / "parquet"
    parquet_dir.mkdir(exist_ok=True)
    for name, sr in results.items():
        sr.to_parquet(parquet_dir / f"{name}.parquet")

    print("\n── Results ──────────────────────────────────────────────")
    print(tbl)

    return results


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Exp 11: Stratified randomization SE correction")
    p.add_argument("--n-sims",  type=int, default=N_SIMS)
    p.add_argument("--n-jobs",  type=int, default=-1)
    p.add_argument("--seed",    type=int, default=42)
    args = p.parse_args()
    run(n_sims=args.n_sims, n_jobs=args.n_jobs, seed=args.seed)
