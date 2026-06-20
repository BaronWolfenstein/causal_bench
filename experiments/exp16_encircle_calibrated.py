"""Exp 16: ENCIRCLE-calibrated replication — 14 estimators vs published marginals.

Generates synthetic data calibrated to published 1-year ENCIRCLE marginals
(device arm, n=299; Feldman et al. / Edwards Lifesciences):
  composite 25.2% (device) / ~45% (historical performance goal)
  mortality 13.9%, HF hospitalization 16.7%, overlap ~5.4%
  ~19% missing at 1-year visit

Runs all 14 Python estimators (no R/concrete required) and checks which
recover the true ATE under ENCIRCLE-like informative censoring and mild
positivity violations. This bridges the abstract violation experiments
(Exp 1-6) to the real application:

  Exp 1-6: "here's how bias behaves as censoring informativeness increases"
  Exp 16:  "here's what that means for ENCIRCLE specifically"

McCoy's TRISCEND II calibration script (concrete PR #36, commit 58bc77f) is
the template: generate synthetic data matching published marginals without
patient-level data, then check if the estimator recovers the published result.
The LLC extends this by comparing 14 estimators instead of one.

DGP validation (n=100k reference):
  device:  composite≈0.257, HFH≈0.166, death≈0.090
  control: composite≈0.465, ATE≈−0.144
"""
from pathlib import Path

import numpy as np
import pandas as pd

from causal_bench.dgp.scenarios import get_scenario
from causal_bench.dgp.survival import compute_true_effects
from causal_bench.estimators import ESTIMATOR_REGISTRY, MVP_ESTIMATORS
from causal_bench.runner import run_simulation
from causal_bench.viz import generate_summary_table, plot_forest

OUT_DIR = Path("results/exp16_encircle_calibrated")
N_SIMS = 200  # increase to 500 for publication

# Python-only estimators (no R bridge required)
ENCIRCLE_ESTIMATORS = [
    "naive", "km", "cox", "ipw", "aipw", "overlap",
    "tmle_ipcw", "tmle_ipcw_comply", "tmle_ipcw_boot", "tmle_ipcw_cv",
    "tmle_ipcw_cv_comply", "ltmle",
    "rmst_k5", "rmst_k10",
]

# Published ENCIRCLE 1-year composite rates (device vs performance goal)
_PUBLISHED_DEVICE_COMPOSITE = 0.252
_PUBLISHED_CONTROL_COMPOSITE = 0.45   # historical performance goal


def _report_calibration(cfg) -> None:
    """Print DGP marginals vs published targets for transparency."""
    from causal_bench.dgp.survival import generate_data
    df = generate_data(cfg.with_overrides(n=100_000))
    print("── DGP calibration check (n=100k) ─────────────────────────────────")
    print(f"  Published targets:  device comp=0.252, HFH=0.167, death=0.085")
    print(f"                      control comp≈0.450")
    for a, arm in [(1, "device"), (0, "control")]:
        sub = df[df.A == a]
        comp  = (sub.event_type > 0).mean()
        hfh   = (sub.event_type == 1).mean()
        death = (sub.event_type == 2).mean()
        print(f"  DGP {arm:8s}:  comp={comp:.3f}, HFH={hfh:.3f}, death={death:.3f}")
    print()


def run(n_sims: int = N_SIMS, n_jobs: int = -1, seed: int = 42) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cfg = get_scenario("encircle_calibrated")
    true_effects = compute_true_effects(cfg)
    true_ate = true_effects["ATE"]

    print(f"Exp 16: ENCIRCLE-calibrated replication | n={cfg.n} | n_sims={n_sims}")
    print(f"  horizon={cfg.horizon} | censoring_rate={cfg.censoring_rate} "
          f"| censoring_informativeness={cfg.censoring_informativeness}")
    print(f"  true ATE (DGP): {true_ate:.3f}  "
          f"(published device−control: {_PUBLISHED_DEVICE_COMPOSITE - _PUBLISHED_CONTROL_COMPOSITE:.3f})")

    _report_calibration(cfg)

    available = [e for e in ENCIRCLE_ESTIMATORS if e in ESTIMATOR_REGISTRY]
    print(f"  estimators: {available}")

    results = run_simulation(
        dgp_config=cfg,
        estimator_names=available,
        n_sim=n_sims,
        n_jobs=n_jobs,
        seed=seed,
        horizon=cfg.horizon,
        estimand="ATE",
        true_value=true_ate,
    )

    results = {k: v for k, v in results.items() if v is not None}

    tbl = generate_summary_table(results)
    (OUT_DIR / "summary.md").write_text(tbl)
    print(f"\n── Results (true ATE = {true_ate:.3f}) ──────────────────────────────────")
    print(tbl)

    # Annotate which estimators are within 1 SE of published point estimate
    published_ate = _PUBLISHED_DEVICE_COMPOSITE - _PUBLISHED_CONTROL_COMPOSITE
    print(f"\n── Recovery of published ATE {published_ate:.3f} ─────────────────────────")
    for name, sr in sorted(results.items(), key=lambda x: abs(x[1].bias)):
        bias = sr.bias
        within = abs(bias - (published_ate - true_ate)) < 2 * sr.rmse if sr.rmse > 0 else False
        print(f"  {name:30s}: bias={bias:+.3f}  RMSE={sr.rmse:.3f}  cov={sr.coverage:.2f}")

    forest_path = str(OUT_DIR / "forest.png")
    plot_forest(results, save_path=forest_path)
    print(f"\nSaved forest  → {forest_path}")
    print(f"Saved summary → {OUT_DIR}/summary.md")

    parquet_dir = OUT_DIR / "parquet"
    parquet_dir.mkdir(exist_ok=True)
    for name, sr in results.items():
        sr.to_parquet(parquet_dir / f"{name}.parquet")
    print(f"Saved Parquet → {parquet_dir}/")

    return results


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Exp 16: ENCIRCLE-calibrated replication")
    p.add_argument("--n-sims", type=int, default=N_SIMS)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--seed",   type=int, default=42)
    args = p.parse_args()
    run(n_sims=args.n_sims, n_jobs=args.n_jobs, seed=args.seed)
