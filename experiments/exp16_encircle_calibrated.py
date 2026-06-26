"""Exp 16: ENCIRCLE-calibrated replication — 14 estimators vs published marginals.

Generates synthetic data calibrated to published 1-year ENCIRCLE marginals
(device arm, n=299; Guerrero, Daniels, Makkar et al., Lancet 2025,
doi:10.1016/S0140-6736(25)02073-2):
  composite 25.2% (device) / 45% (pre-specified performance goal)
  mortality 13.9%, HF hospitalization 16.7%, overlap ~5.4%
  ~19% missing at 1-year visit

Pre-specified estimand (SAP Section C, Lancet supplement):
  Non-hierarchical composite of death + HF rehospitalization at 1 year.
  Primary analysis: KM 1-year composite rate, 95% CI via Wald test with
  Greenwood's formula. Hypothesis: H0: π ≥ 45% vs HA: π < 45%, one-sided
  α = 0.025. Power assumption: true rate 35%, 10% attrition, N = 299.
  "Non-hierarchical" is stated explicitly twice in the SAP — this is NOT a
  prioritised/win-statistic composite and NOT a time-to-first-event hazard
  model.

SCA framing:
  The pre-specified primary analysis is a performance-goal (PG) test against
  the 45% historical rate. The synthetic control arm (SCA) augments this
  design by constructing an external comparator from the TVT Registry,
  replacing the fixed PG with a data-derived comparator. The SCA is layered
  on top of — not replacing — the PG framing. This experiment reports both
  the PG test result and the external-comparator ATE so the two can be
  compared directly.

TVT Registry / TEER SAP note:
  The TEER SAP (Makkar et al., JAMA 2023, doi:10.1001/jama.2023.7089) is
  used as a template for comparator-arm construction conventions (GEE for
  site clustering, backward selection, MI). Its estimand — 30-day MR success
  via GEE logistic — is NOT ENCIRCLE's and is not imported here.

Pre-specified subgroups (SAP Section C — exactly two):
  1. Subject sex (female vs male)
  2. MR etiology (functional vs degenerative)
  The DGP uses W1–W4 as synthetic proxies; W1 approximates a continuous
  severity axis that could be dichotomised to mirror the sex/etiology split,
  but the experiment does not claim to replicate subgroup-level marginals.
  Any subgroup analysis in this experiment uses the synthetic proxies and
  notes this limitation explicitly.

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
from scipy.stats import norm

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
_PUBLISHED_CONTROL_COMPOSITE = 0.45   # pre-specified performance goal (SAP Section C)
_PG_ALPHA = 0.025                     # one-sided α per SAP


def pg_test(
    km_rate: float,
    greenwood_se: float,
    pg: float = _PUBLISHED_CONTROL_COMPOSITE,
    alpha: float = _PG_ALPHA,
) -> dict:
    """One-sided Wald test of KM 1-year composite rate against a performance goal.

    Implements ENCIRCLE's pre-specified primary analysis (SAP Section C):
      H0: π ≥ pg   vs   HA: π < pg,  one-sided α = 0.025

    Parameters
    ----------
    km_rate : float
        KM estimate of 1-year composite event rate (device arm).
    greenwood_se : float
        Greenwood standard error of the KM rate.
    pg : float
        Performance goal (default 0.45 per ENCIRCLE SAP).
    alpha : float
        One-sided significance level (default 0.025 per ENCIRCLE SAP).

    Returns
    -------
    dict with keys: z, p_value, rejects_h0, ci_lower, ci_upper, pg, alpha.
    """
    z = (km_rate - pg) / max(greenwood_se, 1e-9)
    p = float(norm.cdf(z))
    z_crit = norm.ppf(1.0 - alpha)
    return {
        "km_rate":    km_rate,
        "greenwood_se": greenwood_se,
        "z":          z,
        "p_value":    p,
        "rejects_h0": p < alpha,
        "ci_lower":   km_rate - z_crit * greenwood_se,
        "ci_upper":   km_rate + z_crit * greenwood_se,
        "pg":         pg,
        "alpha":      alpha,
    }


def _km_rate_and_greenwood_se(
    T_obs: np.ndarray,
    Delta: np.ndarray,
    horizon: float,
) -> tuple[float, float]:
    """KM 1-year composite event rate with Greenwood SE at a given horizon."""
    order = np.argsort(T_obs)
    t = T_obs[order]
    d = Delta[order].astype(float)
    n_total = len(t)

    S = 1.0
    greenwood_sum = 0.0
    at_risk = n_total
    i = 0

    while i < n_total and t[i] <= horizon:
        t_i = t[i]
        j = i
        events = 0
        while j < n_total and t[j] == t_i:
            if d[j] == 1:
                events += 1
            j += 1
        if events > 0 and at_risk > events:
            S *= (at_risk - events) / at_risk
            greenwood_sum += events / (at_risk * (at_risk - events))
        at_risk -= (j - i)
        i = j

    return float(1.0 - S), float(S * np.sqrt(greenwood_sum))


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
          f"| censoring_informativeness={cfg.censoring.informativeness}")
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

    # Performance-goal test on one synthetic dataset (illustrative — not averaged
    # over replicates; the SCA external-comparator ATE is the replicate-averaged
    # quantity above).  The two estimands should agree in direction: device arm
    # rate < 45% PG, and ATE = device − comparator < 0.
    from causal_bench.dgp.survival import generate_data
    df_demo = generate_data(cfg.with_overrides(seed=0))
    device_only = df_demo[df_demo["A"] == 1]
    km_rate, gw_se = _km_rate_and_greenwood_se(
        device_only["T_obs"].values,
        device_only["Delta"].values,
        horizon=cfg.horizon,
    )
    pg_result = pg_test(km_rate, gw_se)
    print(f"\n── Performance-goal test (one synthetic dataset, seed=0) ────────────")
    print(f"  KM 1-yr composite rate: {pg_result['km_rate']:.3f}  (PG = {pg_result['pg']})")
    print(f"  Greenwood SE:           {pg_result['greenwood_se']:.4f}")
    print(f"  z = {pg_result['z']:.3f},  p = {pg_result['p_value']:.4f}  "
          f"({'REJECT H0' if pg_result['rejects_h0'] else 'fail to reject H0'})")
    print(f"  95% CI: [{pg_result['ci_lower']:.3f}, {pg_result['ci_upper']:.3f}]")
    print(f"  External-comparator ATE (replicate mean): {true_ate:.3f}")
    print(f"  Agreement check: both estimands point {'same direction ✓' if (pg_result['rejects_h0'] and true_ate < 0) or (not pg_result['rejects_h0'] and true_ate >= 0) else 'DIFFERENT directions — investigate'}")

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
