"""Exp 13: Censoring-mechanism sweep — bias, coverage, EIC calibration, IPCW ESS.

Sweeps DGPConfig.censoring_mechanism x its informativeness/strength parameter:
  - "independent"          — C independent of W, A, T_true (pure random dropout)
  - "covariate_dependent"  — C depends on W, A (MAR); censoring_informativeness
                              also sweeps an MNAR-via-U component
  - "informative"          — log C* = beta0 + censoring_beta_T * T_true + eps
                              (MNAR: censoring directly depends on the
                              unobservable event time — IPCW conditional on
                              W, A alone cannot correct this)

Estimators: tmle_ipcw, tmle_ipcw_boot, aipw, concrete_simult (RD row only,
extracted at the DGP horizon — concrete_simult's other family members aren't
comparable to the ATE risk difference these estimators target).

True ATE depends only on the AFT/treatment model, not the censoring
mechanism, so it's computed once and reused across all grid cells.

Reports per (estimator, cell): bias, RMSE, coverage, se_ratio (EIC
calibration diagnostic — ratio of median reported SE to empirical SD of
estimates; near 1.0 means the EIC-based variance estimate is well-calibrated)
and a marginal (non-covariate-adjusted) IPCW effective sample size, which
demonstrates how informative censoring inflates weight variance even before
considering bias.

If concrete is unavailable, "concrete_simult" rows are silently dropped.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numpy.random import SeedSequence
from tqdm import tqdm

from causal_bench.dgp.config import DGPConfig
from causal_bench.dgp.scenarios import get_scenario
from causal_bench.dgp.survival import compute_true_effects, generate_data
from causal_bench.estimators import ESTIMATOR_REGISTRY, get_estimator
from causal_bench.estimators.concrete_simultaneous import ConcreteSimultaneousEstimator
from causal_bench.estimators.tmle_ipcw_boot import TMLEIPCWBootEstimator
from causal_bench.metrics import SimResult
from causal_bench.runner import SIM_TASK_TIMEOUT_SECONDS, build_validated_replicate_configs, classify_error
from pydantic import ValidationError

OUT_DIR = Path("results/exp13_censoring_sweep")
N_SIMS = 60  # increase for publication
HORIZON = 0.7
ESTIMATORS = ["tmle_ipcw", "tmle_ipcw_boot", "aipw", "concrete_simult"]
# Registry default (200 bootstrap reps) takes ~100s/call — far too slow for a
# 9-cell x N_SIMS sweep. Use a lighter bootstrap count here; still enough to
# see whether the bootstrap SE tracks coverage differently from the EIC SE
# under informative censoring.
BOOT_N_BOOTSTRAP = 30

GRID = [
    {"label": "independent",        "censoring_mechanism": "independent"},
    {"label": "covdep_info0.0",      "censoring_mechanism": "covariate_dependent", "censoring_informativeness": 0.0},
    {"label": "covdep_info0.3",      "censoring_mechanism": "covariate_dependent", "censoring_informativeness": 0.3},
    {"label": "covdep_info0.6",      "censoring_mechanism": "covariate_dependent", "censoring_informativeness": 0.6},
    {"label": "covdep_info0.9",      "censoring_mechanism": "covariate_dependent", "censoring_informativeness": 0.9},
    {"label": "informative_betaT-0.8", "censoring_mechanism": "informative", "censoring_beta_T": -0.8},
    {"label": "informative_betaT-0.4", "censoring_mechanism": "informative", "censoring_beta_T": -0.4},
    {"label": "informative_betaT+0.4", "censoring_mechanism": "informative", "censoring_beta_T": 0.4},
    {"label": "informative_betaT+0.8", "censoring_mechanism": "informative", "censoring_beta_T": 0.8},
]


def _ipcw_ess(df: pd.DataFrame, horizon: float) -> float:
    """Marginal (non-covariate-adjusted) IPCW effective sample size.

    Reverse-KM estimate of G(t) = P(C > t) from the censoring indicator,
    weight events by 1/G(T_i-), then ESS = (sum w)^2 / sum(w^2). A coarse
    diagnostic — deliberately ignores covariates so it isolates how the
    censoring *mechanism* alone inflates weight variance.
    """
    from lifelines import KaplanMeierFitter

    T_obs = df["T_obs"].values
    Delta = df["Delta"].values.astype(float)
    pre_horizon_dropout = (Delta == 0) & (T_obs < horizon - 1e-9)

    kmf = KaplanMeierFitter()
    try:
        kmf.fit(T_obs, event_observed=pre_horizon_dropout.astype(float))
    except Exception:
        return float("nan")

    events = Delta == 1
    if events.sum() == 0:
        return float("nan")

    g_at_t = kmf.survival_function_at_times(T_obs[events]).values
    g_at_t = np.clip(g_at_t, 0.05, 1.0)
    w = 1.0 / g_at_t
    return float((w.sum() ** 2) / (w ** 2).sum())


def _run_one_cell_sim(replicate_config_dict: dict, estimator_names: list[str],
                       horizon: float, child_entropy: int) -> dict:
    """Run one simulation replicate.

    replicate_config_dict was already constructed and validated in the parent
    process (see causal_bench.runner.build_validated_replicate_configs) —
    model_construct() here skips re-validation rather than re-running it per
    worker.
    """
    rng = np.random.default_rng(child_entropy)
    rng.integers(0, 2**31)  # discard: matches the draw the parent consumed to pick this replicate's seed, keeping the RNG stream identical
    config = DGPConfig.model_construct(**replicate_config_dict)
    df = generate_data(config, rng=rng)

    out: dict = {}
    errors: dict[str, str] = {}
    for name in estimator_names:
        try:
            if name == "concrete_simult":
                # Only one horizon is ever requested here, so the RD estimand
                # is unambiguous — match by prefix rather than an exact label,
                # since the R bridge caps the horizon (and its label suffix)
                # to last_event * 0.999 under short follow-up / heavy censoring.
                est = ConcreteSimultaneousEstimator(horizons=(horizon,))
                results = est.estimate(df, horizon=horizon, estimand="ATE")
                match = next((r for r in results if r.estimand.startswith("RD_RiskDiff_t")), None)
            elif name == "tmle_ipcw_boot":
                est = TMLEIPCWBootEstimator(n_bootstrap=BOOT_N_BOOTSTRAP)
                results = est.estimate(df, horizon=horizon, estimand="ATE")
                match = next((r for r in results if r.estimand == "ATE"), None)
                if match is None and results:
                    match = results[0]
            else:
                est = get_estimator(name)
                results = est.estimate(df, horizon=horizon, estimand="ATE")
                match = next((r for r in results if r.estimand == "ATE"), None)
                if match is None and results:
                    match = results[0]
        except Exception as e:
            match = None
            errors[name] = classify_error(e)
        out[name] = match

    out["__ess__"] = _ipcw_ess(df, horizon)
    out["__errors__"] = errors
    return out


def run(n_sims: int = N_SIMS, n_jobs: int = -1, seed: int = 42, horizon: float = HORIZON,
        debug_first_replicate: bool = True, task_timeout: float = SIM_TASK_TIMEOUT_SECONDS):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    base_cfg = get_scenario("clean")
    base_cfg = base_cfg.with_overrides(horizon=horizon, censoring_rate=0.3)

    print("Computing true ATE (censoring-mechanism independent)...", flush=True)
    true_ate = compute_true_effects(base_cfg)["ATE"]
    print(f"  True ATE @ horizon={horizon} = {true_ate:.4f}")

    available = [e for e in ESTIMATORS if e == "concrete_simult" or e in ESTIMATOR_REGISTRY]

    all_rows = []
    cell_summaries = {}

    for cell in GRID:
        label = cell["label"]
        cell_kwargs = {k: v for k, v in cell.items() if k != "label"}
        try:
            # with_overrides re-runs DGPConfig's validator, so a bad grid
            # entry (e.g. a typo'd censoring_mechanism) raises here, in the
            # parent, before any worker is dispatched for this cell.
            cfg = base_cfg.with_overrides(**cell_kwargs)
        except ValidationError as e:
            print(f"\nExp 13: cell={label} — invalid DGPConfig overrides {cell_kwargs}, skipping: {e}")
            continue
        config_dict = cfg.__dict__.copy()

        print(f"\nExp 13: cell={label} | mechanism={cfg.censoring_mechanism} | n_sims={n_sims}", flush=True)
        child_entropies = [int(e) for e in SeedSequence(seed ^ hash(label) % (2**31)).generate_state(n_sims)]
        replicate_configs = build_validated_replicate_configs(config_dict, child_entropies)

        if debug_first_replicate and replicate_configs:
            _run_one_cell_sim(replicate_configs[0], available, horizon, child_entropies[0])

        sim_outputs = Parallel(n_jobs=n_jobs, backend="loky", timeout=task_timeout)(
            delayed(_run_one_cell_sim)(replicate_configs[i], available, horizon, child_entropies[i])
            for i in tqdm(range(n_sims), desc=label, total=n_sims)
        )

        error_counts: dict[str, dict[str, int]] = {}
        for o in sim_outputs:
            for name, err_label in o.get("__errors__", {}).items():
                error_counts.setdefault(name, {}).setdefault(err_label, 0)
                error_counts[name][err_label] += 1
        for name, counts in error_counts.items():
            breakdown = ", ".join(f"{k}={v}" for k, v in counts.items())
            print(f"  {name}: {sum(counts.values())}/{n_sims} replicates failed ({breakdown})")

        ess_vals = np.array([o["__ess__"] for o in sim_outputs if np.isfinite(o["__ess__"])])
        mean_ess = float(np.mean(ess_vals)) if len(ess_vals) else float("nan")

        cell_estimator_rows = []
        for name in available:
            pts, ses, cilo, cihi = [], [], [], []
            for o in sim_outputs:
                r = o.get(name)
                if r is not None and np.isfinite(r.point_estimate) and np.isfinite(r.standard_error):
                    pts.append(r.point_estimate)
                    ses.append(r.standard_error)
                    cilo.append(r.ci_lower)
                    cihi.append(r.ci_upper)
            if not pts:
                continue
            sr = SimResult(
                estimator_name=name, estimand="ATE", true_value=true_ate, n_sim=len(pts),
                estimates=np.array(pts), se_estimates=np.array(ses),
                ci_lowers=np.array(cilo), ci_uppers=np.array(cihi),
                nc_estimates=np.zeros(len(pts)),
            )
            row = sr.summary()
            row["cell"] = label
            row["mechanism"] = cfg.censoring_mechanism
            row["ipcw_ess"] = round(mean_ess, 1)
            row["ipcw_ess_pct"] = round(mean_ess / cfg.n * 100, 1) if np.isfinite(mean_ess) else float("nan")
            row["n_converged"] = len(pts)
            cell_estimator_rows.append(row)
            all_rows.append(row)

        cell_summaries[label] = cell_estimator_rows
        for row in cell_estimator_rows:
            print(f"  {row['estimator']:>16s}  bias={row['bias']:+.4f}  rmse={row['rmse']:.4f}  "
                  f"coverage={row['coverage']:.3f}  se_ratio={row['se_ratio']:.3f}  "
                  f"ipcw_ess%={row['ipcw_ess_pct']}")

    tbl = pd.DataFrame(all_rows)
    tbl_path = OUT_DIR / "summary.csv"
    tbl.to_csv(tbl_path, index=False)
    (OUT_DIR / "summary.json").write_text(json.dumps(cell_summaries, indent=2, default=str))
    print(f"\nSaved summary -> {tbl_path}")

    print("\n── Full grid summary ────────────────────────────────────")
    print(tbl.to_string(index=False))

    return tbl


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Exp 13: Censoring mechanism sweep")
    p.add_argument("--n-sims", type=int, default=N_SIMS)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--seed",   type=int, default=42)
    p.add_argument("--horizon", type=float, default=HORIZON)
    p.add_argument("--no-debug-first-replicate", action="store_true",
                   help="Skip the synchronous first-replicate debug pass before each cell's parallel sweep")
    p.add_argument("--task-timeout", type=float, default=SIM_TASK_TIMEOUT_SECONDS)
    args = p.parse_args()
    run(n_sims=args.n_sims, n_jobs=args.n_jobs, seed=args.seed, horizon=args.horizon,
        debug_first_replicate=not args.no_debug_first_replicate, task_timeout=args.task_timeout)
