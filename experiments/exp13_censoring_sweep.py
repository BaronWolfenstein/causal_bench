"""Exp 13: Censoring mechanism sweep.

Sweeps CensoringConfig variants x their informativeness/strength parameter:
  - IndependentCensoringConfig        — MCAR: pure random dropout
  - CovariateDependentCensoringConfig — pure MAR: C depends only on observed W, A;
                                        informativeness scales covariate effects [0, 1]
  - InformativeCensoringConfig        — MNAR via T_true: log C* = beta0 + beta_T*T_true + eps
                                        (IPCW cannot correct without T_true)
  - LatentConfounderCensoringConfig   — MNAR via U: sicker patients (low U) drop out more;
                                        informativeness scales the U contribution

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

ENCIRCLE connection — this experiment mirrors ENCIRCLE's pre-specified
sensitivity analysis (Guerrero et al., Lancet 2025 SAP Section C):
  ENCIRCLE pre-specified: multiple imputation + censoring tipping point.
  The tipping point takes the form of Figure S4 in the Lancet supplement:
  assume j of k censored subjects had the event (worst-case, event-free →
  event), recompute the 1-year KM composite rate vs the 45% performance-goal
  Wald/Greenwood test (one-sided α=0.025), and find j*/k where p crosses the
  critical value. This nonparametric worst-case bound is complementary to the
  IPCW-under-known-mechanism sweep above: exp13's mechanism sweep asks "how
  biased are estimators if censoring is MNAR?"; the tipping point asks "how
  many censored subjects would need to have had an event to overturn the
  conclusion?" Use encircle_censoring_tipping_point() below to run the
  ENCIRCLE-specific sub-case on any dataset from this experiment's DGP.
"""
from __future__ import annotations

import json
from typing import Optional
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numpy.random import SeedSequence
from tqdm import tqdm

from causal_bench.dgp.config import (
    DGPConfig,
    IndependentCensoringConfig,
    CovariateDependentCensoringConfig,
    InformativeCensoringConfig,
    LatentConfounderCensoringConfig,
)
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

def encircle_censoring_tipping_point(
    df: pd.DataFrame,
    horizon: float = 1.0,
    pg: float = 0.45,
    alpha: float = 0.025,
    arm: int = 1,
) -> dict:
    """ENCIRCLE Figure S4 worst-case censoring tipping point (SAP Section C).

    Sweeps j = 0…k censored subjects in the device arm (A=arm) converted from
    event-free to event in worst-case order (earliest censored first — most
    pessimistic for the device), recomputes the 1-year KM composite rate vs the
    45% performance-goal Wald/Greenwood test (one-sided α=0.025) at each j, and
    reports j*/k where p crosses the critical value.

    This is the ENCIRCLE-specific special case of exp13's general censoring
    sensitivity sweep: nonparametric worst-case bound vs IPCW-under-known-
    mechanism. The two are complementary, not alternatives.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset from generate_data() with columns T_obs, Delta, A.
    horizon : float
        Analysis horizon in years (ENCIRCLE: 1.0).
    pg : float
        Performance goal event rate (ENCIRCLE SAP: 0.45).
    alpha : float
        One-sided significance level (ENCIRCLE SAP: 0.025).
    arm : int
        Treatment arm to analyse (1 = device, 0 = control).

    Returns
    -------
    dict with keys:
      k          — total censored subjects before horizon in the arm
      j_star     — smallest j at which H0 is no longer rejected (None if never)
      j_star_over_k — j*/k (robustness fraction; None if never flips)
      sweep      — pd.DataFrame with columns j, km_rate, greenwood_se, z, p, rejects_h0
    """
    from scipy.stats import norm as _norm

    sub = df[df["A"] == arm].copy()
    censored_mask = (sub["Delta"] == 0) & (sub["T_obs"] < horizon - 1e-9)
    censored_idx = sub[censored_mask].sort_values("T_obs").index  # earliest first
    k = len(censored_idx)

    def _km_and_se(t_obs: np.ndarray, delta: np.ndarray) -> tuple[float, float]:
        order = np.argsort(t_obs)
        t = t_obs[order]
        d = delta[order].astype(float)
        S, gw, at_risk, i = 1.0, 0.0, len(t), 0
        while i < len(t) and t[i] <= horizon:
            t_i, j2, events = t[i], i, 0
            while j2 < len(t) and t[j2] == t_i:
                events += d[j2]
                j2 += 1
            if events > 0 and at_risk > events:
                S *= (at_risk - events) / at_risk
                gw += events / (at_risk * (at_risk - events))
            at_risk -= (j2 - i)
            i = j2
        return float(1.0 - S), float(S * np.sqrt(gw))

    z_crit = _norm.ppf(alpha)
    rows = []
    for j in range(k + 1):
        modified = sub.copy()
        if j > 0:
            modified.loc[censored_idx[:j], "Delta"] = 1
        km_rate, gw_se = _km_and_se(modified["T_obs"].values, modified["Delta"].values)
        z = (km_rate - pg) / max(gw_se, 1e-9)
        p = float(_norm.cdf(z))
        rows.append({
            "j": j, "km_rate": km_rate, "greenwood_se": gw_se,
            "z": z, "p": p, "rejects_h0": p < alpha,
        })

    sweep = pd.DataFrame(rows)
    flip_rows = sweep[~sweep["rejects_h0"]]
    j_star = int(flip_rows["j"].iloc[0]) if len(flip_rows) else None
    return {
        "k": k,
        "j_star": j_star,
        "j_star_over_k": j_star / k if (j_star is not None and k > 0) else None,
        "sweep": sweep,
    }


GRID = [
    # MCAR
    {"label": "independent",               "censoring": IndependentCensoringConfig()},
    # Pure MAR (observed W, A only — no U, no T_true)
    {"label": "covdep_info0.0",            "censoring": CovariateDependentCensoringConfig(informativeness=0.0)},
    {"label": "covdep_info0.3",            "censoring": CovariateDependentCensoringConfig(informativeness=0.3)},
    {"label": "covdep_info0.6",            "censoring": CovariateDependentCensoringConfig(informativeness=0.6)},
    {"label": "covdep_info0.9",            "censoring": CovariateDependentCensoringConfig(informativeness=0.9)},
    # MNAR via T_true (IPCW cannot correct without T_true)
    {"label": "informative_betaT-0.8",     "censoring": InformativeCensoringConfig(beta_T=-0.8)},
    {"label": "informative_betaT-0.4",     "censoring": InformativeCensoringConfig(beta_T=-0.4)},
    {"label": "informative_betaT+0.4",     "censoring": InformativeCensoringConfig(beta_T=0.4)},
    {"label": "informative_betaT+0.8",     "censoring": InformativeCensoringConfig(beta_T=0.8)},
    # MNAR via U (latent confounder — mirrors ENCIRCLE; IPCW correctable via L1 proxy)
    {"label": "latent_conf_info0.25",      "censoring": LatentConfounderCensoringConfig(informativeness=0.25)},
    {"label": "latent_conf_info0.6",       "censoring": LatentConfounderCensoringConfig(informativeness=0.6)},
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

        print(f"\nExp 13: cell={label} | mechanism={cfg.censoring.kind} | n_sims={n_sims}", flush=True)
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
            row["mechanism"] = cfg.censoring.kind
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
