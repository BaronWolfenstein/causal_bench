"""Exp 12: Simultaneous coverage across a multi-estimand family.

Runs ConcreteSimultaneousEstimator (RD x 2 horizons, RMST diff, LYL diff,
RMT-IF, win ratio, win odds, net benefit — concrete PR #31 getSimultaneousFamily)
plus ClinicalRMTIFEstimator (concrete PR #33 clinicalRMTIF, multistate engine)
on the same simulated data, and reports three coverage quantities:

  - per-estimand pointwise coverage   (each estimand's own 95% CI)
  - joint pointwise coverage          (ALL estimands covered by their own CI)
  - simultaneous coverage             (ALL estimands covered by the joint
                                        Gaussian-multiplier band from
                                        getSimultaneousFamily)

ClinicalRMTIF is structurally a different estimand from getRMTIF once illness
(cause-1) events occur: causal_bench's competing-risks DGP truncates a
subject's trajectory at the first event, so it has no post-illness follow-up
to define the multistate "favorable time after illness" functional. There is
therefore no closed-form truth available for clinicalRMTIF in this DGP — it
is tracked for point-estimate / SE diagnostics only, not coverage. Likewise
WR_WinOdds has no closed-form truth derived here (its tie-handling definition
in concrete was not reverse-engineered) and is excluded from coverage too.

If concrete is unavailable, the script exits with a clear message.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from numpy.random import SeedSequence
from tqdm import tqdm

from causal_bench.dgp.scenarios import get_scenario
from causal_bench.dgp.survival import (
    compute_true_effects,
    compute_true_rmst,
    compute_true_win_ratio,
    generate_data,
)
from causal_bench.dgp.config import DGPConfig
from causal_bench.estimators import get_estimator
from causal_bench.estimators.concrete_simultaneous import ConcreteSimultaneousEstimator
from causal_bench.metrics import SimResult, SimResultFamily
from causal_bench.runner import SIM_TASK_TIMEOUT_SECONDS, build_validated_replicate_configs, classify_error

OUT_DIR = Path("results/exp12_simultaneous")
N_SIMS = 100  # increase for publication
HORIZONS = (0.4, 0.7)
SIGNIF = 0.05


def _run_one_family_sim(replicate_config_dict: dict, child_entropy: int,
                         horizons: tuple, signif: float) -> dict:
    """Run one simulation replicate. Returns dict keyed by estimand label.

    replicate_config_dict was already constructed and validated in the parent
    process (see causal_bench.runner.build_validated_replicate_configs) —
    model_construct() here skips re-validation rather than re-running it per
    worker.
    """
    rng = np.random.default_rng(child_entropy)
    rng.integers(0, 2**31)  # discard: matches the draw the parent consumed to pick this replicate's seed, keeping the RNG stream identical
    config = DGPConfig.model_construct(**replicate_config_dict)
    df = generate_data(config, rng=rng)
    max_h = max(horizons)

    sim_est = ConcreteSimultaneousEstimator(horizons=horizons, signif=signif)
    nc_val = sim_est.estimate_negative_control(df, horizon=max_h)

    out: dict = {}
    errors: dict[str, str] = {}
    try:
        results = sim_est.estimate(df, horizon=max_h, estimand="ATE")
    except Exception as e:
        results = []
        errors["concrete_simult"] = classify_error(e)
    for r in results:
        ci = r.convergence_info or {}
        out[r.estimand] = dict(
            point=r.point_estimate, se=r.standard_error,
            ci_lo=r.ci_lower, ci_hi=r.ci_upper,
            sim_ci_lo=ci.get("sim_ci_lo", float("nan")),
            sim_ci_hi=ci.get("sim_ci_hi", float("nan")),
            sim_q=ci.get("sim_q", float("nan")),
        )

    try:
        crmtif_est = get_estimator("clinical_RMTIF")
        crmtif_res = crmtif_est.estimate(df, horizon=max_h, estimand="ATE")
    except Exception as e:
        crmtif_res = []
        errors["clinical_RMTIF"] = classify_error(e)
    if crmtif_res:
        r = crmtif_res[0]
        out["clinical_RMTIF"] = dict(
            point=r.point_estimate, se=r.standard_error,
            ci_lo=r.ci_lower, ci_hi=r.ci_upper,
            sim_ci_lo=float("nan"), sim_ci_hi=float("nan"), sim_q=float("nan"),
        )

    out["__nc__"] = nc_val
    out["__errors__"] = errors
    return out


def run(n_sims: int = N_SIMS, n_jobs: int = -1, seed: int = 42,
        horizons: tuple = HORIZONS, signif: float = SIGNIF,
        debug_first_replicate: bool = True, task_timeout: float = SIM_TASK_TIMEOUT_SECONDS):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cfg = get_scenario("competing_risks_base")
    max_h = max(horizons)

    print("Computing true values for the estimand family...", flush=True)
    true_rd = {h: compute_true_effects(cfg.with_overrides(horizon=h))["ATE"] for h in horizons}
    true_rmst_max = compute_true_rmst(cfg.with_overrides(horizon=max_h))["ATE"]
    wr_dict = compute_true_win_ratio(cfg)

    true_values: dict[str, float] = {}
    for h in horizons:
        true_values[f"RD_RiskDiff_t{h:g}"] = true_rd[h]
    true_values[f"RMST_RMSTDiff_t{max_h:g}"] = true_rmst_max
    true_values[f"RMST_LYLDiff_t{max_h:g}"]  = -true_rmst_max
    true_values[f"RMTIF_RMT-IF_t{max_h:g}"]  = true_rmst_max
    true_values[f"WR_WinRatio_t{max_h:g}"]   = wr_dict["ATE"]
    true_values[f"WR_NetBenefit_t{max_h:g}"] = wr_dict["net_benefit"]
    # WR_WinOdds_t<h> and clinical_RMTIF: no closed-form truth — see module docstring.

    print(f"Exp 12: simultaneous coverage | scenario=competing_risks_base | n={cfg.n} | "
          f"horizons={horizons} | n_sims={n_sims}")

    config_dict = cfg.__dict__.copy()
    child_entropies = [int(e) for e in SeedSequence(seed).generate_state(n_sims)]
    replicate_configs = build_validated_replicate_configs(config_dict, child_entropies)

    if debug_first_replicate and replicate_configs:
        _run_one_family_sim(replicate_configs[0], child_entropies[0], horizons, signif)

    sim_outputs = Parallel(n_jobs=n_jobs, backend="loky", timeout=task_timeout)(
        delayed(_run_one_family_sim)(replicate_configs[i], child_entropies[i], horizons, signif)
        for i in tqdm(range(n_sims), desc="Simulations", total=n_sims)
    )

    SIDE_CHANNEL_KEYS = {"__nc__", "__errors__"}
    key_sets = [frozenset(k for k in o if k not in SIDE_CHANNEL_KEYS) for o in sim_outputs]
    common = Counter(key_sets).most_common(1)[0][0]
    filtered = [o for o in sim_outputs if frozenset(k for k in o if k not in SIDE_CHANNEL_KEYS) == common]
    n_used = len(filtered)
    print(f"  {n_used}/{n_sims} sims had the full estimand family converge")

    error_counts: dict[str, dict[str, int]] = {}
    for o in sim_outputs:
        for name, err_label in o.get("__errors__", {}).items():
            error_counts.setdefault(name, {}).setdefault(err_label, 0)
            error_counts[name][err_label] += 1
    for name, counts in error_counts.items():
        breakdown = ", ".join(f"{k}={v}" for k, v in counts.items())
        print(f"  {name}: {sum(counts.values())}/{n_sims} replicates failed ({breakdown})")

    if n_used == 0:
        print("No sims converged — concrete bridge may be unavailable.")
        return None

    nc_arr = np.array([o["__nc__"] for o in filtered])

    members: list[SimResult] = []
    sim_ci_lowers: dict[str, np.ndarray] = {}
    sim_ci_uppers: dict[str, np.ndarray] = {}
    crit_values = None

    for label in sorted(common):
        true_val = true_values.get(label, float("nan"))
        pts  = np.array([o[label]["point"]  for o in filtered])
        ses  = np.array([o[label]["se"]     for o in filtered])
        cilo = np.array([o[label]["ci_lo"]  for o in filtered])
        cihi = np.array([o[label]["ci_hi"]  for o in filtered])
        slo  = np.array([o[label]["sim_ci_lo"] for o in filtered])
        shi  = np.array([o[label]["sim_ci_hi"] for o in filtered])
        sq   = np.array([o[label]["sim_q"]     for o in filtered])

        if np.isfinite(true_val):
            members.append(SimResult(
                estimator_name="clinical_RMTIF" if label == "clinical_RMTIF" else "concrete_simult",
                estimand=label, true_value=true_val, n_sim=n_used,
                estimates=pts, se_estimates=ses, ci_lowers=cilo, ci_uppers=cihi,
                nc_estimates=nc_arr,
            ))
        if np.all(np.isfinite(slo)) and np.all(np.isfinite(shi)):
            sim_ci_lowers[label] = slo
            sim_ci_uppers[label] = shi
        if crit_values is None and np.all(np.isfinite(sq)):
            crit_values = sq

    if crit_values is None:
        crit_values = np.full(n_used, np.nan)

    family = SimResultFamily(
        members=members,
        sim_ci_lowers=sim_ci_lowers,
        sim_ci_uppers=sim_ci_uppers,
        crit_values=crit_values,
    )

    summary = family.summary()
    print("\n── Per-estimand pointwise results ─────────────────────────")
    for row in summary["per_estimand"]:
        print(row)
    print(f"\nJoint pointwise coverage (all estimands, own 95% CIs): "
          f"{summary['joint_pointwise_coverage']}")
    print(f"Simultaneous coverage (all estimands, joint Gaussian-multiplier band): "
          f"{summary['simultaneous_coverage']}")
    print(f"Mean critical value q̂: {summary['mean_crit_value']}")

    no_truth = [l for l in common if not np.isfinite(true_values.get(l, float("nan")))]
    if no_truth:
        print("\n── Tracked without closed-form truth (point-estimate diagnostics only) ──")
        for label in sorted(no_truth):
            pts = np.array([o[label]["point"] for o in filtered])
            print(f"  {label}: mean={pts.mean():.4f}  sd={pts.std(ddof=1):.4f}")

    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nSaved summary -> {OUT_DIR}/summary.json")

    return family


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Exp 12: Simultaneous coverage benchmark")
    p.add_argument("--n-sims", type=int, default=N_SIMS)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--seed",   type=int, default=42)
    p.add_argument("--no-debug-first-replicate", action="store_true",
                   help="Skip the synchronous first-replicate debug pass before the parallel sweep")
    p.add_argument("--task-timeout", type=float, default=SIM_TASK_TIMEOUT_SECONDS)
    args = p.parse_args()
    run(n_sims=args.n_sims, n_jobs=args.n_jobs, seed=args.seed,
        debug_first_replicate=not args.no_debug_first_replicate, task_timeout=args.task_timeout)
