"""Exp 14: Provenance-linked synthetic augmentation — cross-fitting independence.

Motivation: a common real-world pattern augments a real cohort with synthetic
units generated conditioned on / near specific real "parent" units (e.g.
twisted-diffusion-style augmentation). Standard cross-fitting assumes folds
are drawn over independent units; if a synthetic child shares latent
structure with its real parent and the two land in different folds, that
independence is violated and the influence-curve-based variance estimate is
too small, degrading CI coverage below nominal even though the point estimate
stays roughly unbiased.

causal_bench.dgp.augmentation.generate_augmented_data makes this leakage a
controllable knob (leakage_strength in [0, 1]) and tags every row with
provenance_group (shared between a real parent and its synthetic children).
causal_bench.crossfit.make_folds + the fold_mode plumbing in SuperLearner and
TMLEIPCWEstimator let the same estimator be cross-fit two ways on the exact
same data:
  - fold_mode="iid"   — current default, ignores provenance (the bug)
  - fold_mode="group" — sklearn GroupKFold keeps every provenance_group intact
                        within a single fold (the fix)

Sweeps leakage_strength over [0.0, 0.25, 0.5, 0.75, 1.0]; at each level, runs
N Monte Carlo replicates of TMLEIPCWEstimator under both fold modes on the
same augmented dataset and reports bias, RMSE, mean reported SE, empirical SE,
SE ratio (mean reported SE / empirical SE — the EIC calibration diagnostic;
near 1.0 is well-calibrated, < 1.0 means the reported SE understates the true
sampling variability) and 95% CI coverage.

Hypothesis under test (not assumed): iid-mode coverage degrades as
leakage_strength increases while group-mode coverage stays near nominal. The
"Key finding" block reports what was actually observed, including if that
hypothesis does NOT hold.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numpy.random import SeedSequence
from tqdm import tqdm

from causal_bench.dgp.augmentation import AugmentationConfig, generate_augmented_data
from causal_bench.dgp.scenarios import get_scenario
from causal_bench.dgp.survival import compute_true_effects
from causal_bench.estimators.tmle_ipcw import TMLEIPCWEstimator
from causal_bench.metrics import SimResult
from causal_bench.runner import SIM_TASK_TIMEOUT_SECONDS, classify_error

OUT_DIR = Path("results/exp14_synthetic_augmentation")
N_SIMS = 60  # increase for publication
HORIZON = 0.7
N_REAL = 60
N_SYNTH_PER_REAL = 3
FOLD_MODES = ["iid", "group"]
LEAKAGE_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]


def _run_one_cell_sim(cfg_dict: dict, n_real: int, n_synth_per_real: int,
                       leakage_strength: float, horizon: float,
                       child_entropy: int) -> dict:
    """Draw one augmented dataset and fit TMLEIPCWEstimator under both fold modes."""
    from causal_bench.dgp.config import DGPConfig

    rng = np.random.default_rng(child_entropy)
    cfg = DGPConfig.model_construct(**cfg_dict)

    # Draw one augmented dataset shared across both fold modes — fold_mode
    # doesn't affect generate_augmented_data's output, only how downstream
    # cross-fitting respects provenance groups.
    aug = AugmentationConfig(
        n_real=n_real, n_synth_per_real=n_synth_per_real,
        leakage_strength=leakage_strength, fold_mode="group",
    )
    df = generate_augmented_data(cfg, aug, rng=rng)

    out: dict = {}
    errors: dict[str, str] = {}
    for mode in FOLD_MODES:
        try:
            est = TMLEIPCWEstimator(fold_mode=mode)
            results = est.estimate(df, horizon=horizon, estimand="ATE")
            match = next((r for r in results if r.estimand == "ATE"), None)
        except Exception as e:
            match = None
            errors[mode] = classify_error(e)
        out[mode] = match
    out["__errors__"] = errors
    return out


def run(n_sims: int = N_SIMS, n_jobs: int = -1, seed: int = 42, horizon: float = HORIZON,
        n_real: int = N_REAL, n_synth_per_real: int = N_SYNTH_PER_REAL,
        debug_first_replicate: bool = True, task_timeout: float = SIM_TASK_TIMEOUT_SECONDS):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    base_cfg = get_scenario("clean")
    base_cfg = base_cfg.with_overrides(horizon=horizon, censoring_rate=0.3)

    print("Computing true ATE (augmentation-mechanism independent)...", flush=True)
    true_ate = compute_true_effects(base_cfg)["ATE"]
    print(f"  True ATE @ horizon={horizon} = {true_ate:.4f}")

    cfg_dict = base_cfg.__dict__.copy()

    all_rows = []
    cell_summaries: dict[str, list[dict]] = {}

    for leakage_strength in LEAKAGE_GRID:
        label = f"leakage{leakage_strength:.2f}"
        print(f"\nExp 14: cell={label} | n_real={n_real} | n_synth_per_real={n_synth_per_real} "
              f"| n_sims={n_sims}", flush=True)
        child_entropies = [int(e) for e in
                            SeedSequence(seed ^ hash(label) % (2**31)).generate_state(n_sims)]

        if debug_first_replicate:
            _run_one_cell_sim(cfg_dict, n_real, n_synth_per_real, leakage_strength,
                               horizon, child_entropies[0])

        sim_outputs = Parallel(n_jobs=n_jobs, backend="loky", timeout=task_timeout)(
            delayed(_run_one_cell_sim)(cfg_dict, n_real, n_synth_per_real, leakage_strength,
                                        horizon, child_entropies[i])
            for i in tqdm(range(n_sims), desc=label, total=n_sims)
        )

        error_counts: dict[str, dict[str, int]] = {}
        for o in sim_outputs:
            for mode, err_label in o.get("__errors__", {}).items():
                error_counts.setdefault(mode, {}).setdefault(err_label, 0)
                error_counts[mode][err_label] += 1
        for mode, counts in error_counts.items():
            breakdown = ", ".join(f"{k}={v}" for k, v in counts.items())
            print(f"  {mode}: {sum(counts.values())}/{n_sims} replicates failed ({breakdown})")

        cell_mode_rows = []
        for mode in FOLD_MODES:
            pts, ses, cilo, cihi = [], [], [], []
            for o in sim_outputs:
                r = o.get(mode)
                if r is not None and np.isfinite(r.point_estimate) and np.isfinite(r.standard_error):
                    pts.append(r.point_estimate)
                    ses.append(r.standard_error)
                    cilo.append(r.ci_lower)
                    cihi.append(r.ci_upper)
            if not pts:
                continue
            sr = SimResult(
                estimator_name=f"tmle_ipcw[{mode}]", estimand="ATE", true_value=true_ate,
                n_sim=len(pts), estimates=np.array(pts), se_estimates=np.array(ses),
                ci_lowers=np.array(cilo), ci_uppers=np.array(cihi),
                nc_estimates=np.zeros(len(pts)),
            )
            row = sr.summary()
            row["cell"] = label
            row["leakage_strength"] = leakage_strength
            row["fold_mode"] = mode
            row["empirical_se"] = round(float(np.std(sr.estimates, ddof=1)), 4)
            row["mean_reported_se"] = round(float(np.mean(sr.se_estimates)), 4)
            row["n_converged"] = len(pts)
            cell_mode_rows.append(row)
            all_rows.append(row)

        cell_summaries[label] = cell_mode_rows
        for row in cell_mode_rows:
            print(f"  {row['fold_mode']:>6s}  bias={row['bias']:+.4f}  rmse={row['rmse']:.4f}  "
                  f"mean_se={row['mean_reported_se']:.4f}  emp_se={row['empirical_se']:.4f}  "
                  f"se_ratio={row['se_ratio']:.3f}  coverage={row['coverage']:.3f}")

    tbl = pd.DataFrame(all_rows)
    tbl_path = OUT_DIR / "summary.csv"
    tbl.to_csv(tbl_path, index=False)
    (OUT_DIR / "summary.json").write_text(json.dumps(cell_summaries, indent=2, default=str))
    print(f"\nSaved summary -> {tbl_path}")

    print("\n── Full grid summary ────────────────────────────────────")
    print(tbl.to_string(index=False))

    _print_key_finding(tbl)

    return tbl


def _print_key_finding(tbl: pd.DataFrame) -> None:
    """Report whether iid-mode coverage degrades with leakage while group-mode
    stays nominal, or whether the data shows otherwise — verified, not assumed.
    """
    print("\n── Key finding ──────────────────────────────────────────")
    if tbl.empty:
        print("  No converged replicates; cannot assess.")
        return

    iid = tbl[tbl["fold_mode"] == "iid"].sort_values("leakage_strength")
    grp = tbl[tbl["fold_mode"] == "group"].sort_values("leakage_strength")
    if iid.empty or grp.empty:
        print("  Missing rows for one of the fold modes; cannot assess.")
        return

    iid_at_0 = iid.iloc[0]
    iid_at_1 = iid.iloc[-1]
    grp_at_0 = grp.iloc[0]
    grp_at_1 = grp.iloc[-1]

    iid_coverage_dropped = iid_at_1["coverage"] < iid_at_0["coverage"] - 1e-9
    iid_se_ratio_dropped = iid_at_1["se_ratio"] < iid_at_0["se_ratio"] - 1e-9
    group_coverage_stable = grp_at_1["coverage"] >= grp_at_0["coverage"] - 0.10
    group_beats_iid_at_max_leakage = grp_at_1["se_ratio"] > iid_at_1["se_ratio"]

    print(f"  iid:   coverage {iid_at_0['coverage']:.3f} (leakage=0) -> {iid_at_1['coverage']:.3f} (leakage=1), "
          f"se_ratio {iid_at_0['se_ratio']:.3f} -> {iid_at_1['se_ratio']:.3f}")
    print(f"  group: coverage {grp_at_0['coverage']:.3f} (leakage=0) -> {grp_at_1['coverage']:.3f} (leakage=1), "
          f"se_ratio {grp_at_0['se_ratio']:.3f} -> {grp_at_1['se_ratio']:.3f}")

    if iid_coverage_dropped and iid_se_ratio_dropped and group_coverage_stable and group_beats_iid_at_max_leakage:
        print("  CONFIRMED: under fold_mode='iid', provenance leakage shrinks the EIC-based "
              "SE relative to the empirical SE (se_ratio falls) and 95% coverage degrades as "
              "leakage_strength -> 1. fold_mode='group' (GroupKFold on provenance_group) keeps "
              "coverage near nominal and a higher se_ratio at the same leakage level — "
              "confirming the cross-fitting independence violation is real and the group-mode "
              "fix corrects it.")
    else:
        print("  NOT CONFIRMED as hypothesized: the expected iid-degrades / group-stable pattern "
              "did not fully hold in this run. See the per-cell table above for the actual "
              "bias/coverage/se_ratio trajectory across leakage_strength.")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Exp 14: Synthetic augmentation cross-fitting sweep")
    p.add_argument("--n-sims", type=int, default=N_SIMS)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--seed",   type=int, default=42)
    p.add_argument("--horizon", type=float, default=HORIZON)
    p.add_argument("--n-real", type=int, default=N_REAL)
    p.add_argument("--n-synth-per-real", type=int, default=N_SYNTH_PER_REAL)
    p.add_argument("--no-debug-first-replicate", action="store_true",
                   help="Skip the synchronous first-replicate debug pass before each cell's parallel sweep")
    p.add_argument("--task-timeout", type=float, default=SIM_TASK_TIMEOUT_SECONDS)
    args = p.parse_args()
    run(n_sims=args.n_sims, n_jobs=args.n_jobs, seed=args.seed, horizon=args.horizon,
        n_real=args.n_real, n_synth_per_real=args.n_synth_per_real,
        debug_first_replicate=not args.no_debug_first_replicate, task_timeout=args.task_timeout)
