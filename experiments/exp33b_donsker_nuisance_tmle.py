"""Exp 33b: TMLE+IPCW with Donsker-class nuisance learners (phase-2 wiring).

Phase-1 exp33 showed LTB/HAR license AIPW/TMLE without cross-fitting in a
point-treatment DGP. Phase 2 wires those learners into the production
TMLEIPCWEstimator (g_learner/q_learner; Cox censoring model G unchanged) and
compares, on the edwards_realistic survival scenario, the default logistic/
SuperLearner nuisances against LTB and HAR nuisances.

Arms:
- default : g via the default SuperLearner ensemble, Q via IPCW-weighted logistic.
- cv      : same, but the CV-TMLE variant (cross-fitted censoring model G).
- ltb     : LTBClassifier for g, LTBRegressor for Q.
- cv_ltb  : LTB nuisances under CV-TMLE.
- har_q   : HARRegressor for Q (squared-error only, no classifier), default g.

The cv / cv_ltb arms are the spec §8.3 comparison against the existing
tmle_ipcw_cv rows and the documented se_ratio gap.

This is a research comparison, not a unit test: the default n_sims is small so
the script is runnable interactively; raise --n-sims for a real read. Metrics
are bias / RMSE / coverage / mean-SE against the true ATE at the horizon.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from causal_bench.dgp.scenarios import get_scenario
from causal_bench.dgp.survival import compute_true_effects, generate_data
from causal_bench.estimators.tmle_ipcw import TMLEIPCWEstimator

OUT_DIR = Path("results/exp33b_donsker_nuisance_tmle")
SCENARIO = "edwards_realistic"


def _estimators():
    """Arm name -> factory (fresh estimator per sim; learners are cloned inside).

    The cv arms use TMLEIPCWCVEstimator (cross-fitted censoring model G) so the
    run compares against the existing tmle_ipcw AND tmle_ipcw_cv rows (spec
    §8.3). TMLEIPCWCVEstimator inherits g_learner/q_learner from its parent.
    """
    from causal_bench.har import HARRegressor
    from causal_bench.ltb import LTBClassifier, LTBRegressor
    from causal_bench.estimators.tmle_ipcw_cv import TMLEIPCWCVEstimator
    return {
        "default": lambda: TMLEIPCWEstimator(random_state=42),
        "cv": lambda: TMLEIPCWCVEstimator(random_state=42),
        "ltb": lambda: TMLEIPCWEstimator(
            random_state=42, g_learner=LTBClassifier(random_state=0),
            q_learner=LTBRegressor(random_state=0)),
        "cv_ltb": lambda: TMLEIPCWCVEstimator(
            random_state=42, g_learner=LTBClassifier(random_state=0),
            q_learner=LTBRegressor(random_state=0)),
        "har_q": lambda: TMLEIPCWEstimator(
            random_state=42, q_learner=HARRegressor(random_state=0)),
    }


def run(n_sims: int = 50, seed: int = 20260703, arms=None,
        scenario: str = SCENARIO) -> pd.DataFrame:
    cfg = get_scenario(scenario)
    tau0 = compute_true_effects(cfg)["ATE"]
    factories = _estimators()
    if arms is not None:
        factories = {k: v for k, v in factories.items() if k in arms}

    rows = []
    for sim in range(n_sims):
        df = generate_data(cfg, rng=np.random.default_rng(seed + sim))
        for name, make in factories.items():
            try:
                r = make().estimate(df, horizon=cfg.horizon)[0]
                rows.append({
                    "arm": name, "sim": sim, "point": r.point_estimate,
                    "se": r.standard_error,
                    "covered": bool(r.ci_lower <= tau0 <= r.ci_upper),
                })
            except Exception as e:  # a learner blowing up shouldn't kill the grid
                rows.append({"arm": name, "sim": sim, "point": np.nan,
                             "se": np.nan, "covered": False, "error": str(e)})
    return pd.DataFrame(rows), tau0


def summarize(raw: pd.DataFrame, tau0: float) -> pd.DataFrame:
    out = []
    for arm, g in raw.groupby("arm"):
        pts = g["point"].dropna()
        emp_sd = float(pts.std(ddof=1)) if len(pts) > 1 else float("nan")
        out.append({
            "arm": arm, "n_ok": len(pts),
            "bias": float(pts.mean() - tau0) if len(pts) else float("nan"),
            "rmse": float(np.sqrt(np.mean((pts - tau0) ** 2))) if len(pts) else float("nan"),
            "coverage": float(g["covered"].mean()),
            "mean_se": float(g["se"].mean()),
            "emp_sd": emp_sd,
        })
    return pd.DataFrame(out)


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-sims", type=int, default=50)
    ap.add_argument("--seed", type=int, default=20260703)
    ap.add_argument("--arms", nargs="+", default=None,
                    help="subset of {default, cv, ltb, cv_ltb, har_q}")
    ap.add_argument("--scenario", type=str, default=SCENARIO,
                    help="any causal_bench.dgp.scenarios key (e.g. edwards_realistic)")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw, tau0 = run(n_sims=args.n_sims, seed=args.seed, arms=args.arms,
                    scenario=args.scenario)
    raw.to_csv(OUT_DIR / "raw.csv", index=False)
    summ = summarize(raw, tau0)
    summ.to_csv(OUT_DIR / "summary.csv", index=False)
    print(f"true ATE = {tau0:.4f}")
    print(summ.to_string(index=False))


if __name__ == "__main__":
    main()
