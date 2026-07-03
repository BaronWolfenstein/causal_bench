"""Exp 33: do Donsker-class learners license AIPW/TMLE without cross-fitting?

Spec: docs/superpowers/specs/2026-07-02-ltb-har-benchmark-design.md.

Grid: learner in {logistic, xgboost, ltb, har, hal, oracle}
      x crossfit in {off, on} x surface in {jumpy, smooth}, n=700.

Beyond the usual bias/RMSE/coverage/se_ratio, each simulation measures
the two terms of the estimator expansion directly against the DGP truth:

  EP        = (P_n - P)[eif0(f_hat) - eif0(f_0)]   (reported as sqrt(n)*EP)
  remainder = P[eif0(f_hat)] - tau_0

where the population part P[.] is evaluated on a fixed independent
Monte Carlo sample (--mc-n, default 1e5 per surface). The Donsker
theory's claim is precisely that sqrt(n)*EP -> 0 without cross-fitting
for LTB/HAL (and HAR on the smooth surface only); xgboost is the
non-Donsker control.

Crossfit-OFF is the headline arm and is measured EXACTLY: the P_n and P
parts use the same single fitted model, so (P_n - P) is an honest
empirical-process term. Crossfit-ON is a reference arm only and its EP
carries one extra approximation: the P_n part uses per-fold OOF
nuisances, but the P part averages the fold models' predictions
(NuisanceFits.predict) before forming eif0. Because eif0 is nonlinear in
g, eif0(mean_k f_k) != mean_k eif0(f_k), so the crossfit-ON EP column has
a Jensen-type discrepancy the crossfit-OFF column does not. Do not
compare on-vs-off EP magnitudes quantitatively without accounting for it;
a per-fold P evaluation would remove it (deferred, phase 2).

HAR cost note: HARRegressor.predict is O(n_eval * n_train) per call, so
the P-part evaluation dominates runtime for the HAR arm at n_eval=1e5
(minutes/sim, x5 under crossfit for fold-averaging). Pass a smaller
--mc-n (e.g. 10000) for HAR runs; sqrt(n)*EP MC-noise scales as
1/sqrt(mc_n), so 1e4 is ample.

The oracle arm (f_hat = f_0) pins both terms at zero by construction.
HAL runs via the existing rpy2 wrappers at reduced n_sims and is
skipped with a warning when rpy2/hal9001 is unavailable.
"""
from __future__ import annotations

import argparse
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from causal_bench.dgp.point_treatment import (
    SURFACES, draw_point_treatment, true_Q, true_g, true_tau)
from causal_bench.estimators.point import (
    fit_nuisances, oracle_nuisances, point_aipw, point_tmle)

OUT_DIR = Path("results/exp33_donsker_learners")
W_COLS = ["W1", "W2", "W3", "W4"]
LEARNERS = ("logistic", "xgboost", "ltb", "har", "hal", "oracle")
_MC_N = 100_000
_MC_SEED = 424242


def make_learners(name: str, seed: int):
    """(g_learner, q_learner) for a grid arm; None for the oracle arm."""
    if name == "oracle":
        return None
    if name == "logistic":
        from sklearn.linear_model import LogisticRegression
        return (LogisticRegression(max_iter=1000, C=1.0),
                LogisticRegression(max_iter=1000, C=1.0))
    if name == "xgboost":
        from xgboost import XGBClassifier
        mk = lambda: XGBClassifier(n_estimators=300, max_depth=3,
                                   learning_rate=0.05, random_state=seed,
                                   n_jobs=1, verbosity=0)
        return (mk(), mk())
    if name == "ltb":
        from causal_bench.ltb import LTBClassifier
        return (LTBClassifier(random_state=seed), LTBClassifier(random_state=seed))
    if name == "har":
        from causal_bench.har import HARClassifier
        return (HARClassifier(random_state=seed), HARClassifier(random_state=seed))
    if name == "hal":
        from causal_bench.hal import HALClassifier  # raises if rpy2 missing
        return (HALClassifier(), HALClassifier())
    raise ValueError(f"unknown learner {name!r}")


@lru_cache(maxsize=None)
def mc_eval_sample(surface: str, mc_n: int = _MC_N) -> pd.DataFrame:
    """Fixed independent draw used as the population P[.] in EP/remainder."""
    return draw_point_treatment(n=mc_n, surface=surface, seed=_MC_SEED)


def eif0_values(g, Q1, Q0, A, Y) -> np.ndarray:
    """Uncentered efficient influence function value eif0 = Q1-Q0+H(Y-QA)."""
    g = np.clip(g, 0.01, 0.99)
    QA = A * Q1 + (1 - A) * Q0
    H = A / g - (1 - A) / (1 - g)
    return Q1 - Q0 + H * (Y - QA)


def ep_and_remainder(nf, df_sim: pd.DataFrame, surface: str, mc_n: int = _MC_N):
    """(sqrt(n)*EP, remainder) for the fitted nuisances nf on this sim."""
    W = df_sim[W_COLS].values
    A = df_sim["A"].values.astype(float)
    Y = df_sim["Y"].values.astype(float)
    n = len(A)

    diff_sim = (eif0_values(nf.g, nf.Q1, nf.Q0, A, Y)
                - eif0_values(true_g(W, surface), true_Q(1, W, surface),
                              true_Q(0, W, surface), A, Y))

    mc = mc_eval_sample(surface, mc_n)
    W_mc = mc[W_COLS].values
    A_mc = mc["A"].values.astype(float)
    Y_mc = mc["Y"].values.astype(float)
    g_mc, Q1_mc, Q0_mc = nf.predict(W_mc)
    eif_hat_mc = eif0_values(g_mc, Q1_mc, Q0_mc, A_mc, Y_mc)
    diff_mc = eif_hat_mc - eif0_values(
        true_g(W_mc, surface), true_Q(1, W_mc, surface),
        true_Q(0, W_mc, surface), A_mc, Y_mc)

    ep = float(np.mean(diff_sim)) - float(np.mean(diff_mc))
    remainder = float(np.mean(eif_hat_mc)) - true_tau(surface)
    return float(np.sqrt(n)) * ep, remainder


def nuisance_rmse(nf, W: np.ndarray, surface: str) -> tuple[float, float]:
    """(g_rmse, q_rmse) of fitted nuisances vs truth.

    q_rmse is the pooled RMSE over the stacked Q1/Q0 errors: the sum of the
    two mean-squared errors is divided by 2 (not sqrt(2)) inside the sqrt, so
    for constant offsets +a on Q1 and -b on Q0 it equals sqrt((a^2+b^2)/2).
    """
    g_rmse = float(np.sqrt(np.mean((nf.g - true_g(W, surface)) ** 2)))
    q_rmse = float(np.sqrt(np.mean(
        (nf.Q1 - true_Q(1, W, surface)) ** 2
        + (nf.Q0 - true_Q(0, W, surface)) ** 2) / 2.0))
    return g_rmse, q_rmse


def run_cell(learner: str, crossfit: bool, surface: str, n: int,
             n_sims: int, base_seed: int, mc_n: int = _MC_N) -> pd.DataFrame:
    """All simulations for one (learner, crossfit, surface) cell."""
    tau0 = true_tau(surface)
    rows = []
    for sim in range(n_sims):
        seed = base_seed + 1000 * sim
        df = draw_point_treatment(n=n, surface=surface, seed=seed)
        W = df[W_COLS].values
        A = df["A"].values.astype(float)
        Y = df["Y"].values.astype(float)

        if learner == "oracle":
            nf = oracle_nuisances(W, surface)
        else:
            g_l, q_l = make_learners(learner, seed)
            nf = fit_nuisances(W, A, Y, g_l, q_l, crossfit=crossfit,
                               random_state=seed)

        g_rmse, q_rmse = nuisance_rmse(nf, W, surface)
        sqrtn_ep, remainder = ep_and_remainder(nf, df, surface, mc_n)

        for est_name, est in (("aipw", point_aipw), ("tmle", point_tmle)):
            r = est(A, Y, nf)
            rows.append({
                "learner": learner, "crossfit": crossfit, "surface": surface,
                "estimator": est_name, "sim": sim, "n": n,
                "point": r.point, "se": r.se,
                "covered": bool(r.ci_lower <= tau0 <= r.ci_upper),
                "g_rmse": g_rmse, "q_rmse": q_rmse,
                "sqrtn_ep": sqrtn_ep, "remainder": remainder,
            })
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Per (learner, crossfit, surface, estimator) cell aggregates."""
    out = []
    keys = ["learner", "crossfit", "surface", "estimator"]
    for vals, grp in df.groupby(keys):
        tau0 = true_tau(vals[keys.index("surface")])
        emp_sd = float(grp["point"].std(ddof=1)) if len(grp) > 1 else float("nan")
        out.append(dict(
            zip(keys, vals),
            n_sims=len(grp),
            bias=float(grp["point"].mean() - tau0),
            rmse=float(np.sqrt(np.mean((grp["point"] - tau0) ** 2))),
            coverage=float(grp["covered"].mean()),
            se_ratio=float(grp["se"].mean() / emp_sd) if emp_sd else float("nan"),
            g_rmse=float(grp["g_rmse"].mean()),
            q_rmse=float(grp["q_rmse"].mean()),
            sqrtn_ep_mean=float(grp["sqrtn_ep"].mean()),
            sqrtn_ep_sd=float(grp["sqrtn_ep"].std(ddof=1)) if len(grp) > 1
                        else float("nan"),
            remainder_mean=float(grp["remainder"].mean()),
        ))
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-sims", type=int, default=500)
    ap.add_argument("--n-sims-hal", type=int, default=50)
    ap.add_argument("--n", type=int, default=700)
    ap.add_argument("--seed", type=int, default=20260702)
    ap.add_argument("--skip-hal", action="store_true")
    ap.add_argument("--learners", nargs="+", default=list(LEARNERS))
    ap.add_argument("--mc-n", type=int, default=_MC_N,
                    help="MC eval-sample size for EP/remainder. HAR predict is "
                         "O(n_eval*n_train); use ~10000 for the HAR arm.")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    for surface in SURFACES:
        for learner in args.learners:
            if learner == "hal":
                if args.skip_hal:
                    continue
                try:
                    make_learners("hal", 0)
                except Exception as e:  # rpy2/hal9001 absent
                    warnings.warn(f"skipping HAL arm: {e}")
                    continue
            n_sims = args.n_sims_hal if learner == "hal" else args.n_sims
            crossfits = (False,) if learner == "oracle" else (False, True)
            for crossfit in crossfits:
                print(f"[exp33] {surface} / {learner} / crossfit={crossfit} "
                      f"({n_sims} sims)")
                frames.append(run_cell(learner, crossfit, surface,
                                       n=args.n, n_sims=n_sims,
                                       base_seed=args.seed, mc_n=args.mc_n))
    raw = pd.concat(frames, ignore_index=True)
    raw.to_csv(OUT_DIR / "raw.csv", index=False)
    summ = summarize(raw)
    summ.to_csv(OUT_DIR / "summary.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(summ.sort_values(["surface", "estimator", "learner", "crossfit"])
                  .to_string(index=False))


if __name__ == "__main__":
    main()
