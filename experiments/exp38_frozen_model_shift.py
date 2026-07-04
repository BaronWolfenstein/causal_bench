"""Exp 38: Positivity/propensity performance under train-vs-deploy covariate
shift (LENS-faithful, causal_bench #83).

Motivated by Seegmiller & Preum, "LENS: Measuring Distribution Shift in User
Prompts" (ACL 2026): moderate shift between a model's TRAINING distribution
and its DEPLOYMENT distribution costs ~73% average performance loss. That is
a train-vs-deploy shift measurement, distinct from exp2's positivity_severity
(a static dial evaluated cross-sectionally: fit AND evaluate on the SAME
draw). This experiment adds the missing axis: fit nuisances once on a
baseline TRAIN draw, freeze them, and evaluate the FROZEN model's AIPW
performance on a DEPLOY draw whose covariate distribution has shifted —
contrasted against a REFIT model that gets to see the deploy distribution.

Covariate shift is operationalized in the textbook causal-inference sense:
P(A|W) = true_g(W) and P(Y|A,W) = true_Q(a,W) keep their standard
point_treatment functional forms (causal_bench.dgp.point_treatment); only
P(W) moves (W1's mean shifts). This is a genuine distribution shift, not a
change in the causal mechanism -- exactly the "covariate distribution shift"
LENS's own abstract names.

The frozen-vs-refit gap at each shift level is the number this experiment
was built to produce: if frozen_bias >> refit_bias as shift grows, that is
LENS's finding transplanted into causal estimation -- a model that cannot
adapt degrades far faster than one that can.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

from causal_bench.dgp.point_treatment import SURFACES, true_Q, true_g
from causal_bench.estimators.point import NuisanceFits, fit_nuisances, point_aipw

SURFACE = "smooth"          # avoid confounding the shift story with the jumpy
                            # gate's own discontinuity; see exp33 for that axis
SHIFT_GRID = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]   # W1 mean-shift, deploy vs train
N_TRAIN = 700
N_DEPLOY = 700
N_REPS = 40
MC_N_TRUE_TAU = 200_000

OUT_DIR = Path(__file__).parent.parent / "results" / "exp38_frozen_model_shift"
W_COLS = ["W1", "W2", "W3", "W4"]


def draw_shifted(n: int, seed: int, mean_shift: float = 0.0,
                 surface: str = SURFACE) -> pd.DataFrame:
    """W1 ~ N(mean_shift, 1), W2-W4 ~ N(0,1) independent; A, Y drawn from the
    standard point_treatment functional forms (true_g/true_Q) evaluated at
    the SHIFTED W. mean_shift=0.0 is the baseline (unshifted) distribution.

    Independent (uncorrelated) covariates here, unlike point_treatment's
    mildly-correlated draw -- a deliberate simplification for this
    experiment's own DGP wrapper; true_g/true_Q only require a W array, so
    this is a valid, self-contained covariate-shift generator, not a reuse
    of point_treatment's private internals.
    """
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((n, 4))
    W[:, 0] += mean_shift
    A = rng.binomial(1, true_g(W, surface))
    pY = np.where(A == 1, true_Q(1, W, surface), true_Q(0, W, surface))
    Y = rng.binomial(1, pY)
    return pd.DataFrame({
        "W1": W[:, 0], "W2": W[:, 1], "W3": W[:, 2], "W4": W[:, 3],
        "A": A.astype(int), "Y": Y.astype(int),
    })


@lru_cache(maxsize=None)
def true_tau_shifted(mean_shift: float, surface: str = SURFACE,
                     mc_n: int = MC_N_TRUE_TAU) -> float:
    """True ATE at the SHIFTED covariate distribution (differs from the
    unshifted point_treatment.true_tau once mean_shift != 0, since the
    covariate distribution the ATE marginalizes over has moved)."""
    df = draw_shifted(mc_n, seed=999_999, mean_shift=mean_shift, surface=surface)
    W = df[W_COLS].values
    return float(np.mean(true_Q(1, W, surface) - true_Q(0, W, surface)))


def _make_learners():
    return LogisticRegression(max_iter=1000), LogisticRegression(max_iter=1000)


def run_shift_grid(n_reps: int = N_REPS, seed: int = 42) -> pd.DataFrame:
    """For each deploy shift level, over n_reps replicates: fit nuisances on
    a fresh TRAIN draw (mean_shift=0.0), freeze them, evaluate on a fresh
    DEPLOY draw at the target shift (FROZEN), and separately fit fresh
    nuisances directly on that DEPLOY draw (REFIT). Returns one row per
    (shift, rep, arm) with the AIPW point estimate and bias."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records = []

    for shift in SHIFT_GRID:
        tau0 = true_tau_shifted(shift)
        print(f"  shift={shift:.2f}  true_tau={tau0:+.4f}", flush=True)

        for rep in range(n_reps):
            rep_seed = seed + rep * 1000 + int(shift * 1000)

            train_df = draw_shifted(N_TRAIN, seed=rep_seed, mean_shift=0.0)
            deploy_df = draw_shifted(N_DEPLOY, seed=rep_seed + 500_000, mean_shift=shift)

            W_train = train_df[W_COLS].values
            A_train = train_df["A"].values.astype(float)
            Y_train = train_df["Y"].values.astype(float)
            W_deploy = deploy_df[W_COLS].values
            A_deploy = deploy_df["A"].values.astype(float)
            Y_deploy = deploy_df["Y"].values.astype(float)

            g_l, q_l = _make_learners()
            train_nf = fit_nuisances(W_train, A_train, Y_train, g_l, q_l,
                                     crossfit=False, random_state=rep_seed)

            # FROZEN: the train-fitted model's predictions on deploy covariates.
            g_frozen, Q1_frozen, Q0_frozen = train_nf.predict(W_deploy)
            frozen_nf = NuisanceFits(g_frozen, Q1_frozen, Q0_frozen, models=[])
            r_frozen = point_aipw(A_deploy, Y_deploy, frozen_nf)

            # REFIT: fresh nuisances fit directly on deploy.
            g_l2, q_l2 = _make_learners()
            deploy_nf = fit_nuisances(W_deploy, A_deploy, Y_deploy, g_l2, q_l2,
                                      crossfit=False, random_state=rep_seed + 1)
            r_refit = point_aipw(A_deploy, Y_deploy, deploy_nf)

            records.append({"covariate_shift": shift, "rep": rep, "arm": "frozen",
                            "point": r_frozen.point, "true_tau": tau0,
                            "bias": r_frozen.point - tau0})
            records.append({"covariate_shift": shift, "rep": rep, "arm": "refit",
                            "point": r_refit.point, "true_tau": tau0,
                            "bias": r_refit.point - tau0})

    return pd.DataFrame(records)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Mean bias, RMSE, and the frozen/refit degradation gap per shift level."""
    rows = []
    for shift, grp in df.groupby("covariate_shift"):
        frozen = grp[grp.arm == "frozen"]["bias"]
        refit = grp[grp.arm == "refit"]["bias"]
        rows.append({
            "covariate_shift": shift,
            "frozen_bias": float(frozen.mean()),
            "frozen_rmse": float(np.sqrt(np.mean(frozen ** 2))),
            "refit_bias": float(refit.mean()),
            "refit_rmse": float(np.sqrt(np.mean(refit ** 2))),
            "degradation_gap": float(np.sqrt(np.mean(frozen ** 2))
                                     - np.sqrt(np.mean(refit ** 2))),
        })
    return pd.DataFrame(rows)


def plot_degradation(summary: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(summary["covariate_shift"], summary["frozen_rmse"], marker="o", color="#d62728",
           label="Frozen (train-fitted, faces shift)", linewidth=2)
    ax.plot(summary["covariate_shift"], summary["refit_rmse"], marker="o", color="#1f77b4",
           label="Refit (sees deploy distribution)", linewidth=2)
    ax.set_xlabel("Deploy covariate shift (W1 mean shift)")
    ax.set_ylabel("RMSE vs true ATE at that shift")
    ax.set_title("Exp 38: Frozen vs refit AIPW under train-vs-deploy covariate shift\n"
                "(the LENS axis: does an already-fitted model degrade faster?)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {out_path}")


def main() -> None:
    print(f"\nExp 38: Frozen vs refit AIPW under train-vs-deploy covariate shift")
    print(f"  Shift grid: {SHIFT_GRID}, n_reps={N_REPS}, surface={SURFACE}\n")

    df = run_shift_grid()
    df.to_csv(OUT_DIR / "raw.csv", index=False)

    summary = summarize(df)
    summary.to_csv(OUT_DIR / "summary.csv", index=False)
    print("\nSummary (frozen vs refit RMSE by shift level):")
    print(summary.to_string(index=False))

    plot_degradation(summary, OUT_DIR / "degradation.png")
    print("\nDone.")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Exp 38: Frozen-vs-refit under covariate shift")
    p.add_argument("--n-reps", type=int, default=N_REPS)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = run_shift_grid(n_reps=args.n_reps, seed=args.seed)
    df.to_csv(OUT_DIR / "raw.csv", index=False)
    summary = summarize(df)
    summary.to_csv(OUT_DIR / "summary.csv", index=False)
    print(summary.to_string(index=False))
    plot_degradation(summary, OUT_DIR / "degradation.png")
