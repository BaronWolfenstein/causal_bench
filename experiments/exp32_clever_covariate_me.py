"""Exp 32: Σ_x measurement error propagated into the TMLE clever covariate (#66).

The production wiring exp31 (#58) demonstrated only on a stand-in OLS estimator.
Here the estimand is the actual TMLE+IPCW ATE, whose **clever covariate**
H(A,W) is built from the propensity g(W). A confounder W1 drives both treatment
and outcome; the analyst sees W1_obs = W1_true + ε (ε ~ N(0, σ_x²)). Because the
estimator reads its propensity covariates from the dataframe columns, the whole
correction reduces to *which W1 column the estimator sees* — g, and therefore
the clever covariate, are automatically rebuilt on it:

- **oracle**    — W1 = W1_true (unattainable benchmark).
- **naive**     — W1 = W1_obs as if exact → residual confounding in the ATE.
- **corrected** — W1 = E[W1_true | W1_obs, W2..W4, A] (multivariate regression
                  calibration, σ_x² from a reliability study), so g and the
                  clever covariate are built on the de-attenuated confounder.

Learner note: TMLE+IPCW fits g with the repo's SuperLearner ensemble. The
residual-confounding mechanism is **learner-agnostic** — it is a property of the
noisy *input*, not of g's learner — so the conclusion transfers to a HAL g (the
#57 primary). No GP dependency.

Variance note: the TMLE SE treats W1_hat as fixed, so it understates the
calibration uncertainty (same limitation exp31 fixed); the corrected arm's
honest interval uses ``causal_bench.bootstrap.row_bootstrap_ci`` over the full
RC + TMLE pipeline (see ``corrected_bootstrap_ci``).

Regulatory caveat (see the theory note filed on #66): regression calibration is
a *first-order* correction. It does not blanket-preserve the TMLE second-order
remainder rate under a nonparametric g; the defensible postures are (i)
exactness under a linear/GLM working model, (ii) a bounded O(σ_x²) sensitivity
bias, or (iii) a consistent errors-in-variables estimator (corrected-score /
EIV-GP) whose own rate then enters the remainder product. Regime (ii) is
implemented here: ``rc_residual_variance`` is the exact O(σ_x²) driver
τ²_resid = Var(W1_true | observed), and ``residual_bias_report`` shows the
corrected arm's residual bias is dominated by the CI half-width (|bias|/SE ≈
0.07–0.13 at reliability-plausible σ_x, ≤ 0.25 out to σ_x=1.5).
"""
from pathlib import Path

import numpy as np
import pandas as pd

from causal_bench.dgp.survival import DGPConfig, generate_data
from causal_bench.estimators.tmle_ipcw import TMLEIPCWEstimator

OUT_DIR = Path("results/exp32_clever_covariate_me")
_Z_COLS = ["W2", "W3", "W4", "A"]     # error-free covariates RC conditions on


def simulate_me_survival(sigma_x: float, n: int = 1500, seed: int = 0,
                         positivity: float = 0.5):
    """Survival data whose confounder W1 drives A and the hazard; returns
    (df_with_W1_true, w1_true, w1_obs) with w1_obs = w1_true + N(0, σ_x²)."""
    rng = np.random.default_rng(seed)
    w1_true = rng.standard_normal(n)
    cfg = DGPConfig(n=n, positivity_severity=positivity)
    df = generate_data(cfg, W1=w1_true, rng=rng)
    w1_obs = df["W1"].to_numpy() + rng.normal(0.0, sigma_x, len(df))
    return df, df["W1"].to_numpy().copy(), w1_obs


def _rc_system(df: pd.DataFrame, w1_obs: np.ndarray, sigma_x: float):
    """Normal-equation pieces for E[W1_true | W1_obs, Z] under classical error.

    Uses observed moments plus the reliability-study σ_x²: cov(W1_true, W1_obs)
    = var(W1_obs) − σ_x²; cov(W1_true, Z) = cov(W1_obs, Z) for the error-free Z
    (error ⟂ Z). Returns (P, mu, coef, c, var_w1_true)."""
    Z = df[_Z_COLS].to_numpy(float)
    P = np.column_stack([w1_obs, Z])
    mu = P.mean(0)
    Sigma = np.cov(P, rowvar=False, ddof=0)
    var_w1_true = max(P[:, 0].var() - sigma_x**2, 1e-6)
    c = np.empty(P.shape[1])
    c[0] = var_w1_true                                  # cov(W1_true, W1_obs)
    c[1:] = Sigma[0, 1:]                                 # cov(W1_true, Z)=cov(W1_obs, Z)
    coef = np.linalg.solve(Sigma + 1e-9 * np.eye(len(c)), c)
    return P, mu, coef, c, var_w1_true


def regression_calibrate_w1(df: pd.DataFrame, w1_obs: np.ndarray,
                            sigma_x: float) -> np.ndarray:
    """E[W1_true | W1_obs, W2, W3, W4, A] — the RC-corrected confounder."""
    P, mu, coef, _, _ = _rc_system(df, w1_obs, sigma_x)
    return mu[0] + (P - mu) @ coef                      # mean(W1_true)=mean(W1_obs)


def rc_residual_variance(df: pd.DataFrame, w1_obs: np.ndarray,
                         sigma_x: float) -> float:
    """τ²_resid = Var(W1_true | W1_obs, Z) — the confounder variation RC CANNOT
    recover (the unpredictable part), hence the driver of RC's residual bias.

    This is the exact, computable **O(σ_x²)** quantity behind the bounded-bias
    sensitivity claim (regime (ii) of the remainder-rate theory note on #66):
    it → 0 as σ_x → 0, and τ²_resid/σ_x² → 1 as σ_x → 0. Both arms leave this
    variance unadjusted; the corrected arm's residual bias is bounded by the
    confounding strength times τ²_resid, which the report below shows is
    dominated by the CI half-width."""
    _, _, coef, c, var_w1_true = _rc_system(df, w1_obs, sigma_x)
    return float(var_w1_true - c @ coef)                # var − cᵀΣ⁻¹c


def arm_frame(df: pd.DataFrame, arm: str, w1_true: np.ndarray,
              w1_obs: np.ndarray, sigma_x: float) -> pd.DataFrame:
    """A copy of df whose W1 column is the oracle / naive / corrected version —
    swapping it rebuilds g and the clever covariate on that confounder."""
    out = df.copy()
    if arm == "oracle":
        out["W1"] = w1_true
    elif arm == "naive":
        out["W1"] = w1_obs
    elif arm == "corrected":
        out["W1"] = regression_calibrate_w1(df, w1_obs, sigma_x)
    else:
        raise ValueError(f"unknown arm: {arm!r}")
    return out


def estimate_arm_tmle(df_arm: pd.DataFrame, horizon: float = 1.0,
                      n_folds: int = 3) -> dict:
    r = TMLEIPCWEstimator(n_folds=n_folds).estimate(df_arm, horizon=horizon)[0]
    return {"point": float(r.point_estimate), "se": float(r.standard_error),
            "ci_lo": float(r.ci_lower), "ci_hi": float(r.ci_upper)}


def reference_truth(n: int = 40000, seed: int = 777, positivity: float = 0.5,
                    horizon: float = 1.0) -> float:
    """MC truth ψ₀: the oracle ATE (adjusting for W1_true) at large n."""
    df, w1_true, w1_obs = simulate_me_survival(0.0, n=n, seed=seed, positivity=positivity)
    return estimate_arm_tmle(arm_frame(df, "oracle", w1_true, w1_obs, 0.0), horizon)["point"]


def corrected_bootstrap_ci(df: pd.DataFrame, w1_obs: np.ndarray, sigma_x: float,
                           B: int = 120, seed: int = 0, horizon: float = 1.0):
    """Row-bootstrap CI for the corrected arm that re-runs RC + TMLE per
    replicate, capturing the calibration variance the TMLE SE omits."""
    from causal_bench.bootstrap import row_bootstrap_ci

    work = df.copy()
    work["_w1obs"] = w1_obs

    def estimator(sub: pd.DataFrame) -> float:
        s = sub.copy()
        s["W1"] = regression_calibrate_w1(s, s["_w1obs"].to_numpy(), sigma_x)
        return estimate_arm_tmle(s.drop(columns="_w1obs"), horizon)["point"]

    return row_bootstrap_ci(estimator, work, B=B, method="percentile", seed=seed)


def run_me_sweep(sigmas, psi0: float, n_sims: int = 30, n: int = 1200,
                 seed: int = 32) -> pd.DataFrame:
    """Per (σ_x, arm): mean ATE, bias vs ψ₀, |bias|, and CI coverage of ψ₀."""
    rows = []
    for i, s in enumerate(sigmas):
        acc = {a: {"bias": [], "cover": []} for a in ("oracle", "naive", "corrected")}
        for r in range(n_sims):
            df, w1t, w1o = simulate_me_survival(float(s), n=n, seed=seed + 100 * i + r)
            for a in acc:
                e = estimate_arm_tmle(arm_frame(df, a, w1t, w1o, float(s)))
                acc[a]["bias"].append(e["point"] - psi0)
                acc[a]["cover"].append(e["ci_lo"] <= psi0 <= e["ci_hi"])
        for a, d in acc.items():
            b = np.array(d["bias"])
            rows.append({"sigma_x": float(s), "arm": a,
                         "bias_mean": float(b.mean()), "abs_bias": float(np.abs(b).mean()),
                         "coverage": float(np.mean(d["cover"]))})
    return pd.DataFrame(rows)


def residual_bias_report(sigmas, n_sims: int = 12, n: int = 1500,
                         seed: int = 32) -> pd.DataFrame:
    """Bounded-bias sensitivity table (regime (ii), #66 theory note).

    Per σ_x: the exact O(σ_x²) driver τ²_resid = Var(W1_true | observed), the
    corrected arm's empirical residual bias vs oracle (on shared replicates),
    the mean corrected-arm SE, and the ratio |bias| / SE. The residual RC bias
    is bounded by (confounding strength)·τ²_resid; the report demonstrates it is
    dominated by the CI half-width at reliability-plausible σ_x — the defensible
    regulatory claim, in place of a false "RC preserves the remainder rate"."""
    rows = []
    for i, s in enumerate(sigmas):
        t2, dbias, ses = [], [], []
        for r in range(n_sims):
            df, w1t, w1o = simulate_me_survival(float(s), n=n, seed=seed + 50 * i + r)
            t2.append(rc_residual_variance(df, w1o, float(s)))
            o = estimate_arm_tmle(arm_frame(df, "oracle", w1t, w1o, float(s)))["point"]
            c = estimate_arm_tmle(arm_frame(df, "corrected", w1t, w1o, float(s)))
            dbias.append(c["point"] - o)
            ses.append(c["se"])
        t2, dbias, ses = np.array(t2), np.array(dbias), np.array(ses)
        rows.append({"sigma_x": float(s), "tau2_resid": float(t2.mean()),
                     "corrected_bias_vs_oracle": float(dbias.mean()),
                     "abs_bias": float(np.abs(dbias).mean()),
                     "mean_se": float(ses.mean()),
                     "abs_bias_over_se": float(np.abs(dbias).mean() / ses.mean())})
    return pd.DataFrame(rows)


def run(seed: int = 32):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    psi0 = reference_truth()
    sweep = run_me_sweep([0.0, 0.5, 1.0, 1.5], psi0, n_sims=30, n=1200, seed=seed)
    sweep.to_parquet(OUT_DIR / "sweep.parquet", index=False)
    bound = residual_bias_report([0.25, 0.5, 1.0, 1.5], n_sims=20, n=1200, seed=seed)
    bound.to_parquet(OUT_DIR / "residual_bias_bound.parquet", index=False)
    pd.set_option("display.float_format", lambda v: f"{v:0.4f}")
    print(f"reference truth psi0 = {psi0:.4f}")
    print(sweep.to_string(index=False))
    print("\nbounded-bias sensitivity (tau2_resid is O(sigma_x^2); "
          "residual bias dominated by SE):")
    print(bound.to_string(index=False))
    return psi0, sweep, bound


if __name__ == "__main__":
    run()
