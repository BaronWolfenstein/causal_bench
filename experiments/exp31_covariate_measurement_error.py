"""Exp 31: covariate measurement-error sensitivity (Σ_x) — residual confounding.

The exp3-sibling: the *measured-confounder-with-noise* analog of unmeasured
confounding (issue #58). A confounder X_true drives both treatment A and outcome
Y; the analyst sees only a noisy proxy X_obs = X_true + ε, ε ~ N(0, σ_x²).
Adjusting for a noisy proxy leaves **residual confounding** — the back-door path
is not fully blocked — so the causal estimate stays biased even with a
"correctly specified" adjustment set.

Three arms on shared replicates:
- **oracle**   — adjust for X_true (unattainable; the ground-truth benchmark).
- **naive**    — adjust for X_obs as if exact (the default an analyst runs).
- **corrected**— regression calibration: replace X_obs by E[X_true | X_obs, A]
                 (Carroll-Ruppert-Stefanski-Crainiceanu), using the known σ_x²
                 from a reliability study, then re-estimate.

Readout (sensitivity-gradient family): bias and CI coverage of the treatment
effect vs σ_x magnitude, oracle/naive/corrected overlaid; the residual-
confounding **tipping point** (the σ_x at which the naive CI stops covering the
truth), and the fraction of the oracle→naive gap the correction recovers.

Corrected-arm variance: regression calibration restores the *point estimate*,
but the classical OLS SE does not propagate the calibration uncertainty, so with
that SE the corrected arm under-covers at large σ_x. The fix (built here) is a
**row-resampling bootstrap** that re-runs regression-calibration + OLS inside
each replicate — capturing the calibration variance — via the generic
``causal_bench.bootstrap.row_bootstrap_ci`` (complement to the influence-curve
bootstrap; no new dependency, no SIMEX). ``estimate_arm(..., ci="bootstrap")``
selects it; ``compare_corrected_coverage`` shows it restores coverage toward
nominal (e.g. 0.60 → ~1.0 at σ_x=1.5).

GP-propensity sensitivity (issue #57 — HAL stays primary): under covariate
noise a stationary-kernel GP conflates the noise with the kernel length-scale
(it smooths over a noisier input space), biasing the propensity surface. This is
reported as a diagnostic — the fitted length-scale on X_obs inflates vs X_true
and grows with σ_x. At a hard eligibility cutoff the kernel must additionally be
cutoff-aware (a stationary Matérn smooths across the break); that interaction is
documented, not built here.
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR = Path("results/exp31_covariate_me")

TAU_TRUE = 1.0     # true treatment effect on Y
ALPHA = 1.0        # confounding strength into treatment (logit P(A) = ALPHA*X_true)
BETA = 2.0         # confounding strength into outcome (Y = tau*A + BETA*X_true + e)


def simulate_covariate_me(sigma_x: float, n: int = 4000, seed: int = 0,
                          tau: float = TAU_TRUE, alpha: float = ALPHA,
                          beta: float = BETA) -> pd.DataFrame:
    """Confounder X_true drives A and Y; the analyst observes X_obs = X_true + ε."""
    rng = np.random.default_rng(seed)
    x_true = rng.normal(0.0, 1.0, n)
    p = 1.0 / (1.0 + np.exp(-alpha * x_true))
    a = (rng.random(n) < p).astype(float)
    y = tau * a + beta * x_true + rng.normal(0.0, 1.0, n)
    x_obs = x_true + rng.normal(0.0, sigma_x, n)
    return pd.DataFrame({"X_true": x_true, "X_obs": x_obs, "A": a, "Y": y})


def _ols_effect(y: np.ndarray, cols: list[np.ndarray]) -> tuple[float, float]:
    """OLS of y on [1, *cols]; return (coef on cols[0], its classical SE).

    cols[0] is the treatment A, so the returned coefficient is the adjusted
    treatment effect. SE is the classical homoskedastic OLS standard error.
    """
    M = np.column_stack([np.ones_like(y)] + cols)
    XtX_inv = np.linalg.inv(M.T @ M)
    beta_hat = XtX_inv @ M.T @ y
    resid = y - M @ beta_hat
    sigma2 = float(resid @ resid) / (len(y) - M.shape[1])
    cov = sigma2 * XtX_inv
    return float(beta_hat[1]), float(np.sqrt(cov[1, 1]))


def _regression_calibration(df: pd.DataFrame, sigma_x: float) -> np.ndarray:
    """E[X_true | X_obs, A] under classical additive error and known σ_x².

    Uses observed moments plus the reliability-study σ_x²:
    var(X_true) = var(X_obs) − σ_x²; cov(X_true, X_obs) = var(X_true);
    cov(X_true, A) = cov(X_obs, A) (error ⟂ A). Solve the normal equations for
    the linear predictor of X_true from (X_obs, A). This is the standard
    regression-calibration correction (Carroll et al.).
    """
    w, a = df["X_obs"].to_numpy(), df["A"].to_numpy()
    mw, ma = w.mean(), a.mean()
    vW, vA = w.var(), a.var()
    cWA = np.cov(w, a, ddof=0)[0, 1]
    vX = max(vW - sigma_x**2, 1e-6)           # var(X_true), floored
    b = np.array([vX, cWA])                    # [cov(X,W), cov(X,A)]
    Sigma = np.array([[vW, cWA], [cWA, vA]])
    coefs = np.linalg.solve(Sigma, b)
    return mw + coefs[0] * (w - mw) + coefs[1] * (a - ma)


def _corrected_point(df: pd.DataFrame, sigma_x: float) -> float:
    """Regression-calibration + OLS treatment-effect estimate (the full pipeline
    the bootstrap must re-run per replicate to capture calibration variance)."""
    adj = _regression_calibration(df, sigma_x)
    tau_hat, _ = _ols_effect(df["Y"].to_numpy(), [df["A"].to_numpy(), adj])
    return tau_hat


def estimate_arm(df: pd.DataFrame, arm: str, sigma_x: float,
                 ci: str = "classical", B: int = 400, seed: int = 0) -> dict:
    """Adjusted treatment-effect estimate + 95% CI for one arm.

    ci='classical' uses the homoskedastic OLS SE. ci='bootstrap' (corrected arm
    only) row-resamples and re-runs regression-calibration + OLS per replicate,
    capturing the calibration uncertainty the plug-in SE understates.
    """
    a, y = df["A"].to_numpy(), df["Y"].to_numpy()
    if arm == "oracle":
        adj = df["X_true"].to_numpy()
    elif arm == "naive":
        adj = df["X_obs"].to_numpy()
    elif arm == "corrected":
        adj = _regression_calibration(df, sigma_x)
    else:
        raise ValueError(f"unknown arm: {arm!r}")
    tau_hat, se = _ols_effect(y, [a, adj])

    if ci == "bootstrap":
        if arm != "corrected":
            raise ValueError("bootstrap CI is only defined for the corrected arm here")
        from causal_bench.bootstrap import row_bootstrap_ci
        lo, hi = row_bootstrap_ci(lambda sub: _corrected_point(sub, sigma_x),
                                  df, B=B, method="percentile", seed=seed)
    elif ci == "classical":
        lo, hi = tau_hat - 1.96 * se, tau_hat + 1.96 * se
    else:
        raise ValueError(f"unknown ci method: {ci!r}")
    return {"arm": arm, "tau_hat": tau_hat, "se": se, "ci_lo": lo, "ci_hi": hi}


def run_sigma_x_sweep(sigmas, n_sims: int = 200, n: int = 2000, seed: int = 31,
                      tau: float = TAU_TRUE) -> pd.DataFrame:
    """Per (σ_x, arm): mean bias, mean |bias|, and CI coverage over replicates."""
    rows = []
    for i, s in enumerate(sigmas):
        acc = {arm: {"bias": [], "cover": []} for arm in ("oracle", "naive", "corrected")}
        for r in range(n_sims):
            df = simulate_covariate_me(float(s), n=n, seed=seed + 1000 * i + r, tau=tau)
            for arm in acc:
                e = estimate_arm(df, arm, float(s))
                acc[arm]["bias"].append(e["tau_hat"] - tau)
                acc[arm]["cover"].append(e["ci_lo"] <= tau <= e["ci_hi"])
        for arm, d in acc.items():
            bias = np.array(d["bias"])
            rows.append({"sigma_x": float(s), "arm": arm,
                         "bias_mean": float(bias.mean()),
                         "abs_bias": float(np.abs(bias).mean()),
                         "coverage": float(np.mean(d["cover"]))})
    return pd.DataFrame(rows)


def recovery_fraction(sweep: pd.DataFrame) -> pd.DataFrame:
    """Fraction of the oracle→naive |bias| gap the corrected arm closes, per σ_x."""
    rows = []
    for s, g in sweep.groupby("sigma_x"):
        ab = g.set_index("arm")["abs_bias"]
        gap = ab["naive"] - ab["oracle"]
        frac = (ab["naive"] - ab["corrected"]) / gap if gap > 1e-9 else float("nan")
        rows.append({"sigma_x": float(s), "gap": float(gap),
                     "recovery_fraction": float(frac)})
    return pd.DataFrame(rows)


def tipping_point(sweep: pd.DataFrame, coverage_floor: float = 0.90) -> float:
    """Smallest σ_x at which the NAIVE arm's CI coverage drops below the floor —
    the residual-confounding tipping point. NaN if naive never fails on the grid."""
    naive = sweep[sweep.arm == "naive"].sort_values("sigma_x")
    failed = naive[naive.coverage < coverage_floor]
    return float(failed["sigma_x"].iloc[0]) if len(failed) else float("nan")


def compare_corrected_coverage(sigmas, n_sims: int = 40, n: int = 1200,
                               B: int = 300, seed: int = 31) -> pd.DataFrame:
    """Corrected-arm CI coverage under classical SE vs the row-bootstrap, per σ_x.

    The classical SE understates the regression-calibration variance, so it
    under-covers at large σ_x; the bootstrap (which re-runs calibration per
    replicate) restores coverage toward nominal. This is the fix for the
    corrected-arm coverage limitation, with no new dependency and no SIMEX.
    """
    rows = []
    for i, s in enumerate(sigmas):
        cover = {"classical": [], "bootstrap": []}
        for r in range(n_sims):
            df = simulate_covariate_me(float(s), n=n, seed=seed + 1000 * i + r)
            for m in ("classical", "bootstrap"):
                e = estimate_arm(df, "corrected", float(s), ci=m, B=B, seed=seed + r)
                cover[m].append(e["ci_lo"] <= TAU_TRUE <= e["ci_hi"])
        rows.append({"sigma_x": float(s),
                     "coverage_classical": float(np.mean(cover["classical"])),
                     "coverage_bootstrap": float(np.mean(cover["bootstrap"]))})
    return pd.DataFrame(rows)


def gp_length_scale(df: pd.DataFrame, on: str = "X_obs") -> float:
    """Fitted RBF length-scale of a GP propensity surface P(A | X) using column
    `on`. Under input noise the length-scale inflates (the GP smooths over a
    noisier input) — the length-scale↔noise conflation of issue #58."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel

    x = df[on].to_numpy().reshape(-1, 1)
    a = df["A"].to_numpy()
    idx = np.argsort(x.ravel())[:: max(1, len(x) // 400)]   # subsample for speed
    kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3)) + \
        WhiteKernel(noise_level=0.25, noise_level_bounds=(1e-3, 1e1))
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=0)
    gp.fit(x[idx], a[idx])
    return float(gp.kernel_.k1.length_scale)


def run(seed: int = 31):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sigmas = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]
    sweep = run_sigma_x_sweep(sigmas, n_sims=200, n=2000, seed=seed)
    rec = recovery_fraction(sweep)
    sweep.to_parquet(OUT_DIR / "sigma_x_sweep.parquet", index=False)
    rec.to_parquet(OUT_DIR / "recovery_fraction.parquet", index=False)
    tp = tipping_point(sweep)

    ls = pd.DataFrame([
        {"sigma_x": s,
         "ls_X_true": gp_length_scale(simulate_covariate_me(s, n=2000, seed=seed), "X_true"),
         "ls_X_obs": gp_length_scale(simulate_covariate_me(s, n=2000, seed=seed), "X_obs")}
        for s in [0.0, 0.5, 1.0, 1.5]
    ])
    ls.to_parquet(OUT_DIR / "gp_length_scale.parquet", index=False)

    pd.set_option("display.float_format", lambda v: f"{v:0.3f}")
    print(sweep.to_string(index=False))
    print("\nrecovery of oracle->naive gap:\n", rec.to_string(index=False))
    print(f"\nresidual-confounding tipping point (naive coverage < 0.90): sigma_x = {tp}")
    print("\nGP length-scale conflation (X_obs inflates vs X_true):\n", ls.to_string(index=False))
    return sweep, rec, ls


if __name__ == "__main__":
    run()
