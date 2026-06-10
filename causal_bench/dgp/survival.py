"""Survival DGP for causal_bench.

Implements an AFT model with Weibull-distributed survival times (Gumbel noise),
informative/non-informative censoring, unmeasured confounding, and a negative
control outcome.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from causal_bench.dgp.config import DGPConfig


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


@lru_cache(maxsize=256)
def _calibrate_censoring_scale(censoring_rate: float, horizon: float,
                                censoring_informativeness: float = 0.0) -> float:
    """Scale factor so achieved censoring_rate matches target under given informativeness."""
    if censoring_rate <= 0:
        return 1e10
    rng = np.random.default_rng(0)
    n = 5000
    U = rng.standard_normal(n)
    W1 = rng.standard_normal(n)
    W3 = rng.standard_normal(n)
    A = rng.binomial(1, 0.5, n).astype(float)
    log_T = 0.0 + 0.4 * W1 + 0.3 * U + rng.gumbel(0, 1, n)
    T_true = np.exp(log_T)
    # Include MAR and MNAR components so calibration matches actual generate_data
    log_C_base = (1.5 - 0.2 * W1 + 0.1 * W3 - 0.1 * A
                  + 0.4 * U * censoring_informativeness
                  + rng.gumbel(0, 1, n))
    mnar_weight = max(0.0, censoring_informativeness - 0.5) * 2.0
    if mnar_weight > 0:
        log_C_base -= mnar_weight * (T_true < np.median(T_true)).astype(float)
    C_base = np.exp(log_C_base)
    lo, hi = 0.01, 100.0
    for _ in range(40):
        mid = (lo + hi) / 2
        C = C_base * mid
        censor_rate = np.mean((C < T_true) & (C < horizon))
        if censor_rate > censoring_rate:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def generate_data(config: DGPConfig, rng: np.random.Generator | None = None) -> pd.DataFrame:
    """Generate one simulated clinical trial dataset.

    Parameters
    ----------
    config:
        DGP configuration dataclass.
    rng:
        Optional numpy random Generator.  If None a fresh generator is seeded
        from ``config.seed``.

    Returns
    -------
    pd.DataFrame with columns: T_obs, Delta, A, W1, W2, W3, W4, compliance,
    enrollment_time, Y_neg.  U is never included.
    """
    if rng is None:
        rng = np.random.default_rng(config.seed)

    n = config.n

    # --- Latent + observed covariates ---
    U = rng.standard_normal(n)
    W1 = rng.standard_normal(n)
    W2 = rng.binomial(1, 0.5, n).astype(float)
    W3 = rng.standard_normal(n)
    W4 = rng.binomial(1, 0.3, n).astype(float)
    enrollment_time = rng.uniform(0, config.enrollment_period, n)

    # --- Treatment assignment ---
    # Logit intercept adjusted for treatment_prevalence baseline
    p = np.clip(config.treatment_prevalence, 1e-6, 1 - 1e-6)
    logit_A = (
        np.log(p / (1 - p))
        + 0.3 * W1
        + 0.2 * W2
        - 0.2 * W3
        + 0.1 * W4
        + 0.5 * U * config.unmeasured_confounding_strength
        + 0.8 * W1 * W3 * config.positivity_severity
    )
    A = rng.binomial(1, _sigmoid(logit_A)).astype(float)

    # --- Survival time (AFT model with Gumbel noise) ---
    gumbel_noise = rng.gumbel(0, 1, n)
    # Intercept 0.0 (not 1.0) so that median T ≈ 1.0 and ~25-40% events occur within horizon=1.0
    log_T = (
        0.0
        + 0.4 * W1
        - 0.3 * W2
        + 0.2 * W3
        - 0.2 * W4
        + 0.3 * U
        + config.true_tau * A
        + config.enrollment_drift * enrollment_time
        + config.outcome_nonlinearity * (W1 ** 2 - 1)
        + config.effect_heterogeneity * A * W1
        + gumbel_noise
    )
    T_true = np.exp(log_T)

    # --- Compliance covariate (correlated with U, observed) ---
    rho = np.sqrt(config.compliance_censoring_r2)
    compliance_raw = rho * U + np.sqrt(1.0 - rho ** 2) * rng.standard_normal(n)
    compliance = _sigmoid(compliance_raw)

    # --- Censoring ---
    scale_factor = _calibrate_censoring_scale(config.censoring_rate, config.horizon,
                                               config.censoring_informativeness)

    gumbel_c = rng.gumbel(0, 1, n)
    log_C_base = (
        1.5
        - 0.2 * W1
        + 0.1 * W3
        - 0.1 * A
        + 0.4 * U * config.censoring_informativeness
        + gumbel_c
    )
    # MNAR component: early events are more likely to be censored
    mnar_weight = max(0.0, config.censoring_informativeness - 0.5) * 2
    if mnar_weight > 0:
        median_T = np.median(T_true)
        log_C_base -= mnar_weight * (T_true < median_T).astype(float)

    C = np.exp(log_C_base) * scale_factor

    # --- Observed outcomes ---
    T_obs = np.minimum(T_true, np.minimum(C, config.horizon))
    Delta = ((T_true <= C) & (T_true <= config.horizon)).astype(float)

    # --- Negative control outcome (no treatment effect) ---
    Y_neg = 0.5 * W1 - 0.3 * W3 + 0.4 * U + rng.normal(0, 0.5, n)

    return pd.DataFrame({
        "T_obs": T_obs,
        "Delta": Delta,
        "A": A,
        "W1": W1,
        "W2": W2,
        "W3": W3,
        "W4": W4,
        "compliance": compliance,
        "enrollment_time": enrollment_time,
        "Y_neg": Y_neg,
    })


def compute_true_effects(config: DGPConfig, n_ref: int = 50_000) -> dict:
    """Estimate true ATE and ATT via a large reference population.

    Uses shared covariates and shared Gumbel noise so that only treatment
    assignment varies between potential-outcome arms.

    Parameters
    ----------
    config:
        DGP configuration.
    n_ref:
        Size of the reference population (default 50 000).

    Returns
    -------
    dict with keys "ATE" and "ATT" (floats).
    """
    rng = np.random.default_rng(config.seed ^ 0xDEADBEEF)

    # Shared covariates
    U = rng.standard_normal(n_ref)
    W1 = rng.standard_normal(n_ref)
    W2 = rng.binomial(1, 0.5, n_ref).astype(float)
    W3 = rng.standard_normal(n_ref)
    W4 = rng.binomial(1, 0.3, n_ref).astype(float)
    enrollment_time = rng.uniform(0, config.enrollment_period, n_ref)

    # Observed treatment (for ATT)
    p = np.clip(config.treatment_prevalence, 1e-6, 1 - 1e-6)
    logit_A = (
        np.log(p / (1 - p))
        + 0.3 * W1
        + 0.2 * W2
        - 0.2 * W3
        + 0.1 * W4
        + 0.5 * U * config.unmeasured_confounding_strength
        + 0.8 * W1 * W3 * config.positivity_severity
    )
    A_obs = rng.binomial(1, _sigmoid(logit_A)).astype(float)

    # Shared Gumbel noise for potential outcomes
    gumbel_noise = rng.gumbel(0, 1, n_ref)

    def _log_T(a_val: float) -> np.ndarray:
        return (
            0.0
            + 0.4 * W1
            - 0.3 * W2
            + 0.2 * W3
            - 0.2 * W4
            + 0.3 * U
            + config.true_tau * a_val
            + config.enrollment_drift * enrollment_time
            + config.outcome_nonlinearity * (W1 ** 2 - 1)
            + config.effect_heterogeneity * a_val * W1
            + gumbel_noise
        )

    T1 = np.exp(_log_T(1.0))
    T0 = np.exp(_log_T(0.0))

    # Binary event indicator within horizon
    Y1 = (T1 <= config.horizon).astype(float)
    Y0 = (T0 <= config.horizon).astype(float)

    diff = Y1 - Y0
    ATE = float(np.mean(diff))
    ATT = float(np.mean(diff[A_obs == 1]))

    return {"ATE": ATE, "ATT": ATT}
