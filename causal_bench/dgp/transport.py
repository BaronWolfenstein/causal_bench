"""Transport DGP for Exp 11: trial-to-commercial generalizability.

Generates TWO populations from the same underlying causal model but with
different covariate distributions. The trial (source) oversamples sicker
patients (higher W1 mean); the commercial (target) has the broader real-world
distribution. Transport heterogeneity controls how much the treatment effect
varies with W1, determining whether naive transport (just reuse the trial ATE)
is biased.

Two heterogeneity patterns:
  "symmetric"  — ATE is the same in both populations in aggregate, but
                 quantile-specific ATEs diverge (GALILEO insight: overall
                 numbers look fine, subgroup analysis reveals problems).
  "asymmetric" — ATE is systematically different in trial vs commercial
                 because W1 has a different mean in each population.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


class TransportConfig(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    # Shared outcome model (same structural equations for both populations)
    true_tau: float = -0.3
    horizon: float = Field(1.0, gt=0.0)
    outcome_nonlinearity: float = 0.0

    # Effect modification: treatment effect varies linearly with W1
    # 0 = homogeneous tau; 1 = tau doubles when W1 shifts by 1 SD
    transport_heterogeneity: float = Field(0.0, ge=0.0, le=1.0)

    # Divergence pattern determines how heterogeneity produces transport bias
    # "none":       tau_eff = true_tau everywhere (no HTE regardless of heterogeneity)
    # "symmetric":  tau_eff = true_tau + hetero*true_tau*(W1 - pop_mean(W1))
    #               → E[ATE] identical in both populations; quantile ATEs diverge
    # "asymmetric": tau_eff = true_tau + hetero*true_tau*W1
    #               → E[ATE] differs because E[W1] differs across populations
    divergence_pattern: Literal["none", "symmetric", "asymmetric"] = "none"

    # Trial (source): sicker, narrower covariate range
    n_trial: int = Field(700, ge=10, le=50_000)
    trial_W1_mean: float = 0.5
    trial_W1_sd: float = 0.8
    trial_W3_mean: float = 0.3
    trial_W3_sd: float = 0.7
    trial_treatment_prevalence: float = Field(0.5, ge=0.0, le=1.0)
    trial_censoring_rate: float = Field(0.25, ge=0.0, lt=1.0)
    trial_censoring_informativeness: float = Field(0.3, ge=0.0, le=1.0)

    # Commercial (target): broader, real-world covariate distribution
    n_commercial: int = Field(2000, ge=10, le=100_000)
    commercial_W1_mean: float = 0.0
    commercial_W1_sd: float = 1.0
    commercial_W3_mean: float = 0.0
    commercial_W3_sd: float = 1.0
    commercial_treatment_prevalence: float = Field(0.5, ge=0.0, le=1.0)
    commercial_censoring_rate: float = Field(0.15, ge=0.0, lt=1.0)
    commercial_censoring_informativeness: float = Field(0.1, ge=0.0, le=1.0)

    seed: int = 42


def _tau_effective(
    config: TransportConfig,
    W1: np.ndarray,
    W1_pop_mean: float,
) -> np.ndarray:
    """Return patient-level effective tau given divergence pattern."""
    if config.divergence_pattern == "none":
        return np.full(len(W1), config.true_tau)
    elif config.divergence_pattern == "symmetric":
        # Zero-mean HTE within each population → aggregate ATE = true_tau always.
        # But patients with the same W1 value have different tau_eff when the population
        # means differ, so the quantile ATEs diverge between trial and commercial.
        return config.true_tau + config.transport_heterogeneity * config.true_tau * (W1 - W1_pop_mean)
    else:  # "asymmetric"
        # HTE un-centered → aggregate ATE = true_tau * (1 + hetero * E[W1]),
        # which differs between populations because E[W1] differs.
        return config.true_tau + config.transport_heterogeneity * config.true_tau * W1


def _generate_arm(
    config: TransportConfig,
    n: int,
    W1_mean: float,
    W1_sd: float,
    W3_mean: float,
    W3_sd: float,
    treatment_prevalence: float,
    censoring_rate: float,
    censoring_informativeness: float,
    rng: np.random.Generator,
    population: str,
) -> pd.DataFrame:
    """Generate one population dataset (trial or commercial)."""
    U = rng.standard_normal(n)
    W1 = W1_mean + W1_sd * rng.standard_normal(n)
    W2 = rng.binomial(1, 0.5, n).astype(float)
    W3 = W3_mean + W3_sd * rng.standard_normal(n)
    W4 = rng.binomial(1, 0.3, n).astype(float)

    # Treatment assignment (includes some confounding via W1/W3 in both populations)
    p = np.clip(treatment_prevalence, 1e-6, 1 - 1e-6)
    logit_A = (np.log(p / (1 - p)) + 0.3 * W1 + 0.2 * W2 - 0.2 * W3 + 0.1 * W4)
    A = rng.binomial(1, _sigmoid(logit_A)).astype(float)

    # Patient-specific tau
    tau_eff = _tau_effective(config, W1, W1_pop_mean=W1_mean)

    # Survival time (AFT + Gumbel noise, same structural equations as base DGP)
    gumbel = rng.gumbel(0, 1, n)
    log_T = (
        0.0 + 0.4 * W1 - 0.3 * W2 + 0.2 * W3 - 0.2 * W4 + 0.3 * U
        + tau_eff * A
        + config.outcome_nonlinearity * (W1 ** 2 - 1)
        + gumbel
    )
    T_true = np.exp(log_T)

    # Censoring: covariate-dependent + optional MNAR component
    # Scale factor so that censoring_rate fraction are censored before horizon
    gumbel_c = rng.gumbel(0, 1, n)
    log_C_base = (
        1.5 - 0.2 * W1 + 0.1 * W3 - 0.1 * A
        + 0.4 * U * censoring_informativeness
        + gumbel_c
    )
    mnar_weight = max(0.0, censoring_informativeness - 0.5) * 2.0
    if mnar_weight > 0:
        log_C_base -= mnar_weight * (T_true < np.median(T_true)).astype(float)

    # Calibrate censoring scale via bisection on a small reference
    rng_cal = np.random.default_rng(0)
    n_cal = 5000
    _U = rng_cal.standard_normal(n_cal)
    _W1 = W1_mean + W1_sd * rng_cal.standard_normal(n_cal)
    _W3 = W3_mean + W3_sd * rng_cal.standard_normal(n_cal)
    _A = rng_cal.binomial(1, 0.5, n_cal).astype(float)
    _gT = rng_cal.gumbel(0, 1, n_cal)
    _T = np.exp(0.0 + 0.4 * _W1 - 0.3 * W3_mean + 0.3 * _U + config.true_tau * _A + _gT)
    _gC = rng_cal.gumbel(0, 1, n_cal)
    _C_base = np.exp(1.5 - 0.2 * _W1 + 0.1 * _W3 - 0.1 * _A
                     + 0.4 * _U * censoring_informativeness + _gC)
    lo, hi = 0.01, 100.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if np.mean((_C_base * mid < _T) & (_C_base * mid < config.horizon)) > censoring_rate:
            lo = mid
        else:
            hi = mid
    scale = (lo + hi) / 2

    C = np.exp(np.clip(log_C_base, -700, 700)) * scale
    T_obs = np.minimum(T_true, np.minimum(C, config.horizon))
    Delta = ((T_true <= C) & (T_true <= config.horizon)).astype(float)
    Y_binary = Delta  # binary event within horizon

    return pd.DataFrame({
        "T_obs": T_obs,
        "Delta": Delta,
        "Y": Y_binary,
        "A": A,
        "W1": W1,
        "W2": W2,
        "W3": W3,
        "W4": W4,
        "U": U,
        "tau_eff": tau_eff,
        "population": population,
    })


def generate_transport_data(
    config: TransportConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate paired trial and commercial datasets.

    Returns
    -------
    trial_df, commercial_df — each with columns T_obs, Delta, Y, A, W1..W4, population.
    """
    rng = np.random.default_rng(config.seed)

    trial_df = _generate_arm(
        config,
        n=config.n_trial,
        W1_mean=config.trial_W1_mean,
        W1_sd=config.trial_W1_sd,
        W3_mean=config.trial_W3_mean,
        W3_sd=config.trial_W3_sd,
        treatment_prevalence=config.trial_treatment_prevalence,
        censoring_rate=config.trial_censoring_rate,
        censoring_informativeness=config.trial_censoring_informativeness,
        rng=rng,
        population="trial",
    )

    commercial_df = _generate_arm(
        config,
        n=config.n_commercial,
        W1_mean=config.commercial_W1_mean,
        W1_sd=config.commercial_W1_sd,
        W3_mean=config.commercial_W3_mean,
        W3_sd=config.commercial_W3_sd,
        treatment_prevalence=config.commercial_treatment_prevalence,
        censoring_rate=config.commercial_censoring_rate,
        censoring_informativeness=config.commercial_censoring_informativeness,
        rng=rng,
        population="commercial",
    )

    return trial_df, commercial_df


def compute_true_ates(
    config: TransportConfig,
    n_ref: int = 50_000,
) -> dict[str, float]:
    """True ATEs in trial and commercial populations via large reference sample.

    Returns dict with keys "trial_ate" and "commercial_ate".
    """
    rng = np.random.default_rng(config.seed ^ 0xFEEDBEEF)

    def _pop_ate(W1_mean: float, W1_sd: float, W3_mean: float, W3_sd: float) -> float:
        W1 = W1_mean + W1_sd * rng.standard_normal(n_ref)
        W2 = rng.binomial(1, 0.5, n_ref).astype(float)
        W3 = W3_mean + W3_sd * rng.standard_normal(n_ref)
        W4 = rng.binomial(1, 0.3, n_ref).astype(float)
        U = rng.standard_normal(n_ref)
        gumbel = rng.gumbel(0, 1, n_ref)

        tau_eff = _tau_effective(config, W1, W1_pop_mean=W1_mean)
        base = 0.0 + 0.4 * W1 - 0.3 * W2 + 0.2 * W3 - 0.2 * W4 + 0.3 * U + gumbel
        T1 = np.exp(base + tau_eff)
        T0 = np.exp(base)
        Y1 = (T1 <= config.horizon).astype(float)
        Y0 = (T0 <= config.horizon).astype(float)
        return float(np.mean(Y1 - Y0))

    trial_ate = _pop_ate(
        config.trial_W1_mean, config.trial_W1_sd,
        config.trial_W3_mean, config.trial_W3_sd,
    )
    commercial_ate = _pop_ate(
        config.commercial_W1_mean, config.commercial_W1_sd,
        config.commercial_W3_mean, config.commercial_W3_sd,
    )
    return {"trial_ate": trial_ate, "commercial_ate": commercial_ate}
