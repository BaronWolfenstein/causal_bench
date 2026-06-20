"""Hawthorne DGP for Exp 12: panel data with staggered app deployment.

Generates a SITE × PERIOD panel dataset. Each site adopts a monitoring app
at a different calendar time (staggered deployment). The observed outcome Y
for a site-period is a composite of:

  1. Durable effect:   permanent improvement proportional to compliance
  2. Hawthorne effect: transient spike at adoption that decays with a known halflife
  3. Learning curve:   operators improve over experience, plateauing after N periods
  4. Secular trend:    outcome improves over calendar time regardless of app
  5. Site heterogeneity: fixed site-level baseline differences

The TRUE decomposition is stored in the DataFrame so estimators can be
evaluated against it.

Estimands for Exp 12:
  ATT(event_time=k) = true_total_effect at k periods since deployment
                    = durable_effect * compliance_k + hawthorne_effect * 2^(-k/halflife)
                      + learning_curve * min(k, plateau)
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, model_validator


class HawthorneConfig(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    n_sites: int = Field(20, ge=4)
    n_periods: int = Field(12, ge=4)
    patients_per_site_period: int = Field(10, ge=1)

    # Staggered deployment: period in [deploy_window[0], deploy_window[1])
    # Sites with index >= n_treated_sites are never-treated controls
    deploy_window: tuple[int, int] = (3, 9)
    n_treated_sites: Optional[int] = None  # None = 80% of n_sites

    # App compliance: logistic ramp from 0 → compliance_steady_state
    compliance_ramp_speed: float = Field(0.5, gt=0.0)
    compliance_steady_state: float = Field(0.7, ge=0.0, le=1.0)
    compliance_noise: float = Field(0.1, ge=0.0)

    # Treatment effect decomposition
    durable_effect: float = -0.03     # permanent improvement per 10pp compliance
    hawthorne_effect: float = -0.05   # transient spike at adoption
    hawthorne_halflife: float = Field(2.0, gt=0.0)   # periods until Hawthorne halves

    # Time trends
    secular_trend: float = -0.01      # outcome improves over calendar time
    learning_curve: float = 0.02      # outcome improvement per period of experience
    learning_curve_plateau: int = Field(6, ge=1)

    # Site heterogeneity
    site_effect_sd: float = Field(0.02, ge=0.0)
    site_response_sd: float = Field(0.01, ge=0.0)

    # Outcome noise
    outcome_sd: float = Field(0.02, ge=0.0)
    baseline_rate: float = 0.5

    seed: int = 42

    @model_validator(mode="after")
    def _check_deploy_window(self) -> "HawthorneConfig":
        lo, hi = self.deploy_window
        if lo < 1 or hi > self.n_periods or lo >= hi:
            raise ValueError(
                f"deploy_window={self.deploy_window} must satisfy "
                f"1 <= lo < hi <= n_periods={self.n_periods}"
            )
        return self


def generate_hawthorne_data(config: HawthorneConfig) -> pd.DataFrame:
    """Generate site × period panel dataset.

    Returns
    -------
    DataFrame with columns:
        site, period, Y, D, compliance, deploy_period, event_time,
        durable_comp, hawthorne_comp, learning_comp, secular_comp, true_total
    """
    rng = np.random.default_rng(config.seed)

    n_treated = config.n_treated_sites if config.n_treated_sites is not None \
                else max(1, int(config.n_sites * 0.8))
    n_never = config.n_sites - n_treated

    # Site-level random effects
    site_effects = rng.normal(0, config.site_effect_sd, config.n_sites)
    site_responses = rng.normal(0, config.site_response_sd, config.n_sites)  # effect heterogeneity

    # Deployment periods for treated sites (never-treated get deploy_period=None)
    lo, hi = config.deploy_window
    deploy_periods: list[Optional[int]] = [
        int(rng.integers(lo, hi)) for _ in range(n_treated)
    ] + [None] * n_never

    rows = []
    for s in range(config.n_sites):
        dp = deploy_periods[s]
        for t in range(1, config.n_periods + 1):
            deployed = dp is not None and t >= dp
            t_since = int(t - dp) if deployed else 0

            # Compliance: logistic ramp after deployment, 0 before
            if deployed:
                compliance_mean = config.compliance_steady_state * (
                    1 - np.exp(-config.compliance_ramp_speed * t_since)
                )
                compliance = float(np.clip(
                    compliance_mean + rng.normal(0, config.compliance_noise),
                    0.0, 1.0,
                ))
            else:
                compliance = 0.0

            # Effect components (each is 0 for never-treated and pre-deployment)
            if deployed:
                durable_comp = config.durable_effect * compliance * (1 + site_responses[s])
                hawthorne_comp = config.hawthorne_effect * np.exp(
                    -t_since * np.log(2) / config.hawthorne_halflife
                )
                learning_comp = config.learning_curve * min(t_since, config.learning_curve_plateau)
            else:
                durable_comp = hawthorne_comp = learning_comp = 0.0

            secular_comp = config.secular_trend * t
            true_total = durable_comp + hawthorne_comp + learning_comp

            Y = (
                config.baseline_rate
                + site_effects[s]
                + secular_comp
                + true_total
                + rng.normal(0, config.outcome_sd)
            )

            rows.append({
                "site": s,
                "period": t,
                "Y": float(Y),
                "D": int(deployed),
                "compliance": compliance,
                "deploy_period": dp,
                "event_time": t_since if deployed else None,
                "durable_comp": durable_comp,
                "hawthorne_comp": hawthorne_comp,
                "learning_comp": learning_comp,
                "secular_comp": secular_comp,
                "true_total": true_total,
            })

    return pd.DataFrame(rows)


def true_effect_decomposition(
    config: HawthorneConfig,
    max_event_time: Optional[int] = None,
) -> pd.DataFrame:
    """Return the true effect decomposition at each event-time.

    Returns DataFrame with columns:
        event_time, durable_comp, hawthorne_comp, learning_comp, true_total
    Using the average compliance at each event-time (across rng realisations).
    """
    if max_event_time is None:
        max_event_time = config.n_periods - 1

    rows = []
    for k in range(max_event_time + 1):
        compliance_mean = config.compliance_steady_state * (
            1 - np.exp(-config.compliance_ramp_speed * k)
        )
        durable = config.durable_effect * compliance_mean
        hawthorne = config.hawthorne_effect * np.exp(-k * np.log(2) / config.hawthorne_halflife)
        learning = config.learning_curve * min(k, config.learning_curve_plateau)
        rows.append({
            "event_time": k,
            "durable_comp": durable,
            "hawthorne_comp": hawthorne,
            "learning_comp": learning,
            "true_total": durable + hawthorne + learning,
        })
    return pd.DataFrame(rows)
