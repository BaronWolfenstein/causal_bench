"""Calendar-time (era) confounding + the laundering trap (#173/exp42, ENCIRCLE).

An external/synthetic control is historical; the trial is concurrent → calendar
era drives BOTH membership (A: concurrent vs historical) and outcome (secular
standard-of-care trend). Era is a confounder. The trap specific to a manifold /
embedding propensity: the frozen-encoder embedding captures patient STATE, which
is only an *imperfect* proxy for era — adjusting for the state proxy **launders**
era and leaves residual confounding, while putting era in EXPLICITLY recovers.

Honest null: `tau = 0`, so any nonzero estimate is calendar confounding.

  E ~ N(0,1)     calendar era (historical → concurrent)
  X ~ N(0,1)     ordinary baseline confounder
  A = 1{ β_ea·E + β_x·X + noise }        membership driven by era
  S = E + η·noise                        patient-state proxy (imperfect era mirror)
  Y = τ·A + β_ey·E + β_x·X + ε           secular trend in E; NOT a function of S given E
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class CalendarConfig:
    beta_ea: float = 1.5     # era → membership (concurrent vs historical)
    beta_ey: float = 1.5     # era → outcome (secular standard-of-care trend)
    beta_x: float = 0.6      # ordinary baseline confounder → both
    state_noise: float = 1.0 # how imperfectly patient-state mirrors era (the launder gap)
    tau: float = 0.0         # true effect (null → estimate == calendar bias)
    sigma_y: float = 1.0


def draw_calendar(n: int, seed: int, config: CalendarConfig = CalendarConfig()) -> pd.DataFrame:
    """Observed columns E (era), X (baseline confounder), S (patient-state proxy
    for era), A (membership), Y (outcome)."""
    rng = np.random.default_rng(seed)
    E = rng.standard_normal(n)
    X = rng.standard_normal(n)
    logit_a = config.beta_ea * E + config.beta_x * X
    A = rng.binomial(1, 1.0 / (1.0 + np.exp(-logit_a))).astype(float)
    S = E + config.state_noise * rng.standard_normal(n)          # imperfect era mirror
    Y = (config.tau * A + config.beta_ey * E + config.beta_x * X
         + config.sigma_y * rng.standard_normal(n))
    return pd.DataFrame({"E": E, "X": X, "S": S, "A": A, "Y": Y})


def true_tau(config: CalendarConfig = CalendarConfig()) -> float:
    return config.tau


def adjusted_effect(df: pd.DataFrame, adjustment_cols) -> float:
    """ATE = OLS coefficient on A in Y ~ A + adjustment_cols."""
    n = len(df)
    Xmat = np.column_stack([np.ones(n), df["A"].to_numpy(),
                            *[df[c].to_numpy() for c in adjustment_cols]])
    beta, *_ = np.linalg.lstsq(Xmat, df["Y"].to_numpy(), rcond=None)
    return float(beta[1])
