"""Immortal-time-bias DGP — the estimator-proof honest null (#21/exp23, ENCIRCLE).

A device/procedure patient must SURVIVE from eligibility (time-zero) to implant.
If the analysis classifies patients as "device" from time-zero, the eligibility→
implant waiting window is *immortal* for the device group (they could not have
died in it, or they would be controls) — so the device looks protective even when
it does nothing. The bias lives in the **data construction (mis-aligned time-zero)**,
not in any estimator, so covariate adjustment / AIPW cannot remove it. Only a
DESIGN fix — a landmark (shown here) or clone-censor-weight for a grace period —
recovers the null.

Honest null: `hazard_ratio = 1.0` (device has no effect), so any nonzero estimated
effect is pure immortal-time bias.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ImmortalTimeConfig:
    horizon: float = 3.0         # administrative follow-up end
    landmark: float = 1.0        # the design fix: align time-zero at L
    hazard_ratio: float = 1.0    # TRUE device effect (1.0 = honest null)
    base_rate: float = 0.5       # baseline event rate
    beta_x: float = 0.5          # covariate → hazard (a real confounder to adjust for)
    implant_rate: float = 0.7    # rate of the eligibility→implant waiting time


def draw_immortal_time(n: int, seed: int, config: ImmortalTimeConfig = ImmortalTimeConfig()) -> pd.DataFrame:
    """Columns: X (baseline covariate), w (implant/waiting time), T (true event
    time from eligibility), treated (received device: survived to implant before
    horizon), Y (death within horizon). All times measured from eligibility."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal(n)
    rate = config.base_rate * np.exp(config.beta_x * X)
    T = rng.exponential(1.0 / rate)                       # event time; device-independent under null
    w = rng.exponential(1.0 / config.implant_rate, n)     # eligibility → implant waiting time
    # device received iff the patient survives to implant AND implant is within follow-up
    treated = ((w < T) & (w < config.horizon)).astype(float)
    Y = (T <= config.horizon).astype(float)               # death within horizon
    return pd.DataFrame({"X": X, "w": w, "T": T, "treated": treated, "Y": Y})


def naive_risk_difference(df: pd.DataFrame) -> float:
    """P(death | received device) − P(death | not) with time-zero at eligibility —
    the immortal-time-biased contrast. Negative = spurious 'device protective'."""
    t = df["treated"].to_numpy().astype(bool)
    return float(df["Y"].to_numpy()[t].mean() - df["Y"].to_numpy()[~t].mean())


def adjusted_effect(df: pd.DataFrame, cols=("X",)) -> float:
    """OLS coefficient on `treated` in Y ~ treated + X — 'adjust for confounders'.
    Still immortal-time-biased: the bias is not confounding by X."""
    n = len(df)
    Xmat = np.column_stack([np.ones(n), df["treated"].to_numpy(),
                            *[df[c].to_numpy() for c in cols]])
    beta, *_ = np.linalg.lstsq(Xmat, df["Y"].to_numpy(), rcond=None)
    return float(beta[1])


def landmark_risk_difference(df: pd.DataFrame, config: ImmortalTimeConfig = ImmortalTimeConfig()) -> float:
    """The DESIGN fix. Restrict to patients alive at the landmark L, classify by
    device status AT L, and count events only in (L, horizon]. Aligned time-zero
    removes the immortal window → recovers the null."""
    L, H = config.landmark, config.horizon
    alive_at_L = df["T"].to_numpy() > L
    d = df[alive_at_L]
    treated_L = (d["w"].to_numpy() <= L)                  # device by the landmark
    event_after_L = (d["T"].to_numpy() <= H)              # death in (L, H] (all are > L)
    return float(event_after_L[treated_L].mean() - event_after_L[~treated_L].mean())
