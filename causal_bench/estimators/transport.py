"""Transport estimators for Exp 11: trial-to-commercial generalizability.

Each estimator answers: "What is the ATE in the commercial population?"
given access to a labeled trial dataset and an (unlabeled) commercial dataset.

  naive           — Use the trial ATE as-is (ignores population shift)
  ipsw            — Inverse probability of sampling weighting (density ratio)
  g_transport     — Outcome regression on trial, predict at commercial covariates
  dr_transport    — Doubly-robust: augmented IPSW + outcome regression
  quantile        — Quantile-specific ATEs by W1 quintile in both populations
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler


_COVARIATES = ["W1", "W2", "W3", "W4"]


@dataclass
class TransportEstimate:
    method: str
    trial_ate: float
    commercial_ate: float   # estimated ATE in the commercial population
    se: float = float("nan")
    weights_max: float = float("nan")   # max IPSW weight (overlap diagnostic)
    weights_effective_n: float = float("nan")  # ESS


def _binary_outcome(df: pd.DataFrame) -> np.ndarray:
    """Return binary event indicator from whichever column is available."""
    if "Y" in df.columns:
        return df["Y"].values.astype(float)
    return df["Delta"].values.astype(float)


def _naive_ate(df: pd.DataFrame) -> float:
    """Simple difference-in-means on observed binary outcomes."""
    treated = _binary_outcome(df[df["A"] == 1])
    control = _binary_outcome(df[df["A"] == 0])
    return float(np.mean(treated) - np.mean(control))


# ─── 1. Naive ────────────────────────────────────────────────────────────────

def transport_naive(
    trial_df: pd.DataFrame,
    commercial_df: pd.DataFrame,
) -> TransportEstimate:
    """Use the trial ATE as-is — assumes trial population = commercial population."""
    trial_ate = _naive_ate(trial_df)
    commercial_obs_ate = _naive_ate(commercial_df)
    return TransportEstimate(
        method="naive",
        trial_ate=trial_ate,
        commercial_ate=trial_ate,  # naive: copy trial ATE to commercial
    )


# ─── 2. IPSW ─────────────────────────────────────────────────────────────────

def transport_ipsw(
    trial_df: pd.DataFrame,
    commercial_df: pd.DataFrame,
) -> TransportEstimate:
    """Inverse probability of sampling weighting.

    Estimate P(S=trial | W) via logistic regression on the pooled dataset.
    Weight each trial patient by P(S=commercial | W_i) / P(S=trial | W_i)
    to match the commercial covariate distribution, then compute weighted
    difference-in-means.
    """
    # Pool datasets with population indicator
    trial_X = trial_df[_COVARIATES].values
    comm_X = commercial_df[_COVARIATES].values

    X_all = np.vstack([trial_X, comm_X])
    S_all = np.concatenate([
        np.ones(len(trial_df)),
        np.zeros(len(commercial_df)),
    ])

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X_all)

    lr = LogisticRegression(max_iter=500, C=1.0)
    lr.fit(X_sc, S_all)

    # P(S=trial | W) for trial patients
    trial_X_sc = scaler.transform(trial_X)
    p_trial = lr.predict_proba(trial_X_sc)[:, 1]
    p_trial = np.clip(p_trial, 0.02, 0.98)  # truncate extreme weights

    # Density ratio: w_i ∝ P(commercial) / P(trial)
    w_raw = (1.0 - p_trial) / p_trial
    w_norm = w_raw / w_raw.sum() * len(trial_df)

    # Propensity score within trial (for treated/control split)
    Y = _binary_outcome(trial_df)
    A = trial_df["A"].values.astype(float)

    # Hajek-style weighted ATE
    mask1 = A == 1
    mask0 = A == 0
    mu1 = np.sum(w_norm[mask1] * Y[mask1]) / np.sum(w_norm[mask1])
    mu0 = np.sum(w_norm[mask0] * Y[mask0]) / np.sum(w_norm[mask0])
    commercial_ate = float(mu1 - mu0)
    trial_ate = _naive_ate(trial_df)

    ess = float((w_raw.sum() ** 2) / (w_raw ** 2).sum())

    return TransportEstimate(
        method="ipsw",
        trial_ate=trial_ate,
        commercial_ate=commercial_ate,
        weights_max=float(w_norm.max()),
        weights_effective_n=ess,
    )


# ─── 3. G-transport (outcome regression) ─────────────────────────────────────

def transport_g(
    trial_df: pd.DataFrame,
    commercial_df: pd.DataFrame,
) -> TransportEstimate:
    """G-formula transport: fit E[Y(a)|W] on trial, predict at commercial W.

    ATE_commercial = E_commercial[E[Y(1)|W]] - E_commercial[E[Y(0)|W]]
    """
    trial_X = trial_df[_COVARIATES].values
    Y = _binary_outcome(trial_df)
    A = trial_df["A"].values.astype(float)

    # Fit one outcome model including A as a covariate
    XA = np.column_stack([trial_X, A])
    scaler = StandardScaler()
    XA_sc = scaler.fit_transform(XA)

    from sklearn.linear_model import LogisticRegression as LR
    m = LR(max_iter=500, C=1.0)
    m.fit(XA_sc, Y)

    # Predict at commercial covariates under A=1 and A=0
    comm_X = commercial_df[_COVARIATES].values
    n_comm = len(commercial_df)

    def _pred(a_val: float) -> np.ndarray:
        XA_pred = np.column_stack([comm_X, np.full(n_comm, a_val)])
        return m.predict_proba(scaler.transform(XA_pred))[:, 1]

    mu1 = float(np.mean(_pred(1.0)))
    mu0 = float(np.mean(_pred(0.0)))
    commercial_ate = mu1 - mu0
    trial_ate = _naive_ate(trial_df)

    return TransportEstimate(
        method="g_transport",
        trial_ate=trial_ate,
        commercial_ate=commercial_ate,
    )


# ─── 4. DR-transport (doubly-robust) ─────────────────────────────────────────

def transport_dr(
    trial_df: pd.DataFrame,
    commercial_df: pd.DataFrame,
) -> TransportEstimate:
    """Doubly-robust transport: IPSW + outcome regression augmentation.

    Consistent if either the sampling model (IPSW density ratio) or the
    outcome model (g-transport) is correctly specified.

    Uses the augmented IPSW estimator (Dahabreh et al. 2020):
        ATE_DR = E_commercial[ŷ(1,W) - ŷ(0,W)]
               + (1/n_trial) * sum [ w_i * A_i*(Y_i - ŷ(1,W_i))/π_i
                                   - w_i * (1-A_i)*(Y_i - ŷ(0,W_i))/(1-π_i) ]
    where w_i = P(commercial|W_i)/P(trial|W_i) and π_i = P(A=1|W_i) in trial.
    """
    trial_X = trial_df[_COVARIATES].values
    comm_X = commercial_df[_COVARIATES].values
    Y = _binary_outcome(trial_df)
    A = trial_df["A"].values.astype(float)

    # --- Sampling model (density ratio) ---
    X_all = np.vstack([trial_X, comm_X])
    S_all = np.concatenate([np.ones(len(trial_df)), np.zeros(len(comm_X))])
    scaler_s = StandardScaler()
    X_all_sc = scaler_s.fit_transform(X_all)
    lr_s = LogisticRegression(max_iter=500, C=1.0)
    lr_s.fit(X_all_sc, S_all)
    p_trial = np.clip(lr_s.predict_proba(scaler_s.transform(trial_X))[:, 1], 0.02, 0.98)
    w = (1 - p_trial) / p_trial
    w /= w.mean()

    # --- Outcome model ---
    XA = np.column_stack([trial_X, A])
    scaler_o = StandardScaler()
    XA_sc = scaler_o.fit_transform(XA)
    lr_o = LogisticRegression(max_iter=500, C=1.0)
    lr_o.fit(XA_sc, Y)

    def _pred_trial(a_val: float) -> np.ndarray:
        XA_p = np.column_stack([trial_X, np.full(len(trial_df), a_val)])
        return lr_o.predict_proba(scaler_o.transform(XA_p))[:, 1]

    def _pred_comm(a_val: float) -> np.ndarray:
        XA_p = np.column_stack([comm_X, np.full(len(comm_X), a_val)])
        return lr_o.predict_proba(scaler_o.transform(XA_p))[:, 1]

    # --- Propensity score within trial ---
    scaler_p = StandardScaler()
    trial_X_sc = scaler_p.fit_transform(trial_X)
    lr_p = LogisticRegression(max_iter=500, C=1.0)
    lr_p.fit(trial_X_sc, A)
    pi = np.clip(lr_p.predict_proba(trial_X_sc)[:, 1], 0.05, 0.95)

    # --- DR estimator ---
    mu1_hat_comm = float(np.mean(_pred_comm(1.0)))
    mu0_hat_comm = float(np.mean(_pred_comm(0.0)))
    g_component = mu1_hat_comm - mu0_hat_comm

    mu1_hat = _pred_trial(1.0)
    mu0_hat = _pred_trial(0.0)
    aug_component = float(np.mean(
        w * (A * (Y - mu1_hat) / pi - (1 - A) * (Y - mu0_hat) / (1 - pi))
    ))

    commercial_ate = g_component + aug_component
    trial_ate = _naive_ate(trial_df)

    return TransportEstimate(
        method="dr_transport",
        trial_ate=trial_ate,
        commercial_ate=commercial_ate,
        weights_max=float(w.max()),
    )


# ─── 5. Quantile transport ────────────────────────────────────────────────────

def transport_quantile(
    trial_df: pd.DataFrame,
    commercial_df: pd.DataFrame,
    quantiles: tuple[float, ...] = (0.10, 0.25, 0.50, 0.75, 0.90),
) -> dict[float, dict[str, float]]:
    """Quantile-specific ATEs by W1 percentile bin.

    W1 is the key treatment effect modifier. Bins patients by W1 quintile
    (defined on the commercial population) and estimates the ATE within each
    bin in both populations.

    Returns
    -------
    dict keyed by quantile label. Each value is a dict with:
        "W1_threshold"  — upper W1 boundary of this quantile bin
        "trial_ate"     — ATE for trial patients in this W1 bin
        "commercial_ate" — ATE for commercial patients in this W1 bin
        "n_trial"       — number of trial patients in this bin
        "n_commercial"  — number of commercial patients in this bin
    """
    # Define bins on commercial W1 distribution (the target population)
    thresholds = np.quantile(commercial_df["W1"].values, quantiles).tolist()
    thresholds = [-np.inf] + thresholds + [np.inf]

    results = {}
    for i, q in enumerate(quantiles):
        lo_w, hi_w = thresholds[i], thresholds[i + 1]
        mask_t = (trial_df["W1"] > lo_w) & (trial_df["W1"] <= hi_w)
        mask_c = (commercial_df["W1"] > lo_w) & (commercial_df["W1"] <= hi_w)

        sub_t = trial_df[mask_t]
        sub_c = commercial_df[mask_c]

        def _ate(df: pd.DataFrame) -> float:
            if len(df) < 5 or df["A"].nunique() < 2:
                return float("nan")
            return float(_naive_ate(df))

        results[q] = {
            "W1_threshold": float(thresholds[i + 1]),
            "trial_ate": _ate(sub_t),
            "commercial_ate": _ate(sub_c),
            "n_trial": int(mask_t.sum()),
            "n_commercial": int(mask_c.sum()),
        }

    return results


# ─── Convenience wrapper ──────────────────────────────────────────────────────

def run_all_transport_estimators(
    trial_df: pd.DataFrame,
    commercial_df: pd.DataFrame,
) -> dict[str, TransportEstimate]:
    """Run all four aggregate transport estimators and return results by name."""
    return {
        "naive":       transport_naive(trial_df, commercial_df),
        "ipsw":        transport_ipsw(trial_df, commercial_df),
        "g_transport": transport_g(trial_df, commercial_df),
        "dr_transport": transport_dr(trial_df, commercial_df),
    }
