"""Detection metrics for exogenous-shift detectors (#46).

Scores a detector's per-turn signal against the hidden shock label: how well does
|nc_residual| discriminate the step after an exogenous shock from quiet steps?
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


def detection_roc(scored_df: pd.DataFrame, e_label, target_fpr: float = 0.1) -> dict:
    """ROC AUC and power-at-fixed-FPR for the |nc_residual| detector.

    Positive class = the previous step fired a shock (so the latent state jumped).
    Rows with a NaN residual (first turn of each trajectory) are dropped. Returns
    NaN metrics if the label is degenerate (all/none positive).
    """
    score = scored_df["nc_residual"].abs().to_numpy()
    y = np.asarray(e_label, dtype=float)
    mask = ~np.isnan(score)
    score, y = score[mask], y[mask]
    if y.sum() == 0 or y.sum() == len(y):
        return {"auc": float("nan"), "power_at_fpr": float("nan")}
    auc = float(roc_auc_score(y, score))
    fpr, tpr, _ = roc_curve(y, score)
    power = float(np.interp(target_fpr, fpr, tpr))
    return {"auc": auc, "power_at_fpr": power}


def threshold_at_fpr(scored_df: pd.DataFrame, e_label, target_fpr: float = 0.1) -> float:
    """Detection cutoff c for the |nc_residual| detector at a target false-positive rate.

    c is the (1 − target_fpr) quantile of |nc_residual| over quiet steps (previous
    step fired no shock). Flagging when |nc_residual| > c then false-alarms on
    quiet steps at ≈ target_fpr. NaN residuals (first turns) are dropped.
    """
    score = scored_df["nc_residual"].abs().to_numpy()
    y = np.asarray(e_label, dtype=float)
    mask = ~np.isnan(score)
    quiet = score[mask][y[mask] == 0]
    return float(np.quantile(quiet, 1.0 - target_fpr))
