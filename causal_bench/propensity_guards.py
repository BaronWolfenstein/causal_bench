"""Propensity guards for the ENCIRCLE external-control arm (#173/#174, feeds #99).

Two CPU guards on WHAT enters the propensity — the failure modes exp40 (#174) and
exp42 (#173) demonstrate, plus the embedding-contamination check the manifold
propensity needs because its geometry is built ON the embedding:

- ``outcome_adaptive_screen`` — bias-amplification guard: keep only covariates
  associated with the OUTCOME (drop pure instruments). Screened on the
  covariate–outcome association, NOT conditioning on treatment (that opens a
  collider Z→A←U→Y and would keep the instrument).
- ``era_contamination`` / ``residualize_era`` — calendar guard. Era must be an
  explicit covariate, AND the embedding itself must be checked for era leakage:
  if the frozen encoder encodes calendar, the k-NN graph / heat kernel / geodesics
  built on it are partly an *era* graph, so era is not purely downstream — it
  contaminates the manifold. Check = how well the embedding predicts era; remedy
  = partial era out of the embedding before building the geometry.

All CPU. These sit upstream of / downstream of the A100 geometry kernels, never
inside them (the kernels are era/instrument-agnostic feature producers).
"""
from __future__ import annotations

import numpy as np


def outcome_adaptive_screen(X, y, feature_names=None, *, t_thresh: float = 1.96):
    """Keep feature j iff it is associated with `y` in the covariate–outcome model
    `y ~ X` (|t| on its coefficient > `t_thresh`). Instruments (predict treatment,
    not outcome) drop out; confounders / outcome-predictors stay. Returns kept
    feature names if `feature_names` given, else kept column indices.

    Do NOT pass the treatment as a column of `X`: conditioning on the treatment
    opens a collider for instruments under unmeasured confounding."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, p = X.shape
    Xmat = np.column_stack([np.ones(n), X])
    beta, *_ = np.linalg.lstsq(Xmat, y, rcond=None)
    resid = y - Xmat @ beta
    dof = max(n - Xmat.shape[1], 1)
    sigma2 = float(resid @ resid) / dof
    se = np.sqrt(np.diag(sigma2 * np.linalg.inv(Xmat.T @ Xmat)))
    tvals = beta / se
    keep = [j for j in range(p) if abs(tvals[1 + j]) > t_thresh]
    return [feature_names[j] for j in keep] if feature_names is not None else keep


def era_contamination(embedding, era, *, cv: int = 5, seed: int = 0,
                      r2_flag: float = 0.1) -> dict:
    """How well does the embedding predict calendar era? Cross-validated R² of
    `era ~ embedding` (ridge). High R² ⟹ the embedding encodes era, so the
    manifold geometry built on it is era-contaminated — era is NOT purely a
    downstream covariate and must be residualized (below). Returns
    {r2, contaminated, per_dim_r2}."""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import cross_val_score, KFold
    Z = np.asarray(embedding, dtype=float)
    e = np.asarray(era, dtype=float)
    kf = KFold(n_splits=cv, shuffle=True, random_state=seed)
    r2 = float(np.mean(cross_val_score(Ridge(alpha=1.0), Z, e, cv=kf, scoring="r2")))
    # cheap per-dimension screen: which embedding axes individually track era
    per_dim = np.array([abs(np.corrcoef(Z[:, j], e)[0, 1]) for j in range(Z.shape[1])])
    return {"r2": r2, "contaminated": r2 > r2_flag, "per_dim_abscorr": per_dim}


def residualize_era(embedding, era):
    """Partial era out of the embedding: return `Z − era·β̂` (OLS of each embedding
    column on era, plus intercept). The remedy — feed the residualized embedding
    into `build_knn_laplacian` so the manifold is patient-state, not calendar."""
    Z = np.asarray(embedding, dtype=float)
    e = np.asarray(era, dtype=float).reshape(-1, 1)
    E = np.column_stack([np.ones(len(e)), e])
    beta, *_ = np.linalg.lstsq(E, Z, rcond=None)
    return Z - E @ beta
