"""CFG guidance calibration — sweep `guidance_scale` so guided generation LANDS
in the rare region R (indistinguishable from real rare, separation AUC ~ 0.5)
rather than overshooting past it (AUC → 1, which the diagnostic reads as a failed
B″ landing → `smc_required`).

Self-contained: scores landing with an sklearn separation AUC, mirroring the
diagnostic's B″ landing metric WITHOUT importing `run_diagnostic` — this keeps
the generative → diagnostics dependency one-way. The tuning knob is
`guidance_scale`: too low may undershoot, too high overshoots; the returned
`best_scale` is the landing (diffuse_directly) regime.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from .guidance import generate_guided


def landing_auc(guided: np.ndarray, real_rare: np.ndarray) -> float:
    """Separation AUC between guided samples and real rare samples, symmetrized
    to [0.5, 1]. 0.5 ⟺ indistinguishable (guided landed in R); 1.0 ⟺ fully
    separable (guided missed/overshot R). Same signal as the diagnostic's B″
    landing fidelity check, computed locally."""
    guided = np.asarray(guided, float)
    real_rare = np.asarray(real_rare, float)
    X = np.vstack([guided, real_rare])
    y = np.r_[np.ones(len(guided)), np.zeros(len(real_rare))]
    proba = LogisticRegression(max_iter=1000).fit(X, y).predict_proba(X)[:, 1]
    auc = roc_auc_score(y, proba)
    return float(max(auc, 1.0 - auc))


def calibrate_guidance(n, cond_score_fn, uncond_score_fn, sch, real_rare, *,
                       scales=(0.0, 0.5, 1.0, 2.0, 3.0), seed: int = 0,
                       dim: int = 1) -> dict:
    """Sweep CFG `guidance_scale` over `scales`; pick the one whose guided samples
    land in R (``landing_auc`` closest to 0.5). Each scale is generated from the
    SAME initial noise (the RNG is reseeded per scale) so the differences reflect
    the scale, not the draw.

    Returns ``{'table': {scale: auc}, 'best_scale': w*, 'best_auc': auc*}``.
    """
    table: dict = {}
    for w in scales:
        rng = np.random.default_rng(seed)                # identical noise per scale
        guided = generate_guided(n, cond_score_fn, uncond_score_fn, sch, rng,
                                 guidance_scale=float(w), dim=dim)
        table[float(w)] = landing_auc(guided, real_rare)
    best = min(table, key=lambda w: abs(table[w] - 0.5))
    return {"table": table, "best_scale": best, "best_auc": table[best]}
