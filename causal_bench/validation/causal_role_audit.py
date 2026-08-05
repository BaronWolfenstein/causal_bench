"""Latent-aware collider audit for NAMED causal-role variables (issue #216).

Closes the hidden-parent-collider residual the SDR detection layer flagged: an M-bias collider
`Cm <- Ha, Hy` with latent parents cannot be oriented by PC/`orient_colliders`, and constraint-based
FCI is uninformative on the fully-connected latent-collider (all-circles). LiNGAM exploits
non-Gaussianity to orient it and flag latent confounding.

Ensemble of latent-aware LiNGAM variants with COMPLEMENTARY error modes (verified on the M-structure):
RCD has false negatives (misses latent pairs); ParceLiNGAM has false positives (over-flags). For an
audit the asymmetry is decisive -- a missed collider (false negative) lets bias through -- so we take
the CONSERVATIVE union of latent-confounded flags and rank by agreement, cross-checked against pcalg
FCI where R is available. Certifies the "confounders, not colliders" assumption the double-score
reduction depends on, ON NAMED VARIABLES (ENCIRCLE baseline covariates; not the raw embedding).

Backends by environment: RCD + ParceLiNGAM are pure Python (`lingam`) -> run everywhere incl. the
R-free box; the pcalg FCI cross-check needs R -> Mac/production, `None` otherwise (degrade gracefully).
LiNGAM assumes (near-)linearity + non-Gaussian noise; on mixed/Gaussian data the FCI-nonparametric
cross-check is the more applicable leg. See the design spec (2026-08-05-...-collider-audit-design.md).
"""
from __future__ import annotations

import numpy as np


def _latent_pairs(adj, names):
    """Latent-confounded pairs from a lingam adjacency matrix (a NaN entry = latent common cause)."""
    out = set()
    n = len(names)
    for i in range(n):
        for j in range(i + 1, n):
            if np.isnan(adj[i, j]) or np.isnan(adj[j, i]):
                out.add(frozenset((names[i], names[j])))
    return out


def lingam_ensemble(X, names):
    """RCD + ParceLiNGAM (both latent-aware). Returns per-method latent-confounded pairs plus the
    conservative union and the agreement set."""
    import lingam
    rcd = _latent_pairs(lingam.RCD().fit(X).adjacency_matrix_, names)
    parce = _latent_pairs(lingam.BottomUpParceLiNGAM().fit(X).adjacency_matrix_, names)
    return {"rcd": rcd, "parcelingam": parce, "union": rcd | parce, "agree": rcd & parce}


def fci_cross_check(X, names, alpha=0.05):
    """pcalg FCI (R via rpy2). Returns the set of bidirected (latent-confounded) pairs it orients,
    or `None` if R/pcalg/rpy2 is unavailable (e.g. the box). An empty set = a PAG that oriented
    nothing latent (often all-circles on a tangled structure)."""
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import numpy2ri
        from rpy2.robjects.conversion import localconverter
        ro.r("suppressMessages(library(pcalg))")
    except Exception:
        return None
    try:
        C = np.corrcoef(np.asarray(X, float).T)
        with localconverter(ro.default_converter + numpy2ri.converter):
            ro.globalenv["C"] = C
        ro.globalenv["nm"] = ro.StrVector(list(names))
        ro.globalenv["nn"] = ro.IntVector([len(X)])
        ro.globalenv["al"] = ro.FloatVector([alpha])
        amat = np.asarray(ro.r("""
            p <- length(nm)
            ss <- list(C = matrix(C, p, p), n = nn[1])
            rownames(ss$C) <- nm; colnames(ss$C) <- nm
            pag <- fci(ss, indepTest = gaussCItest, alpha = al[1], labels = nm)
            pag@amat
        """), dtype=float)
    except Exception:
        return None
    bi = set()
    p = len(names)
    for i in range(p):
        for j in range(i + 1, p):
            if amat[i, j] == 2 and amat[j, i] == 2:          # arrowhead both ends = bidirected (latent)
                bi.add(frozenset((names[i], names[j])))
    return bi


def audit(X, names, treatment="A", outcome="Y"):
    """The audit verdict. Flags candidate covariates that are latent-confounded (collider signature)
    and should be EXCLUDED from the adjustment set, with a confidence tag from method agreement.
    `X` is (n, p) with columns `names`; LiNGAM needs non-Gaussian noise for orientation power."""
    ens = lingam_ensemble(X, names)
    fci = fci_cross_check(X, names)                          # set | None
    candidates = [v for v in names if v not in (treatment, outcome)]

    flagged = {}
    for pair in ens["union"]:
        methods = [m for m in ("rcd", "parcelingam") if pair in ens[m]]
        if fci is None:
            conf = "lingam-only (no R cross-check)"
        elif pair in fci:
            conf = "high (LiNGAM + FCI agree)"
        else:
            conf = "method-dependent (FCI does not orient it)"
        if len(methods) == 1:
            conf += "; single-method flag"
        flagged[tuple(sorted(pair))] = {"methods": methods, "confidence": conf}

    exclude = sorted({v for pair in ens["union"] for v in pair if v in candidates})
    safe = [v for v in candidates if v not in exclude]
    return {"flagged_pairs": flagged, "exclude_covariates": exclude, "safe_adjustment_set": safe,
            "rcd": ens["rcd"], "parcelingam": ens["parcelingam"], "fci_bidirected": fci}


def sim_latent_collider(n=5000, tau=0.8, seed=0):
    """Self-validation DGP: an M-bias collider with HIDDEN parents, non-Gaussian (Laplace) noise so
    LiNGAM applies. Ha->A, Hy->Y, Cm<-Ha,Hy (Ha,Hy latent). Observed {A, Y, Cm}; Cm is the covariate
    that MUST be excluded (adjusting for it opens the M-bias path). Returns (X, names)."""
    rng = np.random.default_rng(seed)
    Ha = rng.laplace(size=n); Hy = rng.laplace(size=n)
    A = 0.9 * Ha + rng.laplace(size=n)
    Y = tau * A + 0.9 * Hy + rng.laplace(size=n)
    Cm = 0.9 * Ha + 0.9 * Hy + 0.4 * rng.laplace(size=n)
    return np.column_stack([A, Y, Cm]), ["A", "Y", "Cm"]
