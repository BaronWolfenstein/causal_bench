"""Ablation metrics on pooled SMB embeddings.

Per-model (one encoder):
  1. separability  : rare-vs-common ROC-AUC (5-fold CV logistic)   [primary]
  2. sigreg        : mean Epps-Pulley T after whitening; full / common / rare
  3. disentangle   : within-rare (indication fixed) treated-vs-untreated AUC
                     -> is the treatment signal recoverable independent of severity?

Regime-level (both backbones of one regime):
  4. agl           : cross-BACKBONE probit-agreement line quality (R^2, slope)
                     using TRAINED, bootstrap-diverse probes. ID = common (held-out),
                     OOD = rare. High R^2 => label-free OOD validity holds for the
                     guard => trustworthy E_gen!=E_eval check.

This is the corrected form of the dumped recipe: probes are TRAINED (AGL relates
agreement to accuracy, which untrained random probes do not have), diversity comes
from bootstrap + random subspace (not from re-seeding a convex linear map), and the
primary decoupling axis is cross-BACKBONE (independent failure), not cross-seed.
"""
from __future__ import annotations
import numpy as np
from numpy.random import default_rng
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score


# --------------------------------------------------------------------------- #
# 1. separability
# --------------------------------------------------------------------------- #
def separability_auc(emb, y, seed=0):
    y = np.asarray(y)
    if y.sum() < 5 or (len(y) - y.sum()) < 5:
        return float("nan")
    Xs = StandardScaler().fit_transform(emb)
    clf = LogisticRegression(max_iter=2000, C=1.0)
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    p = cross_val_predict(clf, Xs, y, cv=cv, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, p))


def fewshot_auc(emb, y, k=10, n_draws=25, seed=0):
    """Mean held-out AUC when training on only k examples PER CLASS.

    This is the non-saturating discriminator: when full-data separability is at
    ceiling, k-shot exposes representation quality (a better-structured space needs
    fewer labels). Used for the dynamics-defined `fast_prog` phenotype.
    """
    y = np.asarray(y)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    if len(pos) < k + 5 or len(neg) < k + 5:
        return float("nan")
    Xs = StandardScaler().fit_transform(emb)
    rng = default_rng(seed)
    aucs = []
    for _ in range(n_draws):
        tp = rng.choice(pos, k, replace=False); tn = rng.choice(neg, k, replace=False)
        tr = np.concatenate([tp, tn])
        te = np.setdiff1d(np.arange(len(y)), tr)
        clf = LogisticRegression(max_iter=1000, C=1.0).fit(Xs[tr], y[tr])
        p = clf.predict_proba(Xs[te])[:, 1]
        if y[te].sum() and (len(y[te]) - y[te].sum()):
            aucs.append(roc_auc_score(y[te], p))
    return float(np.mean(aucs)) if aucs else float("nan")


# --------------------------------------------------------------------------- #
# 2. SIGReg / Epps-Pulley on whitened random projections
# --------------------------------------------------------------------------- #
def _whiten(emb, k):
    """PCA-whiten to k comps (robust when n < H); returns [n,k] with unit variance."""
    X = emb - emb.mean(0, keepdims=True)
    # economy SVD; components ordered by singular value
    U, S, _ = np.linalg.svd(X, full_matrices=False)
    k = min(k, (S > 1e-8).sum())
    if k < 1:
        return None
    Z = U[:, :k] * np.sqrt(X.shape[0] - 1)     # unit-variance scores
    return Z


def _epps_pulley_1d(x):
    """Epps-Pulley T for a standardized 1D sample; trapezoid on t in [-5,5], 17 pts."""
    x = (x - x.mean()) / (x.std() + 1e-9)
    n = len(x)
    t = np.linspace(-5, 5, 17)
    cos = np.cos(np.outer(t, x)).mean(1)        # Re phi_n(t)
    sin = np.sin(np.outer(t, x)).mean(1)        # Im phi_n(t)
    phiN = np.exp(-t ** 2 / 2)
    integrand = ((cos - phiN) ** 2 + sin ** 2) * phiN     # w(t)=phi_N(t)
    return float(n * np.trapz(integrand, t))


def sigreg_T(emb, n_dirs=64, k=40, seed=0):
    """Mean Epps-Pulley T across random projections of the whitened embeddings."""
    if len(emb) < 8:
        return float("nan")
    Z = _whiten(emb, k)
    if Z is None:
        return float("nan")
    rng = default_rng(seed)
    dirs = rng.standard_normal((Z.shape[1], n_dirs))
    dirs /= np.linalg.norm(dirs, axis=0, keepdims=True)
    proj = Z @ dirs                              # [n, n_dirs]
    return float(np.mean([_epps_pulley_1d(proj[:, j]) for j in range(n_dirs)]))


# --------------------------------------------------------------------------- #
# 3. treatment / indication disentanglement
# --------------------------------------------------------------------------- #
def within_rare_treatment_auc(emb, is_rare, A, seed=0):
    """Among rare (metastatic, stage fixed) patients, can we recover treatment A?
    AUC>0.5 => treatment signal is present independent of the indication axis."""
    m = np.asarray(is_rare) == 1
    Xr, Ar = emb[m], np.asarray(A)[m]
    if Ar.sum() < 5 or (len(Ar) - Ar.sum()) < 5:
        return float("nan")
    Xs = StandardScaler().fit_transform(Xr)
    clf = LogisticRegression(max_iter=2000, C=1.0)
    cv = StratifiedKFold(min(5, int(min(Ar.sum(), len(Ar) - Ar.sum()))), shuffle=True, random_state=seed)
    p = cross_val_predict(clf, Xs, Ar, cv=cv, method="predict_proba")[:, 1]
    return float(roc_auc_score(Ar, p))


def per_model_metrics(emb, labels, seed=0):
    y_rare = labels["is_rare"].to_numpy()
    y_fast = labels["fast_prog"].to_numpy()
    out = {
        # coarse vocabulary axis (saturates -> demo targetability check, not a discriminator)
        "separability_auc": separability_auc(emb, y_rare, seed),
        # DYNAMICS axis, few-shot -> the real discriminator of representation quality
        "fewshot_fast_k10": fewshot_auc(emb, y_fast, k=10, seed=seed),
        "fewshot_fast_k25": fewshot_auc(emb, y_fast, k=25, seed=seed),
        "fullshot_fast_auc": separability_auc(emb, y_fast, seed),
        # geometry
        "sigreg_T_full": sigreg_T(emb, seed=seed),
        "sigreg_T_common": sigreg_T(emb[y_rare == 0], seed=seed),
        "sigreg_T_rare": sigreg_T(emb[y_rare == 1], seed=seed),
        # treatment recoverability (visible-code -> expected high; kept for reference)
        "within_rare_treatment_auc": within_rare_treatment_auc(
            emb, y_rare, labels["A"].to_numpy(), seed),
    }
    return out


# --------------------------------------------------------------------------- #
# 4. cross-backbone AGL (regime level)
# --------------------------------------------------------------------------- #
def _train_probe(X, y, rng, subspace=0.5):
    """One bootstrap + random-subspace logistic probe. Returns (clf, cols)."""
    n, d = X.shape
    bi = rng.integers(0, n, n)                    # bootstrap rows
    cols = rng.choice(d, max(2, int(d * subspace)), replace=False)  # random subspace
    clf = LogisticRegression(max_iter=1000, C=0.5).fit(X[bi][:, cols], y[bi])
    return clf, cols


def regime_agl(emb_a, emb_b, labels, n_probes=8, seed=0):
    """Cross-backbone probit-agreement line quality for one regime.

    emb_a, emb_b : pooled embeddings from the two backbones (same patients/order).
    Trains probes on Y (progression) using ID=common; evaluates agreement on
    ID (held-out common) and OOD (rare). Returns R^2 and slope of the OLS fit
    probit(Agr_OOD) ~ probit(Agr_ID) over all probe pairs, plus the cross-backbone
    subset (the load-bearing pairs). Higher R^2 = AGL holds for this regime.
    """
    # AGL task = the dynamics label (fast_prog); shift = common (ID) -> rare (OOD).
    # A subtle shared-task label gives the agreement spread the probit line needs;
    # the coarse rare-vocabulary label saturated agreement to ~1 (no signal).
    rng = default_rng(seed)
    y = labels["fast_prog"].to_numpy()
    rare = labels["is_rare"].to_numpy() == 1
    common = ~rare
    idx_c = np.where(common)[0]
    rng.shuffle(idx_c)
    half = len(idx_c) // 2
    tr, te = idx_c[:half], idx_c[half:]           # common train / common test (ID)
    ood = np.where(rare)[0]                        # OOD
    if min(len(tr), len(te), len(ood)) < 8 or y[tr].sum() < 3:
        return {"agl_R2": float("nan"), "agl_slope": float("nan"),
                "agl_R2_crossbackbone": float("nan"), "n_pairs": 0}

    probes = []   # (backbone_id, predict_id_fn)
    for bb, emb in enumerate((emb_a, emb_b)):
        Xs = StandardScaler().fit(emb[tr])
        Xtr, Xte, Xood = Xs.transform(emb[tr]), Xs.transform(emb[te]), Xs.transform(emb[ood])
        for _ in range(n_probes):
            clf, cols = _train_probe(Xtr, y[tr], rng)
            pid = clf.predict(Xte[:, cols])
            pood = clf.predict(Xood[:, cols])
            probes.append((bb, pid, pood))

    agr_id, agr_ood, cross = [], [], []
    for i in range(len(probes)):
        for j in range(i + 1, len(probes)):
            bi, idi, oodi = probes[i]
            bj, idj, oodj = probes[j]
            agr_id.append((idi == idj).mean())
            agr_ood.append((oodi == oodj).mean())
            cross.append(bi != bj)
    agr_id = np.clip(np.array(agr_id), 1e-3, 1 - 1e-3)
    agr_ood = np.clip(np.array(agr_ood), 1e-3, 1 - 1e-3)
    cross = np.array(cross)
    pid, pood = norm.ppf(agr_id), norm.ppf(agr_ood)

    def _fit(x, yv):
        if len(x) < 3 or np.std(x) < 1e-6:
            return float("nan"), float("nan")
        A = np.vstack([x, np.ones_like(x)]).T
        (slope, _), *_ = np.linalg.lstsq(A, yv, rcond=None)
        pred = A @ np.linalg.lstsq(A, yv, rcond=None)[0]
        ss = 1 - ((yv - pred) ** 2).sum() / (((yv - yv.mean()) ** 2).sum() + 1e-9)
        return float(ss), float(slope)

    r2_all, slope_all = _fit(pid, pood)
    r2_cross, _ = _fit(pid[cross], pood[cross]) if cross.any() else (float("nan"), 0)
    return {"agl_R2": r2_all, "agl_slope": slope_all,
            "agl_R2_crossbackbone": r2_cross, "n_pairs": int(len(agr_id))}
