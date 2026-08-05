"""Misclassification correction for DECODED-subgroup estimands (issue #149).

When subgroup membership is *decoded* from an embedding by a probe (with error) and you then compute
a subgroup-stratified estimand, the misclassification biases it — the naive per-decoded-subgroup mean
mixes true subgroups. The correction (non-differential misclassification, the confusion-matrix "matrix
method" / Rogan-Gladen): estimate the confusion matrix `M[i,j] = P(pred=i | true=j)` from a LABELED
validation subsample, then invert the mixing to recover the true-subgroup estimand.

Assumption: **non-differential** misclassification, `S_hat ⊥ Y | S` (the probe's error does not depend
on the outcome given the true subgroup) — the standard, checkable condition for this correction. The
labeled validation subsample is the one legitimate embedding-transfer anchor: it makes the decoded
subgroup usable for a causal estimand without trusting the probe blindly.

Composes with the pooled-Q subgroup estimators: correct the per-arm decoded-subgroup means, then take
the difference for a decoded-subgroup treatment effect.
"""
from __future__ import annotations

import numpy as np


def confusion_matrix(s_true, s_pred, K):
    """M[i, j] = P(pred = i | true = j), estimated on the labeled validation subsample."""
    s_true = np.asarray(s_true); s_pred = np.asarray(s_pred)
    M = np.zeros((K, K))
    for j in range(K):
        m = s_true == j
        if m.sum():
            for i in range(K):
                M[i, j] = np.mean(s_pred[m] == i)
    return M


def naive_subgroup_means(Y, s_pred, K):
    """E[Y | pred = i] — the biased, uncorrected decoded-subgroup estimand."""
    Y = np.asarray(Y, float); s_pred = np.asarray(s_pred)
    return np.array([Y[s_pred == i].mean() if (s_pred == i).any() else np.nan for i in range(K)])


def corrected_subgroup_means(Y, s_pred, M):
    """Misclassification-corrected E[Y | true = j] for each subgroup.

    Under non-differential misclassification, for observed category i:
      E[Y·1{pred=i}] = sum_j M[i,j] E[Y·1{true=j}]   and   P(pred=i) = sum_j M[i,j] P(true=j).
    Invert both and divide -> E[Y|true=j] = (M^{-1} wY)[j] / (M^{-1} p)[j]."""
    Y = np.asarray(Y, float); s_pred = np.asarray(s_pred); n = len(Y); K = M.shape[0]
    wY = np.array([Y[s_pred == i].sum() / n for i in range(K)])   # E[Y·1{pred=i}]
    p = np.array([(s_pred == i).mean() for i in range(K)])        # P(pred=i)
    Minv = np.linalg.pinv(M)
    trueY = Minv @ wY
    trueP = Minv @ p
    return trueY / np.clip(trueP, 1e-6, None)


def corrected_subgroup_effect(Y, A, s_pred, M):
    """Decoded-subgroup treatment effect: correct each arm's decoded-subgroup means, then difference.
    (Randomized/ignorable A within subgroup; corrects the misclassification, not confounding.)"""
    Y = np.asarray(Y, float); A = np.asarray(A)
    mu1 = corrected_subgroup_means(Y[A == 1], np.asarray(s_pred)[A == 1], M)
    mu0 = corrected_subgroup_means(Y[A == 0], np.asarray(s_pred)[A == 0], M)
    return mu1 - mu0


def simulate(n=8000, K=3, err=0.25, seed=0, val_frac=0.2):
    """Self-validating DGP: K subgroups with distinct outcome means; membership DECODED with a known
    non-differential error rate `err`; a labeled validation subsample of size `val_frac*n`. Returns
    (Y, s_true, s_pred, val_idx, true_means)."""
    rng = np.random.default_rng(seed)
    s_true = rng.integers(0, K, n)
    means = 1.0 + 2.0 * np.arange(K)                       # true E[Y|S=j] = 1, 3, 5
    Y = means[s_true] + rng.normal(size=n)
    # non-differential decode: with prob (1-err) correct, else a random OTHER class (independent of Y)
    keep = rng.random(n) >= err
    other = (s_true + rng.integers(1, K, n)) % K
    s_pred = np.where(keep, s_true, other)
    val_idx = rng.choice(n, int(val_frac * n), replace=False)
    return {"Y": Y, "s_true": s_true, "s_pred": s_pred, "val_idx": val_idx,
            "true_means": means.astype(float), "K": K}


def report(n=8000, K=3, err=0.25, seed=0):
    s = simulate(n=n, K=K, err=err, seed=seed)
    M = confusion_matrix(s["s_true"][s["val_idx"]], s["s_pred"][s["val_idx"]], K)  # from LABELED subsample
    naive = naive_subgroup_means(s["Y"], s["s_pred"], K)
    corr = corrected_subgroup_means(s["Y"], s["s_pred"], M)
    return {"true": s["true_means"], "naive": naive, "corrected": corr,
            "naive_bias": float(np.mean(np.abs(naive - s["true_means"]))),
            "corrected_bias": float(np.mean(np.abs(corr - s["true_means"])))}
