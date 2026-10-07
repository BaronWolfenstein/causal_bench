"""Ratio confidence intervals — Fieller's theorem & MOVER — for ratio estimands (the win ratio, ratios of means).

Why, not just how: the win ratio WR = P(win)/P(loss) and other ratio metrics have a *denominator that can be
small or heavy-tailed*, where the delta method's symmetric CI (WR ± z·SE on WR or logWR) understates uncertainty
and can even put the lower bound below 0. **Fieller** inverts the test H0: num − θ·den = 0 directly, giving an
exact, asymmetric interval that widens correctly as the denominator loses significance (and, honestly, becomes
*unbounded* when the denominator is not significantly ≠ 0 — a feature the delta method hides). **MOVER**
(Method of Variance Estimates Recovery, Zou–Donner) combines two interval endpoints into a CI for their
difference without solving a joint quadratic — the practical route for a *difference of two ratios* (R_T − R_C).

Self-contained numpy/scipy (no R): `win_ratio_ci` computes P(win), P(loss) and their rigorous two-sample
U-statistic (DeLong-style) variance/covariance from samples, then applies Fieller/MOVER — so these CIs can be
used as an asymmetric, small-denominator-robust alternative to the concrete win-ratio estimator's log-WR Wald CI.
It also takes optional cluster labels (a subject id under repeated measures, a site id) for the cluster-robust
U-statistic variance; with no labels it reduces exactly to the independent DeLong estimator.
"""
from __future__ import annotations
import numpy as np
from scipy.stats import norm, t as _tdist


def _crit(alpha, df):
    return float(_tdist.ppf(1 - alpha / 2, df)) if df is not None else float(norm.ppf(1 - alpha / 2))


def fieller_ci(num, den, v_num, v_den, cov, *, alpha=0.05, df=None):
    """Fieller CI for θ = num/den given Var(num)=v_num, Var(den)=v_den, Cov(num,den)=cov.
    Returns (point, lo, hi, status). status:
      'bounded'   — a finite [lo,hi] (the usual case; den significantly ≠ 0).
      'exterior'  — the CI is the COMPLEMENT (−∞,lo]∪[hi,∞): den not significant, θ poorly identified.
      'unbounded' — the whole real line (no finite bound)."""
    z2 = _crit(alpha, df) ** 2
    A = den * den - z2 * v_den                       # coefficient on θ²
    B = -2.0 * (num * den - z2 * cov)                # coefficient on θ
    C = num * num - z2 * v_num
    disc = B * B - 4.0 * A * C
    point = num / den if den != 0 else np.nan
    if disc < 0:
        return point, -np.inf, np.inf, "unbounded"   # quadratic has no real roots ⇒ no finite bound
    r1 = (-B - np.sqrt(disc)) / (2.0 * A)
    r2 = (-B + np.sqrt(disc)) / (2.0 * A)
    lo, hi = min(r1, r2), max(r1, r2)
    if A > 0:
        return point, lo, hi, "bounded"              # standard confidence interval [lo,hi]
    return point, lo, hi, "exterior"                 # A<0: CI is (−∞,lo]∪[hi,∞) — denominator not significant


def delta_ratio_ci(num, den, v_num, v_den, cov, *, alpha=0.05, df=None):
    """Delta-method (symmetric) CI for num/den — for comparison with Fieller."""
    r = num / den
    var = (v_num - 2 * r * cov + r * r * v_den) / (den * den)
    se = np.sqrt(max(var, 0.0)); z = _crit(alpha, df)
    return r, r - z * se, r + z * se


def mover_difference(est1, lo1, hi1, est2, lo2, hi2, *, corr=0.0):
    """MOVER (Zou–Donner) CI for est1 − est2 from their individual CIs. `corr` = correlation of the two
    estimators (0 if independent). Recovers asymmetry/exactness of the per-term intervals into the difference."""
    d = est1 - est2
    L = d - np.sqrt((est1 - lo1) ** 2 + (hi2 - est2) ** 2 - 2 * corr * (est1 - lo1) * (hi2 - est2))
    U = d + np.sqrt((hi1 - est1) ** 2 + (est2 - lo2) ** 2 - 2 * corr * (hi1 - est1) * (est2 - lo2))
    return d, L, U


def _delong_components(M):
    """Two-sample U-statistic (DeLong) variance of mean(M) for an m×n indicator matrix M[i,j]:
    Var = Var_i(row means)/m + Var_j(col means)/n. Returns (mean, var, row_means, col_means)."""
    m, n = M.shape
    rm = M.mean(1); cm = M.mean(0)
    var = rm.var(ddof=1) / m + cm.var(ddof=1) / n
    return float(M.mean()), float(var), rm, cm


def _cluster_contrib(centered, clusters):
    """Collapse per-unit (already 1/k-scaled) centered contributions onto their cluster. `clusters=None` ⇒ each
    unit is its own cluster (the independent DeLong leg). Returns the length-G vector of per-cluster summed
    contributions, so that Var_leg = [G/(G-1)]·Σ_g s_g² is the design-based (sandwich) cluster-robust variance."""
    if clusters is None:
        return centered
    _, inv = np.unique(np.asarray(clusters), return_inverse=True)
    s = np.zeros(inv.max() + 1)
    np.add.at(s, inv, centered)
    return s


def _winloss_moments(W, L, cluster_treat=None, cluster_control=None):
    """(P_win, P_loss) and their two-sample U-statistic variance/covariance, cluster-robust when cluster labels
    are given. Each leg (treated rows, control cols) is clustered independently — correct when a cluster lies
    within ONE arm (the subject-level repeated-measures / within-arm-site case). With no clusters this reduces
    EXACTLY to the DeLong estimator (singleton clusters give the same ddof=1 row/col-mean variance)."""
    m, n = W.shape
    pw = float(W.mean()); pl = float(L.mean())
    aw = (W.mean(1) - pw) / m; al = (L.mean(1) - pl) / m          # treated-leg (row) influence, scaled 1/m
    bw = (W.mean(0) - pw) / n; bl = (L.mean(0) - pl) / n          # control-leg (col) influence, scaled 1/n
    sw_t = _cluster_contrib(aw, cluster_treat);  sl_t = _cluster_contrib(al, cluster_treat)
    sw_c = _cluster_contrib(bw, cluster_control); sl_c = _cluster_contrib(bl, cluster_control)
    Gt, Gc = len(sw_t), len(sw_c)
    ft = Gt / (Gt - 1) if Gt > 1 else 0.0                        # finite-cluster (sandwich) correction per leg
    fc = Gc / (Gc - 1) if Gc > 1 else 0.0
    vw = ft * float(sw_t @ sw_t) + fc * float(sw_c @ sw_c)
    vl = ft * float(sl_t @ sl_t) + fc * float(sl_c @ sl_c)
    cov = ft * float(sw_t @ sl_t) + fc * float(sw_c @ sl_c)       # shared pairs ⇒ Cov(P_win,P_loss) ≠ 0
    return pw, pl, vw, vl, cov, Gt, Gc


def win_ratio_ci(t_treat, t_control, *, alpha=0.05, method="fieller", higher_is_better=True,
                 cluster_treat=None, cluster_control=None):
    """Win ratio WR = P(T_treat 'wins') / P(T_treat 'loses') from two independent samples, with a Fieller or
    delta CI built on the rigorous two-sample U-statistic variance/covariance of (P_win, P_loss). Ties are
    neither wins nor losses. `method` ∈ {'fieller','delta'}.

    Clustering: pass `cluster_treat` (length m) and/or `cluster_control` (length n) — e.g. a subject id when a
    subject contributes several correlated observations (repeated measures), or a site id — to get the
    cluster-robust U-statistic variance. Each leg is clustered within its own arm; omitting the labels recovers
    the independent DeLong variance exactly. Returns a dict with point, CI, status, components, cluster counts."""
    t = np.asarray(t_treat, float)[:, None]; c = np.asarray(t_control, float)[None, :]
    W = (t > c).astype(float) if higher_is_better else (t < c).astype(float)
    L = (t < c).astype(float) if higher_is_better else (t > c).astype(float)
    m, n = W.shape
    if cluster_treat is not None and len(np.asarray(cluster_treat)) != m:
        raise ValueError(f"cluster_treat must have length m={m}; got {len(np.asarray(cluster_treat))}")
    if cluster_control is not None and len(np.asarray(cluster_control)) != n:
        raise ValueError(f"cluster_control must have length n={n}; got {len(np.asarray(cluster_control))}")
    pw, pl, vw, vl, cov, Gt, Gc = _winloss_moments(W, L, cluster_treat, cluster_control)
    if method == "fieller":
        point, lo, hi, status = fieller_ci(pw, pl, vw, vl, cov, alpha=alpha)
    else:
        point, lo, hi = delta_ratio_ci(pw, pl, vw, vl, cov, alpha=alpha); status = "bounded"
    return {"win_ratio": point, "lo": lo, "hi": hi, "status": status,
            "p_win": pw, "p_loss": pl, "method": method,
            "var_win": vw, "var_loss": vl, "cov": float(cov),
            "n_clusters_treat": Gt, "n_clusters_control": Gc,
            "clustered": cluster_treat is not None or cluster_control is not None}
