"""exp47 (#196) — borrowing × interim-looks composition.

Composes external-control borrowing with repeated interim analyses (`sequential.py`): at each
of K accumulating looks, form a borrowed posterior on the current-cohort data, then monitor the
resulting estimate/SE sequence with each rule (naive / O'Brien-Fleming α-spending / Howard
confidence sequence). Under the current NULL (μ=0) with a historical prior in conflict (δ>0),
report the **cumulative** Type-I (reject at ANY look) per (borrowing policy × monitoring rule × δ).

The point (Qian/EitW; FDA Jan-2026 Bayesian draft): neither `sequential.py` (monitors a
non-borrowing statistic) nor exp41/exp44 (single analysis) tests the *composition*. Borrowing
shifts the per-look statistic AND breaks the independent-increments assumption OBF boundaries
are built on, so borrowing × monitoring can inflate cumulative Type-I beyond either alone.

Fast: conjugate Normal-Normal borrow (no MCMC); the robust-MAP *adaptive* mitigation is the
exp44 MCMC version — here `power` is a fixed-discount conjugate stand-in.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

from causal_bench.sequential import confidence_sequence, lan_demets_obf_boundaries


def _conjugate_borrow(y_bar, se, prior_mean, prior_sd):
    """Normal-Normal posterior for μ from likelihood N(y_bar, se²) × prior N(prior_mean, prior_sd²)."""
    pv, lv = prior_sd ** 2, se ** 2
    post_var = 1.0 / (1.0 / pv + 1.0 / lv)
    return post_var * (prior_mean / pv + y_bar / lv), float(np.sqrt(post_var))


def simulate_looks(delta, policy, *, K=5, n_per_look=100, n_hist=400, sigma=1.0,
                   vague_sd=10.0, discount=0.3, rng=None):
    """One replicate under the current NULL (μ=0). Historical prior sits at μ=δ (the conflict).
    Returns the per-look (estimate, se) sequence AFTER borrowing.
      flat  — vague prior (no borrow);  map — full borrow at δ;  power — discounted borrow."""
    rng = rng or np.random.default_rng(0)
    m_hist, s_hist = float(delta), sigma / np.sqrt(n_hist)          # borrowed prior (μ_hist=δ)
    obs = rng.normal(0.0, sigma, K * n_per_look)                    # current cohort: TRULY NULL
    est, ses = [], []
    for k in range(1, K + 1):
        nk = k * n_per_look
        y_bar, se = float(obs[:nk].mean()), sigma / np.sqrt(nk)
        if policy == "flat":
            pm, ps = 0.0, vague_sd
        elif policy == "map":
            pm, ps = m_hist, s_hist
        elif policy == "power":                                    # discount → borrow less
            pm, ps = m_hist, s_hist / np.sqrt(discount)
        else:
            raise ValueError(f"unknown policy {policy!r}")
        e, s = _conjugate_borrow(y_bar, se, pm, ps)
        est.append(e)
        ses.append(s)
    return est, ses


def composition_type_i(deltas, policies=("flat", "map", "power"), *, K=5, n_reps=3000,
                       n_per_look=100, alpha=0.05, seed=0, **kw):
    """Cumulative Type-I (reject at ANY look) per (policy × monitoring rule × δ). Boundaries are
    computed ONCE (deterministic in the info fractions); per-rep is just comparisons."""
    info_fractions = [k / K for k in range(1, K + 1)]
    obf_c = lan_demets_obf_boundaries(info_fractions, alpha=alpha)   # OBF critical values, once
    z_alpha = norm.ppf(1.0 - alpha / 2.0)
    rows = []
    for policy in policies:
        for d in deltas:
            hits = {"naive": 0, "obf": 0, "confidence_sequence": 0}
            for r in range(n_reps):
                est, ses = simulate_looks(d, policy, K=K, n_per_look=n_per_look,
                                          rng=np.random.default_rng(seed + r), **kw)
                z = [e / s for e, s in zip(est, ses)]
                if any(abs(zk) > z_alpha for zk in z):
                    hits["naive"] += 1
                if any(abs(zk) > ck for zk, ck in zip(z, obf_c)):
                    hits["obf"] += 1
                cs = confidence_sequence(est, ses, alpha=alpha)
                if any(c["cs_lower"] > 0 or c["cs_upper"] < 0 for c in cs):
                    hits["confidence_sequence"] += 1
            for m, h in hits.items():
                rows.append({"policy": policy, "delta": float(d), "method": m,
                             "type_i": h / n_reps, "n_reps": n_reps})
    return rows
