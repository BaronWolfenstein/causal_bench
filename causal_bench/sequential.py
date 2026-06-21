"""Sequential monitoring utilities for CED post-coverage surveillance.

Implements three approaches for repeated analysis of accumulating trial data:
  1. Naive repeated testing — no multiplicity correction (inflated type I error)
  2. O'Brien-Fleming alpha spending via Lan-DeMets spending function — handles
     irregular look times; critical values computed from multivariate normal
     rectangular probabilities under the independent-increments structure
  3. Confidence sequences (Howard et al. 2021) — anytime-valid, covers true
     parameter at >=1-alpha simultaneously at ALL looks, not just pre-specified

Lan-DeMets OBF spending:
  α*(t) = 2 − 2Φ(z_{α/2} / √t),  t = I_k / I_K  (information fraction)

  The critical value c_k at look k is the unique solution to:
    P(|Z_j| ≤ c_j for j < k,  |Z_k| > c_k) = α*(t_k) − α*(t_{k-1})

  where (Z_1, ..., Z_k) ~ MVN(0, Σ) with Σ_{ij} = √(t_{min(i,j)} / t_{max(i,j)})
  (independent-increments property of TMLE z-statistics, exact asymptotically).

  The Lan-DeMets generalization of fixed-K OBF is necessary for CED because
  CMS may request updates on an irregular schedule or milestone-triggered looks.
  The spending function controls total α regardless of the number and timing of
  looks, provided α*(·) is non-decreasing and α*(1) = α.

Independent-increments caveat:
  At finite N, TMLE z-statistics across looks are not exactly independent-
  increments because the Super Learner nuisance fits update at each look. The
  boundaries are slightly anti-conservative or conservative at early looks with
  small N. The simulation in Exp 15 quantifies this empirically.

References:
  Lan & DeMets (1983). Discrete sequential boundaries for clinical trials. Biometrika.
  Howard et al. (2021). Time-uniform confidence sequences. Annals of Statistics.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm
from scipy.stats import multivariate_normal as _mvn


# ─── Lan-DeMets OBF spending function ────────────────────────────────────────

def lan_demets_spending(t: float, alpha: float = 0.05) -> float:
    """Cumulative alpha spent by information fraction t under OBF spending.

    α*(t) = 2 − 2Φ(z_{α/2} / √t)

    Satisfies α*(0⁺) → 0 and α*(1) = α.  Conservative early (little α spent
    when t is small) and decisive late (remaining α spent near t=1).
    """
    if t <= 0.0:
        return 0.0
    z_half = norm.ppf(1.0 - alpha / 2.0)
    return float(2.0 * (1.0 - norm.cdf(z_half / math.sqrt(t))))


def _mvn_rect_prob(bounds: list[float], cov: np.ndarray) -> float:
    """P(|Z_i| ≤ b_i for all i) for Z ~ MVN(0, cov).

    Uses inclusion-exclusion over the 2^k corners of the rectangular region
    [-b_i, b_i].  Exact for any k; cost is O(2^k) MVN CDF evaluations.
    For k ≤ 10 (≤ 1024 evaluations) this is fast.
    """
    k = len(bounds)
    if k == 0:
        return 1.0
    if k == 1:
        # Fast path: univariate
        return float(2 * norm.cdf(bounds[0] / math.sqrt(cov[0, 0])) - 1)

    total = 0.0
    mean_zero = np.zeros(k)
    for s_int in range(2 ** k):
        s = [(s_int >> i) & 1 for i in range(k)]
        corner = np.array([bounds[i] if s[i] else -bounds[i] for i in range(k)])
        sign = (-1) ** sum(1 - si for si in s)
        total += sign * float(_mvn.cdf(corner, mean=mean_zero, cov=cov))
    return float(total)


def lan_demets_obf_boundaries(
    info_fractions: list[float],
    alpha: float = 0.05,
) -> list[float]:
    """O'Brien-Fleming critical values at irregular look times via Lan-DeMets.

    Parameters
    ----------
    info_fractions : t_1 < t_2 < ... < t_K, each in (0, 1].
        t_k = I_k / I_K where I_k is the Fisher information (≈ cumulative n)
        at look k.  For K equally-spaced annual looks: [1/K, 2/K, ..., 1].
    alpha : overall type I error level.

    Returns
    -------
    list of K critical values c_1, ..., c_K.
    Reject H_0 at look k if |Z_k| > c_k.
    """
    t = np.asarray(info_fractions, dtype=float)
    K = len(t)

    # Incremental alpha at each look
    cum_spent = np.array([lan_demets_spending(ti, alpha) for ti in t])
    alpha_inc = np.diff(cum_spent, prepend=0.0)

    # Correlation matrix: Corr(Z_i, Z_j) = √(t_min / t_max)
    Sigma = np.array([
        [math.sqrt(min(t[i], t[j]) / max(t[i], t[j])) for j in range(K)]
        for i in range(K)
    ])
    # Small ridge for numerical stability (info fractions close together)
    Sigma += np.eye(K) * 1e-8

    boundaries: list[float] = []

    for k in range(K):
        sub_cov = Sigma[:k + 1, :k + 1]
        prev = boundaries[:]          # c_1, ..., c_{k-1}
        target = float(alpha_inc[k])

        if k == 0:
            # P(|Z_1| > c_1) = α_1  →  c_1 = Φ^{-1}(1 − α_1/2)
            c_k = float(norm.ppf(1.0 - target / 2.0))
        else:
            p_prev = _mvn_rect_prob(prev, sub_cov[:k, :k])

            def _p_reject(c: float) -> float:
                """P(|Z_j|≤c_j ∀j<k, |Z_k|>c)."""
                return p_prev - _mvn_rect_prob(prev + [c], sub_cov)

            try:
                c_k = float(brentq(
                    lambda c: _p_reject(c) - target,
                    lo := 1.0,
                    hi := 20.0,
                    xtol=1e-5,
                ))
            except ValueError:
                # Fallback: approximate OBF formula (valid for equally-spaced looks)
                z_half = norm.ppf(1.0 - alpha / 2.0)
                c_k = float(z_half / math.sqrt(float(t[k])))

        boundaries.append(c_k)

    return boundaries


# ─── Fixed-K OBF (backward-compatible approximation) ─────────────────────────

def obf_boundary(k: int, K: int, alpha: float = 0.05) -> float:
    """O'Brien-Fleming critical value at look k of K equally-spaced looks.

    Approximation: c_k ≈ z_{α/2} / √(k/K).  Exact only when looks are
    pre-specified and equally spaced.  Use lan_demets_obf_boundaries() for
    irregular looks or when exact spending is required.
    """
    z_alpha = norm.ppf(1.0 - alpha / 2.0)
    return z_alpha / math.sqrt(k / K)


# ─── Confidence sequences ─────────────────────────────────────────────────────

def confidence_sequence(
    estimates: list[float],
    ses: list[float],
    alpha: float = 0.05,
    v: float = 1.0,
) -> list[dict]:
    """Howard et al. (2021) mixture-martingale confidence sequence.

    Anytime-valid: the true parameter is simultaneously covered at every look
    with probability ≥ 1 − alpha, without requiring pre-specified looks or
    stopping rules.

    Parameters
    ----------
    estimates : point estimates at each look
    ses       : standard errors at each look
    alpha     : miscoverage level (default 0.05)
    v         : prior information weight (unit-information prior)
    """
    results = []
    for t_idx in range(len(estimates)):
        V_t = 1.0 / (ses[t_idx] ** 2)
        width = math.sqrt(
            (v + V_t) / (V_t ** 2)
            * 2.0 * math.log(math.sqrt((v + V_t) / v) / alpha)
        )
        results.append({
            "look": t_idx + 1,
            "estimate": estimates[t_idx],
            "cs_lower": estimates[t_idx] - width,
            "cs_upper": estimates[t_idx] + width,
        })
    return results


# ─── SequentialResult ─────────────────────────────────────────────────────────

@dataclass
class SequentialResult:
    trajectory: str             # "stable", "degrading", "step_change", "null"
    method: str                 # "naive", "obf", "confidence_sequence"
    true_values: list[float]    # true ATE at each look
    estimates: list[float]
    ci_lowers: list[float]
    ci_uppers: list[float]
    rejects_null: list[bool]
    first_rejection_look: Optional[int]
    ever_rejected: bool
    false_rejection: bool       # rejected when true ATE == 0
    coverage_all_looks: bool    # true value inside CI at every look
    info_fractions: list[float] = field(default_factory=list)  # t_k = I_k/I_K
    boundaries: list[float] = field(default_factory=list)      # c_k per look


# ─── apply_sequential_methods ─────────────────────────────────────────────────

def apply_sequential_methods(
    estimates: list[float],
    ses: list[float],
    true_values: list[float],
    trajectory: str,
    K: int,
    alpha: float = 0.05,
    info_fractions: Optional[list[float]] = None,
) -> dict[str, SequentialResult]:
    """Apply all three monitoring approaches to a sequence of estimates.

    Parameters
    ----------
    estimates      : point estimates at each look (length K)
    ses            : standard errors at each look (length K)
    true_values    : true ATE at each look (for coverage / false-rejection checks)
    trajectory     : DGP trajectory label
    K              : total pre-planned number of looks
    alpha          : overall type I error level
    info_fractions : optional list of length K with t_k = I_k / I_K ∈ (0, 1].
        Defaults to equally-spaced [1/K, 2/K, ..., 1].
        Pass actual cumulative sample sizes when looks are irregularly timed:
          n_cumulative = [n_1, n_2, ..., n_K]
          info_fractions = [n / n_K for n in n_cumulative]

    Returns
    -------
    dict mapping method name → SequentialResult (keys: "naive", "obf",
    "confidence_sequence").
    """
    if info_fractions is None:
        info_fractions = [k / K for k in range(1, K + 1)]

    z_alpha = norm.ppf(1.0 - alpha / 2.0)
    obf_boundaries = lan_demets_obf_boundaries(info_fractions, alpha=alpha)
    cs_results = confidence_sequence(estimates, ses, alpha=alpha)

    results = {}
    for method in ("naive", "obf", "confidence_sequence"):
        lowers, uppers, rejects, used_boundaries = [], [], [], []

        for k in range(K):
            est = estimates[k]
            se = ses[k]

            if method == "naive":
                lo = est - z_alpha * se
                hi = est + z_alpha * se
                rej = abs(est / se) > z_alpha
                used_boundaries.append(z_alpha)

            elif method == "obf":
                c_k = obf_boundaries[k]
                lo = est - c_k * se
                hi = est + c_k * se
                rej = abs(est / se) > c_k
                used_boundaries.append(c_k)

            else:  # confidence_sequence
                cs = cs_results[k]
                lo = cs["cs_lower"]
                hi = cs["cs_upper"]
                rej = lo > 0 or hi < 0
                used_boundaries.append(float("nan"))

            lowers.append(lo)
            uppers.append(hi)
            rejects.append(rej)

        first_rej = next((k + 1 for k, r in enumerate(rejects) if r), None)
        coverage = all(
            lowers[k] <= true_values[k] <= uppers[k] for k in range(K)
        )

        results[method] = SequentialResult(
            trajectory=trajectory,
            method=method,
            true_values=true_values,
            estimates=estimates,
            ci_lowers=lowers,
            ci_uppers=uppers,
            rejects_null=rejects,
            first_rejection_look=first_rej,
            ever_rejected=any(rejects),
            false_rejection=any(rejects) and all(tv == 0.0 for tv in true_values),
            coverage_all_looks=coverage,
            info_fractions=list(info_fractions),
            boundaries=used_boundaries,
        )

    return results
