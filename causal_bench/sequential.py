"""Sequential monitoring utilities for CED post-coverage surveillance.

Implements three approaches for repeated analysis of accumulating trial data:
  1. Naive repeated testing — no multiplicity correction (inflated type I error)
  2. O'Brien-Fleming alpha spending — controls overall alpha across pre-specified looks
  3. Confidence sequences (Howard et al. 2021) — anytime-valid, covers true parameter
     at >=1-alpha simultaneously at ALL looks, not just pre-specified ones

Reference: Howard et al. (2021). Time-uniform, nonparametric, nonasymptotic
confidence sequences. Annals of Statistics.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.stats import norm


@dataclass
class SequentialResult:
    trajectory: str            # "stable", "degrading", "step_change", "null"
    method: str                # "naive", "obf", "confidence_sequence"
    true_values: list[float]   # true ATE at each look
    estimates: list[float]
    ci_lowers: list[float]
    ci_uppers: list[float]
    rejects_null: list[bool]
    first_rejection_look: Optional[int]
    ever_rejected: bool
    false_rejection: bool      # rejected when true ATE == 0
    coverage_all_looks: bool   # true value inside CI at every look


def obf_boundary(k: int, K: int, alpha: float = 0.05) -> float:
    """O'Brien-Fleming critical value at look k of K total looks.

    Conservative early (wide boundary), liberal late (narrow boundary).
    Controls overall type I error at alpha across K equally-spaced looks.
    """
    info_fraction = k / K
    z_alpha = norm.ppf(1 - alpha / 2)
    return z_alpha / math.sqrt(info_fraction)


def confidence_sequence(
    estimates: list[float],
    ses: list[float],
    alpha: float = 0.05,
    v: float = 1.0,
) -> list[dict]:
    """Howard et al. (2021) mixture-martingale confidence sequence.

    Returns anytime-valid intervals: the true parameter is simultaneously
    covered at every look with probability >= 1 - alpha, without requiring
    pre-specified looks or stopping rules.

    Parameters
    ----------
    estimates : sequence of point estimates at each look
    ses       : sequence of standard errors at each look
    alpha     : miscoverage level (default 0.05)
    v         : prior information weight (default 1.0 — unit-information prior)
    """
    results = []
    for t in range(len(estimates)):
        V_t = 1.0 / (ses[t] ** 2)
        width = math.sqrt(
            (v + V_t) / (V_t ** 2)
            * 2.0 * math.log(math.sqrt((v + V_t) / v) / alpha)
        )
        results.append({
            "look": t + 1,
            "estimate": estimates[t],
            "cs_lower": estimates[t] - width,
            "cs_upper": estimates[t] + width,
        })
    return results


def apply_sequential_methods(
    estimates: list[float],
    ses: list[float],
    true_values: list[float],
    trajectory: str,
    K: int,
    alpha: float = 0.05,
) -> dict[str, SequentialResult]:
    """Apply all three monitoring approaches to a sequence of estimates.

    Parameters
    ----------
    estimates   : TMLE+IPCW point estimates at each look (length K)
    ses         : standard errors at each look (length K)
    true_values : true ATE at each look (for coverage and false-rejection checks)
    trajectory  : DGP trajectory label
    K           : total pre-planned number of looks
    alpha       : overall type I error level

    Returns
    -------
    dict mapping method name to SequentialResult
    """
    z_alpha = norm.ppf(1 - alpha / 2)
    cs_results = confidence_sequence(estimates, ses, alpha=alpha)

    results = {}
    for method in ("naive", "obf", "confidence_sequence"):
        lowers, uppers, rejects = [], [], []

        for k in range(1, K + 1):
            est = estimates[k - 1]
            se = ses[k - 1]

            if method == "naive":
                lo = est - z_alpha * se
                hi = est + z_alpha * se
                rej = abs(est / se) > z_alpha

            elif method == "obf":
                boundary = obf_boundary(k, K, alpha)
                lo = est - boundary * se
                hi = est + boundary * se
                rej = abs(est / se) > boundary

            else:  # confidence_sequence
                cs = cs_results[k - 1]
                lo = cs["cs_lower"]
                hi = cs["cs_upper"]
                rej = lo > 0 or hi < 0

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
        )

    return results
