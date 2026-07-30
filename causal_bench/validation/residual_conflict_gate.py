"""Residual-conflict diagnostic gate for external-control borrowing (exp44 Part-B, first step).

The gate that decides whether the exp44 dynamic-borrowing layer is even needed. After matching
trial vs external patients on a representation Z (the frozen SMB embedding, or a propensity
score), test whether a **control / negative-control OUTCOME** is still dependent on **SOURCE**
given Z:

    Y_control  ⫫  Source | Z ?

Independence ⇒ the matched SCA already achieves outcome-exchangeability → **skip borrowing**.
Dependence  ⇒ residual outcome conflict the matching did not remove (e.g. the echo modality gap
or era drift an EHR-only frozen encoder can't capture) → the dynamic-borrowing layer (exp44) is
warranted, aimed at that residual.

Engine: `detectors.zero_flow_ci.zero_flow_ci_test`. Three caveats it inherits (all load-bearing):
  1. Use a **control / negative-control** outcome — raw Y conflates the *treatment effect* with
     conflict (the trial arm has the device, the registry is the control).
  2. Z must be **pre-treatment**. A learned embedding that encodes outcome-associated / post-
     baseline information turns the conditioning set into a collider trap → spurious "conflict"
     (the test is provably fooled by conditioning on a collider).
  3. Small n → `underpowered`; per #194 the diagnostic is unreliable in BOTH directions at
     subgroup n (miss real conflict / fabricate it from noise). Read the power caveat per region.
"""
from __future__ import annotations

import numpy as np

from causal_bench.detectors.zero_flow_ci import zero_flow_ci_test


def residual_conflict_gate(outcome, source, embedding, *, alpha: float = 0.05,
                           n_perm: int = 100, rng=None) -> dict:
    """Go/no-go on external-control borrowing. Tests `outcome ⫫ source | embedding`.

    Parameters
    ----------
    outcome : (n,) a CONTROL or NEGATIVE-CONTROL outcome (NOT the raw treated-vs-control outcome).
    source  : (n,) 0/1 indicator, external/registry vs trial.
    embedding : (n, d) the pre-treatment representation to condition on (frozen embedding / PS).

    Returns a dict: verdict ('refutes'=dependent / 'supports'=independent / 'underpowered'),
    p, n, `residual_conflict` (bool | None), and a plain-language `recommendation`.
    """
    res = zero_flow_ci_test(np.asarray(outcome, float), np.asarray(source, float),
                            np.asarray(embedding, float), alpha=alpha, n_perm=n_perm, rng=rng)
    if res.verdict == "underpowered":
        return {"verdict": res.verdict, "p": res.p_value, "n": res.effective_n,
                "residual_conflict": None,
                "recommendation": "underpowered — inconclusive; do not certify exchangeability"}
    trigger = res.verdict == "refutes"                       # dependence ⇒ residual conflict
    return {"verdict": res.verdict, "p": res.p_value, "n": res.effective_n,
            "residual_conflict": bool(trigger),
            "recommendation": ("residual conflict after matching → build/apply the exp44 "
                               "borrowing layer for this residual"
                               if trigger else
                               "matched SCA achieves outcome-exchangeability → skip borrowing")}


def gate_by_region(outcome, source, embedding, region, *, alpha: float = 0.05,
                   n_perm: int = 100, min_n: int = 50, rng=None) -> dict:
    """Localize residual conflict: run the gate within each region (e.g. calendar-era or a
    modality-driven stratum) so a global 'supports' can't hide a concentrated conflict. Returns
    {region: gate-result}; regions with < min_n rows are marked underpowered."""
    outcome = np.asarray(outcome, float)
    source = np.asarray(source, float)
    embedding = np.asarray(embedding, float)
    region = np.asarray(region)
    out = {}
    for r in np.unique(region):
        m = region == r
        if int(m.sum()) < min_n:
            out[r] = {"verdict": "underpowered", "residual_conflict": None, "n": int(m.sum()),
                      "recommendation": "too few in region — inconclusive"}
            continue
        out[r] = residual_conflict_gate(outcome[m], source[m], embedding[m],
                                        alpha=alpha, n_perm=n_perm, rng=rng)
    return out
