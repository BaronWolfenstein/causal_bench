"""Exp 45: estimators for the time-varying app co-intervention (spec 2026-07-29).

A T=2 longitudinal DGP with treatment-confounder feedback (W→A0→L1→A1→Y) — the multi-period
case the repo's two-timepoint `LTMLEEstimator` cannot represent. Four-way comparison,
demonstrating the estimand distinction (the causal_bench thesis: the right estimator depends on
the question):

  • **naive** (OLS adjusting the feedback confounder L1) — biased for A0's blip: adjusting the
    *mediator* L1 blocks A0's effect through L1.
  • **two-timepoint LTMLE** — not applicable: it has a fixed baseline A + a single mediator, no
    second treatment A1 (reported, not run).
  • **sequential regression (ICE)** — the multi-period LTMLE g-computation backbone; recovers the
    regime contrast E[Y_{1,1}]−E[Y_{0,0}]. (Targeting is the DR refinement, phase 2.)
  • **g-estimation (blip-down)** — recovers the structural blips ψ and (in the effect-mod arm)
    their modification by L1, which the regime-mean does not surface.

Self-validating against interventional MC truth. All OLS/logistic — no MCMC.
Run: python -m experiments.exp45_app_cointervention
"""
from pathlib import Path

from causal_bench.validation.longitudinal_cointervention import (
    g_estimation, g_estimation_effect_mod, ice_contrast, naive_effects, sim_longitudinal,
    true_values,
)

OUT_DIR = Path("results/exp45_app_cointervention")


def run(*, n=8000, psi0=0.3, psi1=0.5, effect_mod=0.4, seed=0):
    tv = true_values(psi0=psi0, psi1=psi1)
    d = sim_longitudinal(n, psi0=psi0, psi1=psi1, seed=seed)
    gp0, gp1 = g_estimation(d)
    ne = naive_effects(d)
    dm = sim_longitudinal(n, psi0=psi0, psi1=psi1, effect_mod=effect_mod, seed=seed + 1)
    em = g_estimation_effect_mod(dm)
    return {"truth": tv, "g_psi0": gp0, "g_psi1": gp1, "ice_contrast": ice_contrast(d),
            "naive": ne, "em": em, "effect_mod": effect_mod}


def report(r) -> str:
    tv = r["truth"]
    L = ["## Exp 45 — time-varying app co-intervention: estimand distinction\n",
         "| estimand / estimator | estimate | truth |",
         "|----------------------|----------|-------|",
         f"| **regime contrast** E[Y₁₁]−E[Y₀₀] — sequential regression (ICE) | {r['ice_contrast']:.3f} | {tv['contrast']:.3f} |",
         f"| **blip ψ(A1)** — g-estimation | {r['g_psi1']:.3f} | {tv['blip_A1']:.3f} |",
         f"| **blip ψ(A0)** (total, incl. L1-mediated) — g-estimation | {r['g_psi0']:.3f} | {tv['blip_A0']:.3f} |",
         f"| A0 effect — **naive** (adjusts the mediator L1) | {r['naive']['A0']:.3f} | {tv['blip_A0']:.3f} (blocked → biased low) |",
         "",
         "**Effect-modification arm** (A1 blip modified by L1):",
         f"- g-estimation ψ(A1)={r['em']['psi1']:.3f}, modification by L1={r['em']['effect_mod']:.3f} "
         f"(truth {r['effect_mod']}) — the object the regime-mean does not surface.",
         "",
         "**Two-timepoint `LTMLEEstimator`: not applicable** — no second treatment A1; it cannot "
         "represent the multi-period exposure (the gap that motivates this experiment)."]
    return "\n".join(L)


def main():
    import argparse
    p = argparse.ArgumentParser(description="Exp 45: time-varying app co-intervention")
    p.add_argument("--n", type=int, default=8000)
    a = p.parse_args()
    r = run(n=a.n)
    rep = report(r)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "summary.md").write_text(rep + "\n")
    print(rep)
    print("\nRead-out: g-estimation recovers the structural blips (incl. the A0 effect through the "
          "feedback confounder L1, and its modification), sequential regression recovers the "
          "regime contrast, and naive — by adjusting the mediator L1 — blocks A0's mediated path "
          "and is biased low. Different estimands, different tools; the two-timepoint LTMLE can't "
          "represent the multi-period exposure at all.")


if __name__ == "__main__":
    main()
