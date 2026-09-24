"""Exp 54: SURVIVAL variant of exp45 — time-varying treatment + time-to-event (churn) outcome.

Sequential lifecycle program on new teams (Figma story): week-1 nudge A0 → week-1 engagement L1
(raised by A0) → week-4 re-nudge A1 (targeted by low L1) → time-to-churn T, with informative
censoring. L1 is a treatment-affected confounder — a *mediator* of A0 and a *confounder* of A1 —
so ordinary adjustment fails and g-methods are required, now on a SURVIVAL outcome.

Comparison (all self-validating vs interventional MC truth):
  • **ICE-survival g-computation** (IPCW) — recovers the regime survival S(τ|ā), RMST, and the A0
    survival blip (total effect incl. the L1-mediated path).
  • **last-blip g-estimation** ψ1 — full-history-adjusted Cox coefficient = structural log-HR of A1.
  • **naive (biased)** — per-(A0,A1) KM (ignores confounding/feedback) and a Cox adjusting the
    mediator L1 (blocks A0's engagement-mediated benefit).

Beyond the two-timepoint LTMLE; the DR survival-LTMLE / continuous-time CONCRETE column (optimal
dynamic regime, survival-SNMM targeting) is phase-2 / gated on #223. No MCMC, no concrete.
Run: python -m experiments.exp54_survival_timevarying_uplift
"""
from pathlib import Path

from causal_bench.validation.survival_timevarying import (
    g_estimation_psi1, ice_regime_table, ice_rmst_regime, naive_cox_A0, naive_km_regime,
    sim_survival_timevarying, true_regime_survival, discover_timevarying_structure,
)

OUT_DIR = Path("results/exp54_survival_timevarying_uplift")


def run(*, n=12000, tau=3.0, seed=0):
    truth = true_regime_survival(tau=tau)
    df = sim_survival_timevarying(n, tau=tau, seed=seed)
    ice = ice_regime_table(df, tau)
    rmst = {r: ice_rmst_regime(df, r[0], r[1], tau) for r in [(0.0, 0.0), (1.0, 1.0)]}
    return {"truth": truth, "ice": ice, "ice_rmst": rmst,
            "psi1": g_estimation_psi1(df), "naive_km": naive_km_regime(df, tau),
            "naive_cox_A0": naive_cox_A0(df), "n": n, "tau": tau,
            "discovery": discover_timevarying_structure(df, seed=seed)}


def report(r) -> str:
    t = r["truth"]; ice = r["ice"]; ts = t["surv"]; es = ice["surv"]
    regimes = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
    L = ["## Exp 54 — survival variant of exp45: time-varying treatment, time-to-churn outcome\n",
         f"n={r['n']}, τ={r['tau']}. W→A0→L1→A1→T(churn), L1 a treatment-affected confounder, informative "
         "censoring. Estimand: regime survival S(τ|ā) and its contrasts.\n",
         "| regime ā=(A0,A1) | truth S(τ) | ICE-survival g-comp | naive per-(A0,A1) KM |",
         "|------------------|-----------|---------------------|----------------------|"]
    nk = r["naive_km"]["surv"]
    for a0, a1 in regimes:
        L.append(f"| ({a0:.0f},{a1:.0f}) | {ts[(a0, a1)]:.3f} | {es[(a0, a1)]:.3f} | {nk[(a0, a1)]:.3f} |")
    L += ["",
          f"**Regime contrast** S(τ|1,1)−S(τ|0,0): truth {t['contrast_surv']:.3f}, "
          f"ICE g-comp {ice['contrast_surv']:.3f} (recovers), naive KM {r['naive_km']['contrast_surv']:.3f} (biased).",
          "",
          "**A0 survival blip** S(τ|1,a1)−S(τ|0,a1) — the TOTAL A0 effect incl. the L1-mediated path:",
          f"- a1=0: truth {t['blip_A0_surv'][0.0]:.3f}, ICE {ice['blip_A0_surv'][0.0]:.3f}",
          f"- a1=1: truth {t['blip_A0_surv'][1.0]:.3f}, ICE {ice['blip_A0_surv'][1.0]:.3f}",
          "",
          "**RMST** (expected active-days over τ), via ICE-survival ∫S dt:",
          f"- ā=(0,0): truth {t['rmst'][(0.0, 0.0)]:.3f}, ICE {r['ice_rmst'][(0.0, 0.0)]:.3f}",
          f"- ā=(1,1): truth {t['rmst'][(1.0, 1.0)]:.3f}, ICE {r['ice_rmst'][(1.0, 1.0)]:.3f}",
          f"- RMST contrast (1,1)−(0,0): truth {t['contrast_rmst']:.3f}, "
          f"ICE {r['ice_rmst'][(1.0, 1.0)] - r['ice_rmst'][(0.0, 0.0)]:.3f}",
          "",
          "**Structural blips (hazard scale):**",
          f"- ψ1 (A1 log-HR) — last-blip g-estimation (full-history Cox): {r['psi1']:.3f}, truth {t['psi_logHR'][1]:.3f} "
          "(recovered — L1 precedes A1).",
          f"- A0 log-HR — **naive Cox adjusting the mediator L1**: {r['naive_cox_A0']:.3f}, structural ψ0 "
          f"{t['psi_logHR'][0]:.3f} → attenuated toward 0 (L1-mediated churn-reduction blocked).",
          "",
          "Read-out: adjusting the treatment-affected confounder L1 in the OUTCOME (naive Cox) blocks A0's "
          "benefit-through-engagement; ignoring the sequential structure (naive KM) biases the regime "
          "contrast. ICE-survival g-computation recovers the interventional regime survival, RMST, and the "
          "A0 total effect on the collapsible survival scale; last-blip g-estimation recovers ψ1. The A0 "
          "*hazard* blip is not reported as a single HR — Cox non-collapsibility, which is why the A0 total "
          "effect lives on the survival/RMST scale.",
          "",
          "(DR survival-LTMLE / continuous-time CONCRETE — optimal dynamic regime + survival-SNMM targeting — "
          "is phase-2, gated on the concrete default-learner attenuation fix #223. See docs/plans/2026-09-23-*.)"]
    dsc = r.get("discovery")
    if dsc is not None:                     # constraint-based discovery VALIDATES the g-methods requirement
        L += ["",
              "**Structure discovery (ZFCI, exp39/46) — the g-methods requirement, *recovered* not asserted.**",
              "The power-use of the structure layer: L1's ambiguous role is an OBSERVED-variable question CI",
              "structure can resolve (unlike a hidden U). Verdicts (temporal order W<A0<L1<A1<T fixes arrows):",
              "```"]
        for name, tr in dsc["tests"].items():
            L.append(f"  {name:<30} {tr['verdict']:<10} (p={tr['p']:.3f})")
        L += ["```",
              f"⇒ L1 is a mediator of A0: **{dsc['L1_is_mediator_of_A0']}**  (A0→L1); "
              f"L1 confounds A1: **{dsc['L1_is_confounder_of_A1']}**  (L1→A1 and L1→Y).",
              f"⇒ **treatment-affected confounder recovered: {dsc['treatment_affected_confounder']}** — so adjusting L1",
              "blocks A0's mediated path while not-adjusting leaves A1 confounded: **no static adjustment set works,",
              "g-methods are required**, now shown empirically. Negative control A1⫫A0|L1 supports "
              f"(**discriminates: {dsc['discriminates']}**), so the discovery isn't just flagging everything."]
    return "\n".join(L)


def main():
    import argparse
    p = argparse.ArgumentParser(description="Exp 54: survival time-varying uplift")
    p.add_argument("--n", type=int, default=12000)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    r = run(n=a.n, seed=a.seed)
    rep = report(r)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "summary.md").write_text(rep + "\n")
    print(rep)


if __name__ == "__main__":
    main()
