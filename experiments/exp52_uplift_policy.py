"""Exp 52: UPLIFT TARGETING POLICY VALUE (doubly robust) + Qini.

Turns CATE into what uplift modeling is actually for — a targeting policy ("treat the top-k% by predicted
uplift") — and evaluates its VALUE the causal-correct way. Under confounding, the usual naive Qini (top-k
outcome mean) is BIASED; the doubly-robust (AIPW) off-policy value is unbiased. DR-learner CATE for ranking,
AIPW for evaluation. Self-validating against interventional MC truth (optimal policy value).
Run: python -m experiments.exp52_uplift_policy
"""
from pathlib import Path
import numpy as np

from causal_bench.validation.uplift_policy import (
    sim_uplift, true_values, dr_learner_cate, aipw_policy_value, policy_value_ess, naive_policy_value, qini,
    sim_uplift_confounded, confounded_trap, structure_layer_check,
)

OUT_DIR = Path("results/exp52_uplift_policy")


def run(*, n=8000, seed=0):
    tv = true_values()
    d = sim_uplift(n, seed=seed)
    tau_hat, nuis = dr_learner_cate(d, folds=2, seed=seed)
    # the estimated optimal policy: treat where predicted uplift > 0
    pi_hat = (tau_hat > 0).astype(int)
    v_dr_pihat = aipw_policy_value(nuis, pi_hat)
    v_dr_none = aipw_policy_value(nuis, np.zeros(n, int))
    v_dr_all = aipw_policy_value(nuis, np.ones(n, int))
    # naive value of the SAME policy (biased under confounding)
    v_nv_pihat = naive_policy_value(d, pi_hat); v_nv_none = naive_policy_value(d, np.zeros(n, int))
    ess_pihat, essf_pihat = policy_value_ess(nuis, pi_hat)      # overlap of the DEPLOYED targeting policy
    q = qini(tau_hat, nuis, d)
    return {"truth": tv, "n": n, "frac_treated_hat": float(pi_hat.mean()),
            "v_dr_pihat": v_dr_pihat, "v_dr_none": v_dr_none, "v_dr_all": v_dr_all,
            "v_nv_pihat": v_nv_pihat, "v_nv_none": v_nv_none, "qini": q,
            "ess_pihat": ess_pihat, "essf_pihat": essf_pihat}


def run_confounded(*, n=8000, seed=0):
    """The necessary-but-not-sufficient scenario: a hidden confounder U, true effect 0, healthy overlap. Runs all
    three diagnostics — overlap ESS, (implicit) calibration, and the STRUCTURE layer (ZFCI/MB) — each blind to U."""
    d = sim_uplift_confounded(n, seed=seed)
    r = confounded_trap(d, folds=2, seed=seed)
    r["structure"] = structure_layer_check(d, seed=seed)
    return r


def report(r) -> str:
    t = r["truth"]; q = r["qini"]
    ach = t["achievable"]
    cap_dr = (r["v_dr_pihat"] - r["v_dr_none"]) / ach                    # fraction of achievable uplift captured
    # naive "capture" using the biased value (to show it misleads)
    cap_nv = (r["v_nv_pihat"] - r["v_nv_none"]) / ach
    # Qini of a perfect-ranking oracle (area of the concave optimal curve) for normalization context
    L = ["## Exp 52 — uplift targeting policy value (doubly robust) + Qini\n",
         f"n={r['n']}. Heterogeneous confounded CATE; policy = treat where predicted uplift > 0.\n",
         "| quantity | value | truth / note |",
         "|----------|-------|--------------|",
         f"| achievable uplift  V(π*)−V(none) | — | **{ach:.3f}** (optimal targets {100*t['frac_treated_opt']:.0f}%) |",
         f"| our policy treats | {100*r['frac_treated_hat']:.0f}% | vs optimal {100*t['frac_treated_opt']:.0f}% |",
         f"| **DR** policy value gain  V̂(π̂)−V̂(none) | {r['v_dr_pihat']-r['v_dr_none']:.3f} | captures **{100*cap_dr:.0f}%** of achievable |",
         f"| **naive** policy value gain (confounded) | {r['v_nv_pihat']-r['v_nv_none']:.3f} | \"{100*cap_nv:.0f}%\" — biased, misleads |",
         f"| **Qini coefficient** (DR) | {q['qini_dr']:.3f} | area of DR uplift curve above random targeting |",
         f"| ↳ **overlap of deployed policy** — Kish ESS = {r['ess_pihat']:.0f}/{r['n']} ({100*r['essf_pihat']:.0f}%) | (positivity check) | worst-bin ESS {100*q['min_ess_frac_dr']:.0f}% |",
         "",
         "Uplift curve (DR value gain vs targeted fraction k), with per-bin overlap ESS:",
         "```",
         "k       : " + " ".join(f"{k:.2f}" for k in q["ks"][::4]),
         "DR      : " + " ".join(f"{u:.2f}" for u in q["uplift_dr"][::4]),
         "naive   : " + " ".join(f"{u:.2f}" for u in q["uplift_naive"][::4]),
         "rand    : " + " ".join(f"{u:.2f}" for u in q["rand"][::4]),
         "ESS%    : " + " ".join(f"{100*e:.0f}" for e in q["ess_frac_dr"][::4]),
         "```",
         "",
         "The **ESS%** row is the off-policy overlap of each top-k targeting policy (the DR value's effective n):",
         "low-ESS bins are variance-dominated, so trust the uplift curve only where ESS stays high — the same",
         "Kish ESS the QEC-decoder RL-gym uses as its PPO reuse gate / hardness order-parameter.",
         "",
         "Read-out: ranking users by DR-learner CATE and treating the top fraction captures most of the",
         "achievable uplift; the DR (AIPW) policy value is unbiased, while the naive top-k outcome mean — the",
         "usual Qini — is biased by confounding (treated users differ systematically). The differentiator vs a",
         "standard uplift-tree + naive-Qini pipeline: the *evaluation* is doubly robust, so the targeting ROI",
         "you report to the business is trustworthy under confounding."]
    if r.get("confounded") is not None:
        c = r["confounded"]
        L += ["",
              "## The overlap ESS is NECESSARY but NOT SUFFICIENT (unmeasured-confounder variant)\n",
              "Same AIPW machinery, but now a hidden binary U (\"enterprise mandate\") confounds A and Y, and the",
              "TRUE treatment effect is **exactly 0**. A depends on U (hidden) + W1 (observed), so the observed",
              "propensity is smoothly distributed and the overlap diagnostic flashes green:",
              "",
              "| quantity | value | note |",
              "|----------|-------|------|",
              f"| propensity spread ê(W) | [{c['e_min']:.2f}, {c['e_max']:.2f}] | no extreme weights, no thresholding |",
              f"| **ATE overlap Kish ESS** (all units) | **{100*c['essf']:.0f}%** | overlap looks *healthy* — green light |",
              f"| true interventional contrast V(all)−V(none) | **{c['contrast_true']:.3f}** | τ≡0 by construction |",
              f"| **DR (AIPW) contrast** | **{c['contrast_dr']:+.3f}** | confidently BIASED despite the green ESS |",
              "",
              "The ESS reports on *positivity/overlap*; it is blind to the A←U→Y back-door that W does not close.",
              "A doubly-robust estimator cannot rescue a mis-specified estimand — the same lesson as the",
              "collider/estimand-discipline thread (#206/#216): you cannot math your way out of a violated DAG.",
              "This is the converse failure to the top-k contrast collapse above (loud, overlap-driven); here the",
              "diagnostic is silent and green while the answer is wrong. ESS is a *necessary* gate, not a sufficient one."]
        s = c.get("structure")
        if s is not None:                       # third leg: the STRUCTURE layer (ZFCI/MB) is blind to U too
            L += ["",
                  "**And the STRUCTURE layer (ZFCI / Markov blanket, exp39/46) is equally blind to U — the third leg.**",
                  f"- Recovered predictive Markov blanket of Y: **{{{', '.join(s['mb_Y'])}}}** — a structure pipeline",
                  "  would use these features, and A∈MB(Y) is a *prediction* fact (A helps predict Y), not evidence A",
                  "  *causes* Y.",
                  f"- ZFCI test  A ⫫ Y | (W1,W2):  **{s['ci_verdict']}** (p={s['ci_p']:.3f}) — residual A–Y dependence",
                  "  after adjusting the OBSERVED confounders. But that dependence is equally consistent with a causal",
                  "  A→Y *and* with A←U→Y confounding; no observed-variable CI test can distinguish them. So the",
                  "  structure evidence is **necessary but not sufficient** to license the causal read — the same blind",
                  "  spot as ESS and calibration. Three green-or-ambiguous diagnostics, one hidden U: only a sensitivity",
                  "  analysis (the E-value, exp57) bounds the gap none of them can see."]
    return "\n".join(L)


def main():
    import argparse
    p = argparse.ArgumentParser(description="Exp 52: uplift targeting policy value")
    p.add_argument("--n", type=int, default=8000)
    a = p.parse_args()
    r = run(n=a.n)
    r["confounded"] = run_confounded(n=a.n)
    rep = report(r)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "summary.md").write_text(rep + "\n")
    print(rep)


if __name__ == "__main__":
    main()
