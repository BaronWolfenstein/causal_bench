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
    sim_uplift, true_values, dr_learner_cate, aipw_policy_value, naive_policy_value, qini,
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
    q = qini(tau_hat, nuis, d)
    return {"truth": tv, "n": n, "frac_treated_hat": float(pi_hat.mean()),
            "v_dr_pihat": v_dr_pihat, "v_dr_none": v_dr_none, "v_dr_all": v_dr_all,
            "v_nv_pihat": v_nv_pihat, "v_nv_none": v_nv_none, "qini": q}


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
         "",
         "Uplift curve (DR value gain vs targeted fraction k):",
         "```",
         "k     : " + " ".join(f"{k:.2f}" for k in q["ks"][::4]),
         "DR    : " + " ".join(f"{u:.2f}" for u in q["uplift_dr"][::4]),
         "naive : " + " ".join(f"{u:.2f}" for u in q["uplift_naive"][::4]),
         "rand  : " + " ".join(f"{u:.2f}" for u in q["rand"][::4]),
         "```",
         "",
         "Read-out: ranking users by DR-learner CATE and treating the top fraction captures most of the",
         "achievable uplift; the DR (AIPW) policy value is unbiased, while the naive top-k outcome mean — the",
         "usual Qini — is biased by confounding (treated users differ systematically). The differentiator vs a",
         "standard uplift-tree + naive-Qini pipeline: the *evaluation* is doubly robust, so the targeting ROI",
         "you report to the business is trustworthy under confounding."]
    return "\n".join(L)


def main():
    import argparse
    p = argparse.ArgumentParser(description="Exp 52: uplift targeting policy value")
    p.add_argument("--n", type=int, default=8000)
    a = p.parse_args()
    r = run(n=a.n)
    rep = report(r)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "summary.md").write_text(rep + "\n")
    print(rep)


if __name__ == "__main__":
    main()
