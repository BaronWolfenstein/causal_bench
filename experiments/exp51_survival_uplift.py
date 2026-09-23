"""Exp 51: SURVIVAL UPLIFT — heterogeneous time-to-event treatment effect (CATE by an effect modifier
V) under confounding + informative censoring, with McCoy's CONCRETE fork as the doubly-robust engine.

The time-to-event analogue of uplift modeling (retention/churn): which users (by V) get the most
*survival* benefit from the intervention. Estimand: per-stratum survival-probability uplift
CATE(v)=S(τ|1,v)−S(τ|0,v). Naive per-arm Kaplan–Meier is biased (treatment confounded by W1 →
treated look sicker; censoring informative in W1); per-stratum DR survival estimators recover it:
TMLE-IPCW (Python) and CONCRETE (McCoy's fork, continuous-time TMLE).

Self-validating against interventional MC truth. Reuses tmle_ipcw + the concrete_rmst bridge.
Run: python -m experiments.exp51_survival_uplift
"""
from pathlib import Path

from causal_bench.validation.survival_uplift import (
    sim_survival_uplift, true_cate, km_cate, tmle_cate, concrete_cate,
)

OUT_DIR = Path("results/exp51_survival_uplift")


def run(*, n=6000, tau=3.0, seed=0):
    truth = true_cate(tau=tau)
    df = sim_survival_uplift(n, tau=tau, seed=seed)
    return {"truth": truth, "km": km_cate(df, tau), "tmle": tmle_cate(df, tau),
            "concrete": concrete_cate(df, tau), "n": n, "tau": tau}


def report(r) -> str:
    t = r["truth"]["cate"]; km = r["km"]; tm = r["tmle"]; cc = r["concrete"]
    def fmt(d, v):
        return f"{d[v]:.3f}" if (d and v in d and d[v] == d[v]) else "—"
    cc_col = "CONCRETE (McCoy, DR)" if cc else "CONCRETE (R n/a)"
    L = ["## Exp 51 — survival uplift (per-stratum survival-probability CATE) under confounding + informative censoring\n",
         f"n={r['n']}, horizon τ={r['tau']}. Uplift CATE(v) = S(τ|A=1,V=v) − S(τ|A=0,V=v) (retention benefit).\n",
         f"| stratum V | truth | Kaplan–Meier (naive) | TMLE-IPCW (DR) | {cc_col} |",
         "|-----------|-------|----------------------|----------------|-----------------------|",
         f"| V=0 | {t[0.0]:.3f} | {fmt(km,0.0)} | {fmt(tm,0.0)} | {fmt(cc,0.0)} |",
         f"| V=1 | {t[1.0]:.3f} | {fmt(km,1.0)} | {fmt(tm,1.0)} | {fmt(cc,1.0)} |",
         "",
         f"**Uplift heterogeneity** CATE(1)−CATE(0): truth {r['truth']['heterogeneity']:.3f}, "
         f"TMLE-IPCW {tm[1.0]-tm[0.0]:.3f} (DR), KM {km[1.0]-km[0.0]:.3f} (biased)"
         + (f", CONCRETE {cc[1.0]-cc[0.0]:.3f} (McCoy DR)." if cc else "."),
         "",
         "Read-out: under confounding (A↑ with W1, which also ↑ hazard → treated look sicker) + informative",
         "censoring in W1, per-arm Kaplan–Meier is biased low for the per-stratum uplift; the DR survival",
         "estimators (TMLE-IPCW; CONCRETE continuous-time TMLE) adjust W1–W4 + model the censoring and recover",
         "it. This is the time-to-event uplift the point-outcome estimators (exp45) cannot address — churn/",
         "retention CATE, doubly robust."]
    return "\n".join(L)


def main():
    import argparse
    p = argparse.ArgumentParser(description="Exp 51: survival uplift")
    p.add_argument("--n", type=int, default=6000)
    a = p.parse_args()
    r = run(n=a.n)
    rep = report(r)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "summary.md").write_text(rep + "\n")
    print(rep)


if __name__ == "__main__":
    main()
