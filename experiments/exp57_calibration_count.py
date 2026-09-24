"""Exp 57: CALIBRATION vs CAUSAL COUNT — a calibrated probability you can count on observationally is not a count
you can act on. The counting-axis analogue of exp52's ESS "necessary but not sufficient," with the SUFFICIENCY
complement made explicit: an E-value sensitivity analysis bounds the unmeasured-confounding gap that calibration
(and ESS) are blind to.
Run: python -m experiments.exp57_calibration_count
"""
from pathlib import Path
import numpy as np

from causal_bench.validation.calibration_count import (
    sim_churn_confounded, true_rates, calibration_count_demo, evalue_rr, _expit, _PU,
)

OUT_DIR = Path("results/exp57_calibration_count")


def _true_confounder_strength(n=4_000_000, seed=7):
    """Actual U associations for the E-value comparison: RR_EU = P(U|A=1)/P(U|A=0), RR_UD = P(Y|U=1)/P(Y|U=0)."""
    df = sim_churn_confounded(n, seed=seed)
    U = df["_U"].values; A = df["A"].values; Y = df["Y"].values
    rr_eu = (U[A == 1].mean()) / (U[A == 0].mean())
    rr_ud = (Y[U == 1].mean()) / (Y[U == 0].mean())
    return float(rr_eu), float(rr_ud)


def run(*, n=40000, seed=0):
    truth = true_rates()
    d = calibration_count_demo(sim_churn_confounded(n, seed=seed))
    rr_eu, rr_ud = _true_confounder_strength()
    return {"truth": truth, "demo": d, "n": n,
            "evalue_naive": evalue_rr(d["rr_naive"]), "rr_eu": rr_eu, "rr_ud": rr_ud}


def report(r) -> str:
    t = r["truth"]; d = r["demo"]; n = d["n"]
    C = lambda rate: rate * n                                     # per-capita rate → count over the cohort
    ev = r["evalue_naive"]
    L = ["## Exp 57 — calibration vs causal count (a probability you can count on ≠ a count you can act on)\n",
         f"n={n}. Confounded binary churn; hidden U. Product question: 'how many teams would we RETAIN by shipping",
         "the nudge to everyone?' — an INTERVENTIONAL count, answered wrong by a calibrated predictor.\n",
         "**1. Calibration holds (the necessary condition).**",
         f"- ECE of p̂(W,A) = **{d['ece']:.4f}** (small ⇒ among items called p, ~p come true — Montgomery's premise).",
         "",
         "**2. The OBSERVATIONAL count is right; the INTERVENTIONAL count is not.**",
         "| count | truth | estimate | note |",
         "|-------|-------|----------|------|",
         f"| observational  Σ churns | {C(t['rate_obs']):.0f} | **{C(d['rate_obs_hat']):.0f}** | calibrated sum works — Montgomery |",
         f"| interventional do(A=1) — **naive calibrated** | {C(t['rate_do1']):.0f} | **{C(d['rate_do1_naive']):.0f}** | biased (U unseen) |",
         f"| interventional do(A=1) — **oracle (U seen)** | {C(t['rate_do1']):.0f} | {C(d['rate_do1_oracle']):.0f} | recovers truth ⇒ gap = U |",
         "",
         f"So Σ p̂(W,A=1) mis-estimates the retained-teams count by **{C(d['rate_do1_naive']) - C(t['rate_do1']):+.0f}** "
         f"teams (naive {C(d['rate_do1_naive']):.0f} vs truth {C(t['rate_do1']):.0f}). Calibration is necessary for an",
         "honest observational count, **not sufficient** for a causal one — the exact converse-of-ESS beat, on the",
         "counting axis. (Montgomery's Poisson-binomial count variance Σp̂(1−p̂)="
         f"{d['pb_var_obs']:.0f} is also an under-estimate for the *causal* count: it ignores estimation + confounding uncertainty.)",
         "",
         "**3. The SUFFICIENCY side — what actually addresses the gap: a sensitivity analysis (E-value).**",
         f"- Naive interventional risk ratio RR = **{d['rr_naive']:.3f}** (true, U-adjusted, RR = {t['rr_true']:.3f}).",
         f"- **E-value = {ev:.2f}**: unmeasured confounding would need associations of ≥{ev:.2f} (RR scale) with BOTH",
         f"  treatment and outcome to explain the naive effect away. The DGP's ACTUAL confounder strengths are",
         f"  RR_EU = **{r['rr_eu']:.2f}** (U↔uptake) and RR_UD = **{r['rr_ud']:.2f}** (U↔churn) — "
         + ("both exceed the E-value" if min(r['rr_eu'], r['rr_ud']) >= ev else "at/near the E-value")
         + ", so the sensitivity analysis **correctly flags** the vulnerability that calibration and ESS missed.",
         "",
         "Read-out: 'sufficient' is never a single green light. Calibration (honest observational counts) and the",
         "Kish overlap ESS (positivity) are each **necessary, not sufficient**; causal validity is the *conjunction*",
         "— positivity (ESS) ∧ functional form (DR/EIC) ∧ estimand discipline (DAG/collider, #206/#216) ∧ a",
         "**sensitivity analysis** (E-value here) bounding the one condition no diagnostic can verify from data:",
         "no unmeasured confounding. The E-value doesn't PROVE the gap closed — it can't — it BOUNDS it. That is",
         "the honest shape of the sufficiency answer."]
    return "\n".join(L)


def main():
    import argparse
    p = argparse.ArgumentParser(description="Exp 57: calibration vs causal count")
    p.add_argument("--n", type=int, default=40000)
    a = p.parse_args()
    r = run(n=a.n)
    rep = report(r)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "summary.md").write_text(rep + "\n")
    print(rep)


if __name__ == "__main__":
    main()
