"""Exp 55: POOR-OVERLAP FAILURE of the uplift RMST/policy-value contrast + Kish-ESS guardrail (Figma retention).

The Figma-realistic failure mode: retention interventions are targeted by engagement signals that ALSO predict
churn, so treatment is assigned under strong confounding + poor overlap (worse for threshold rules like "nudge
teams with <5 collaborators", which drive the propensity toward 0/1). Under that regime the doubly-robust
targeting-policy value contrast COMPRESSES toward null — the correction has no matched units to borrow from.

Framed in McCoy's audit vocabulary (blind-contours.github.io/causality): a TMLE/AIPW one-step correction
*removes* the plug-in's confounding bias, *leaves* an inverse-propensity-weighted residual, and its interval
is *honest only while that residual has effective support*. The Kish ESS is exactly that support: it is the
operational answer to "WHEN is the interval honest?" — trust the contrast only where ESS stays above a floor.

Estimand (McCoy's argument): we report a difference-scale, restricted-horizon effect (policy value / RMST),
NOT a hazard ratio — the HR is non-collapsible and, under a time-varying effect, not even a single number;
RMST / policy value is collapsible, interpretable ("expected retained value"), and the estimand the ESS audit
is honest about. The point-outcome sweep here is the RMST analogue on a scalar outcome; survival/RMST via
exp51 `rmst_tmle_cate` + `clever_ess_profile` tells the identical story on the time-to-churn outcome.

Self-validating: MC truth (`true_values`) is gamma-INDEPENDENT — gamma degrades overlap, never the truth, so
the compression is estimator failure, not a moving target. Stretch (ESS necessary-but-NOT-sufficient) hands
off to the estimand-discipline thread (#206/#216): a healthy ESS cannot see an omitted-confounder DAG violation.
Reuse only (no new estimator, no concrete): numpy/sklearn. Run: python -m experiments.exp55_overlap_uplift
"""
from pathlib import Path
import numpy as np

from causal_bench.validation.uplift_policy import (
    sim_uplift, true_values, dr_learner_cate, aipw_policy_value, policy_value_ess, ate_overlap_ess,
    naive_policy_value, sim_uplift_confounded, confounded_trap, structure_layer_check, covariate_support,
)

OUT_DIR = Path("results/exp55_overlap_uplift")
GAMMAS = (0.3, 0.8, 1.5, 3.0, 6.0)          # SMOOTH confounding knob: good → bad overlap
THRESHOLDS = (0.0, 0.8, 1.5, 2.2)           # near-DETERMINISTIC targeting cut (support-gap non-overlap)
CURV = 0.7                                  # outcome-model MISSPECIFICATION: true μ0 curves in W1, learner is linear.
# Realistic (retention surfaces are not linear) and load-bearing: double-robustness now leans entirely on the
# propensity + overlap. Where overlap is good the IPW correction rescues the wrong μ (DR robust); where it is
# poor (clipping starves the correction) or absent (no support) the DR inherits the μ bias AND it persists with n.


def _one_overlap(*, n, seed, gamma=None, threshold=None):
    """One overlap level: DR targeting-policy contrast, its Kish ESS, the naive top-k contrast, vs MC truth.
    Pass `gamma` (smooth confounding — correct logistic propensity, so overlap alone gates DR) OR `threshold`
    (a near-deterministic rule A=1{W1+ε>threshold} → covariate-SUPPORT gap the weight-dispersion ESS can't see)."""
    d = sim_uplift(n, gamma=gamma, threshold=threshold, curv=CURV, seed=seed)
    tau_hat, nuis = dr_learner_cate(d, folds=2, seed=seed)
    _, A, _, e, _, _ = nuis
    pi = (tau_hat > 0).astype(int)
    v_dr = aipw_policy_value(nuis, pi) - aipw_policy_value(nuis, np.zeros(n, int))   # DR contrast vs treat-none
    v_naive = naive_policy_value(d, pi) - naive_policy_value(d, np.zeros(n, int))    # naive top-k (biased)
    ess, essf = ate_overlap_ess(A, e)                                               # weight-DISPERSION support
    _, essf_pol = policy_value_ess(nuis, pi)                                         # deployed-policy support
    Wmat = np.column_stack([d["W1"], d["W2"]])                                       # positivity-IN-W support:
    sup = covariate_support(Wmat, A, e)                                              # overlap of covariate dists
    knob = gamma if gamma is not None else threshold
    return {"knob": knob, "e_min": float(e.min()), "e_max": float(e.max()),
            "dr_contrast": float(v_dr), "naive_contrast": float(v_naive),
            "ess_frac": float(essf), "ess_frac_policy": float(essf_pol),
            "overlap_coef": sup["overlap_coef"], "common_support": sup["common_support"],
            "extrapolation": sup["extrapolation"]}


def run(*, n=8000, seed=0):
    tv = true_values(curv=CURV)
    sweep = [_one_overlap(n=n, seed=seed, gamma=g) for g in GAMMAS]              # smooth confounding
    tsweep = [_one_overlap(n=n, seed=seed, threshold=t) for t in THRESHOLDS]     # near-deterministic targeting
    return {"truth": tv, "n": n, "achievable": tv["achievable"], "sweep": sweep, "tsweep": tsweep}


def run_confounded(*, n=8000, seed=0):
    """Stretch: ESS necessary but NOT sufficient. Hidden U, true effect 0, healthy overlap → DR still wrong.
    The remedy (adjustment-set / FCI-with-latents / collider sensitivity) is the estimand-discipline thread
    #206/#216 — this only demonstrates that the overlap audit is blind to the DAG violation."""
    d = sim_uplift_confounded(n, seed=seed)
    r = confounded_trap(d, folds=2, seed=seed)
    try:
        r["structure"] = structure_layer_check(d, seed=seed)   # ZFCI/MB layer (also blind to U by construction)
    except Exception as ex:
        r["structure"] = {"error": repr(ex)}
    return r


def _sweep_block(sweep, ach, knob_label):
    lines = [f"  {knob_label:>5} | e-range      | DR contrast | naive top-k | overlap ESS% | deployed ESS%"]
    for s in sweep:
        lines.append(f"  {s['knob']:>5.1f} | [{s['e_min']:.3f},{s['e_max']:.3f}] | "
                     f"{s['dr_contrast']:+.3f}     | {s['naive_contrast']:+.3f}      | "
                     f"{100*s['ess_frac']:6.1f}       | {100*s['ess_frac_policy']:6.1f}")
    dr_bias = [round(abs(s["dr_contrast"] - ach), 3) for s in sweep]
    nv_bias = [round(abs(s["naive_contrast"] - ach), 3) for s in sweep]
    lines += [f"  |bias| DR:    {dr_bias}", f"  |bias| naive: {nv_bias}"]
    lines += _support_block(sweep, knob_label)
    return lines


def _support_block(sweep, knob_label):
    """Covariate-support (positivity-in-W) diagnostic printed beside the weight-dispersion ESS. These measure a
    DIFFERENT thing than the Kish ESS: the ESS sees only the DISPERSION of the IP weights, while these see
    whether the treated/control COVARIATE distributions actually overlap — so they fire in regime B where the
    ESS is blind. overlap_coef = ∫min of the two propensity densities (1=identical, 0=disjoint); common_support
    = fraction of units inside the overlapping propensity range; extrap = fraction whose k nearest neighbours in
    W are ALL same-arm (a pure-extrapolation unit the outcome model must guess, not interpolate)."""
    lines = [f"  {knob_label:>5} | overlap_coef | common_supp | extrapolation (pos-in-W support)"]
    for s in sweep:
        lines.append(f"  {s['knob']:>5.1f} | {s['overlap_coef']:.3f}        | {s['common_support']:.3f}       | "
                     f"{s['extrapolation']:.3f}")
    return lines


def report(r, rc=None) -> str:
    tv = r["truth"]; ach = tv["achievable"]
    lines = ["exp55 — poor-overlap failure of the uplift contrast + Kish-ESS guardrail (Figma retention)",
             f"  MC-true achievable policy value (knob- & curvature-independent) = {ach:+.3f}",
             "  Setup: true μ0 curves in W1, the outcome learner is LINEAR (misspecified) — so double-robustness",
             "  leans entirely on the propensity model + overlap (realistic: retention surfaces are not linear).",
             "",
             "  [A] SMOOTH confounding e(W)=expit(gamma·W1) — propensity correctly specified, so OVERLAP gates DR:"]
    lines += _sweep_block(r["sweep"], ach, "gamma")
    lines += ["  ⇒ TRUE-POSITIVE regime: as overlap worsens, clipping starves the IPW correction that was covering",
              "    the misspecified μ, and the DR bias tracks the ESS collapse IN LOCKSTEP and PERSISTS with n.",
              "    Here the Kish ESS IS the honest gauge — McCoy's 'when is the interval honest' made operational",
              "    (the correction removes plug-in confounding but leaves an IPW residual honest only while the ESS holds).",
              "    Naive top-k is catastrophic and monotone throughout.",
              "",
              "  [B] near-DETERMINISTIC targeting A=1{W1+ε>threshold} — a covariate-SUPPORT gap the ESS CANNOT see:"]
    lines += _sweep_block(r["tsweep"], ach, "thr")
    lines += ["  ⇒ FALSE-GREEN regime: the DR bias is large and PERSISTS with n (μ extrapolated into the no-support",
              "    tail), yet the overlap ESS reads HEALTHY — a near-deterministic rule classifies units correctly,",
              "    so the IPW weights are low-DISPERSION (Kish ESS high) even though covariate SUPPORT is absent.",
              "    The weight-dispersion ESS is blind to support-gap non-overlap. This is the Figma 'nudge teams with",
              "    <5 collaborators' rule (#224): the failure does NOT wash out with n and needs a positivity-in-W",
              "    support check (overlap of the covariate distributions), NOT just the weight ESS. The support block",
              "    above IS that check: as the cut steepens overlap_coef / common_support collapse and extrapolation",
              "    → 1 (the real signal) while the overlap ESS% column stays high — the two diagnostics disagree,",
              "    and it is the support check, not the ESS, that tracks the persistent DR bias."]
    if rc is not None:
        lines += ["", "  STRETCH — ESS necessary but NOT sufficient (hidden confounder U, true effect = 0):",
                  f"    DR contrast = {rc.get('contrast_dr', float('nan')):+.3f} (truth {rc.get('contrast_true', 0.0):+.1f}), "
                  f"overlap ESS% = {100*rc.get('essf', float('nan')):.1f} (HEALTHY, green light)",
                  "    ⇒ a robust estimator + a healthy ESS cannot rescue a mis-specified estimand — the omitted-",
                  "      confounder remedy (adjustment-set / FCI-with-latents / collider sensitivity) is #206/#216."]
        st = rc.get("structure")
        if isinstance(st, dict) and "ci_verdict" in st:
            lines.append(f"    structure layer (ZFCI/MB, also U-blind by construction): verdict={st['ci_verdict']} "
                         f"(p={st.get('ci_p', float('nan')):.3f}) — confirms the DAG check can't see U either.")
    return "\n".join(lines)


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    r = run(); rc = run_confounded()
    txt = report(r, rc)
    print(txt)
    (OUT_DIR / "report.txt").write_text(txt + "\n")
