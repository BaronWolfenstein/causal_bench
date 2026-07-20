"""Borrowing-calibration fidelity engine on the joint DGP — the reusable core of
exp41 (#144 step C). Runs the BP-decoded-labels pipeline end to end and reports the
frequentist operating characteristics of an identifiability-informed borrowing prior.

Pipeline per replicate: ``sample_joint_cohort`` → ``decode_cohort_labels(θ₀)`` →
per-**decoded**-subgroup ``(theta_hat, se)`` → ``fit_three_level_meta(theta_hat, se,
tau_sd=policy)`` → decision on the population effect μ. The estimator only ever sees
BP-decoded labels; the true labels are used solely to compute the oracle τ and the
known μ for scoring.

``tau_policy``:
- ``flat``    — a fixed ``tau_sd`` (the naive baseline);
- ``oracle``  — the true between-subgroup effect SD at the level (best case);
- ``canonical`` — ``tau_base · canonical_tau_discount(decode_acc)``: the analyst's base
  effect-scale prior, *discounted* by the level's decode accuracy at θ₀ (the
  identifiability-informed prior under test — a discount, not an absolute setter).

This is the ENGINE (a library function); the exp41 experiment script sweeps regimes ×
θ₀ × grammar configs × policies and compares reject/coverage curves. Requires the 3.12
``[bayes]`` stack (PyMC/NumPyro) via ``fit_three_level_meta``.
"""
from __future__ import annotations

import numpy as np

from causal_bench.dgp.joint_hierarchy import (
    make_joint_hierarchy, sample_joint_cohort, decode_cohort_labels, true_tau_by_level,
)
from causal_bench.diagnostics.borrowing_informativeness import canonical_tau_discount


def population_effect(spec: dict) -> float:
    """The true population-average treatment effect μ implied by the spec's effect
    tables (subgroups uniform): ``w_group·mean(group_effect) + w_member·mean(member_
    effect)``. Used as ``true_effect`` for coverage and as the null target when the
    effect tables are centered."""
    return float(spec["w_group"] * spec["group_effect"].mean()
                 + spec["w_member"] * spec["member_effect"].mean())


def _subgroup_estimates(Y, A, sub, n_sub, *, min_per_arm=3):
    """Per-decoded-subgroup treatment-effect estimates (mean-difference), SEs, and the
    surviving subgroup indices. Subgroups without at least ``min_per_arm`` units in each
    arm are dropped — ``kept`` maps rows of ``(theta_hat, se)`` back to subgroup ids (so a
    partial-null test can locate the null subgroup after drops)."""
    th, se, kept = [], [], []
    for k in range(n_sub):
        m = sub == k
        y1, y0 = Y[m & (A == 1)], Y[m & (A == 0)]
        if len(y1) < min_per_arm or len(y0) < min_per_arm:
            continue
        th.append(y1.mean() - y0.mean())
        se.append(np.sqrt(y1.var(ddof=1) / len(y1) + y0.var(ddof=1) / len(y0)))
        kept.append(k)
    return np.asarray(th), np.asarray(se), np.asarray(kept, int)


def _policy_tau_sd(policy, level, spec, decoded, *, flat_tau_sd, tau_base, tau_sd_min):
    if policy == "flat":
        return flat_tau_sd
    if policy == "oracle":
        key = "tau_group" if level == "group" else "tau_member"
        return max(true_tau_by_level(spec)[key], 1e-3)          # HalfNormal scale must be > 0
    if policy == "canonical":
        acc = decoded["group_decode_acc" if level == "group" else "member_decode_acc"]
        k = spec["g"] if level == "group" else spec["b_size"]
        # tau_sd = tau_base · learnability-discount (NOT an absolute map — see #144/exp41):
        # a well-decoded level recovers the base scale; a poorly-decoded one pools harder.
        return max(tau_base * canonical_tau_discount(acc, k), tau_sd_min)
    raise ValueError(f"unknown policy {policy!r}")


def joint_fidelity(spec: dict, *, level: str = "group", policy: str = "canonical",
                   theta0: float = 0.7, n_reps: int = 20, n_units: int = 4000,
                   depth: int = 7, sigma: float = 0.5, flat_tau_sd: float = 0.5,
                   tau_base: float = 0.5, tau_sd_min: float = 0.05, draws: int = 500,
                   tune: int = 500, chains: int = 2, seed: int = 0,
                   chain_method: str = "sequential",
                   tail_ess_threshold: float = 100.0, null_subgroup: int | None = None) -> dict:
    """Operating characteristics of the borrowing prior at one (level, policy, θ₀, spec)
    cell. ``reject_rate`` is the population-μ decision (Type-I under a null spec, power
    under an alt). When ``null_subgroup`` is set (a partial-null spec, e.g.
    ``make_partial_null_spec``), also reports ``subgroup_reject_rate`` — the per-subgroup
    Type-I of that truly-null subgroup, the estimand where strong borrowing inflates size
    by dragging it toward non-null siblings.

    Reports (per the #144 review): ``mean_ci_width`` (μ CI width — the primary honest OC,
    since reject≈0 at small K is a size-≈0 test, not "nominal"), ``coverage``, and BOTH
    the tail-ESS-gated ``reject_rate`` and the flagged-included ``reject_rate_uncond``
    (dropping flagged fits is selection-on-data). Subgroups = the DECODED labels at
    ``level``; the prior is set by ``policy``."""
    from causal_bench.estimators.three_level_bhm import fit_three_level_meta, tail_ess_ok

    mu_true = population_effect(spec)
    tau_true = true_tau_by_level(spec)["tau_group" if level == "group" else "tau_member"]
    rejects, covers, taus, widths, sub_rejects = [], [], [], [], []
    rejects_all, n_flagged, n_used = [], 0, 0
    for r in range(n_reps):
        coh = sample_joint_cohort(spec, n_units, depth, sigma=sigma, seed=seed + r)
        dec = decode_cohort_labels(spec, coh, theta0=theta0, seed=seed + 1000 + r)
        sub = dec["group_decoded" if level == "group" else "member_decoded"]
        n_sub = spec["g"] if level == "group" else spec["b_size"]
        th, se, kept = _subgroup_estimates(coh["Y"], coh["A"], sub, n_sub)
        if len(th) < 2:
            continue
        tau_sd = _policy_tau_sd(policy, level, spec, dec, flat_tau_sd=flat_tau_sd,
                                tau_base=tau_base, tau_sd_min=tau_sd_min)
        fit = fit_three_level_meta(th, se, tau_sd=tau_sd, true_effect=mu_true,
                                   draws=draws, tune=tune, chains=chains, seed=seed + r,
                                   chain_method=chain_method,
                                   return_theta=null_subgroup is not None)
        rejects_all.append(fit["rejects_null"])                 # flagged-included sensitivity
        if not tail_ess_ok(fit, threshold=tail_ess_threshold):
            n_flagged += 1
            continue
        rejects.append(fit["rejects_null"])
        covers.append(fit["covers_truth"])
        taus.append(tau_sd)
        widths.append(fit["ci_hi"] - fit["ci_lo"])
        if null_subgroup is not None:
            pos = np.where(kept == null_subgroup)[0]
            if len(pos):                                        # null subgroup survived drops
                sub_rejects.append(bool(fit["theta_g_rejects"][pos[0]]))
        n_used += 1
    return {
        "reject_rate": float(np.mean(rejects)) if rejects else float("nan"),
        "reject_rate_uncond": float(np.mean(rejects_all)) if rejects_all else float("nan"),
        "subgroup_reject_rate": float(np.mean(sub_rejects)) if sub_rejects else float("nan"),
        "coverage": float(np.mean(covers)) if covers else float("nan"),
        "mean_ci_width": float(np.mean(widths)) if widths else float("nan"),
        "mean_tau_sd": float(np.mean(taus)) if taus else float("nan"),
        "mu_true": mu_true, "tau_true": float(tau_true),
        "n_flagged": n_flagged, "n_used": n_used,
    }


def make_scenario_spec(g, b_size, s, m, *, level: str, mu: float = 0.0, tau: float = 0.0,
                       seed: int = 0) -> dict:
    """A spec with a KNOWN population effect μ = ``mu`` and between-subgroup SD τ =
    ``tau`` at ``level`` (the other level carries no effect). The level's effect table is
    standardized (mean 0, unit SD) then set to mean ``mu`` and SD ``tau``, weight 1.
    Scenarios: global null ``(mu=0, tau=0)``; heterogeneous null ``(mu=0, tau>0)`` — the
    case where borrowing threatens Type I; alternative ``(mu≠0, ...)``."""
    spec = make_joint_hierarchy(g, b_size, s, m, w_group=0.0, w_member=0.0, seed=seed)
    key, w = ("group_effect", "w_group") if level == "group" else ("member_effect", "w_member")
    e = spec[key] - spec[key].mean()
    sd = e.std()
    z = (e / sd) if sd > 1e-9 else e                            # mean 0, unit SD
    spec[key] = mu + tau * z                                    # mean μ, SD τ
    spec[w] = 1.0
    return spec


def make_null_spec(g, b_size, s, m, *, level: str, tau_scale: float, seed: int = 0) -> dict:
    """Backward-compatible wrapper: a null spec (μ = 0) with between-subgroup SD
    ``tau_scale``. See ``make_scenario_spec``."""
    return make_scenario_spec(g, b_size, s, m, level=level, mu=0.0, tau=tau_scale, seed=seed)


def make_partial_null_spec(g, b_size, s, m, *, level: str, sibling_effect: float,
                           null_idx: int = 0, seed: int = 0) -> dict:
    """A **partial null**: subgroup ``null_idx`` is truly null (θ = 0) while every sibling
    has the SAME-sign effect ``sibling_effect`` (same sign maximizes the population mean μ,
    hence the borrowing drag on the null subgroup — the adversarial case for per-subgroup
    Type-I). Pass ``null_subgroup=null_idx`` to ``joint_fidelity`` to measure whether
    borrowing makes the null subgroup's posterior reject θ = 0."""
    spec = make_joint_hierarchy(g, b_size, s, m, w_group=0.0, w_member=0.0, seed=seed)
    key, w = ("group_effect", "w_group") if level == "group" else ("member_effect", "w_member")
    k = g if level == "group" else b_size
    e = np.full(k, float(sibling_effect))
    e[null_idx] = 0.0                                           # the truly-null subgroup
    spec[key] = e
    spec[w] = 1.0
    return spec
