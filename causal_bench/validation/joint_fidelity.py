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
- ``canonical`` — set from the level's **decode accuracy** at θ₀ via
  ``canonical_tau_prior`` (the identifiability-informed prior under test).

This is the ENGINE (a library function); the exp41 experiment script sweeps regimes ×
θ₀ × grammar configs × policies and compares reject/coverage curves. Requires the 3.12
``[bayes]`` stack (PyMC/NumPyro) via ``fit_three_level_meta``.
"""
from __future__ import annotations

import numpy as np

from causal_bench.dgp.joint_hierarchy import (
    make_joint_hierarchy, sample_joint_cohort, decode_cohort_labels, true_tau_by_level,
)
from causal_bench.diagnostics.borrowing_informativeness import canonical_tau_prior


def population_effect(spec: dict) -> float:
    """The true population-average treatment effect μ implied by the spec's effect
    tables (subgroups uniform): ``w_group·mean(group_effect) + w_member·mean(member_
    effect)``. Used as ``true_effect`` for coverage and as the null target when the
    effect tables are centered."""
    return float(spec["w_group"] * spec["group_effect"].mean()
                 + spec["w_member"] * spec["member_effect"].mean())


def _subgroup_estimates(Y, A, sub, n_sub, *, min_per_arm=3):
    """Per-decoded-subgroup treatment-effect estimates (mean-difference) and SEs.
    Subgroups without at least ``min_per_arm`` units in each arm are dropped (can't form
    a stable estimate) — returns only the resolvable subgroups' ``(theta_hat, se)``."""
    th, se = [], []
    for k in range(n_sub):
        m = sub == k
        y1, y0 = Y[m & (A == 1)], Y[m & (A == 0)]
        if len(y1) < min_per_arm or len(y0) < min_per_arm:
            continue
        th.append(y1.mean() - y0.mean())
        se.append(np.sqrt(y1.var(ddof=1) / len(y1) + y0.var(ddof=1) / len(y0)))
    return np.asarray(th), np.asarray(se)


def _policy_tau_sd(policy, level, spec, decoded, *, flat_tau_sd, tau_sd_min, tau_sd_max):
    if policy == "flat":
        return flat_tau_sd
    if policy == "oracle":
        key = "tau_group" if level == "group" else "tau_member"
        return max(true_tau_by_level(spec)[key], 1e-3)          # HalfNormal scale must be > 0
    if policy == "canonical":
        acc = decoded["group_decode_acc" if level == "group" else "member_decode_acc"]
        k = spec["g"] if level == "group" else spec["b_size"]
        return canonical_tau_prior(acc, k, tau_sd_min=tau_sd_min, tau_sd_max=tau_sd_max)
    raise ValueError(f"unknown policy {policy!r}")


def joint_fidelity(spec: dict, *, level: str = "group", policy: str = "canonical",
                   theta0: float = 0.7, n_reps: int = 20, n_units: int = 4000,
                   depth: int = 7, sigma: float = 0.5, flat_tau_sd: float = 0.5,
                   tau_sd_min: float = 0.05, tau_sd_max: float = 1.0, draws: int = 500,
                   tune: int = 500, chains: int = 2, seed: int = 0,
                   tail_ess_threshold: float = 100.0) -> dict:
    """Operating characteristics of the borrowing prior at one (level, policy, θ₀, spec)
    cell. Returns ``{reject_rate, coverage, mean_tau_sd, mu_true, tau_true, n_flagged,
    n_used}`` — under a null spec (μ=0) ``reject_rate`` is Type-I; under an alt it is
    power. Subgroups = the DECODED labels at ``level``; the prior is set by ``policy``."""
    from causal_bench.estimators.three_level_bhm import fit_three_level_meta, tail_ess_ok

    mu_true = population_effect(spec)
    tau_true = true_tau_by_level(spec)["tau_group" if level == "group" else "tau_member"]
    rejects, covers, taus, n_flagged, n_used = [], [], [], 0, 0
    for r in range(n_reps):
        coh = sample_joint_cohort(spec, n_units, depth, sigma=sigma, seed=seed + r)
        dec = decode_cohort_labels(spec, coh, theta0=theta0, seed=seed + 1000 + r)
        sub = dec["group_decoded" if level == "group" else "member_decoded"]
        n_sub = spec["g"] if level == "group" else spec["b_size"]
        th, se = _subgroup_estimates(coh["Y"], coh["A"], sub, n_sub)
        if len(th) < 2:
            continue
        tau_sd = _policy_tau_sd(policy, level, spec, dec, flat_tau_sd=flat_tau_sd,
                                tau_sd_min=tau_sd_min, tau_sd_max=tau_sd_max)
        fit = fit_three_level_meta(th, se, tau_sd=tau_sd, true_effect=mu_true,
                                   draws=draws, tune=tune, chains=chains, seed=seed + r)
        if not tail_ess_ok(fit, threshold=tail_ess_threshold):
            n_flagged += 1
            continue
        rejects.append(fit["rejects_null"])
        covers.append(fit["covers_truth"])
        taus.append(tau_sd)
        n_used += 1
    return {
        "reject_rate": float(np.mean(rejects)) if rejects else float("nan"),
        "coverage": float(np.mean(covers)) if covers else float("nan"),
        "mean_tau_sd": float(np.mean(taus)) if taus else float("nan"),
        "mu_true": mu_true, "tau_true": float(tau_true),
        "n_flagged": n_flagged, "n_used": n_used,
    }


def make_null_spec(g, b_size, s, m, *, level: str, tau_scale: float, seed: int = 0) -> dict:
    """A spec with population effect μ = 0 at ``level`` but between-subgroup SD τ =
    ``tau_scale`` (heterogeneous null — the case where borrowing threatens Type I). The
    level's effect table is centered (mean 0) and scaled to unit SD then ×``tau_scale``;
    the other level carries no effect. ``tau_scale = 0`` ⇒ the global null (μ=τ=0)."""
    spec = make_joint_hierarchy(g, b_size, s, m, w_group=0.0, w_member=0.0, seed=seed)
    key, w = ("group_effect", "w_group") if level == "group" else ("member_effect", "w_member")
    e = spec[key] - spec[key].mean()
    sd = e.std()
    spec[key] = (e / sd) if sd > 1e-9 else e                    # unit SD, mean 0
    spec[w] = tau_scale                                         # τ_true = tau_scale·1 = tau_scale
    return spec
