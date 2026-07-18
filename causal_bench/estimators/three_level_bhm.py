"""Three-level BHM + two-vs-three-level OC fidelity harness (#40).

exp19's borrowing OC uses the **conjugate two-level** Normal-Normal robust-MAP
kernel at the inner loop — fast and correct *for the two-level model*. But the
model we would submit is a **three-level** hierarchical BHM (population /
subgroup / patient) whose between-subgroup scale τ is non-conjugate and needs
MCMC. The conjugate kernel that ignores τ **underestimates the SE** → is
anti-conservative → inflates Type-I. #16's `approximation_deviation` bounds the
mixture-vs-MAP *mean* gap analytically; it does **not** tell us whether the
*operating characteristics* (Type-I, power, coverage) of the three-level model
match the two-level approximation. Only fitting both on shared replicates does.

This module provides that direct empirical check:
- `fit_three_level_bhm` — PyMC three-level Normal hierarchy with `τ ~ HalfNormal`,
  sampled via the **NumPyro/JAX** NUTS backend (the A100 Fidelity-path GPU
  sampler; CPU here). Reports R-hat / bulk-ESS / **tail-ESS** (load-bearing —
  Type-I is a tail event on a rare-subpop estimand).
- `fit_two_level_conjugate` — the conjugate kernel that pools subgroups (ignores τ).
- `run_fidelity` — both kernels on the same replicate subsample → OC per kernel
  and the ΔType-I / Δpower / Δcoverage the fidelity argument turns on.

Needs the 3.12 `[bayes]` + `[bayes-gpu]` extras (pymc/arviz + numpyro/jax); lazy
imports so CPU-only / 3.10 installs stay clean. Estimand: the population mean
effect vs the null (0).
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def simulate_three_level(*, n_subgroups: int = 8, n_per: int = 25,
                         true_effect: float = 0.0, tau: float = 0.3,
                         sigma: float = 1.0, seed: int = 0) -> dict:
    """Hierarchical Gaussian data: subgroup means ``~ N(true_effect, τ²)``,
    patient obs ``~ N(subgroup_mean, σ²)``. Returns ``{y, subgroup, true_effect}``
    with ``y`` shape ``(n_subgroups·n_per,)`` and integer ``subgroup`` labels."""
    rng = np.random.default_rng(seed)
    sub_means = rng.normal(true_effect, tau, size=n_subgroups)
    subgroup = np.repeat(np.arange(n_subgroups), n_per)
    y = rng.normal(sub_means[subgroup], sigma)
    return {"y": y, "subgroup": subgroup, "true_effect": float(true_effect)}


def _decision(effect: float, se: float, true_effect: float, z: float = 1.96) -> dict:
    lo, hi = effect - z * se, effect + z * se
    return {"effect": float(effect), "se": float(se), "ci_lo": float(lo),
            "ci_hi": float(hi), "rejects_null": bool(lo > 0 or hi < 0),
            "covers_truth": bool(lo <= true_effect <= hi)}


def fit_two_level_conjugate(data: dict, *, prior_mean: float = 0.0,
                            prior_sd: float = 10.0) -> dict:
    """Two-level conjugate Normal-Normal on the **pooled** outcome (ignores the
    subgroup level → ignores τ). Posterior mean of the population effect with a
    weak registry prior; the SE is the pooled standard error, which *omits*
    between-subgroup variance — the anti-conservative approximation."""
    y = np.asarray(data["y"], float)
    n = len(y)
    lik_mean, lik_var = float(y.mean()), float(y.var(ddof=1) / n)
    pv = prior_sd ** 2
    post_var = 1.0 / (1.0 / pv + 1.0 / lik_var)
    post_mean = post_var * (prior_mean / pv + lik_mean / lik_var)
    return _decision(post_mean, np.sqrt(post_var), data["true_effect"])


def _diagnostics(idata, var: str) -> dict:
    import arviz as az
    r_hat = float(az.rhat(idata)[var])
    bulk = float(az.ess(idata, method="bulk")[var])
    tail = float(az.ess(idata, method="tail")[var])
    return {"r_hat": r_hat, "bulk_ess": bulk, "tail_ess": tail}


def fit_three_level_bhm(data: dict, *, draws: int = 500, tune: int = 500,
                        chains: int = 2, seed: int = 0,
                        sampler: str = "numpyro") -> dict:
    """Three-level BHM in PyMC: population ``mu``, subgroup effects
    ``~ N(mu, τ²)`` with ``τ ~ HalfNormal`` (non-conjugate), patient obs
    ``~ N(subgroup_effect, σ²)``. Sampled via ``nuts_sampler=sampler`` (default
    'numpyro' → JAX; GPU on the box). Returns the population-effect decision plus
    R-hat / bulk-ESS / tail-ESS."""
    import pymc as pm

    y = np.asarray(data["y"], float)
    sub = np.asarray(data["subgroup"], int)
    n_sub = int(sub.max()) + 1
    with pm.Model():
        mu = pm.Normal("mu", 0.0, 10.0)
        tau = pm.HalfNormal("tau", 1.0)
        z = pm.Normal("z", 0.0, 1.0, shape=n_sub)          # non-centered: avoids the funnel
        sub_eff = pm.Deterministic("sub_eff", mu + tau * z)
        sigma = pm.HalfNormal("sigma", 2.0)
        pm.Normal("y", sub_eff[sub], sigma, observed=y)
        idata = pm.sample(draws=draws, tune=tune, chains=chains,
                          nuts_sampler=sampler, progressbar=False,
                          random_seed=seed,
                          nuts_sampler_kwargs={"target_accept": 0.9},
                          idata_kwargs={"log_likelihood": False})
    post = idata.posterior["mu"]
    effect, se = float(post.mean()), float(post.std())
    out = _decision(effect, se, data["true_effect"])
    out.update(_diagnostics(idata, "mu"))
    return out


def fit_three_level_meta(theta_hat, se, *, draws: int = 500, tune: int = 500,
                         chains: int = 2, seed: int = 0, sampler: str = "numpyro",
                         true_effect: float = 0.0, mu_sd: float = 1.0,
                         tau_sd: float = 0.5) -> dict:
    """Three-level model on **subgroup summaries** (the exp19-compatible form): a
    Bayesian random-effects meta-analysis over per-subgroup effect estimates.

        μ ~ N(0, mu_sd²);  τ ~ HalfNormal(tau_sd);
        θ_g ~ N(μ, τ²);    θ̂_g ~ N(θ_g, se_g²)   [se_g fixed, the within-subgroup SE]

    ``μ`` is the population effect. Unlike an ESS-weighted pooled variance (which
    ignores between-subgroup heterogeneity), this **propagates τ into the μ
    posterior**, so the SE is honest when subgroups disagree. Sampled via the
    NumPyro/JAX backend; reports R-hat / bulk-ESS / tail-ESS."""
    import pymc as pm

    theta_hat = np.asarray(theta_hat, float)
    se = np.asarray(se, float)
    n_g = len(theta_hat)
    with pm.Model():
        mu = pm.Normal("mu", 0.0, mu_sd)
        tau = pm.HalfNormal("tau", tau_sd)
        z = pm.Normal("z", 0.0, 1.0, shape=n_g)            # non-centered: avoids the funnel
        theta = pm.Deterministic("theta", mu + tau * z)
        pm.Normal("obs", theta, se, observed=theta_hat)
        idata = pm.sample(draws=draws, tune=tune, chains=chains,
                          nuts_sampler=sampler, progressbar=False,
                          random_seed=seed,
                          nuts_sampler_kwargs={"target_accept": 0.9},
                          idata_kwargs={"log_likelihood": False})
    post = idata.posterior["mu"]
    out = _decision(float(post.mean()), float(post.std()), true_effect)
    out.update(_diagnostics(idata, "mu"))
    return out


def tail_ess_ok(fit: dict, *, threshold: float = 100.0) -> bool:
    """Tail-ESS gate — Type-I is a tail event on a rare-subpop estimand, so a fit
    with tail-ESS below ``threshold`` is flagged (not silently averaged in)."""
    return bool(fit["tail_ess"] >= threshold)


def run_fidelity(*, n_reps: int = 20, tau: float = 0.3, effect_alt: float = 0.5,
                 n_subgroups: int = 10, n_per: int = 30, draws: int = 500,
                 tune: int = 500, chains: int = 2, seed: int = 0,
                 sampler: str = "numpyro", tail_ess_threshold: float = 100.0) -> dict:
    """Fit BOTH kernels on the SAME replicate subsample under null (effect 0) and
    alternative (``effect_alt``); report each kernel's Type-I / power / coverage
    and the deltas. Tail-ESS-flagged three-level fits are counted (and excluded
    from the three-level OC). Returns
    ``{two_level, three_level, delta, n_mcmc_fits, n_tail_ess_flagged}``."""
    acc = {k: {"null_rej": [], "alt_rej": [], "cover": []}
           for k in ("two_level", "three_level")}
    n_fits = 0
    n_flagged = 0
    for r in range(n_reps):
        for scen, eff in (("null", 0.0), ("alt", effect_alt)):
            d = simulate_three_level(n_subgroups=n_subgroups, n_per=n_per,
                                     true_effect=eff, tau=tau, seed=seed + r * 2 + (scen == "alt"))
            two = fit_two_level_conjugate(d)
            three = fit_three_level_bhm(d, draws=draws, tune=tune, chains=chains,
                                        seed=seed + r, sampler=sampler)
            n_fits += 1
            flagged = not tail_ess_ok(three, threshold=tail_ess_threshold)
            n_flagged += int(flagged)
            for k, fit, keep in (("two_level", two, True),
                                 ("three_level", three, not flagged)):
                if not keep:
                    continue
                if scen == "null":
                    acc[k]["null_rej"].append(fit["rejects_null"])
                else:
                    acc[k]["alt_rej"].append(fit["rejects_null"])
                acc[k]["cover"].append(fit["covers_truth"])

    def _oc(a):
        return {"type_i": float(np.mean(a["null_rej"])) if a["null_rej"] else float("nan"),
                "power": float(np.mean(a["alt_rej"])) if a["alt_rej"] else float("nan"),
                "coverage": float(np.mean(a["cover"])) if a["cover"] else float("nan")}

    two_oc, three_oc = _oc(acc["two_level"]), _oc(acc["three_level"])
    delta = {q: two_oc[q] - three_oc[q] for q in ("type_i", "power", "coverage")}
    return {"two_level": two_oc, "three_level": three_oc, "delta": delta,
            "n_mcmc_fits": n_fits, "n_tail_ess_flagged": n_flagged}
