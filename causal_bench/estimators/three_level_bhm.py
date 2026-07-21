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
                        sampler: str = "numpyro", chain_method: str = "sequential") -> dict:
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
                          chain_method=chain_method,
                          nuts_sampler_kwargs={"target_accept": 0.9},
                          idata_kwargs={"log_likelihood": False})
    post = idata.posterior["mu"]
    effect, se = float(post.mean()), float(post.std())
    out = _decision(effect, se, data["true_effect"])
    out.update(_diagnostics(idata, "mu"))
    return out


def _build_tau(pm, tau_prior, tau_sd):
    """Construct the between-subgroup SD prior τ. ``tau_prior`` is a tagged pair
    ``(family, params)`` — ``("halfnormal", (scale,))`` or ``("lognormal", (mu_log,
    sigma_log))`` (the van Zwet empirical policy). When ``tau_prior`` is None the legacy
    ``HalfNormal(tau_sd)`` is used, so callers passing only ``tau_sd`` are unchanged."""
    if tau_prior is None:
        return pm.HalfNormal("tau", tau_sd)
    family, params = tau_prior
    if family == "halfnormal":
        return pm.HalfNormal("tau", params[0])
    if family == "lognormal":
        mu_log, sigma_log = params
        return pm.LogNormal("tau", mu=mu_log, sigma=sigma_log)
    raise ValueError(f"unknown tau_prior family {family!r}")


def fit_three_level_meta(theta_hat, se, *, draws: int = 500, tune: int = 500,
                         chains: int = 2, seed: int = 0, sampler: str = "numpyro",
                         chain_method: str = "sequential",
                         true_effect: float = 0.0, mu_sd: float = 1.0,
                         tau_sd: float = 0.5, tau_prior: tuple | None = None,
                         return_theta: bool = False) -> dict:
    """Three-level model on **subgroup summaries** (the exp19-compatible form): a
    Bayesian random-effects meta-analysis over per-subgroup effect estimates.

        μ ~ N(0, mu_sd²);  τ ~ <tau_prior>;
        θ_g ~ N(μ, τ²);    θ̂_g ~ N(θ_g, se_g²)   [se_g fixed, the within-subgroup SE]

    ``μ`` is the population effect. Unlike an ESS-weighted pooled variance (which
    ignores between-subgroup heterogeneity), this **propagates τ into the μ
    posterior**, so the SE is honest when subgroups disagree. Sampled via the
    NumPyro/JAX backend; reports R-hat / bulk-ESS / tail-ESS.

    The τ prior is ``tau_prior=(family, params)`` — ``("halfnormal", (scale,))`` or
    ``("lognormal", (mu_log, sigma_log))``. Omitting it falls back to
    ``HalfNormal(tau_sd)`` (legacy). The lognormal family is what the empirical
    (van Zwet) borrowing policy uses (see ``joint_fidelity.empirical_tau_prior``)."""
    import pymc as pm

    theta_hat = np.asarray(theta_hat, float)
    se = np.asarray(se, float)
    n_g = len(theta_hat)
    with pm.Model():
        mu = pm.Normal("mu", 0.0, mu_sd)
        tau = _build_tau(pm, tau_prior, tau_sd)
        z = pm.Normal("z", 0.0, 1.0, shape=n_g)            # non-centered: avoids the funnel
        theta = pm.Deterministic("theta", mu + tau * z)
        pm.Normal("obs", theta, se, observed=theta_hat)
        idata = pm.sample(draws=draws, tune=tune, chains=chains,
                          nuts_sampler=sampler, progressbar=False,
                          random_seed=seed,
                          chain_method=chain_method,
                          nuts_sampler_kwargs={"target_accept": 0.9},
                          idata_kwargs={"log_likelihood": False})
    post = idata.posterior["mu"]
    out = _decision(float(post.mean()), float(post.std()), true_effect)
    out.update(_diagnostics(idata, "mu"))
    if return_theta:
        # per-subgroup posteriors θ_g = μ + τ·z_g — needed for per-subgroup (partial-null)
        # size: strong borrowing drags a truly-null subgroup toward non-null siblings.
        th = idata.posterior["theta"]
        out["theta_g_mean"] = th.mean(("chain", "draw")).values
        out["theta_g_sd"] = th.std(("chain", "draw")).values
        lo = out["theta_g_mean"] - 1.96 * out["theta_g_sd"]
        hi = out["theta_g_mean"] + 1.96 * out["theta_g_sd"]
        out["theta_g_rejects"] = (lo > 0) | (hi < 0)          # per-subgroup reject-null
    return out


def tail_ess_ok(fit: dict, *, threshold: float = 100.0) -> bool:
    """Tail-ESS gate — Type-I is a tail event on a rare-subpop estimand, so a fit
    with tail-ESS below ``threshold`` is flagged (not silently averaged in)."""
    return bool(fit["tail_ess"] >= threshold)


# ── compile-once direct-NumPyro path (the OC-loop fast path) ─────────────────────
# pymc rebuilds the model graph and re-JITs every fit_three_level_meta call (~22s
# steady, dominated by recompilation). This path builds ONE NumPyro MCMC per
# (n_pad, draws, tune, chains) and reuses its compiled sampler across reps: only
# the FIRST fit pays compilation, the rest are warmup+sampling. Variable decoded
# subgroup counts are padded to a fixed n_pad with an uninformative se, so shape
# stays constant (no recompile) and padded subgroups don't move the mu/tau posterior.
_NUMPYRO_MCMC_CACHE: dict = {}


def _meta_model_numpyro(theta_hat, se, mu_sd, tau_sd):
    import numpyro
    import numpyro.distributions as dist
    n = theta_hat.shape[0]
    mu = numpyro.sample("mu", dist.Normal(0.0, mu_sd))
    tau = numpyro.sample("tau", dist.HalfNormal(tau_sd))
    z = numpyro.sample("z", dist.Normal(0.0, 1.0).expand([n]))   # non-centered
    theta = mu + tau * z
    numpyro.sample("obs", dist.Normal(theta, se), obs=theta_hat)


def _meta_model_numpyro_lognormal(theta_hat, se, mu_sd, tau_mu_log, tau_sigma_log):
    """Empirical (van Zwet) variant: τ ~ LogNormal. A SEPARATE model fn (not a branch
    inside ``_meta_model_numpyro``) so JAX's compile cache keys the two τ families to two
    entries — each compiles once; the scalar params still change without recompiling."""
    import numpyro
    import numpyro.distributions as dist
    n = theta_hat.shape[0]
    mu = numpyro.sample("mu", dist.Normal(0.0, mu_sd))
    tau = numpyro.sample("tau", dist.LogNormal(tau_mu_log, tau_sigma_log))
    z = numpyro.sample("z", dist.Normal(0.0, 1.0).expand([n]))   # non-centered
    theta = mu + tau * z
    numpyro.sample("obs", dist.Normal(theta, se), obs=theta_hat)


def fit_three_level_meta_fast(theta_hat, se, *, draws: int = 500, tune: int = 500,
                              chains: int = 2, seed: int = 0, tau_sd: float = 0.5,
                              tau_prior: tuple | None = None,
                              mu_sd: float = 1.0, true_effect: float = 0.0,
                              n_pad: int | None = None, chain_method: str = "vectorized",
                              return_theta: bool = False) -> dict:
    """Statistically equivalent to `fit_three_level_meta` but compiles once and
    reuses across reps (validated against it). `n_pad` fixes the subgroup dimension
    (default = len(theta_hat)); pass the level's max (e.g. spec['g']) so all reps
    share one compiled sampler. mu_sd/tau params are passed as arrays so changing the
    prior between reps does NOT recompile.

    ``tau_prior=(family, params)`` selects the τ prior — ``("halfnormal", (scale,))`` or
    ``("lognormal", (mu_log, sigma_log))`` (the empirical policy). Each family has its own
    compiled model (one compile apiece), so a --fast sweep runs all policies on the same
    compile-once sampler. Omitting ``tau_prior`` falls back to ``HalfNormal(tau_sd)``."""
    import jax
    import jax.numpy as jnp
    import arviz as az
    from numpyro.infer import MCMC, NUTS
    th = np.asarray(theta_hat, float)
    s = np.asarray(se, float)
    n_g = len(th)
    n_pad = int(n_pad) if n_pad else n_g
    if n_pad < n_g:
        raise ValueError(f"n_pad {n_pad} < n_g {n_g}")
    thp = np.zeros(n_pad, float); thp[:n_g] = th
    sp = np.full(n_pad, 1e6, float); sp[:n_g] = s              # padded rows uninformative

    # Resolve the τ prior to (model_fn, extra scalar args). halfnormal and lognormal are
    # distinct model fns so JAX compiles each once (see _meta_model_numpyro_lognormal).
    family, params = tau_prior if tau_prior is not None else ("halfnormal", (tau_sd,))
    if family == "halfnormal":
        model_fn = _meta_model_numpyro
        tau_args = (jnp.asarray(float(params[0])),)
    elif family == "lognormal":
        model_fn = _meta_model_numpyro_lognormal
        tau_args = (jnp.asarray(float(params[0])), jnp.asarray(float(params[1])))
    else:
        raise ValueError(f"unknown tau_prior family {family!r}")

    # Fresh MCMC each call (reusing the object breaks on re-run with vectorized
    # chains). jax's global compile cache keys on the model fn + shapes, so with a
    # fixed n_pad every rep hits the cache — only the first pays compilation.
    mcmc = MCMC(NUTS(model_fn, target_accept_prob=0.9),
                num_warmup=tune, num_samples=draws, num_chains=chains,
                chain_method=chain_method, progress_bar=False)
    mcmc.run(jax.random.PRNGKey(int(seed)), jnp.asarray(thp), jnp.asarray(sp),
             jnp.asarray(float(mu_sd)), *tau_args)
    mu_cd = np.asarray(mcmc.get_samples(group_by_chain=True)["mu"])   # (chains, draws)
    out = _decision(float(mu_cd.mean()), float(mu_cd.std()), true_effect)
    out.update({"r_hat": float(az.rhat(mu_cd)),                       # (chain, draw) array
                "bulk_ess": float(az.ess(mu_cd, method="bulk")),
                "tail_ess": float(az.ess(mu_cd, method="tail", prob=(0.05, 0.95)))})
    return out


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
