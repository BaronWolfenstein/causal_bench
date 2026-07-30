"""exp44 — two-epoch borrowing under prior-data conflict (δ-offset), Part A core.

Historical epoch carries a true effect μ_hist = δ; the current epoch is truly null (μ=0). We
build a prior FROM the historical data (a meta-analytic-predictive / MAP prior on the population
mean μ), borrow it into the current-epoch analysis, and test H0: μ_current = 0. **δ is the
prior-data-conflict magnitude**: δ=0 is exchangeable (no conflict), δ>0 makes the borrowed prior
disagree with the null current data. The read-out is Type-I(δ) per policy — the size a
point-calibration at δ=0 would miss (Qian/EitW; FDA Jan-2026 Bayesian draft).

Policies (μ-borrowing axis): `flat` (no borrowing) · `map` (MAP prior from historical) ·
`robust_map` (Schmidli mixture: MAP + a vague component that the posterior down-weights under
conflict) · `pooled` (naive full pooling — the adversary). See the spec
docs/superpowers/specs/2026-07-29-two-epoch-borrowing-conflict-design.md.

Requires the 3.12 `[bayes]` stack (PyMC/NumPyro); lazy imports so CPU/3.10 installs stay clean.
"""
from __future__ import annotations

import numpy as np


def sim_epoch(n_sub: int, n_per: int, *, mu: float, tau: float, sigma: float,
              rng: np.random.Generator):
    """One epoch as subgroup-level effect summaries: θ_g ~ N(μ, τ²), θ̂_g ~ N(θ_g, se²),
    se = σ/√n_per. Returns (theta_hat, se), the meta-analysis input."""
    theta = rng.normal(mu, tau, n_sub)
    se = np.full(n_sub, sigma / np.sqrt(n_per))
    theta_hat = rng.normal(theta, se)
    return theta_hat, se


def _fit_mu(theta_hat, se, *, mu_prior, tau_sd=0.5, draws=500, tune=500,
            chains=2, seed=0):
    """Random-effects meta on (θ̂, se) with a settable prior on the population mean μ.
    `mu_prior` is one of:
      ("normal", mean, sd)                          — flat / map (a single Normal)
      ("robust", w_vague, vague_sd, map_mean, map_sd) — robust-MAP mixture
    Returns {effect, se, ci_lo, ci_hi, rejects_null, mu_samples}."""
    import pymc as pm

    theta_hat = np.asarray(theta_hat, float)
    se = np.asarray(se, float)
    n_g = len(theta_hat)
    with pm.Model():
        if mu_prior[0] == "normal":
            _, m, s = mu_prior
            mu = pm.Normal("mu", float(m), float(s))
        elif mu_prior[0] == "robust":
            _, w_vague, vague_sd, map_mean, map_sd = mu_prior
            mu = pm.NormalMixture("mu", w=[1.0 - w_vague, w_vague],
                                  mu=[float(map_mean), 0.0],
                                  sigma=[float(map_sd), float(vague_sd)])
        else:
            raise ValueError(f"unknown mu_prior {mu_prior[0]!r}")
        tau = pm.HalfNormal("tau", tau_sd)
        z = pm.Normal("z", 0.0, 1.0, shape=n_g)               # non-centered
        theta = pm.Deterministic("theta", mu + tau * z)
        pm.Normal("obs", theta, se, observed=theta_hat)
        idata = pm.sample(draws=draws, tune=tune, chains=chains, nuts_sampler="numpyro",
                          progressbar=False, random_seed=seed,
                          nuts_sampler_kwargs={"target_accept": 0.9},
                          idata_kwargs={"log_likelihood": False})
    post = np.asarray(idata.posterior["mu"]).ravel()
    lo, hi = np.quantile(post, [0.025, 0.975])
    return {"effect": float(post.mean()), "se": float(post.std()),
            "ci_lo": float(lo), "ci_hi": float(hi),
            "rejects_null": bool(lo > 0 or hi < 0), "mu_samples": post}


def map_prior_from_historical(theta_hat_h, se_h, *, tau_sd=0.5, draws=500, tune=500,
                              chains=2, seed=0):
    """The MAP prior for a NEW exchangeable study's μ = the posterior of the hyper-mean μ from
    the historical meta-analysis, summarized as (mean, sd). (μ is shared by an exchangeable new
    study; its posterior IS the predictive prior for the current epoch's μ.)"""
    fit = _fit_mu(theta_hat_h, se_h, mu_prior=("normal", 0.0, 10.0),
                  tau_sd=tau_sd, draws=draws, tune=tune, chains=chains, seed=seed)
    return fit["effect"], max(fit["se"], 1e-3)


def run_conflict(delta, policy, *, n_sub=8, n_per=40, tau=0.15, sigma=1.0,
                 w_vague=0.5, vague_sd=1.0, tau_sd=0.5, draws=500, tune=500,
                 chains=2, seed=0):
    """One replicate: historical epoch at μ=δ, current epoch at μ=0 (the null). Build the
    borrowed prior per `policy`, fit the current epoch, and return whether it rejects H0: μ=0.
    Under the current null this is a Type-I event; sweeping δ traces Type-I(δ)."""
    rng = np.random.default_rng(seed)
    th_h, se_h = sim_epoch(n_sub, n_per, mu=float(delta), tau=tau, sigma=sigma, rng=rng)
    th_c, se_c = sim_epoch(n_sub, n_per, mu=0.0, tau=tau, sigma=sigma, rng=rng)

    if policy == "flat":
        prior = ("normal", 0.0, vague_sd)
        fit = _fit_mu(th_c, se_c, mu_prior=prior, tau_sd=tau_sd, draws=draws, tune=tune,
                      chains=chains, seed=seed)
    elif policy == "pooled":                                   # naive: pool both epochs
        fit = _fit_mu(np.concatenate([th_h, th_c]), np.concatenate([se_h, se_c]),
                      mu_prior=("normal", 0.0, vague_sd), tau_sd=tau_sd,
                      draws=draws, tune=tune, chains=chains, seed=seed)
    elif policy in ("map", "robust_map"):
        m, s = map_prior_from_historical(th_h, se_h, tau_sd=tau_sd, draws=draws, tune=tune,
                                         chains=chains, seed=seed)
        prior = (("normal", m, s) if policy == "map"
                 else ("robust", w_vague, vague_sd, m, s))
        fit = _fit_mu(th_c, se_c, mu_prior=prior, tau_sd=tau_sd, draws=draws, tune=tune,
                      chains=chains, seed=seed)
    else:
        raise ValueError(f"unknown policy {policy!r}")
    return {"delta": float(delta), "policy": policy, "rejects_null": fit["rejects_null"],
            "effect": fit["effect"], "ci_lo": fit["ci_lo"], "ci_hi": fit["ci_hi"]}


def type_i_curve(deltas, policy, *, n_reps=20, seed=0, **kw):
    """Type-I(δ) for a policy: reject-rate under the current null across a δ sweep."""
    out = []
    for d in deltas:
        rej = [run_conflict(d, policy, seed=seed + r, **kw)["rejects_null"] for r in range(n_reps)]
        out.append({"delta": float(d), "policy": policy, "type_i": float(np.mean(rej)),
                    "n_reps": n_reps})
    return out
