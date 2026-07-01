"""Bayesian cumulative-link mixed model (CLMM) estimator — issue #27.

Model
-----
    support ~ 0 + f(W) + A + (1 | site),  family = cumulative()

The response `support` must be an ordered integer or categorical column.
The intercepts are the threshold parameters (one per category boundary);
the treatment coefficient `A` is the **marginal cumulative log-OR** on the
latent scale (proportional-odds parameterisation).

Backend decision: bambi (brms-like formula API on PyMC)
--------------------------------------------------------
We chose bambi-native over an rpy2/brms bridge for three reasons:

1. R-free: bambi wraps PyMC directly, no R, rpy2, or subprocess overhead.
   This keeps the `bayes` optional-dep stack pure-Python and avoids the
   R-version pinning that already complicates the concrete bridge.

2. Formula parity: bambi's formula mini-language (`y ~ x + (1|g)`) mirrors
   brms / lme4 syntax, so the statistical model is legible to R-trained
   readers without a translation layer.

3. Full posterior access: bambi/PyMC returns ArviZ InferenceData, giving
   us the raw posterior samples for the treatment coefficient.  We derive
   the posterior SD as the reported SE and the 95% HDI as the credible
   interval — no bootstrap, no delta method.

The downside vs. brms is that bambi's `family="cumulative"` is younger
and has fewer built-in link options (logit only in versions < 0.15).  This
is acceptable for the benchmark role: we need proportional-odds logit, and
violation of that assumption is exactly what the benchmark is designed to
detect.

Usage
-----
    from causal_bench.estimators.clmm_ordinal import CLMMOrdinalEstimator
    est = CLMMOrdinalEstimator(draws=1000, tune=1000, chains=2)
    results = est.estimate(df)  # df must have: support, A, site, covariates

    # In SimResult diagnostics (run_simulation / coverage machinery):
    #   result.point_estimate  → posterior mean of cumulative log-OR for A
    #   result.standard_error  → posterior std dev of A coefficient
    #   result.ci_lower / ci_upper → 95% highest-density interval (HDI)
    #   result.ess              → posterior ESS for A
    #   result.convergence_info → {"r_hat": <float>, "n_eff": <float>}

References
----------
Bürkner & Vuorre (2019). Ordinal regression models in psychological research.
  Advances in Methods and Practices in Psychological Science 2(1):77-101.
Kurz (2023). Doing Bayesian Data Analysis in brms and the tidyverse (online book).
Bambi docs: https://bambinos.github.io/bambi/
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

from causal_bench.estimators.base import BaseEstimator
from causal_bench.metrics import EstimatorResult

# ---------------------------------------------------------------------------
# Optional-dependency sentinel
# ---------------------------------------------------------------------------

try:
    import bambi as bmb  # noqa: F401
    import arviz as az   # noqa: F401
    _BAMBI_AVAILABLE = True
except ImportError:
    _BAMBI_AVAILABLE = False


# ---------------------------------------------------------------------------
# Estimator
# ---------------------------------------------------------------------------

class CLMMOrdinalEstimator(BaseEstimator):
    """Bayesian CLMM with partial pooling over sites (bambi/PyMC backend).

    Estimates a **marginal cumulative log-OR** for binary treatment `A`
    under a proportional-odds cumulative-link mixed model with site random
    intercepts.  Returns the posterior mean, posterior SD (used as SE), and
    95% highest-density credible interval — fed directly into the coverage /
    SE-calibration diagnostics without bootstrapping.

    Parameters
    ----------
    outcome_col : str
        Ordered ordinal response column (integer or ordered Categorical).
    treatment_col : str
        Binary treatment indicator column (default "A").
    covariate_cols : list of str
        Covariate columns entered as linear fixed effects (default W1-W4).
    site_col : str
        Site/cluster column for the random intercept (default "site").
    draws : int
        Number of posterior draws per chain (default 1000).
    tune : int
        Number of tuning/warmup steps per chain (default 1000).
    chains : int
        Number of MCMC chains (default 2).
    target_accept : float
        NUTS target acceptance rate (default 0.9).
    progressbar : bool
        Show sampler progress bar (default False — quiet in tests/batch).
    random_seed : int or None
        Seed for reproducibility (default None).
    hdi_prob : float
        Probability mass for the highest-density credible interval (default 0.95).
    **sampler_kwargs
        Any extra keyword arguments forwarded verbatim to `model.fit()`.
    """

    name = "clmm_ordinal"

    def __init__(
        self,
        outcome_col: str = "support",
        treatment_col: str = "A",
        covariate_cols: list[str] | None = None,
        site_col: str = "site",
        draws: int = 1000,
        tune: int = 1000,
        chains: int = 2,
        target_accept: float = 0.9,
        progressbar: bool = False,
        random_seed: int | None = None,
        hdi_prob: float = 0.95,
        **sampler_kwargs: Any,
    ):
        self._outcome_col    = outcome_col
        self._treatment_col  = treatment_col
        self._covariate_cols = covariate_cols or ["W1", "W2", "W3", "W4"]
        self._site_col       = site_col
        self._draws          = draws
        self._tune           = tune
        self._chains         = chains
        self._target_accept  = target_accept
        self._progressbar    = progressbar
        self._random_seed    = random_seed
        self._hdi_prob       = hdi_prob
        self._sampler_kwargs = sampler_kwargs

    # ------------------------------------------------------------------
    # BaseEstimator interface
    # ------------------------------------------------------------------

    def estimate(
        self,
        df: pd.DataFrame,
        horizon: float = 1.0,
        estimand: str = "cumulative_log_OR",
    ) -> list[EstimatorResult]:
        """Fit the Bayesian CLMM and return posterior summaries.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain columns: `outcome_col`, `treatment_col`, `site_col`,
            and any columns named in `covariate_cols` that are present.
            Missing covariate columns are silently dropped from the formula.
        horizon : float
            Unused for CLMM (present for BaseEstimator API compatibility).
        estimand : str
            Label attached to the returned EstimatorResult (default
            "cumulative_log_OR").

        Returns
        -------
        list of EstimatorResult
            Typically a single element for the treatment log-OR.
            Returns [] with a warning if bambi is unavailable.
        """
        if not _BAMBI_AVAILABLE:
            warnings.warn(
                f"{self.name}: bambi/PyMC not installed — skipping. "
                "Install with: pip install 'causal_bench[bayes]'  "
                "(or: pip install bambi)",
                stacklevel=2,
            )
            return []

        import bambi as bmb
        import arviz as az

        # ---- validate required columns --------------------------------
        required = [self._outcome_col, self._treatment_col]
        missing_req = [c for c in required if c not in df.columns]
        if missing_req:
            warnings.warn(
                f"{self.name}: required columns {missing_req} missing — skipping",
                stacklevel=2,
            )
            return []

        # ---- prepare data -------------------------------------------
        # Keep only rows with no NaN in outcome or treatment
        cols_needed = [self._outcome_col, self._treatment_col]
        present_covars = [c for c in self._covariate_cols if c in df.columns]
        has_site = self._site_col in df.columns
        if has_site:
            cols_needed.append(self._site_col)
        cols_needed.extend(present_covars)

        data = df[cols_needed].copy().dropna()
        if len(data) < 20:
            warnings.warn(
                f"{self.name}: fewer than 20 complete rows after dropna — skipping",
                stacklevel=2,
            )
            return []

        # Convert outcome to ordered Categorical so bambi sees the right dtype
        y_vals = sorted(data[self._outcome_col].unique())
        data[self._outcome_col] = pd.Categorical(
            data[self._outcome_col],
            categories=y_vals,
            ordered=True,
        )
        if len(y_vals) < 2:
            warnings.warn(
                f"{self.name}: outcome column has fewer than 2 unique levels — skipping",
                stacklevel=2,
            )
            return []

        # ---- build formula ------------------------------------------
        # Intercept is suppressed (0 +) because the cumulative family uses
        # threshold parameters as its intercept-equivalent; bambi also warns
        # when an explicit intercept is requested for ordinal families.
        fixed_terms = " + ".join([self._treatment_col] + present_covars)
        if has_site:
            formula = f"{self._outcome_col} ~ 0 + {fixed_terms} + (1 | {self._site_col})"
        else:
            formula = f"{self._outcome_col} ~ 0 + {fixed_terms}"

        # ---- fit -------------------------------------------------------
        try:
            model = bmb.Model(formula, data, family="cumulative")
            idata = model.fit(
                draws=self._draws,
                tune=self._tune,
                chains=self._chains,
                target_accept=self._target_accept,
                progressbar=self._progressbar,
                random_seed=self._random_seed,
                **self._sampler_kwargs,
            )
        except Exception as exc:
            warnings.warn(
                f"{self.name}: MCMC sampling failed — {exc}", stacklevel=2
            )
            return []

        # ---- extract treatment posterior ----------------------------
        try:
            post = idata.posterior[self._treatment_col]
            # post has dims (chain, draw) — flatten to 1D
            samples = np.asarray(post.values).reshape(-1)
        except KeyError:
            warnings.warn(
                f"{self.name}: treatment variable '{self._treatment_col}' "
                "not found in posterior — check formula / column name",
                stacklevel=2,
            )
            return []

        if len(samples) == 0 or not np.all(np.isfinite(samples)):
            warnings.warn(
                f"{self.name}: posterior samples for {self._treatment_col} "
                "are empty or non-finite — skipping",
                stacklevel=2,
            )
            return []

        point_est  = float(np.mean(samples))
        post_sd    = float(np.std(samples, ddof=1))
        post_sd    = max(post_sd, 1e-8)          # safety floor; never triggers in practice

        # 95% HDI as the credible interval
        hdi_arr = az.hdi(samples, hdi_prob=self._hdi_prob)
        ci_lower = float(hdi_arr[0])
        ci_upper = float(hdi_arr[1])

        # Enforce CI brackets point estimate (floating-point edge)
        ci_lower = min(ci_lower, point_est)
        ci_upper = max(ci_upper, point_est)

        # ---- convergence diagnostics --------------------------------
        try:
            summary = az.summary(idata, var_names=[self._treatment_col], round_to=4)
            r_hat = float(summary["r_hat"].iloc[0]) if "r_hat" in summary.columns else float("nan")
            n_eff = float(summary["ess_bulk"].iloc[0]) if "ess_bulk" in summary.columns else float(len(samples))
        except Exception:
            r_hat = float("nan")
            n_eff = float(len(samples))

        return [
            EstimatorResult(
                name=self.name,
                estimand=estimand,
                point_estimate=point_est,
                standard_error=post_sd,
                ci_lower=ci_lower,
                ci_upper=ci_upper,
                ess=n_eff,
                convergence_info={"r_hat": r_hat, "n_eff": n_eff},
            )
        ]
