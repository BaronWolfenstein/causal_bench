"""Variance-preserving SDE (discrete DDPM form) with a PLUGGABLE score. An exact
analytic Gaussian score validates the whole pipeline on CPU; the torch score net
drops into `score_fn` unchanged. x_t = sqrt(a_t) x0 + sqrt(1-a_t) eps."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Schedule:
    n_steps: int = 200
    beta_min: float = 1e-4
    beta_max: float = 2e-2

    @property
    def betas(self) -> np.ndarray:
        return np.linspace(self.beta_min, self.beta_max, self.n_steps)

    @property
    def alphas_bar(self) -> np.ndarray:
        return np.cumprod(1.0 - self.betas)


def alpha_bar(sch: Schedule, t: int) -> float:
    return float(sch.alphas_bar[t])


def forward_sample(x0, t, sch, rng):
    a = alpha_bar(sch, t)
    return np.sqrt(a) * x0 + np.sqrt(1 - a) * rng.standard_normal(np.shape(x0))


def gaussian_score(x_t, t, mu, cov, sch):
    """Score of the marginal of N(mu, cov) under VP noising at step t:
    marginal = N(sqrt(a) mu, a cov + (1-a) I); score = -inv(cov_t)(x - mean_t)."""
    a = alpha_bar(sch, t)
    mean_t = np.sqrt(a) * mu
    cov_t = a * cov + (1 - a) * np.eye(cov.shape[0])
    inv = np.linalg.inv(cov_t)
    return -(x_t - mean_t) @ inv.T


def tweedie_denoise(x_t, t, score, sch):
    """E[x0 | x_t] = (x_t + (1-a) score) / sqrt(a)  (Tweedie for VP-SDE)."""
    a = alpha_bar(sch, t)
    return (x_t + (1 - a) * score) / np.sqrt(a)


def ddpm_reverse(x_T, score_fn, sch, rng):
    """Ancestral reverse using the score. Posterior mean at step t:
    (1/sqrt(alpha_t)) (x_t + beta_t * score); add sqrt(beta_t) noise except at t=0."""
    x = np.array(x_T, float)
    betas = sch.betas
    for t in range(sch.n_steps - 1, -1, -1):
        alpha_t = 1.0 - betas[t]
        s = score_fn(x, t)
        mean = (x + betas[t] * s) / np.sqrt(alpha_t)
        if t > 0:
            x = mean + np.sqrt(betas[t]) * rng.standard_normal(x.shape)
        else:
            x = mean
    return x
