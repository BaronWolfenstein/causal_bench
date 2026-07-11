"""Schedule-targeted guidance — item 5 of the #122 roadmap, analytic-score CPU
prototype. Synthetic; the real-embedding version swaps the analytic score for the
trained score net (gated on the box), no other change.

Idea: the diffusion schedule is a MAP of *where decisions happen* — the coarse
class is decided near the high-noise transition (``t_coarse*``), fine detail only
below the low-noise one (``t_fine*``). So to synthesize a chosen (rare) coarse
class WITHOUT collapsing its fine diversity: apply class-conditional CFG guidance
in the **high-noise phase** (fix the class while it is being decided) and **release
it below ``t_fine*``** (let the fine subclass resample freely). Uniform (always-on)
guidance over-commits and collapses fine diversity; no guidance rarely reaches a
rare class at all. Schedule-targeting gets both.

The score of a Gaussian mixture (and its class-restricted version) is analytic, so
the whole thing validates on CPU with no trained net. Reuses `vpsde` (schedule +
ddpm_reverse) and the same CFG structure as `guidance.py`. numpy only.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from causal_bench.generative.vpsde import Schedule, alpha_bar, ddpm_reverse


def mixture_score(x: np.ndarray, t: int, means: np.ndarray, sch: Schedule,
                  sigma_within: float, weights: Optional[np.ndarray] = None) -> np.ndarray:
    """Analytic score ∇ log pₜ(x) of a VP-noised isotropic Gaussian mixture with
    component means ``means`` (K, d) and within-class σ. At noise ``t`` each
    component is ``N(√ā·μ_k, vₜ I)`` with ``vₜ = ā·σ² + (1−ā)``; the score is the
    responsibility-weighted pull toward the scaled means."""
    means = np.asarray(means, float)
    a = alpha_bar(sch, t)
    v = a * sigma_within ** 2 + (1.0 - a)
    scaled = np.sqrt(a) * means                          # (K, d)
    d2 = ((x[:, None, :] - scaled[None]) ** 2).sum(-1)   # (n, K)
    logr = -d2 / (2.0 * v)
    if weights is not None:
        logr = logr + np.log(np.asarray(weights))[None]
    logr -= logr.max(1, keepdims=True)
    r = np.exp(logr)
    r /= r.sum(1, keepdims=True)
    return (r[:, :, None] * (scaled[None] - x[:, None, :])).sum(1) / v


def make_schedule_guided_score(all_means: np.ndarray, target_means: np.ndarray,
                               sch: Schedule, sigma_within: float, *,
                               w_max: float, t_gate: int) -> Callable:
    """CFG score guiding toward the ``target_means`` (a class's fine subclasses),
    with guidance strength ``w_max`` applied only at steps ``t ≥ t_gate`` (the
    high-noise phase) and **0 below** — the schedule-targeting. ``t_gate`` is the
    step index of ``t_fine*`` (guide the class, free the detail)."""
    def score_fn(x, t):
        s_all = mixture_score(x, t, all_means, sch, sigma_within)
        if t < t_gate or w_max == 0.0:
            return s_all
        s_tgt = mixture_score(x, t, target_means, sch, sigma_within)
        return s_all + w_max * (s_tgt - s_all)
    return score_fn


def generate(score_fn: Callable, n: int, dim: int, sch: Schedule,
             rng: np.random.Generator) -> np.ndarray:
    """DDPM ancestral sampling from noise with the (possibly guided) score."""
    return ddpm_reverse(rng.standard_normal((n, dim)), score_fn, sch, rng)


def coarse_hit_rate(samples: np.ndarray, coarse_means: np.ndarray,
                    target_coarse: int) -> float:
    """Fraction of samples whose nearest coarse-class mean is ``target_coarse``."""
    d2 = ((samples[:, None, :] - np.asarray(coarse_means)[None]) ** 2).sum(-1)
    return float((d2.argmin(1) == target_coarse).mean())


def fine_diversity(samples: np.ndarray, target_fine_means: np.ndarray) -> float:
    """Normalized entropy of the samples' assignment across the target class's fine
    subclasses (1.0 = uniformly spread over all subclasses, 0.0 = collapsed to one).
    Guards against schedule-targeting silently collapsing the rare mode."""
    fm = np.asarray(target_fine_means)
    d2 = ((samples[:, None, :] - fm[None]) ** 2).sum(-1)
    assign = d2.argmin(1)
    counts = np.bincount(assign, minlength=len(fm)).astype(float)
    p = counts / counts.sum()
    nz = p[p > 0]
    return float(-(nz * np.log(nz)).sum() / np.log(len(fm))) if len(fm) > 1 else 1.0
