"""theta <-> VP-SDE time mapping (#137, non-gated half). Connects the RHM grammar's
corruption parameter theta (tree_reconstruction.py / rhm_grammar.py) to a diffusion
schedule time t, on two channels that need NOT agree:

1. **Raw discrete tokens** — closed form. The D3PM uniform kernel's retention
   probability at schedule time t is ``alpha_bar(t) + (1-alpha_bar(t))/v``. The
   RHM grammar's corruption channel (keep w.p. theta, else replace with a uniform
   draw over v) has retention ``theta + (1-theta)/v`` — the SAME expression with
   ``theta = alpha_bar(t)``. So on the token channel, theta simply IS alpha_bar(t)
   under this parameterization; "calibration" is just inverting the schedule.

2. **Frozen-encoder embeddings** — no token channel exists; the "grammar" is
   implicit in the manifold. Here we measure the class-overlap order parameter
   directly on VP-SDE-noised embeddings, using the SAME normalization as the
   grammar's overlap ((K*accuracy - 1)/(K-1), 0 at chance / 1 at perfect), so the
   two channels are on a common footing without an explicit theta(t) curve-fit.
   Requires known/synthetic labels (real-embedding calibration is gated on-box;
   this file's embedding_transition_scan runs on synthetic hierarchical Gaussians
   we already control — see hierarchy_probe.sample_hierarchical_gaussian).

numpy only.
"""
from __future__ import annotations

import numpy as np

from causal_bench.generative.vpsde import Schedule, alpha_bar


def theta_to_vpsde_time(theta: float, sch: Schedule) -> int:
    """Token channel: theta IS alpha_bar(t) under the shared-retention-formula
    parameterization (see module docstring); inverting is inverting the
    (monotonically decreasing) alpha_bar schedule. Returns the nearest schedule
    step t (int index into ``sch.alphas_bar``)."""
    ab = sch.alphas_bar
    return int(np.argmin(np.abs(ab - theta)))


def flip_rate(theta: float, v: int) -> float:
    """The net token flip-away probability, D3PM notation: ``(1-theta)*(v-1)/v``
    (the uniform-channel corruption expressed as a per-token error rate)."""
    return (1.0 - theta) * (v - 1) / v


def embedding_channel_overlap(Xt: np.ndarray, means: np.ndarray, labels: np.ndarray) -> float:
    """Class-overlap order parameter on embeddings, normalized like the grammar's:
    ``(K * recovery_accuracy - 1) / (K - 1)`` via nearest-(scaled)-mean assignment
    (0 at chance, 1 at perfect recovery; K = number of classes)."""
    d2 = ((Xt[:, None, :] - means[None, :, :]) ** 2).sum(-1)
    acc = float((d2.argmin(1) == labels).mean())
    k = len(means)
    return (k * acc - 1.0) / (k - 1)


def embedding_transition_scan(X: np.ndarray, labels: np.ndarray, means: np.ndarray, *,
                              sch: Schedule | None = None, n_grid: int = 25,
                              rng: np.random.Generator | None = None) -> dict:
    """Sweep VP-SDE noise on labeled (synthetic) embeddings; returns overlap vs
    schedule-time fraction on the SAME order-parameter footing as
    ``rhm_transition_scan`` (grammar), so the two channels are comparable without
    a discrete grammar on the embedding side. ``t_star`` (a schedule fraction) is
    located via the susceptibility peak (overlap falls as noise/t grows, mirroring
    the grammar's overlap rising as theta grows). Returns ``{t_frac, overlap,
    susceptibility, t_star}``."""
    sch = sch or Schedule(n_steps=200)
    rng = rng or np.random.default_rng(0)
    X = np.asarray(X, float)
    steps = np.linspace(1, sch.n_steps - 1, n_grid).astype(int)
    t_frac = steps / sch.n_steps
    ov = np.array([
        embedding_channel_overlap(
            np.sqrt(alpha_bar(sch, int(t))) * X
            + np.sqrt(1.0 - alpha_bar(sch, int(t))) * rng.normal(size=X.shape),
            np.sqrt(alpha_bar(sch, int(t))) * means, labels)
        for t in steps
    ])
    susc = -np.diff(ov) / np.diff(t_frac)              # overlap FALLS as t grows
    mid = 0.5 * (t_frac[:-1] + t_frac[1:])
    return {"t_frac": t_frac, "overlap": ov, "susceptibility": susc,
            "t_star": float(mid[int(np.argmax(susc))])}


def transition_report(theta_c: float, v: int, sch: Schedule, embedding_scan: dict) -> dict:
    """Side-by-side transition location on both channels: the token-channel t*
    (inverting the grammar's ``theta_c``, e.g. from ``rhm_fss_collapse``/#136)
    vs. the embedding-channel ``t_star`` (from ``embedding_transition_scan``), and
    the gap between them. The gap is a REPORTED FINDING, not expected to be zero —
    the two channels' transitions need not coincide (see module docstring)."""
    token_t_frac = theta_to_vpsde_time(theta_c, sch) / sch.n_steps
    return {"token_t_frac": token_t_frac, "embedding_t_star": embedding_scan["t_star"],
            "gap": abs(token_t_frac - embedding_scan["t_star"])}
