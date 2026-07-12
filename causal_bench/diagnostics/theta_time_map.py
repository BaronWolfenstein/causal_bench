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

**Two order parameters, deliberately different in kind.** ``embedding_channel_
overlap`` is an ORACLE readout — it assigns via the TRUE (known) class means, so
it measures what's geometrically present in the space. ``linear_probe_overlap``
is an ESTIMATED readout — it fits a shared-covariance linear discriminant (LDA)
from a labeled reference SAMPLE (as one would have to on real data, where the
true means aren't known), then reads the fitted posterior of the true class. On
this synthetic isotropic-Gaussian mixture the two should agree up to
finite-sample noise; on real embeddings with anisotropic per-class covariance
they could diverge — that divergence is itself the diagnostic (see the
adapter-vs-probe discussion on #137: a probe/oracle mismatch is evidence the raw
embedding geometry needs reshaping before diffusion).

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


def _fit_shared_cov_lda(X: np.ndarray, labels: np.ndarray, *, reg: float = 1e-3) -> dict:
    """Fit a shared-covariance linear discriminant from a labeled SAMPLE (not the
    true generative parameters): per-class empirical means, a pooled (within-class)
    covariance, and empirical priors. ``reg*I`` is added before inversion for
    numerical stability. Returns ``{classes, means, cov_inv, log_priors}``."""
    classes = np.unique(labels)
    d = X.shape[1]
    means = np.array([X[labels == c].mean(0) for c in classes])
    n = len(X)
    pooled = np.zeros((d, d))
    # np.errstate: macOS Accelerate BLAS emits spurious divide/overflow warnings on
    # small-matrix matmul with perfectly finite inputs/outputs (verified — not a
    # real numerical issue here).
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        for c, mu in zip(classes, means):
            Xc = X[labels == c] - mu
            pooled += Xc.T @ Xc
    pooled = pooled / n + reg * np.eye(d)
    cov_inv = np.linalg.inv(pooled)
    log_priors = np.array([np.log((labels == c).mean()) for c in classes])
    return {"classes": classes, "means": means, "cov_inv": cov_inv, "log_priors": log_priors}


def _lda_posterior(Xt: np.ndarray, params: dict) -> np.ndarray:
    """LDA discriminant posterior: ``score_k(x) = x^T Sigma^-1 mu_k - 0.5 mu_k^T
    Sigma^-1 mu_k + log(prior_k)``, softmax-normalized over classes. Returns
    ``(n, K)`` posteriors, columns aligned to ``params['classes']``."""
    means, cov_inv, log_priors = params["means"], params["cov_inv"], params["log_priors"]
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        linear = Xt @ cov_inv @ means.T
        bias = -0.5 * np.sum((means @ cov_inv) * means, axis=1) + log_priors
    scores = linear + bias
    scores -= scores.max(1, keepdims=True)
    p = np.exp(scores)
    return p / p.sum(1, keepdims=True)


def linear_probe_overlap(Xt_eval: np.ndarray, labels_eval: np.ndarray, params: dict) -> float:
    """Class-overlap order parameter via the FITTED linear probe's posterior of
    the true class (soft, not hard accuracy — matching the grammar's own
    ``mean_p`` convention), normalized like ``embedding_channel_overlap``:
    ``(K * mean_posterior_true - 1) / (K - 1)``."""
    post = _lda_posterior(Xt_eval, params)
    class_idx = np.searchsorted(params["classes"], labels_eval)
    mean_p = float(post[np.arange(len(labels_eval)), class_idx].mean())
    k = len(params["classes"])
    return (k * mean_p - 1.0) / (k - 1)


def linear_probe_transition_scan(X: np.ndarray, labels: np.ndarray, *,
                                 sch: Schedule | None = None, n_grid: int = 25,
                                 rng: np.random.Generator | None = None,
                                 reg: float = 1e-3, ref_frac: float = 0.5) -> dict:
    """Sweep VP-SDE noise, splitting ``X`` once into a reference set (fits the LDA
    probe) and an eval set (scored by it) — at EACH noise level both are noised
    independently and the probe is refit on the noised reference, mirroring how
    one would calibrate a probe on real noised data (no access to true means).
    Same ``{t_frac, overlap, susceptibility, t_star}`` shape as
    ``embedding_transition_scan``, for direct comparison."""
    sch = sch or Schedule(n_steps=200)
    rng = rng or np.random.default_rng(0)
    X = np.asarray(X, float)
    n = len(X)
    perm = rng.permutation(n)
    n_ref = int(ref_frac * n)
    ref_idx, eval_idx = perm[:n_ref], perm[n_ref:]
    steps = np.linspace(1, sch.n_steps - 1, n_grid).astype(int)
    t_frac = steps / sch.n_steps
    ov = []
    for t in steps:
        a = alpha_bar(sch, int(t))
        noise_ref = np.sqrt(a) * X[ref_idx] + np.sqrt(1 - a) * rng.normal(size=X[ref_idx].shape)
        noise_eval = np.sqrt(a) * X[eval_idx] + np.sqrt(1 - a) * rng.normal(size=X[eval_idx].shape)
        params = _fit_shared_cov_lda(noise_ref, labels[ref_idx], reg=reg)
        ov.append(linear_probe_overlap(noise_eval, labels[eval_idx], params))
    ov = np.asarray(ov)
    susc = -np.diff(ov) / np.diff(t_frac)
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
