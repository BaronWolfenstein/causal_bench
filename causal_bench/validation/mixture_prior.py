"""Normal-mixture approximation to a heavy-tailed treatment-effect prior.

Method (Pananos & Buffington, Datadog, "Approximating Heavy-Tailed Priors With Normal
Mixtures"): the conjugate normal model theta_hat | theta ~ N(theta, s^2), theta ~ g(theta)
gives an analytic normal posterior ONLY when g is normal. To keep the analytic posterior
while using a heavy-tailed prior h (e.g. a van-Zwet-style / Student-t treatment-effect prior),
approximate h by a K-component mixture of normals on a FIXED grid of (mu_k, tau_k), estimating
only the weights pi. The posterior is then a weighted set of conjugate normal updates -- no MCMC.

Weights minimise Pearson chi-square divergence from the mixture g to the target h,
    chi2(g||h) = \int (g-h)^2 / h dtheta = pi' C pi - 1,  C_jk = \int phi_j phi_k / h dtheta,
which, since C is a PSD Gram matrix, is a convex quadratic program on the simplex
    pi_hat = argmin_{pi>=0, 1'pi=1}  1/2 pi' C pi + lambda/2 ||pi||^2   (small ridge lambda).
We solve it with scipy SLSQP (K is small; no cvxpy dependency), keeping this in the torch-free
core venv.

PRINCIPAL LIMITATION (the paper's, kept explicit): a finite normal mixture is NOT heavy-tailed
asymptotically -- its tail is that of the widest component. This is an operational-range body
approximation, good between the target's lo/hi percentiles, not a model of the asymptotic tail.
So it is the right tool for fast calibrated shrinkage over the observed z-range, and the WRONG
tool if the science is in the extreme tail (cf. the alpha=1/Levy spectral-diffusion work).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm


def normal_grid(tau_lo: float, tau_hi: float, K: int, *, mu: float = 0.0):
    """Fixed mixture support: K components, common mean ``mu``, log-spaced sds in [tau_lo, tau_hi].

    A symmetric target about ``mu`` is well covered by common-mean components (the paper's
    choice); for an asymmetric target pass a grid of means instead via :func:`fit_weights`.
    """
    if tau_lo <= 0 or tau_hi <= tau_lo or K < 2:
        raise ValueError("require 0 < tau_lo < tau_hi and K >= 2")
    taus = np.logspace(np.log10(tau_lo), np.log10(tau_hi), K)
    mus = np.full(K, float(mu))
    return mus, taus


def gram_matrix(mus, taus, target_pdf, theta_grid):
    """C_jk = \int phi(theta;mu_j,tau_j) phi(theta;mu_k,tau_k) / h(theta) dtheta on ``theta_grid``.

    Computed by trapezoidal quadrature on ``theta_grid`` (cover the target's lo/hi percentiles
    finely). The integrand decays like a Gaussian over any polynomial growth of 1/h, so C is
    finite for heavy-tailed h as long as the grid spans the operational range.
    """
    mus = np.asarray(mus, float)
    taus = np.asarray(taus, float)
    theta = np.asarray(theta_grid, float)
    h = np.asarray(target_pdf(theta), float)
    if np.any(h <= 0) or not np.all(np.isfinite(h)):
        raise ValueError("target_pdf must be finite and strictly positive on theta_grid")
    # Phi[k, t] = phi(theta_t; mu_k, tau_k)
    Phi = norm.pdf(theta[None, :], loc=mus[:, None], scale=taus[:, None])
    inv_h = 1.0 / h
    # C_jk = trapz over theta of Phi_j * Phi_k * inv_h
    weighted = Phi * inv_h[None, :]                      # [K, T]
    C = np.trapezoid(weighted[:, None, :] * Phi[None, :, :], theta, axis=2)  # [K, K]
    C = 0.5 * (C + C.T)                                  # symmetrise numerical dust
    return C


def fit_weights(C, *, ridge: float = 1e-8):
    """Solve the simplex-constrained convex QP  min 1/2 pi'(C+ridge I)pi  s.t. pi>=0, 1'pi=1.

    scipy SLSQP with analytic gradient. ``ridge`` makes it strictly convex (the paper uses 1e-8).
    """
    C = np.asarray(C, float)
    K = C.shape[0]
    A = C + ridge * np.eye(K)

    def obj(pi):
        return 0.5 * pi @ A @ pi

    def grad(pi):
        return A @ pi

    cons = ({"type": "eq", "fun": lambda p: np.sum(p) - 1.0, "jac": lambda p: np.ones(K)},)
    bounds = [(0.0, 1.0)] * K
    x0 = np.full(K, 1.0 / K)
    res = minimize(obj, x0, jac=grad, bounds=bounds, constraints=cons,
                   method="SLSQP", options={"maxiter": 500, "ftol": 1e-12})
    pi = np.clip(res.x, 0.0, None)
    s = pi.sum()
    if s <= 0:
        raise RuntimeError("QP returned a degenerate (all-zero) weight vector")
    return pi / s


def mixture_pdf(theta, pi, mus, taus):
    """Density of the fitted normal mixture g(theta) = sum_k pi_k phi(theta; mu_k, tau_k)."""
    theta = np.asarray(theta, float)
    comp = norm.pdf(theta[..., None], loc=mus, scale=taus)   # [..., K]
    return comp @ np.asarray(pi, float)


@dataclass
class Posterior:
    """Mixture-of-normals posterior from one conjugate update of each component."""
    w: np.ndarray          # mixture weights (sum to 1)
    mu: np.ndarray         # component posterior means
    tau: np.ndarray        # component posterior sds

    def mean(self) -> float:
        return float(self.w @ self.mu)

    def cdf(self, theta) -> np.ndarray:
        theta = np.asarray(theta, float)
        comp = norm.cdf(theta[..., None], loc=self.mu, scale=self.tau)
        return comp @ self.w


def posterior(theta_hat: float, s: float, pi, mus, taus) -> Posterior:
    """Analytic posterior for theta_hat | theta ~ N(theta, s^2), theta ~ mixture(pi, mus, taus).

    Each component gets the conjugate normal update; component weights are re-weighted by the
    marginal likelihood N(theta_hat; mu_k, sqrt(s^2 + tau_k^2)).
    """
    pi = np.asarray(pi, float)
    mus = np.asarray(mus, float)
    taus = np.asarray(taus, float)
    s2 = s * s
    tau2 = taus * taus
    post_var = 1.0 / (1.0 / s2 + 1.0 / tau2)
    post_mu = post_var * (theta_hat / s2 + mus / tau2)
    # log marginal likelihood per component, for a stable softmax re-weighting
    marg_sd = np.sqrt(s2 + tau2)
    logw = np.log(pi) + norm.logpdf(theta_hat, loc=mus, scale=marg_sd)
    logw -= logw.max()
    w = np.exp(logw)
    w /= w.sum()
    return Posterior(w=w, mu=post_mu, tau=np.sqrt(post_var))


def shrinkage_curve(z_grid, s, pi, mus, taus):
    """z_tilde = E[theta|theta_hat]/s as a function of z = theta_hat/s (theta_hat = z*s).

    The conjugate-normal baseline shrinks z proportionally (a straight line); the mixture shrinks
    small z hard and large z little -- the nonlinear adaptive shrinkage that motivates the method.
    """
    z_grid = np.asarray(z_grid, float)
    out = np.empty_like(z_grid)
    for i, z in enumerate(z_grid):
        out[i] = posterior(z * s, s, pi, mus, taus).mean() / s
    return out


def sbc_ranks(target_sampler, s, pi, mus, taus, *, n: int, rng=None):
    """Simulation-based calibration ranks for the mixture approximation.

    For each of ``n`` replicates: draw theta ~ target (the EXACT heavy-tailed prior), draw
    theta_hat ~ N(theta, s^2), form the mixture posterior, and take rank = F_post(theta), the
    posterior CDF at the true value. Under a correct posterior these ranks are Uniform(0,1);
    systematic departure => the mixture distorts the posterior. Analytic (no posterior sampling).
    """
    rng = np.random.default_rng(rng)
    thetas = np.asarray(target_sampler(n, rng), float)
    theta_hats = thetas + rng.normal(0.0, s, size=n)
    ranks = np.empty(n, float)
    for i in range(n):
        ranks[i] = posterior(theta_hats[i], s, pi, mus, taus).cdf(thetas[i])
    return ranks
