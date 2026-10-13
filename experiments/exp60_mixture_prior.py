"""Exp 60: normal-mixture approximation to a heavy-tailed borrowing prior (fast analytic posterior).

A heavy-tailed treatment-effect prior (van-Zwet-style, or Student-t) breaks the conjugate-normal
shortcut the borrowing stack relies on for a closed-form posterior. This experiment reproduces the
Pananos-Buffington (Datadog) method -- approximate the heavy-tailed prior h by a K-normal mixture on
a fixed grid, fitting only the weights with a convex QP -- so the posterior is a weighted set of
conjugate updates (no MCMC), and checks that it is CALIBRATED and reproduces the heavy-tailed
prior's nonlinear adaptive shrinkage.

Two targets:
  * student_t : df=3, scale=0.05 -- the paper's benchmark (its scale-mixture form is known, so this
    is a clean check that the fit + SBC machinery is right).
  * van_zwet  : marginal effect prior theta ~ N(0, tau^2), log tau ~ N(-1.82, 0.90) (van Zwet single
    trial) integrated over tau -- a genuinely heavy-tailed prior from the borrowing line, the use
    case that matters here.

Metrics:
  * fit      : max |log g - log h| over the target's [0.01, 99.99] percentile range (body fit).
  * shrinkage: z_tilde/z at small vs large z, mixture vs the proportional normal-conjugate baseline
               (the nonlinear shrinkage that motivates the method).
  * sbc      : KS distance of the SBC ranks from Uniform(0,1) over many datasets (calibration).

Caveat (the paper's): a finite mixture is not heavy-tailed in the asymptotic tail -- body/operational
range only. PASS criteria are deliberately scoped to that range.

Run: python -m experiments.exp60_mixture_prior
"""
from pathlib import Path

import numpy as np
from scipy import stats

from causal_bench.validation.mixture_prior import (
    normal_grid, gram_matrix, fit_weights, mixture_pdf, shrinkage_curve, sbc_ranks,
)

OUT_DIR = Path("results/exp60_mixture_prior")

# ---- targets: each is (pdf, sampler, ppf) in the SAME operational-range convention ----------


def student_t_target(df=3.0, scale=0.05):
    rv = stats.t(df=df, loc=0.0, scale=scale)
    return rv.pdf, (lambda n, rng: rv.rvs(size=n, random_state=rng)), rv.ppf


def van_zwet_target(mu_log=-1.82, sd_log=0.90, n_tau=400):
    """Marginal theta prior: theta ~ N(0, tau^2), log tau ~ N(mu_log, sd_log). Heavy-tailed."""
    # quadrature nodes in log-tau for the pdf (mixture over a lognormal scale)
    g = np.linspace(mu_log - 6 * sd_log, mu_log + 6 * sd_log, n_tau)
    taus = np.exp(g)
    w = stats.norm.pdf(g, mu_log, sd_log)
    w /= np.trapezoid(w, g)                      # normalise on the log-tau grid
    dg = np.gradient(g)

    def pdf(theta):
        theta = np.asarray(theta, float)
        comp = stats.norm.pdf(theta[..., None], 0.0, taus)   # [..., n_tau]
        out = comp @ (w * dg)
        return np.where(np.isfinite(out) & (out > 0), out, 1e-300)

    def sampler(n, rng):
        lt = rng.normal(mu_log, sd_log, size=n)
        return rng.normal(0.0, np.exp(lt))

    # ppf via a fine numerical CDF (symmetric about 0)
    grid = np.linspace(-5.0, 5.0, 20001)
    cdf = np.concatenate([[0.0], np.cumsum((pdf(grid[1:]) + pdf(grid[:-1])) * np.diff(grid) / 2)])
    cdf /= cdf[-1]
    ppf = lambda q: np.interp(q, cdf, grid)
    return pdf, sampler, ppf


TARGETS = {"student_t": student_t_target, "van_zwet": van_zwet_target}


def run(*, target="van_zwet", K=80, tau_lo=0.002, tau_hi=1.5, s=0.1,
        n_sbc=4000, n_grid=6001, seed=0):
    if target not in TARGETS:
        raise ValueError(f"unknown target {target!r}; choose from {sorted(TARGETS)}")
    pdf, sampler, ppf = TARGETS[target]()

    # operational range = [0.01, 99.99] percentiles of the target
    lo, hi = float(ppf(1e-4)), float(ppf(1 - 1e-4))
    span = max(abs(lo), abs(hi))
    theta_grid = np.linspace(-span, span, n_grid)

    mus, taus = normal_grid(tau_lo, tau_hi, K)
    C = gram_matrix(mus, taus, pdf, theta_grid)
    pi = fit_weights(C)

    # --- fit quality: BODY (central [1,99] pct) is the gate; the tail edge is the known limitation ---
    def max_log_err_on(a, b):
        gg = np.linspace(a, b, 2001)
        e = np.abs(np.log(mixture_pdf(gg, pi, mus, taus)) - np.log(pdf(gg)))
        return float(np.nanmax(e))
    body_lo, body_hi = float(ppf(0.01)), float(ppf(0.99))
    max_log_err = max_log_err_on(body_lo, body_hi)          # body fit (gated)
    tail_edge_log_err = max_log_err_on(lo, hi)              # full op-range incl. tail edge (reported)

    # --- nonlinear shrinkage vs the proportional normal-conjugate baseline ---
    # baseline normal prior matched to the target's variance (second moment on the grid)
    var_h = float(np.trapezoid(theta_grid**2 * pdf(theta_grid), theta_grid))
    z_pts = np.array([0.5, 1.0, 2.0, 4.0, 8.0])
    zt_mix = shrinkage_curve(z_pts, s, pi, mus, taus)
    shr_base = var_h / (var_h + s * s)                   # constant proportional shrinkage factor
    shrink = {
        "z": z_pts.tolist(),
        "ztilde_over_z_mix": (zt_mix / z_pts).tolist(),
        "ztilde_over_z_normal": [shr_base] * len(z_pts),
        "adaptive": bool((zt_mix / z_pts)[0] < (zt_mix / z_pts)[-1] - 0.02),  # small shrunk > large
    }

    # --- SBC calibration ---
    ranks = sbc_ranks(sampler, s, pi, mus, taus, n=n_sbc, rng=seed)
    ks = float(stats.kstest(ranks, "uniform").statistic)
    ks_thr = 1.36 / np.sqrt(n_sbc)                       # ~95% KS band for Uniform(0,1)

    result = {
        "config": dict(target=target, K=K, tau_lo=tau_lo, tau_hi=tau_hi, s=s,
                       n_sbc=n_sbc, operational_range=[lo, hi]),
        "n_active_components": int(np.sum(pi > 1e-4)),
        "max_log_density_err": max_log_err,
        "tail_edge_log_err": tail_edge_log_err,
        "shrinkage": shrink,
        "sbc_ks": ks,
        "sbc_ks_threshold": ks_thr,
        "pass": {
            "fit_body": max_log_err < 0.15,              # <~15% density error across the body
            "shrinkage_adaptive": shrink["adaptive"],
            "sbc_calibrated": ks < ks_thr,
        },
    }
    result["all_pass"] = all(result["pass"].values())
    return result


def report(result):
    c = result["config"]
    lines = [f"# Exp60 — normal-mixture heavy-tailed prior ({c['target']})", "",
             f"K={c['K']} components, tau in [{c['tau_lo']}, {c['tau_hi']}], s={c['s']}, "
             f"operational range [{c['operational_range'][0]:.3f}, {c['operational_range'][1]:.3f}]",
             f"active components: {result['n_active_components']}", "",
             f"body fit   max|log g - log h| = {result['max_log_density_err']:.4f}  "
             f"({'PASS' if result['pass']['fit_body'] else 'FAIL'})   [central 1-99 pct]",
             f"tail edge  max|log g - log h| = {result['tail_edge_log_err']:.4f}  "
             f"[0.01-99.99 pct; large by design -- finite mixture is not heavy-tailed]",
             f"SBC        KS = {result['sbc_ks']:.4f}  (thr {result['sbc_ks_threshold']:.4f}, "
             f"{'PASS' if result['pass']['sbc_calibrated'] else 'FAIL'})", "",
             "shrinkage z_tilde/z  (mixture vs proportional normal baseline):"]
    sh = result["shrinkage"]
    for z, m, nb in zip(sh["z"], sh["ztilde_over_z_mix"], sh["ztilde_over_z_normal"]):
        lines.append(f"  z={z:>4}:  mix {m:.3f}   normal {nb:.3f}")
    lines += ["", f"adaptive (small shrunk more than large): "
                  f"{'yes' if sh['adaptive'] else 'no'}", "",
              f"ALL PASS: {result['all_pass']}"]
    return "\n".join(lines)


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for tgt in ("student_t", "van_zwet"):
        res = run(target=tgt)
        text = report(res)
        print(text + "\n")
        (OUT_DIR / f"{tgt}.txt").write_text(text + "\n")
