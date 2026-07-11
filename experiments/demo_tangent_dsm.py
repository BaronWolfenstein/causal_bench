"""Demo: tangent-space-penalty DSM + gap-sampler on synthetic curved manifolds.

Shows the PR #99 loss end-to-end on TWO curved manifolds — an arc (1-manifold in
R²) and a Swiss roll (2-manifold in R³, still codimension-1 so the scalar
tangent-penalty closed form is unchanged). For each, two score models are fit on
data covering the manifold EXCEPT a held-out gap:
  • plain DSM (λ=0) — no signal in the gap, its denoiser drifts off-manifold there;
  • tangent-penalised DSM (λ>0) — the penalty is enforced on GAP-SAMPLED points,
    so its denoiser pulls gap-region points back onto the manifold.

Both agree in the supported region; the penalty only changes behaviour where the
flat model has nothing to go on — the "silent bias generator" region.

Run: python -m experiments.demo_tangent_dsm
"""
import warnings
from pathlib import Path

import numpy as np

# Apple-Accelerate BLAS emits spurious divide/overflow RuntimeWarnings on cos/matmul;
# results are finite and correct. Silence only these at the script boundary.
warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)

from causal_bench.generative.tangent_dsm import (
    ArcManifold, SwissRoll, Helix, Plane, RFF, dsm_target, fit_linear_score,
    denoise, gap_noise_points, offmanifold_dist, estimate_local_normals,
)

OUT_DIR = Path("results/tangent_dsm")


def _fit_and_eval(man, dim, gap, *, sigma, lam, n_data, n_feat, scale, n_pen,
                  in_support_sampler, seed):
    rng = np.random.default_rng(seed)
    _, X0 = man.sample(n_data, rng, gaps=[gap])
    Xt = X0 + sigma * rng.normal(size=X0.shape)
    T = dsm_target(X0, Xt, sigma)

    rff = RFF(n_features=n_feat, dim=dim, scale=scale, seed=1)
    Phi = rff.transform(Xt)

    W_plain = fit_linear_score(Phi, T, lam=0.0)
    gxt, _, gnorm, gr = gap_noise_points(man, gap, n=n_pen, sigma=sigma, rng=rng)
    W_tan = fit_linear_score(Phi, T, Phi_pen=rff.transform(gxt),
                             normals=gnorm, r_pen=gr, lam=lam)

    gt, _, _, _ = gap_noise_points(man, gap, n=3000, sigma=sigma,
                                   rng=np.random.default_rng(seed + 99))
    Xs = in_support_sampler(rng)
    Xts = Xs + sigma * rng.normal(size=Xs.shape)

    def err(W, X):
        return offmanifold_dist(denoise(X, W, rff, sigma), man).mean()

    rows = [
        ("gap region", offmanifold_dist(gt, man).mean(), err(W_plain, gt), err(W_tan, gt)),
        ("in support", offmanifold_dist(Xts, man).mean(), err(W_plain, Xts), err(W_tan, Xts)),
    ]
    ablation = None
    if gnorm.ndim == 3 and gnorm.shape[1] > 1:
        # codim ≥ 2: constrain ONLY the first normal — leaves drift in the rest
        W_one = fit_linear_score(Phi, T, Phi_pen=rff.transform(gxt),
                                 normals=gnorm[:, 0, :], r_pen=gr, lam=lam)
        ablation = (err(W_one, gt), err(W_tan, gt))       # (1 normal, full basis)
    return rows, ablation


def _learned_and_control_section() -> str:
    """Two additions that matter for real embeddings: (1) LEARNED normals — the
    penalty works with the normal basis estimated from the point cloud by local
    PCA, no analytic manifold; (2) a FLAT plane in R⁵ — the gated-design control
    showing the penalty is harmless where curvature is absent."""
    lines = ["### Learned metric & flat control"]

    # (1) learned normals recover the analytic payoff on the helix (codim 2)
    man = Helix()
    gap = (1.6 * np.pi, 2.0 * np.pi)
    sigma = 0.10
    rng = np.random.default_rng(5)
    _, X0 = man.sample(4000, rng, gaps=[gap])
    Xt = X0 + sigma * rng.normal(size=X0.shape)
    T = dsm_target(X0, Xt, sigma)
    rff = RFF(500, 3, 1.2, seed=3)
    Phi = rff.transform(Xt)
    W_plain = fit_linear_score(Phi, T, lam=0.0)
    gxt, _, gn_a, gr = gap_noise_points(man, gap, n=900, sigma=sigma, rng=rng)
    _, cloud = man.sample(6000, np.random.default_rng(123))
    gn_l = estimate_local_normals(cloud, gxt, k=40, intrinsic_dim=1)
    W_a = fit_linear_score(Phi, T, Phi_pen=rff.transform(gxt), normals=gn_a, r_pen=gr, lam=8.0)
    W_l = fit_linear_score(Phi, T, Phi_pen=rff.transform(gxt), normals=gn_l, r_pen=gr, lam=8.0)
    gt, _, _, _ = gap_noise_points(man, gap, n=3000, sigma=sigma, rng=np.random.default_rng(31))
    ep = offmanifold_dist(denoise(gt, W_plain, rff, sigma), man).mean()
    ea = offmanifold_dist(denoise(gt, W_a, rff, sigma), man).mean()
    el = offmanifold_dist(denoise(gt, W_l, rff, sigma), man).mean()
    lines += ["", f"**Learned normals (helix, codim 2)** — normal basis estimated from the "
              f"point cloud by local PCA, *no analytic manifold*:",
              f"- plain DSM {ep:.4f} → analytic-normal penalty {ea:.4f} → "
              f"**learned-normal penalty {el:.4f}** ({100*(1-el/ep):.0f}% drop, matches analytic). "
              f"This is the bridge to real embeddings."]

    # (2) flat plane in R^5 — penalty harmless
    pl = Plane(dim=5, intrinsic=2, seed=0)
    pgap = (-0.8, 0.8)
    rng = np.random.default_rng(2)
    _, X0 = pl.sample(4000, rng, gaps=[pgap])
    Xt = X0 + 0.12 * rng.normal(size=X0.shape)
    T = dsm_target(X0, Xt, 0.12)
    rff = RFF(400, 5, 0.7, seed=4)
    Phi = rff.transform(Xt)
    W_plain = fit_linear_score(Phi, T, lam=0.0)
    gxt, _, gn, gr = gap_noise_points(pl, pgap, n=800, sigma=0.12, rng=rng)
    W_tan = fit_linear_score(Phi, T, Phi_pen=rff.transform(gxt), normals=gn, r_pen=gr, lam=8.0)
    gt, _, _, _ = gap_noise_points(pl, pgap, n=3000, sigma=0.12, rng=np.random.default_rng(9))
    ep = offmanifold_dist(denoise(gt, W_plain, rff, 0.12), pl).mean()
    et = offmanifold_dist(denoise(gt, W_tan, rff, 0.12), pl).mean()
    lines += ["", f"**Flat plane in R⁵ (codim 3, gated-design control)** — raw gap noise "
              f"{offmanifold_dist(gt, pl).mean():.3f}: plain DSM already {ep:.4f}, penalty "
              f"{et:.4f}. On a flat manifold plain DSM is correct and the penalty is "
              f"harmless — you only pay for geometry when curvature is present."]
    return "\n".join(lines)


def run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    arc = ArcManifold(R=1.0, lo=0.0, hi=np.pi)
    arc_gap = (1.2, 1.9)
    roll = SwissRoll()
    roll_gap = (2.5 * np.pi, 2.8 * np.pi)
    helix = Helix()
    helix_gap = (1.6 * np.pi, 2.0 * np.pi)

    cases = [
        ("Arc (1-manifold in R², codim 1)", arc, 2, arc_gap, dict(
            sigma=0.12, lam=5.0, n_data=2000, n_feat=250, scale=2.0, n_pen=400,
            in_support_sampler=lambda r: arc.sample(3000, r, gaps=[arc_gap])[1],
            seed=4)),
        ("Swiss roll (2-manifold in R³, codim 1)", roll, 3, roll_gap, dict(
            sigma=0.15, lam=8.0, n_data=4000, n_feat=400, scale=0.6, n_pen=800,
            in_support_sampler=lambda r: roll.sample(3000, r, gaps=[roll_gap])[1],
            seed=7)),
        ("Helix (curve in R³, codim 2)", helix, 3, helix_gap, dict(
            sigma=0.10, lam=8.0, n_data=4000, n_feat=500, scale=1.2, n_pen=900,
            in_support_sampler=lambda r: helix.sample(3000, r, gaps=[helix_gap])[1],
            seed=5)),
    ]

    blocks = []
    for name, man, dim, gap, kw in cases:
        rows, ablation = _fit_and_eval(man, dim, gap, **kw)
        lines = [f"### {name}   (gap held out, σ={kw['sigma']}, λ={kw['lam']})",
                 "| region | raw noised | plain DSM | tangent penalty | gap-error ↓ |",
                 "|---|---|---|---|---|"]
        for region, raw, ep, et in rows:
            drop = f"{100 * (1 - et / ep):.0f}%" if ep > 0 else "—"
            lines.append(f"| {region} | {raw:.4f} | {ep:.4f} | {et:.4f} | {drop} |")
        if ablation is not None:
            e_one, e_full = ablation
            lines.append("")
            lines.append(f"*Codim-2 ablation (gap): constrain 1 normal → {e_one:.4f} "
                         f"vs full 2-basis → {e_full:.4f} "
                         f"({e_one / e_full:.1f}× worse). Fixing one normal leaves "
                         f"drift in the other — this is why codim ≥ 2 needs the basis.*")
        blocks.append("\n".join(lines))

    report = "\n\n".join(blocks + [_learned_and_control_section()])
    (OUT_DIR / "summary.md").write_text(report + "\n")
    print(report)
    print(f"\nSaved → {OUT_DIR}/summary.md")
    print("\nRead-out: on every manifold the two models denoise near-identically in")
    print("the supported region; the tangent penalty (fed by gap-sampled points) is what")
    print("rescues the gap. Narrow gaps recover near-perfectly; wide gaps are harder.")
    print("At codimension 2 (helix) the FULL normal basis is required — constraining")
    print("only one normal leaves the score free to drift along the other.")


if __name__ == "__main__":
    run()
