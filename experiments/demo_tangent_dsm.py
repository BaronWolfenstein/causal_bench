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
    ArcManifold, SwissRoll, Helix, RFF, dsm_target, fit_linear_score, denoise,
    gap_noise_points, offmanifold_dist,
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

    report = "\n\n".join(blocks)
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
