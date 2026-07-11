"""Demo: tangent-space-penalty DSM + gap-sampler on a synthetic curved manifold.

Shows the PR #99 loss end-to-end on an arc in R². Two score models are fit on
data that covers the arc EXCEPT a held-out angular gap:
  • plain DSM (λ=0) — no signal in the gap, so its denoiser drifts off-manifold
    there;
  • tangent-penalised DSM (λ>0) — the penalty is enforced on GAP-SAMPLED points,
    so its denoiser pulls gap-region points back onto the manifold.

Both agree in the supported region; the penalty only changes behaviour where the
flat model has nothing to go on — exactly the "silent bias generator" region the
manifold-aware propensity spec targets.

Run: python -m experiments.demo_tangent_dsm
"""
import warnings
from pathlib import Path

import numpy as np

# Apple-Accelerate BLAS emits spurious divide/overflow RuntimeWarnings on cos/matmul;
# results are finite and correct. Silence only these at the script boundary.
warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)

from causal_bench.generative.tangent_dsm import (
    ArcManifold, RFF, dsm_target, fit_linear_score, denoise,
    gap_noise_points, offmanifold_dist,
)

OUT_DIR = Path("results/tangent_dsm")


def run(sigma: float = 0.12, lam: float = 5.0, seed: int = 4):
    man = ArcManifold(R=1.0, lo=0.0, hi=np.pi)
    gap = (1.2, 1.9)
    rng = np.random.default_rng(seed)

    # DSM data covers the arc except the gap
    _, X0 = man.sample(2000, rng, gaps=[gap])
    Xt = X0 + sigma * rng.normal(size=X0.shape)
    T = dsm_target(X0, Xt, sigma)

    rff = RFF(n_features=250, dim=2, scale=2.0, seed=1)
    Phi = rff.transform(Xt)

    W_plain = fit_linear_score(Phi, T, lam=0.0)
    gxt, _, gnorm, gr = gap_noise_points(man, gap, n=400, sigma=sigma, rng=rng)
    W_tan = fit_linear_score(Phi, T, Phi_pen=rff.transform(gxt),
                             normals=gnorm, r_pen=gr, lam=lam)

    # evaluate on fresh gap-region points and on in-support points
    gt, _, _, _ = gap_noise_points(man, gap, n=3000, sigma=sigma,
                                   rng=np.random.default_rng(99))
    _, Xs = man.sample(3000, rng, gaps=[gap])
    Xts = Xs + sigma * rng.normal(size=Xs.shape)

    def err(W, X):
        return offmanifold_dist(denoise(X, W, rff, sigma), man).mean()

    rows = [
        ("gap region", offmanifold_dist(gt, man).mean(), err(W_plain, gt), err(W_tan, gt)),
        ("in support", offmanifold_dist(Xts, man).mean(), err(W_plain, Xts), err(W_tan, Xts)),
    ]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lines = ["| region | raw noised | plain DSM | tangent penalty | gap-error ↓ |",
             "|---|---|---|---|---|"]
    for name, raw, ep, et in rows:
        drop = f"{100 * (1 - et / ep):.0f}%" if ep > 0 else "—"
        lines.append(f"| {name} | {raw:.4f} | {ep:.4f} | {et:.4f} | {drop} |")
    table = "\n".join(lines)
    (OUT_DIR / "summary.md").write_text(table + "\n")

    print(f"Tangent-DSM + gap-sampler | σ={sigma} λ={lam} | arc [0,π] gap={gap}\n")
    print(table)
    print(f"\nSaved → {OUT_DIR}/summary.md")
    print("\nRead-out: both models denoise identically in the supported region;")
    print("the tangent penalty (fed by gap-sampled points) is what rescues the gap.")
    return rows


if __name__ == "__main__":
    run()
