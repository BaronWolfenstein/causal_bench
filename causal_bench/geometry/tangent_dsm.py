"""Tangent-space-penalty DSM — the generation-side Layer-2 of #99 (gated).

Standard denoising score matching matches the predicted noise in the AMBIENT
space, which teaches a flat bias: the learned score field will happily push
samples off the data manifold into invalid intermediate anatomies.  A Riemannian
bias projects the score onto the local NORMAL space `N_xM` (estimated by local
PCA) and penalizes that off-manifold component — near the manifold the true score
is tangent (the density is a normal-direction local max there), so this pulls the
learned field toward tangency without fighting the DSM target off-manifold (the
penalty is weighted by √ᾱ_t, ≈1 near the manifold, →0 in the noised tail).

DESIGN-ONLY / GATED; validated on a synthetic manifold (swiss roll) by confirming
the penalized net's on-manifold score is more tangent than the baseline's.  A
training-time variant of the same score net (does not modify `score_net.py`).
"""
from __future__ import annotations

import numpy as np

from causal_bench.generative.score_net import resolve_device, make_optimizer


def estimate_local_normals(X, k: int = 15, intrinsic_dim: int = 2):
    """Per-point NORMAL basis via local PCA of the k-NN neighborhood.  Returns
    (N, D-intrinsic_dim, D): the bottom principal directions (orthogonal to the
    local tangent `T_xM`).  CPU preprocessing — does not gate the GPU loop."""
    from sklearn.neighbors import NearestNeighbors
    n, D = X.shape
    _, idx = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X)
    normals = np.empty((n, D - intrinsic_dim, D), dtype=np.float32)
    for i in range(n):
        c = X[idx[i, 1:]] - X[idx[i, 1:]].mean(0)         # center the neighborhood
        _, _, Vt = np.linalg.svd(c, full_matrices=True)   # rows = principal dirs
        normals[i] = Vt[intrinsic_dim:]                   # tangent = top d; normal = rest
    return normals


def normal_component_fraction(model, X, normals, sch, t: int = 5, device="auto"):
    """Diagnostic: mean fraction of the score's magnitude lying in the local
    NORMAL space at on-manifold points (0 = fully tangent)."""
    import torch
    dev = resolve_device(device)
    model.to(dev).eval()
    with torch.no_grad():
        xt = torch.as_tensor(X, dtype=torch.float32, device=dev)
        tf = torch.full((len(X),), float(t) / sch.n_steps, device=dev)
        eps = model(xt, tf)
        a = float(sch.alphas_bar[t])
        score = (-eps / np.sqrt(1 - a)).cpu().numpy()      # (N, D)
    nc = np.einsum("imd,id->im", normals, score)           # (N, m) normal components
    return float(np.mean(np.linalg.norm(nc, axis=1) / (np.linalg.norm(score, axis=1) + 1e-9)))


def train_score_tangent(model, X, sch, normals, *, lam: float = 0.0, epochs: int = 20,
                        rng=None, device: str = "auto", opt=None, precision: str = "fp32"):
    """DSM training with the tangent-space penalty.  lam=0 recovers plain DSM.
    `normals`: (N, m, D) local normal bases aligned with rows of X."""
    import torch
    from causal_bench.generative.score_net import _perf_setup
    dev = resolve_device(device)
    model.to(dev)
    rng = rng or np.random.default_rng(0)
    autocast, _ = _perf_setup(precision, dev)
    Xt = torch.as_tensor(np.asarray(X), dtype=torch.float32, device=dev)
    Nt = torch.as_tensor(normals, dtype=torch.float32, device=dev)      # (N, m, D)
    opt = opt or make_optimizer(model)
    ab = torch.as_tensor(sch.alphas_bar, dtype=torch.float32, device=dev)
    for _ in range(epochs):
        model.train()
        t = rng.integers(1, sch.n_steps, size=len(Xt))
        a = ab[torch.as_tensor(t, dtype=torch.long, device=dev)]        # (B,)
        eps = torch.randn_like(Xt)
        xt = torch.sqrt(a)[:, None] * Xt + torch.sqrt(1 - a)[:, None] * eps
        tf = torch.as_tensor(t / sch.n_steps, dtype=torch.float32, device=dev)
        with autocast():
            pred = model(xt, tf)
            dsm = ((pred - eps) ** 2).mean()
            if lam > 0:
                score = -pred / torch.sqrt(1 - a)[:, None]             # (B, D)
                nc = torch.einsum("imd,id->im", Nt, score)             # normal component
                w = torch.sqrt(a)                                       # ≈1 near manifold
                pen = (w * (nc ** 2).sum(dim=1)).mean()
                loss = dsm + lam * pen
            else:
                loss = dsm
        opt.zero_grad(); loss.backward(); opt.step()
    return model
