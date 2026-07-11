"""Torch MLP score net. Same score_fn(x_t, t) contract as the analytic path, so
it drops into vpsde/roundtrip/guidance unchanged. Trains by denoising score
matching on ZCA-whitened stand-in embeddings with an INDEPENDENT timestep per
sample (standard DSM — every minibatch sees the whole noise schedule, not one
level per step); per-sample tail-aware weights (Task 4) reweight the loss toward
the rare region (Test B'). Lazy torch import."""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np


def resolve_device(device: str = "auto"):
    """Pick a torch device: 'auto' -> cuda if available, else mps (Apple),
    else cpu. Explicit strings ('cuda', 'cuda:1', 'cpu', 'mps') pass through."""
    import torch
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _time_embedding(t_frac, dim, torch):
    # sinusoidal embedding of a PER-SAMPLE timestep fraction. t_frac is (B,);
    # returns (B, dim). Built on t_frac's device.
    half = dim // 2
    freqs = torch.exp(torch.arange(half, device=t_frac.device)
                      * -(np.log(10000.0) / max(half - 1, 1)))
    ang = t_frac[:, None] * freqs[None, :]            # (B, half)
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)   # (B, dim)


def ScoreMLP(dim: int, hidden: int = 128):
    import torch
    import torch.nn as nn

    class _Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.temb = 16
            self.net = nn.Sequential(
                nn.Linear(dim + self.temb, hidden), nn.SiLU(),
                nn.Linear(hidden, hidden), nn.SiLU(),
                nn.Linear(hidden, dim),
            )

        def forward(self, x, t_frac):
            # t_frac is per-sample (B,) -> te is (B, temb); no expand needed
            te = _time_embedding(t_frac, self.temb, torch).to(x)
            return self.net(torch.cat([x, te], dim=1))

    return _Net()


def make_torch_score_fn(model, sch, device: str = "auto") -> Callable[[np.ndarray, int], np.ndarray]:
    import torch
    dev = resolve_device(device)
    model.to(dev)

    def score_fn(x_t: np.ndarray, t: int) -> np.ndarray:
        model.eval()
        with torch.no_grad():
            xt = torch.as_tensor(np.asarray(x_t), dtype=torch.float32, device=dev)
            # inference: every particle is at the same step t -> per-sample t_frac
            # vector of shape (B,), matching the batched time embedding
            t_frac = torch.full((xt.shape[0],), float(t) / sch.n_steps,
                                dtype=torch.float32, device=dev)
            eps = model(xt, t_frac)                       # eps-prediction
            a = float(sch.alphas_bar[t])
            score = -eps / np.sqrt(1 - a)                 # score = -eps/sqrt(1-abar)
        return score.detach().cpu().numpy()               # always hand numpy back to the SDE loop

    return score_fn


def train_score(model, X, sch, *, weights: Optional[np.ndarray] = None,
                epochs: int = 20, rng=None, device: str = "auto", _loss_log=None):
    import torch
    dev = resolve_device(device)
    model.to(dev)
    rng = rng or np.random.default_rng(0)
    X = torch.as_tensor(np.asarray(X), dtype=torch.float32, device=dev)
    w = (torch.as_tensor(np.asarray(weights), dtype=torch.float32, device=dev)
         if weights is not None else torch.ones(len(X), device=dev))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    ab = torch.as_tensor(sch.alphas_bar, dtype=torch.float32, device=dev)
    for _ in range(epochs):
        model.train()
        # independent timestep per sample (standard DSM), shape (B,)
        t = rng.integers(1, sch.n_steps, size=len(X))
        t_idx = torch.as_tensor(t, dtype=torch.long, device=dev)
        a = ab[t_idx]                                     # (B,)
        eps = torch.randn_like(X)
        xt = torch.sqrt(a)[:, None] * X + torch.sqrt(1 - a)[:, None] * eps
        t_frac = torch.as_tensor(t / sch.n_steps, dtype=torch.float32, device=dev)
        pred = model(xt, t_frac)
        loss = (w * ((pred - eps) ** 2).mean(dim=1)).mean()   # tail-aware weighted
        opt.zero_grad(); loss.backward(); opt.step()
        if _loss_log is not None:
            _loss_log.append(float(loss.item()))
    return model
