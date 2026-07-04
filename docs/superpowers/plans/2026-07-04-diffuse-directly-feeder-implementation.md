# diffuse_directly Feeder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `diffuse_directly` GPU feeder — a device-agnostic score-diffusion model over embeddings that produces Test-B reconstruction round-trips and Test-B″ CFG generative-landing samples, feeding `causal_bench`'s `run_diagnostic`, validated on planted known-answer toy cases.

**Architecture:** A new sibling repo `diffuse_directly` (torch/GPU deps, kept out of causal_bench's lean footprint) depending on `causal_bench` as a local editable package for `diagnostics.localization` and `diagnostics.embedding_eda`. A conditional DDPM (discrete-time score diffusion) is trained on ZCA-whitened embeddings with a class label {common, rare, null} — the null class enables classifier-free guidance (CFG). Test B reuses the trained model for partial-noise reconstruction; Test B″ reuses it for full generation under CFG. One `run_feeder()` entry point wires source → whiten → train → B → B″ → `run_diagnostic` → persisted artifacts.

**Tech Stack:** Python ≥3.10, PyTorch ≥2.2 (MPS + CUDA backends), numpy, causal_bench (local editable dependency), pytest, uv for environment management.

## Global Constraints

- New repo at `/Users/noahrahman/git/diffuse_directly`; depends on `causal_bench` via a local path dependency (editable), never vendors/copies its code.
- Device-agnostic: every model/tensor call goes through a single `get_device()` — no hardcoded `"cuda"`/`"mps"` elsewhere. Toy validation must run end-to-end on CPU/MPS with no GPU server.
- Diffusion trains and samples in **ZCA-whitened** coordinates (`causal_bench.diagnostics.embedding_eda.zca_whiten`/`zca_unwhiten`); `run_diagnostic` always receives embeddings unwhitened back to the original space.
- Toy `PlantedSource` defines rare/common by construction (embedding-space clusters) — this is a machinery-validation device only, per spec §3; it is not a template for the real `SMBSource` path, which must define rare/common by clinical/outcome label (documented in the module docstring, not enforced in code since `SMBSource` is out of scope here).
- No new dependency on `torch` (or any GPU library) leaks into `causal_bench`. `causal_bench` is consumed read-only.
- Every persisted artifact (model, whitening params, strata membership) is saved via `torch.save`/`np.savez` to a caller-supplied directory — no hardcoded paths.
- Commit after every task; message ends with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- Tests use tiny configs (small `epochs`, `hidden`, `n_steps`) so the full suite runs in well under a minute on CPU; no test requires MPS/CUDA to pass (MPS is exercised manually, not asserted in CI).

---

### Task 1: Repo scaffold + device selection

**Files:**
- Create: `/Users/noahrahman/git/diffuse_directly/pyproject.toml`
- Create: `/Users/noahrahman/git/diffuse_directly/src/diffuse_directly/__init__.py`
- Create: `/Users/noahrahman/git/diffuse_directly/src/diffuse_directly/device.py`
- Test: `/Users/noahrahman/git/diffuse_directly/tests/test_device.py`

**Interfaces:**
- Produces: `get_device(prefer: str = "auto") -> torch.device`. `prefer` ∈ {"auto","cpu","mps","cuda"}; `"auto"` picks cuda > mps > cpu by availability.

- [ ] **Step 1: Create the repo and directory structure**

```bash
mkdir -p /Users/noahrahman/git/diffuse_directly/src/diffuse_directly
mkdir -p /Users/noahrahman/git/diffuse_directly/tests
cd /Users/noahrahman/git/diffuse_directly
git init -q
```

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[project]
name = "diffuse_directly"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.2",
    "numpy>=1.24",
    "causal_bench @ file:///Users/noahrahman/git/causal_bench",
]

[project.optional-dependencies]
dev = ["pytest>=7"]

[tool.setuptools.packages.find]
where = ["src"]

[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 3: Sync the environment**

```bash
cd /Users/noahrahman/git/diffuse_directly
uv venv
uv pip install -e ".[dev]"
```

Expected: installs torch, numpy, and causal_bench (editable, from the local path) with no errors.

- [ ] **Step 4: Write the failing test**

Create `tests/test_device.py`:

```python
import torch

from diffuse_directly.device import get_device


def test_get_device_cpu_explicit():
    d = get_device("cpu")
    assert d.type == "cpu"


def test_get_device_auto_returns_valid_device():
    d = get_device("auto")
    assert d.type in ("cpu", "mps", "cuda")


def test_get_device_invalid_prefer_raises():
    import pytest
    with pytest.raises(ValueError):
        get_device("tpu")
```

- [ ] **Step 5: Run test to verify it fails**

Run: `cd /Users/noahrahman/git/diffuse_directly && .venv/bin/python -m pytest tests/test_device.py -q`
Expected: FAIL (ModuleNotFoundError: `diffuse_directly.device`).

- [ ] **Step 6: Write the implementation**

Create `src/diffuse_directly/__init__.py`:

```python
"""diffuse_directly: device-agnostic score-diffusion feeder for the
rare-detail localisation diagnostic (causal_bench.diagnostics.localization).

Trains a conditional DDPM on ZCA-whitened embeddings and produces the
reconstruction round-trip (Test B) and CFG generative-landing samples
(Test B'') that feed run_diagnostic. Toy-validated on planted embeddings
(no GPU/encoder needed); the real SMB-v1/TVT source is a documented seam
(sources.EmbeddingSource), not built here.
"""
```

Create `src/diffuse_directly/device.py`:

```python
"""Single device-selection choke point -- no other module should call
torch.device(...) directly, so the whole pipeline is CPU/MPS/CUDA-agnostic."""
from __future__ import annotations

import torch

_VALID = {"auto", "cpu", "mps", "cuda"}


def get_device(prefer: str = "auto") -> torch.device:
    if prefer not in _VALID:
        raise ValueError(f"prefer must be one of {_VALID}, got {prefer!r}")
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer == "cuda":
        return torch.device("cuda")
    if prefer == "mps":
        return torch.device("mps")
    # auto: cuda > mps > cpu
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
```

- [ ] **Step 7: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_device.py -q`
Expected: 3 passed.

- [ ] **Step 8: Commit**

```bash
cd /Users/noahrahman/git/diffuse_directly
git add pyproject.toml src/diffuse_directly/__init__.py src/diffuse_directly/device.py tests/test_device.py
git commit -m "feat: repo scaffold + device-agnostic selection

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: EmbeddingSource protocol + PlantedSource

**Files:**
- Create: `src/diffuse_directly/sources.py`
- Test: `tests/test_sources.py`

**Interfaces:**
- Produces:
  - `PlantedConfig(n_rare=40, n_common=200, dim=16, separation=3.0, rare_scale=0.5, common_scale=0.5, seed=0)` (dataclass).
  - `EmbeddingSource` protocol: `sample(self) -> tuple[np.ndarray, np.ndarray]` returning `(rare_emb, common_emb)`.
  - `PlantedSource(config: PlantedConfig)` implementing `EmbeddingSource`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_sources.py`:

```python
import numpy as np

from diffuse_directly.sources import PlantedConfig, PlantedSource


def test_planted_source_shapes():
    src = PlantedSource(PlantedConfig(n_rare=10, n_common=50, dim=8))
    rare, common = src.sample()
    assert rare.shape == (10, 8)
    assert common.shape == (50, 8)


def test_planted_source_separation_controls_shift():
    src = PlantedSource(PlantedConfig(n_rare=20, n_common=20, dim=4,
                                      separation=5.0, seed=1))
    rare, common = src.sample()
    assert rare.mean(axis=0)[0] > common.mean(axis=0)[0] + 3.0


def test_planted_source_deterministic_with_seed():
    cfg = PlantedConfig(seed=42)
    r1, c1 = PlantedSource(cfg).sample()
    r2, c2 = PlantedSource(cfg).sample()
    assert np.array_equal(r1, r2) and np.array_equal(c1, c2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_sources.py -q`
Expected: FAIL (ModuleNotFoundError).

- [ ] **Step 3: Write the implementation**

Create `src/diffuse_directly/sources.py`:

```python
"""EmbeddingSource: the swappable seam between toy validation and the real
SMB-v1/TVT (or MIMIC prototype) path.

PlantedSource defines rare/common by embedding-space construction -- valid
ONLY as a machinery-validation device (spec sec 3, 2026-07-04 design). The
real SMBSource (interface-level, not built here) MUST define rare/common by
clinical/outcome label, never by embedding clustering -- clustering-based
labels on real data would be circular (the localisation diagnostic tests
whether clustering-worthy structure survives generation; using clustering to
define the labels in the first place begs the question).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Tuple, runtime_checkable

import numpy as np


@dataclass
class PlantedConfig:
    n_rare: int = 40
    n_common: int = 200
    dim: int = 16
    separation: float = 3.0
    rare_scale: float = 0.5
    common_scale: float = 0.5
    seed: int = 0


@runtime_checkable
class EmbeddingSource(Protocol):
    def sample(self) -> Tuple[np.ndarray, np.ndarray]: ...


class PlantedSource:
    """Bulk N(0, common_scale^2 I) + a rare cluster shifted by `separation`
    along the first coordinate. Deterministic given the config's seed."""

    def __init__(self, config: PlantedConfig):
        self.config = config

    def sample(self) -> Tuple[np.ndarray, np.ndarray]:
        c = self.config
        rng = np.random.default_rng(c.seed)
        common = rng.normal(0.0, c.common_scale, (c.n_common, c.dim))
        mean_rare = np.zeros(c.dim)
        mean_rare[0] = c.separation
        rare = rng.normal(mean_rare, c.rare_scale, (c.n_rare, c.dim))
        return rare, common
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_sources.py -q`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/diffuse_directly/sources.py tests/test_sources.py
git commit -m "feat: EmbeddingSource protocol + PlantedSource toy generator

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Whitening wrapper

**Files:**
- Create: `src/diffuse_directly/whitening.py`
- Test: `tests/test_whitening.py`

**Interfaces:**
- Consumes: `causal_bench.diagnostics.embedding_eda.zca_whiten(Z, eps=1e-6) -> (Z_white, W, mu)`, `zca_unwhiten(Z_white, W, mu) -> Z`.
- Produces:
  - `WhiteningParams(W: np.ndarray, mu: np.ndarray)` (dataclass).
  - `whiten_pooled(rare, common) -> tuple[np.ndarray, np.ndarray, WhiteningParams]` — pools rare+common, fits ZCA once on the pool, returns `(rare_white, common_white, params)`.
  - `unwhiten(Z_white, params: WhiteningParams) -> np.ndarray`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_whitening.py`:

```python
import numpy as np

from diffuse_directly.whitening import WhiteningParams, unwhiten, whiten_pooled


def test_whiten_pooled_shapes_and_roundtrip():
    rng = np.random.default_rng(0)
    rare = rng.normal(3.0, 2.0, (10, 5))
    common = rng.normal(0.0, 1.0, (40, 5))
    rare_w, common_w, params = whiten_pooled(rare, common)
    assert rare_w.shape == rare.shape and common_w.shape == common.shape
    assert isinstance(params, WhiteningParams)

    rare_back = unwhiten(rare_w, params)
    common_back = unwhiten(common_w, params)
    assert np.allclose(rare_back, rare, atol=1e-6)
    assert np.allclose(common_back, common, atol=1e-6)


def test_whiten_pooled_identity_covariance():
    rng = np.random.default_rng(1)
    rare = rng.normal(0.0, 3.0, (30, 4))
    common = rng.normal(0.0, 3.0, (30, 4))
    rare_w, common_w, _ = whiten_pooled(rare, common)
    pooled_w = np.vstack([rare_w, common_w])
    cov = np.cov(pooled_w, rowvar=False)
    assert np.allclose(cov, np.eye(4), atol=1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_whitening.py -q`
Expected: FAIL (ModuleNotFoundError).

- [ ] **Step 3: Write the implementation**

Create `src/diffuse_directly/whitening.py`:

```python
"""Thin wrapper around causal_bench's ZCA whitening, pooling rare+common so
both modes share one whitening transform (required for Test B/B'' -- the
diagnostic compares rare and common in the SAME space)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from causal_bench.diagnostics.embedding_eda import zca_unwhiten, zca_whiten


@dataclass
class WhiteningParams:
    W: np.ndarray
    mu: np.ndarray


def whiten_pooled(rare: np.ndarray, common: np.ndarray,
                  eps: float = 1e-6) -> Tuple[np.ndarray, np.ndarray, WhiteningParams]:
    n_rare = len(rare)
    pooled = np.vstack([rare, common])
    pooled_white, W, mu = zca_whiten(pooled, eps=eps)
    return pooled_white[:n_rare], pooled_white[n_rare:], WhiteningParams(W=W, mu=mu)


def unwhiten(Z_white: np.ndarray, params: WhiteningParams) -> np.ndarray:
    return zca_unwhiten(Z_white, params.W, params.mu)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_whitening.py -q`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/diffuse_directly/whitening.py tests/test_whitening.py
git commit -m "feat: pooled ZCA whitening wrapper around causal_bench

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Score network

**Files:**
- Create: `src/diffuse_directly/score_net.py`
- Test: `tests/test_score_net.py`

**Interfaces:**
- Produces:
  - `ScoreMLP.NULL_CLASS = 2` (class constant; classes are `0=common, 1=rare, 2=null` for CFG).
  - `ScoreMLP(dim: int, n_steps: int, hidden: int = 128, time_embed_dim: int = 32, class_embed_dim: int = 16)` (`torch.nn.Module`).
  - `ScoreMLP.forward(x_t: Tensor[n,dim], t: Tensor[n] (long), y: Tensor[n] (long)) -> Tensor[n,dim]` (predicted noise `eps`).

- [ ] **Step 1: Write the failing test**

Create `tests/test_score_net.py`:

```python
import torch

from diffuse_directly.score_net import ScoreMLP


def test_score_mlp_output_shape():
    net = ScoreMLP(dim=8, n_steps=50, hidden=32)
    x_t = torch.randn(6, 8)
    t = torch.randint(0, 50, (6,))
    y = torch.tensor([0, 1, 2, 0, 1, 2])
    out = net(x_t, t, y)
    assert out.shape == (6, 8)


def test_score_mlp_null_class_constant():
    assert ScoreMLP.NULL_CLASS == 2


def test_score_mlp_different_classes_give_different_output():
    torch.manual_seed(0)
    net = ScoreMLP(dim=4, n_steps=20, hidden=16)
    x_t = torch.randn(1, 4)
    t = torch.tensor([5])
    out_rare = net(x_t, t, torch.tensor([1]))
    out_null = net(x_t, t, torch.tensor([ScoreMLP.NULL_CLASS]))
    assert not torch.allclose(out_rare, out_null)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_score_net.py -q`
Expected: FAIL (ModuleNotFoundError).

- [ ] **Step 3: Write the implementation**

Create `src/diffuse_directly/score_net.py`:

```python
"""Score network: predicts the noise eps added at timestep t, conditioned on
a class label y in {0=common, 1=rare, 2=null}. The null class is what makes
classifier-free guidance (CFG) possible at sample time (Test B'')."""
from __future__ import annotations

import torch
import torch.nn as nn

N_CLASSES = 3  # common, rare, null


class ScoreMLP(nn.Module):
    NULL_CLASS = 2

    def __init__(self, dim: int, n_steps: int, hidden: int = 128,
                 time_embed_dim: int = 32, class_embed_dim: int = 16):
        super().__init__()
        self.dim = dim
        self.time_embed = nn.Embedding(n_steps, time_embed_dim)
        self.class_embed = nn.Embedding(N_CLASSES, class_embed_dim)
        in_dim = dim + time_embed_dim + class_embed_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        te = self.time_embed(t)
        ce = self.class_embed(y)
        h = torch.cat([x_t, te, ce], dim=-1)
        return self.net(h)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_score_net.py -q`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/diffuse_directly/score_net.py tests/test_score_net.py
git commit -m "feat: conditional score MLP with a null class for CFG

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: Diffusion process utilities

**Files:**
- Create: `src/diffuse_directly/diffusion.py`
- Test: `tests/test_diffusion.py`

**Interfaces:**
- Produces:
  - `NoiseSchedule(n_steps, betas, alphas, alpha_bars)` (dataclass of tensors).
  - `make_schedule(n_steps: int = 100, beta_start: float = 1e-4, beta_end: float = 0.02, device=None) -> NoiseSchedule`.
  - `forward_diffuse(x0: Tensor, t: Tensor, schedule: NoiseSchedule, noise: Tensor | None = None) -> tuple[Tensor, Tensor]` — returns `(x_t, noise_used)`.
  - `cfg_combine(eps_uncond: Tensor, eps_cond: Tensor, guidance_scale: float) -> Tensor`.
  - `reverse_step(x_t: Tensor, t: int, eps_hat: Tensor, schedule: NoiseSchedule, generator: torch.Generator | None = None) -> Tensor` — one DDPM ancestral-sampling step, `x_t -> x_{t-1}`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_diffusion.py`:

```python
import torch

from diffuse_directly.diffusion import (
    cfg_combine, forward_diffuse, make_schedule, reverse_step,
)


def test_make_schedule_shapes_and_monotonic():
    sched = make_schedule(n_steps=10)
    assert sched.betas.shape == (10,)
    assert sched.alpha_bars.shape == (10,)
    # alpha_bar is a cumulative product of values < 1 -> strictly decreasing
    assert torch.all(sched.alpha_bars[1:] < sched.alpha_bars[:-1])


def test_forward_diffuse_at_t0_is_near_original():
    sched = make_schedule(n_steps=100)
    x0 = torch.randn(5, 4)
    t = torch.zeros(5, dtype=torch.long)
    x_t, noise = forward_diffuse(x0, t, sched, noise=torch.zeros(5, 4))
    assert torch.allclose(x_t, x0, atol=1e-2)


def test_forward_diffuse_at_large_t_is_mostly_noise():
    sched = make_schedule(n_steps=100)
    x0 = torch.zeros(5, 4)
    t = torch.full((5,), 99, dtype=torch.long)
    fixed_noise = torch.ones(5, 4)
    x_t, noise = forward_diffuse(x0, t, sched, noise=fixed_noise)
    # x0=0, so x_t should be ~ sqrt(1-alpha_bar_99) * noise -- close to `noise`
    # scaled, and alpha_bar_99 is tiny after 100 steps of the default schedule.
    assert torch.allclose(x_t, noise, atol=0.15)


def test_cfg_combine_scale_one_is_conditional():
    eps_u = torch.zeros(2, 3)
    eps_c = torch.ones(2, 3)
    out = cfg_combine(eps_u, eps_c, guidance_scale=1.0)
    assert torch.allclose(out, eps_c)


def test_cfg_combine_extrapolates_above_one():
    eps_u = torch.zeros(2, 3)
    eps_c = torch.ones(2, 3)
    out = cfg_combine(eps_u, eps_c, guidance_scale=2.0)
    assert torch.allclose(out, 2 * eps_c)


def test_reverse_step_shape_preserved():
    sched = make_schedule(n_steps=20)
    x_t = torch.randn(6, 4)
    eps_hat = torch.randn(6, 4)
    gen = torch.Generator().manual_seed(0)
    x_prev = reverse_step(x_t, t=10, eps_hat=eps_hat, schedule=sched, generator=gen)
    assert x_prev.shape == x_t.shape


def test_reverse_step_at_t0_is_deterministic_no_noise_added():
    sched = make_schedule(n_steps=20)
    x_t = torch.randn(3, 4)
    eps_hat = torch.randn(3, 4)
    out_a = reverse_step(x_t, t=0, eps_hat=eps_hat, schedule=sched)
    out_b = reverse_step(x_t, t=0, eps_hat=eps_hat, schedule=sched)
    assert torch.allclose(out_a, out_b)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_diffusion.py -q`
Expected: FAIL (ModuleNotFoundError).

- [ ] **Step 3: Write the implementation**

Create `src/diffuse_directly/diffusion.py`:

```python
"""DDPM noise schedule, forward diffusion, classifier-free guidance
combination, and the ancestral reverse-sampling step."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class NoiseSchedule:
    n_steps: int
    betas: torch.Tensor
    alphas: torch.Tensor
    alpha_bars: torch.Tensor


def make_schedule(n_steps: int = 100, beta_start: float = 1e-4,
                  beta_end: float = 0.02, device=None) -> NoiseSchedule:
    betas = torch.linspace(beta_start, beta_end, n_steps, device=device)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return NoiseSchedule(n_steps=n_steps, betas=betas, alphas=alphas, alpha_bars=alpha_bars)


def forward_diffuse(x0: torch.Tensor, t: torch.Tensor, schedule: NoiseSchedule,
                    noise: Optional[torch.Tensor] = None):
    """x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * noise. `t` is a
    (n,) long tensor of per-row timesteps."""
    if noise is None:
        noise = torch.randn_like(x0)
    ab = schedule.alpha_bars[t].unsqueeze(-1)          # (n, 1)
    x_t = ab.sqrt() * x0 + (1 - ab).sqrt() * noise
    return x_t, noise


def cfg_combine(eps_uncond: torch.Tensor, eps_cond: torch.Tensor,
               guidance_scale: float) -> torch.Tensor:
    """eps_hat = eps_uncond + guidance_scale * (eps_cond - eps_uncond).
    guidance_scale=1.0 reduces to the plain conditional model; >1 extrapolates
    the conditioning direction (stronger CFG)."""
    return eps_uncond + guidance_scale * (eps_cond - eps_uncond)


@torch.no_grad()
def reverse_step(x_t: torch.Tensor, t: int, eps_hat: torch.Tensor,
                 schedule: NoiseSchedule,
                 generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """One DDPM ancestral-sampling step: x_t -> x_{t-1}, scalar timestep `t`
    shared across the whole batch (as used by the reconstruction/generation
    loops, which step down from a fixed t)."""
    beta_t = schedule.betas[t]
    alpha_t = schedule.alphas[t]
    alpha_bar_t = schedule.alpha_bars[t]

    mean = (1.0 / alpha_t.sqrt()) * (x_t - (beta_t / (1 - alpha_bar_t).sqrt()) * eps_hat)
    if t == 0:
        return mean
    sigma_t = beta_t.sqrt()
    noise = torch.randn(x_t.shape, generator=generator, device=x_t.device)
    return mean + sigma_t * noise
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_diffusion.py -q`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add src/diffuse_directly/diffusion.py tests/test_diffusion.py
git commit -m "feat: DDPM schedule, forward diffusion, CFG combine, reverse step

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: Training loop

**Files:**
- Create: `src/diffuse_directly/train.py`
- Test: `tests/test_train.py`

**Interfaces:**
- Consumes: `ScoreMLP` (Task 4), `make_schedule`/`forward_diffuse` (Task 5).
- Produces:
  - `TrainConfig(n_steps=100, hidden=128, epochs=200, batch_size=64, lr=1e-3, p_uncond=0.1, seed=0)` (dataclass).
  - `train_score_model(rare_white: np.ndarray, common_white: np.ndarray, config: TrainConfig, device) -> tuple[ScoreMLP, NoiseSchedule]`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_train.py`:

```python
import numpy as np
import torch

from diffuse_directly.device import get_device
from diffuse_directly.train import TrainConfig, train_score_model


def test_train_score_model_returns_net_and_schedule():
    rng = np.random.default_rng(0)
    rare = rng.normal(0, 1, (20, 4))
    common = rng.normal(0, 1, (60, 4))
    cfg = TrainConfig(n_steps=10, hidden=16, epochs=5, batch_size=16, seed=0)
    net, sched = train_score_model(rare, common, cfg, device=get_device("cpu"))
    assert sched.n_steps == 10
    x = torch.randn(3, 4)
    t = torch.zeros(3, dtype=torch.long)
    y = torch.zeros(3, dtype=torch.long)
    out = net(x, t, y)
    assert out.shape == (3, 4)


def test_train_score_model_loss_decreases():
    rng = np.random.default_rng(1)
    rare = rng.normal(2.0, 0.3, (30, 3))
    common = rng.normal(0.0, 0.3, (60, 3))
    cfg = TrainConfig(n_steps=20, hidden=32, epochs=1, batch_size=32, seed=1)
    from diffuse_directly.train import _epoch_loss
    net, sched = train_score_model(rare, common, cfg, device=get_device("cpu"))
    # one more epoch's average loss should be lower than a freshly re-run
    # single epoch on a randomly re-initialised net of the same size (sanity
    # that training moved parameters in a loss-reducing direction at all).
    import torch.nn as nn
    from diffuse_directly.score_net import ScoreMLP
    torch.manual_seed(999)
    fresh_net = ScoreMLP(dim=3, n_steps=20, hidden=32).to(get_device("cpu"))
    pooled = np.vstack([rare, common])
    labels = np.array([1] * len(rare) + [0] * len(common))
    fresh_loss = _epoch_loss(fresh_net, pooled, labels, sched, cfg, get_device("cpu"))
    trained_loss = _epoch_loss(net, pooled, labels, sched, cfg, get_device("cpu"))
    assert trained_loss < fresh_loss
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_train.py -q`
Expected: FAIL (ModuleNotFoundError).

- [ ] **Step 3: Write the implementation**

Create `src/diffuse_directly/train.py`:

```python
"""Denoising score-matching training with CFG label dropout: with
probability p_uncond, the true class is replaced by ScoreMLP.NULL_CLASS so
the network learns both the conditional and unconditional score, which is
what cfg_combine() needs at sample time."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from diffuse_directly.diffusion import NoiseSchedule, forward_diffuse, make_schedule
from diffuse_directly.score_net import ScoreMLP

COMMON_CLASS = 0
RARE_CLASS = 1


@dataclass
class TrainConfig:
    n_steps: int = 100
    hidden: int = 128
    epochs: int = 200
    batch_size: int = 64
    lr: float = 1e-3
    p_uncond: float = 0.1
    seed: int = 0


def _epoch_loss(net: ScoreMLP, X: np.ndarray, labels: np.ndarray,
                schedule: NoiseSchedule, config: TrainConfig, device) -> float:
    """Mean denoising loss over one full pass, no gradient step (eval mode)."""
    net.eval()
    x0 = torch.as_tensor(X, dtype=torch.float32, device=device)
    y = torch.as_tensor(labels, dtype=torch.long, device=device)
    with torch.no_grad():
        t = torch.randint(0, schedule.n_steps, (len(X),), device=device)
        x_t, noise = forward_diffuse(x0, t, schedule)
        pred = net(x_t, t, y)
        loss = nn.functional.mse_loss(pred, noise)
    return float(loss.item())


def train_score_model(rare_white: np.ndarray, common_white: np.ndarray,
                      config: TrainConfig, device) -> tuple:
    torch.manual_seed(config.seed)
    X = np.vstack([rare_white, common_white])
    labels = np.array([RARE_CLASS] * len(rare_white) + [COMMON_CLASS] * len(common_white))
    dim = X.shape[1]

    schedule = make_schedule(n_steps=config.n_steps, device=device)
    net = ScoreMLP(dim=dim, n_steps=config.n_steps, hidden=config.hidden).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=config.lr)

    x0_all = torch.as_tensor(X, dtype=torch.float32, device=device)
    y_all = torch.as_tensor(labels, dtype=torch.long, device=device)
    n = len(X)
    rng = np.random.default_rng(config.seed)

    net.train()
    for _ in range(config.epochs):
        perm = rng.permutation(n)
        for start in range(0, n, config.batch_size):
            idx = perm[start:start + config.batch_size]
            idx_t = torch.as_tensor(idx, dtype=torch.long, device=device)
            x0 = x0_all[idx_t]
            y = y_all[idx_t].clone()

            # CFG label dropout: replace with the null class w.p. p_uncond.
            drop_mask = torch.as_tensor(
                rng.random(len(idx)) < config.p_uncond, device=device)
            y[drop_mask] = ScoreMLP.NULL_CLASS

            t = torch.randint(0, schedule.n_steps, (len(idx),), device=device)
            x_t, noise = forward_diffuse(x0, t, schedule)

            pred = net(x_t, t, y)
            loss = nn.functional.mse_loss(pred, noise)

            opt.zero_grad()
            loss.backward()
            opt.step()

    return net, schedule
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_train.py -q`
Expected: 2 passed. (If `test_train_score_model_loss_decreases` is flaky on a given seed, that's a signal to bump `epochs` slightly in the test config, not to loosen the assertion — the point is a real, if noisy, loss decrease.)

- [ ] **Step 5: Commit**

```bash
git add src/diffuse_directly/train.py tests/test_train.py
git commit -m "feat: denoising score-matching training loop with CFG label dropout

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 7: Test-B and Test-B″ feeders

**Files:**
- Create: `src/diffuse_directly/feeder.py`
- Test: `tests/test_feeder.py`

**Interfaces:**
- Consumes: `ScoreMLP` (Task 4), `NoiseSchedule`/`forward_diffuse`/`reverse_step`/`cfg_combine` (Task 5), `COMMON_CLASS`/`RARE_CLASS` (Task 6).
- Produces:
  - `reconstruct_round_trip(model, schedule, x_white: np.ndarray, y_label: int, t_recon: int, device, generator=None) -> np.ndarray` — Test B: forward-diffuse to `t_recon`, then conditionally reverse-denoise back to 0.
  - `generate_cfg_landing(model, schedule, n_samples: int, target_class: int, guidance_scale: float, dim: int, device, generator=None) -> np.ndarray` — Test B″: full reverse process from pure noise, CFG-guided toward `target_class`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_feeder.py`:

```python
import numpy as np
import torch

from diffuse_directly.device import get_device
from diffuse_directly.feeder import generate_cfg_landing, reconstruct_round_trip
from diffuse_directly.train import COMMON_CLASS, RARE_CLASS, TrainConfig, train_score_model


def _tiny_trained_model(seed=0):
    rng = np.random.default_rng(seed)
    rare = rng.normal(3.0, 0.3, (30, 3))
    common = rng.normal(0.0, 0.3, (60, 3))
    cfg = TrainConfig(n_steps=15, hidden=32, epochs=30, batch_size=32, seed=seed)
    net, sched = train_score_model(rare, common, cfg, device=get_device("cpu"))
    return net, sched, rare, common


def test_reconstruct_round_trip_shape():
    net, sched, rare, common = _tiny_trained_model()
    gen = torch.Generator().manual_seed(0)
    recon = reconstruct_round_trip(
        net, sched, rare, y_label=RARE_CLASS, t_recon=5,
        device=get_device("cpu"), generator=gen,
    )
    assert recon.shape == rare.shape


def test_reconstruct_round_trip_stays_near_original_at_small_t_recon():
    net, sched, rare, common = _tiny_trained_model(seed=2)
    gen = torch.Generator().manual_seed(2)
    # a very small t_recon adds almost no noise, so even an undertrained net
    # should reconstruct close to the original (this is a sanity check on the
    # forward/reverse plumbing, not on model quality).
    recon = reconstruct_round_trip(
        net, sched, rare, y_label=RARE_CLASS, t_recon=1,
        device=get_device("cpu"), generator=gen,
    )
    assert np.linalg.norm(recon - rare) / np.linalg.norm(rare) < 1.0


def test_generate_cfg_landing_shape():
    net, sched, rare, common = _tiny_trained_model(seed=3)
    gen = torch.Generator().manual_seed(3)
    guided = generate_cfg_landing(
        net, sched, n_samples=15, target_class=RARE_CLASS,
        guidance_scale=2.0, dim=3, device=get_device("cpu"), generator=gen,
    )
    assert guided.shape == (15, 3)


def test_generate_cfg_landing_is_finite():
    net, sched, rare, common = _tiny_trained_model(seed=4)
    gen = torch.Generator().manual_seed(4)
    guided = generate_cfg_landing(
        net, sched, n_samples=10, target_class=RARE_CLASS,
        guidance_scale=1.5, dim=3, device=get_device("cpu"), generator=gen,
    )
    assert np.all(np.isfinite(guided))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_feeder.py -q`
Expected: FAIL (ModuleNotFoundError).

- [ ] **Step 3: Write the implementation**

Create `src/diffuse_directly/feeder.py`:

```python
"""Test B (reconstruction round-trip) and Test B'' (CFG generative landing)
feeders -- the two localization.py inputs this package exists to produce."""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from diffuse_directly.diffusion import NoiseSchedule, cfg_combine, forward_diffuse, reverse_step
from diffuse_directly.score_net import ScoreMLP


@torch.no_grad()
def reconstruct_round_trip(model: ScoreMLP, schedule: NoiseSchedule,
                          x_white: np.ndarray, y_label: int, t_recon: int,
                          device, generator: Optional[torch.Generator] = None) -> np.ndarray:
    """Test B: diffuse x0 forward to `t_recon` (a partial, not full, noise
    level -- this tests LOCAL denoising fidelity near existing points, per
    the 2026-07-02 diagram's Test B description), then conditionally
    reverse-denoise back to 0 using the TRUE label (no CFG -- Test B is a
    reconstruction check, not a generation check)."""
    model.eval()
    x0 = torch.as_tensor(x_white, dtype=torch.float32, device=device)
    n = len(x_white)
    t0 = torch.full((n,), t_recon, dtype=torch.long, device=device)
    x_t, _ = forward_diffuse(x0, t0, schedule)

    y = torch.full((n,), y_label, dtype=torch.long, device=device)
    for t in range(t_recon, -1, -1):
        t_batch = torch.full((n,), t, dtype=torch.long, device=device)
        eps_hat = model(x_t, t_batch, y)
        x_t = reverse_step(x_t, t, eps_hat, schedule, generator=generator)
    return x_t.cpu().numpy()


@torch.no_grad()
def generate_cfg_landing(model: ScoreMLP, schedule: NoiseSchedule, n_samples: int,
                         target_class: int, guidance_scale: float, dim: int,
                         device, generator: Optional[torch.Generator] = None) -> np.ndarray:
    """Test B'': full reverse process from pure noise x_T ~ N(0,I), CFG-guided
    toward `target_class` at every step. These are HELD-OUT generated
    samples, never round-tripped from real data -- generation, not denoising."""
    model.eval()
    x_t = torch.randn(n_samples, dim, generator=generator, device=device)
    target = torch.full((n_samples,), target_class, dtype=torch.long, device=device)
    null = torch.full((n_samples,), ScoreMLP.NULL_CLASS, dtype=torch.long, device=device)

    for t in range(schedule.n_steps - 1, -1, -1):
        t_batch = torch.full((n_samples,), t, dtype=torch.long, device=device)
        eps_cond = model(x_t, t_batch, target)
        eps_uncond = model(x_t, t_batch, null)
        eps_hat = cfg_combine(eps_uncond, eps_cond, guidance_scale)
        x_t = reverse_step(x_t, t, eps_hat, schedule, generator=generator)
    return x_t.cpu().numpy()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_feeder.py -q`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/diffuse_directly/feeder.py tests/test_feeder.py
git commit -m "feat: Test-B reconstruction round-trip + Test-B'' CFG generative landing

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 8: Diagnostic runner + artifact persistence

**Files:**
- Create: `src/diffuse_directly/run.py`
- Test: `tests/test_run.py`

**Interfaces:**
- Consumes: `EmbeddingSource` (Task 2), `whiten_pooled`/`unwhiten`/`WhiteningParams` (Task 3), `TrainConfig`/`train_score_model`/`COMMON_CLASS`/`RARE_CLASS` (Task 6), `reconstruct_round_trip`/`generate_cfg_landing` (Task 7), `causal_bench.diagnostics.localization.run_diagnostic`.
- Produces:
  - `FeederResult(report, model_state: dict, whitening: WhiteningParams, schedule_config: dict, strata: dict)` (dataclass). `strata = {"rare_idx": np.ndarray, "common_idx": np.ndarray}` — index membership for the future twist project's tail-stratified ESS (spec §7 seam).
  - `run_feeder(source: EmbeddingSource, train_config: TrainConfig, t_recon: int, guidance_scale: float, device=None, artifact_dir: str | None = None) -> FeederResult`.
  - `save_artifacts(result: FeederResult, artifact_dir: str) -> None`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_run.py`:

```python
import numpy as np
import torch

from diffuse_directly.device import get_device
from diffuse_directly.run import FeederResult, run_feeder, save_artifacts
from diffuse_directly.sources import PlantedConfig, PlantedSource
from diffuse_directly.train import TrainConfig


def _tiny_config():
    return TrainConfig(n_steps=15, hidden=32, epochs=30, batch_size=32, seed=0)


def test_run_feeder_produces_report_and_strata():
    source = PlantedSource(PlantedConfig(n_rare=30, n_common=80, dim=4,
                                          separation=4.0, seed=0))
    result = run_feeder(source, _tiny_config(), t_recon=3, guidance_scale=2.0,
                        device=get_device("cpu"))
    assert isinstance(result, FeederResult)
    assert result.report.terminal in (
        "diffuse_directly", "smc_required", "pending_B_prime",
        "pending_cfg_landing_check",
    )
    assert result.strata["rare_idx"].shape == (30,)
    assert result.strata["common_idx"].shape == (80,)


def test_run_feeder_persists_artifacts(tmp_path):
    source = PlantedSource(PlantedConfig(n_rare=20, n_common=60, dim=4,
                                          separation=4.0, seed=1))
    result = run_feeder(source, _tiny_config(), t_recon=3, guidance_scale=2.0,
                        device=get_device("cpu"))
    save_artifacts(result, str(tmp_path))
    assert (tmp_path / "model.pt").exists()
    assert (tmp_path / "whitening.npz").exists()
    assert (tmp_path / "strata.npz").exists()

    loaded_strata = np.load(tmp_path / "strata.npz")
    assert np.array_equal(loaded_strata["rare_idx"], result.strata["rare_idx"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_run.py -q`
Expected: FAIL (ModuleNotFoundError).

- [ ] **Step 3: Write the implementation**

Create `src/diffuse_directly/run.py`:

```python
"""Full feeder pipeline: EmbeddingSource -> whiten -> train -> Test B ->
Test B'' -> run_diagnostic -> persisted artifacts.

Any validity filter a future version applies to generated samples MUST
record a survival probability rather than silently discarding samples (spec
sec 7, IPCW every-kill rule) -- no such filter exists yet in this toy/
embedding-only feeder, so this is a documented seam, not implemented here.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from causal_bench.diagnostics.localization import run_diagnostic

from diffuse_directly.feeder import generate_cfg_landing, reconstruct_round_trip
from diffuse_directly.sources import EmbeddingSource
from diffuse_directly.train import COMMON_CLASS, RARE_CLASS, TrainConfig, train_score_model
from diffuse_directly.whitening import WhiteningParams, unwhiten, whiten_pooled


@dataclass
class FeederResult:
    report: object                  # causal_bench.diagnostics.localization.DiagnosticReport
    model_state: dict
    whitening: WhiteningParams
    schedule_config: dict
    strata: dict                    # {"rare_idx": np.ndarray, "common_idx": np.ndarray}


def run_feeder(source: EmbeddingSource, train_config: TrainConfig, t_recon: int,
              guidance_scale: float, device=None,
              artifact_dir: Optional[str] = None) -> FeederResult:
    if device is None:
        from diffuse_directly.device import get_device
        device = get_device("auto")

    rare, common = source.sample()
    rare_white, common_white, params = whiten_pooled(rare, common)

    net, schedule = train_score_model(rare_white, common_white, train_config, device)

    gen = torch.Generator(device="cpu").manual_seed(train_config.seed)

    # Test B: reconstruction round-trip, per mode.
    rare_recon_w = reconstruct_round_trip(
        net, schedule, rare_white, y_label=RARE_CLASS, t_recon=t_recon,
        device=device, generator=gen)
    common_recon_w = reconstruct_round_trip(
        net, schedule, common_white, y_label=COMMON_CLASS, t_recon=t_recon,
        device=device, generator=gen)
    recon_b = (unwhiten(rare_recon_w, params), unwhiten(common_recon_w, params))

    # Test B'': CFG generative landing. common_ref is the real held-out
    # common set (per spec sec 3 -- generation is only required for rare).
    rare_guided_w = generate_cfg_landing(
        net, schedule, n_samples=len(rare), target_class=RARE_CLASS,
        guidance_scale=guidance_scale, dim=rare_white.shape[1],
        device=device, generator=gen)
    cfg_landing = (unwhiten(rare_guided_w, params), common)

    report = run_diagnostic(
        rare, common,
        recon_b=recon_b,
        cfg_landing=cfg_landing,
    )

    strata = {
        "rare_idx": np.arange(len(rare)),
        "common_idx": np.arange(len(rare), len(rare) + len(common)),
    }
    schedule_config = {
        "n_steps": schedule.n_steps,
        "t_recon": t_recon,
        "guidance_scale": guidance_scale,
    }

    result = FeederResult(
        report=report,
        model_state=net.state_dict(),
        whitening=params,
        schedule_config=schedule_config,
        strata=strata,
    )
    if artifact_dir is not None:
        save_artifacts(result, artifact_dir)
    return result


def save_artifacts(result: FeederResult, artifact_dir: str) -> None:
    os.makedirs(artifact_dir, exist_ok=True)
    torch.save(result.model_state, os.path.join(artifact_dir, "model.pt"))
    np.savez(os.path.join(artifact_dir, "whitening.npz"),
             W=result.whitening.W, mu=result.whitening.mu)
    np.savez(os.path.join(artifact_dir, "strata.npz"),
             rare_idx=result.strata["rare_idx"],
             common_idx=result.strata["common_idx"])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_run.py -q`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/diffuse_directly/run.py tests/test_run.py
git commit -m "feat: run_feeder glue (source->whiten->train->B->B''->diagnostic) + artifact persistence

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 9: Validation suite — the three known-answer toy cases

**Files:**
- Create: `tests/test_validation_suite.py`

**Interfaces:**
- Consumes: everything from Tasks 1–8. No new production code — this task is the spec §5 regression suite that proves the whole pipeline (and specifically Test B'') behaves correctly end-to-end.

- [ ] **Step 1: Write the easy case (well-separated rare, non-thin) — expect `diffuse_directly`**

Create `tests/test_validation_suite.py`:

```python
"""Spec sec 5 validation suite: planted known-answer cases proving the
end-to-end feeder (and specifically that Test B'' earns its keep) behaves
correctly. Runs entirely on CPU -- no GPU/encoder needed."""
from diffuse_directly.device import get_device
from diffuse_directly.run import run_feeder
from diffuse_directly.sources import PlantedConfig, PlantedSource
from diffuse_directly.train import TrainConfig

_DEVICE = get_device("cpu")


def _config(epochs=150, hidden=64, n_steps=25):
    return TrainConfig(n_steps=n_steps, hidden=hidden, epochs=epochs,
                       batch_size=32, seed=0)


def test_easy_well_separated_case_reaches_diffuse_directly():
    # Well-separated (sep=5), non-thin (scale=0.5) rare cluster: standard
    # diffusion should both reconstruct AND generatively land in it.
    source = PlantedSource(PlantedConfig(
        n_rare=40, n_common=150, dim=6, separation=5.0,
        rare_scale=0.5, common_scale=0.5, seed=10))
    result = run_feeder(source, _config(), t_recon=schedule_t_recon(_config()),
                        guidance_scale=3.0, device=_DEVICE)
    assert result.report.terminal == "diffuse_directly"


def schedule_t_recon(config: TrainConfig) -> int:
    """A mid-schedule reconstruction point: enough noise to be a real test,
    not so much the round-trip degenerates into full regeneration."""
    return config.n_steps // 2
```

- [ ] **Step 2: Run test to verify it fails or passes as expected**

Run: `cd /Users/noahrahman/git/diffuse_directly && .venv/bin/python -m pytest tests/test_validation_suite.py -q`
Expected: this first test should PASS given the pipeline built in Tasks 1-8 (there is no new implementation code in this task — if it fails, the bug is in an earlier task, most likely undertrained defaults; try raising `epochs` in `_config()` before touching anything else).

- [ ] **Step 3: Add the reconstruction-faithful-but-CFG-fails case — expect `smc_required`**

Append to `tests/test_validation_suite.py`:

```python
def test_reconstruction_faithful_but_cfg_fails_requires_smc():
    # A rare cluster close enough to the bulk that reconstruction (which only
    # denoises near existing points) is easy, but generation from pure noise
    # under CFG tends to collapse back toward the (much larger) common mode.
    # This is the case Test B'' exists to catch: reconstruction alone would
    # have wrongly certified diffuse_directly.
    source = PlantedSource(PlantedConfig(
        n_rare=15, n_common=300, dim=6, separation=1.2,
        rare_scale=0.3, common_scale=1.0, seed=11))
    # A weak guidance_scale is the mechanism that lets this case demonstrate
    # collapse: strong CFG (as in the easy case) can rescue even a close
    # cluster, which would defeat the point of this fixture.
    result = run_feeder(source, _config(epochs=80), t_recon=schedule_t_recon(_config()),
                        guidance_scale=1.0, device=_DEVICE)
    assert result.report.terminal in ("smc_required", "pending_cfg_landing_check")
    # if this lands on pending_cfg_landing_check the cfg_landing samples were
    # not supplied for some reason -- that indicates a wiring bug in run.py,
    # not a modeling issue, since run_feeder always supplies cfg_landing.
```

- [ ] **Step 4: Run to verify; tune the fixture if it lands on `diffuse_directly` instead**

Run: `.venv/bin/python -m pytest tests/test_validation_suite.py::test_reconstruction_faithful_but_cfg_fails_requires_smc -q`
Expected: PASS with terminal `smc_required`. If the assertion fails because the terminal is `diffuse_directly` (guidance rescued the case despite `guidance_scale=1.0`), lower `separation` further (e.g., 0.8) or increase `common_scale` relative to `rare_scale` — the goal is a cluster CFG cannot reliably land in in a short (25-step) schedule at weak guidance. Do not raise `guidance_scale` to force the failure — that would make the fixture generation-mechanism-dependent rather than geometry-dependent, defeating its purpose as a known-answer case.

- [ ] **Step 5: Add the tail-collapse case — expect `pending_B_prime`**

Append to `tests/test_validation_suite.py`:

```python
def test_thin_tail_collapse_case_reaches_pending_b_prime():
    # A very thin (small rare_scale), small rare cluster: standard-trained
    # diffusion's reconstruction itself collapses the tail toward the bulk
    # (the noise schedule washes out a low-density filament), so Test B
    # fails before Test B'' is ever reached.
    source = PlantedSource(PlantedConfig(
        n_rare=8, n_common=300, dim=6, separation=3.0,
        rare_scale=0.05, common_scale=1.0, seed=12))
    result = run_feeder(source, _config(epochs=60), t_recon=schedule_t_recon(_config()),
                        guidance_scale=2.0, device=_DEVICE)
    assert result.report.terminal == "pending_B_prime"
```

- [ ] **Step 6: Run the full validation suite**

Run: `.venv/bin/python -m pytest tests/test_validation_suite.py -v`
Expected: 3 passed. Wall clock: well under a minute total on CPU (tiny toy configs). If any test is flaky across re-runs with the same seed, the fixture (separation/scale/n_rare) needs adjusting, not the assertion — these are meant to be known-answer, deterministic-given-seed cases.

- [ ] **Step 7: Run the full repo test suite**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: all tests from Tasks 1-9 pass (device, sources, whitening, score_net, diffusion, train, feeder, run, validation_suite).

- [ ] **Step 8: Commit**

```bash
git add tests/test_validation_suite.py
git commit -m "test: spec sec 5 validation suite -- the 3 known-answer toy cases

Proves Test B'' earns its keep: the reconstruction-faithful-but-cfg-fails
case would have been wrongly certified diffuse_directly without it.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Self-Review (completed at plan-writing time)

- **Spec coverage:** §3 EmbeddingSource/PlantedSource → Task 2; whitening → Task 3; base score diffusion → Tasks 4-6; Test B → Task 7; Test B″ → Task 7; diagnostic glue → Task 8; persisted artifacts (sampler + strata) → Task 8; §4 data flow → the full Task 1-8 chain matches the diagram exactly (source→whiten→train→B→B″→run_diagnostic); §5 validation → Task 9 (all three cases); §6 testing → every task has its test step, Task 9 is the CPU/MPS-runnable regression suite. §7 (downstream twist seams) is deliberately NOT implemented — Task 8's `strata` output and `model_state` persistence are the only seams this plan builds, matching the spec's "task-2 seams" callouts; the IPCW kill-rule seam is documented in `run.py`'s docstring rather than code, since no validity filter exists yet to apply it to (correctly out of scope). `SMBSource` (§3) is explicitly deferred (interface-level only, not built) — matches spec.
- **Placeholder scan:** none; every step has complete, runnable code.
- **Type consistency:** `WhiteningParams` (Task 3) consumed identically in Task 8's `run_feeder`; `TrainConfig`/`COMMON_CLASS`/`RARE_CLASS` (Task 6) match their use in Tasks 7-9; `ScoreMLP.NULL_CLASS` (Task 4) used correctly in Task 7's `generate_cfg_landing` and Task 6's label-dropout; `NoiseSchedule` (Task 5) threaded unchanged through Tasks 6-8; `EmbeddingSource.sample() -> (rare, common)` (Task 2) matches `run_feeder`'s first call.
