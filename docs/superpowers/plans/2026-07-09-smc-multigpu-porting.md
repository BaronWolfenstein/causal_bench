# SMC → Multi-GPU Porting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the numpy-first SMC/IPCW hot loop to a device-agnostic array namespace so `run_smc(device="cuda")` becomes a validated end-to-end path, and add the gating + low-communication resampling scaffolding the on-box multi-GPU run needs.

**Architecture:** Keep every hot-loop function device-agnostic by inferring its array namespace (numpy or cupy) *from the arrays it receives* — no `device` string threaded through internals. `run_smc` already converts inputs on-device and returns host numpy at the edges; this plan pushes `xp` into the four functions that still call bare `np.*`. The multi-GPU barrier stays at the collective/library layer (NCCL + CUB via cupy) — no custom kernels. The numpy oracle in `sampling/sharded.py` remains the reference the real distributed run must match byte-for-byte.

**Tech Stack:** Python, numpy (reference), `cupy-cuda12x` (device), `torch.distributed`/NCCL (collectives, on-box only), pytest.

## Global Constraints

- **Collective/library layer only** — no custom CUDA kernels, cooperative groups, grid/warp/block sync, or CPU memory fences. Systematic resampling stays `xp.cumsum` + `xp.searchsorted` (cupy backs these with CUB).
- **Lazy GPU imports** — never import `cupy` or `torch` at module top level; import inside functions so CPU-only installs and CI stay torch/cupy-free.
- **Host RNG is load-bearing** — the resampling RNG stays a host `numpy.random.Generator` producing scalar draws. Identical seed → byte-identical indices across ranks is the invariant `sharded.py` asserts; moving the RNG on-device would break it.
- **CPU path must never regress** — `device="cpu"` uses numpy and every existing `tests/test_smc_*.py` must stay green after each task.
- **`run_smc` returns host numpy regardless of device** (already true via `to_numpy`) so estimators/callers stay device-agnostic.
- Real `cuda==cpu` numerical parity and distributed==serial validation are **on-box** acceptance steps; CI covers them with `pytest.importorskip`, mirroring the repo's existing `bayes`-extra skip pattern.

---

### Task 1: `[gpu]` optional-dependency extra

**Files:**
- Modify: `pyproject.toml:25-` (`[project.optional-dependencies]`)
- Test: `tests/test_gpu_extra.py`

**Interfaces:**
- Produces: a `gpu` extra installing `cupy-cuda12x`; no import-time coupling (nothing in the package imports cupy at module top level).

- [ ] **Step 1: Write the failing test** (guards the lazy-import discipline — importing the sampling package must not drag in cupy)

```python
# tests/test_gpu_extra.py
import sys
import tomllib
from pathlib import Path


def test_gpu_extra_declares_cupy():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())
    gpu = pyproject["project"]["optional-dependencies"]["gpu"]
    assert any(dep.startswith("cupy-cuda12x") for dep in gpu)


def test_importing_sampling_does_not_import_cupy():
    for mod in [m for m in sys.modules if m.startswith("cupy")]:
        del sys.modules[mod]
    import causal_bench.sampling.smc  # noqa: F401
    import causal_bench.sampling.backend  # noqa: F401
    assert not any(m.startswith("cupy") for m in sys.modules), \
        "cupy imported at module load — must stay lazy"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_gpu_extra.py -v`
Expected: FAIL on `test_gpu_extra_declares_cupy` with `KeyError: 'gpu'`.

- [ ] **Step 3: Add the extra**

In `pyproject.toml`, under `[project.optional-dependencies]`, add:

```toml
gpu = ["cupy-cuda12x>=12"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_gpu_extra.py -v`
Expected: PASS (both). If `test_importing_sampling_does_not_import_cupy` fails, a module added a top-level `import cupy` — fix that, don't weaken the test.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml tests/test_gpu_extra.py
git commit -m "feat(gpu): add [gpu] extra (cupy-cuda12x), guard lazy import"
```

---

### Task 2: Array-inferred namespace helper

**Files:**
- Modify: `causal_bench/sampling/backend.py`
- Test: `tests/test_smc_backend.py` (extend)

**Interfaces:**
- Consumes: existing `array_namespace(device)`, `to_numpy(x)`.
- Produces: `get_namespace(*arrays) -> module` — returns `cupy` if any argument is a cupy array, else `numpy`. This is how the hot-loop functions in Tasks 3–5 obtain `xp` without a `device` parameter, satisfying spec §1a ("route through `xp`").

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_smc_backend.py
def test_get_namespace_returns_numpy_for_numpy_arrays():
    import numpy as np
    from causal_bench.sampling.backend import get_namespace
    assert get_namespace(np.zeros(3)) is np
    assert get_namespace(np.zeros(3), np.ones(2)) is np
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_smc_backend.py::test_get_namespace_returns_numpy_for_numpy_arrays -v`
Expected: FAIL with `ImportError: cannot import name 'get_namespace'`.

- [ ] **Step 3: Implement `get_namespace`**

Add to `causal_bench/sampling/backend.py`:

```python
def get_namespace(*arrays):
    """Infer the array namespace (numpy or cupy) from the arrays themselves,
    so hot-loop functions stay device-agnostic without threading a `device`
    string through their signatures. Mirrors `to_numpy`'s module sniff.
    Satisfies spec §1a ('route through xp = array_namespace(device)')."""
    for a in arrays:
        if type(a).__module__.startswith("cupy"):
            import cupy as cp            # lazy: only on the GPU box
            return cp
    return np
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_smc_backend.py -v`
Expected: PASS (all backend tests).

- [ ] **Step 5: Commit**

```bash
git add causal_bench/sampling/backend.py tests/test_smc_backend.py
git commit -m "feat(sampling): add get_namespace array-inferred backend helper"
```

---

### Task 3: Port `weights.py` to the array namespace

**Files:**
- Modify: `causal_bench/sampling/weights.py`
- Test: `tests/test_smc_weights.py` (extend), regression via existing tests

**Interfaces:**
- Consumes: `backend.get_namespace`.
- Produces: `normalize_log_weights(log_w) -> (weights, float)` and `kish_ess(log_w) -> float` operating on numpy **or** cupy arrays; return types unchanged (device array for weights, host `float` for the scalars).

- [ ] **Step 1: Write the failing test** (proves the functions no longer hard-depend on `np` by driving them through a non-numpy duck namespace via a spy)

```python
# append to tests/test_smc_weights.py
import numpy as np
from causal_bench.sampling.weights import normalize_log_weights, kish_ess


def test_normalize_uses_inferred_namespace_not_bare_np(monkeypatch):
    import causal_bench.sampling.weights as W
    calls = {"n": 0}
    real = W.get_namespace
    def spy(*a):
        calls["n"] += 1
        return real(*a)
    monkeypatch.setattr(W, "get_namespace", spy)
    w, log_norm = normalize_log_weights(np.array([-1.0, -2.0, -3.0]))
    assert calls["n"] >= 1                     # went through get_namespace
    assert np.isclose(w.sum(), 1.0)
    assert np.isfinite(log_norm)


def test_kish_ess_uniform_equals_n():
    assert np.isclose(kish_ess(np.zeros(10)), 10.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_smc_weights.py::test_normalize_uses_inferred_namespace_not_bare_np -v`
Expected: FAIL — `weights` has no name `get_namespace` yet (AttributeError in `monkeypatch.setattr`).

- [ ] **Step 3: Port the implementations**

Replace the bodies in `causal_bench/sampling/weights.py` (add the import, route through `xp`; keep the loud all-`-inf` raise):

```python
"""Log-space weight normalization and Kish effective sample size."""
from __future__ import annotations

import numpy as np

from .backend import get_namespace


def normalize_log_weights(log_w) -> tuple:
    """Return (normalized weights, log normalizer). Subtract the max before
    exponentiating so weights never under/overflow. numpy or cupy in."""
    xp = get_namespace(log_w)
    m = xp.max(log_w)
    if not bool(xp.isfinite(m)):
        # all -inf (total weight collapse / positivity failure) or a +inf leaked
        # in — surface it loudly rather than returning silent nan weights.
        raise ValueError(
            "normalize_log_weights: non-finite max log-weight "
            f"({float(m)}) — total weight collapse (every particle out of "
            "support). This is a positivity failure; fix upstream (twist "
            "earlier), do not reweight."
        )
    shifted = xp.exp(log_w - m)
    total = shifted.sum()
    log_norm = m + xp.log(total)
    return shifted / total, float(log_norm)


def kish_ess(log_w) -> float:
    """Kish ESS = (sum w)^2 / sum(w^2) = 1 / sum(w_norm^2)."""
    xp = get_namespace(log_w)
    w, _ = normalize_log_weights(log_w)
    return float(1.0 / xp.sum(w ** 2))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_smc_weights.py -v`
Expected: PASS. Then regression: `pytest tests/test_smc_loop.py tests/test_smc_diagnostics.py -q` — all green (CPU path unchanged).

- [ ] **Step 5: Commit**

```bash
git add causal_bench/sampling/weights.py tests/test_smc_weights.py
git commit -m "feat(sampling): port weights.py to array namespace (§1a)"
```

---

### Task 4: Port `resample.py` to the array namespace

**Files:**
- Modify: `causal_bench/sampling/resample.py`
- Test: `tests/test_smc_resample.py` (extend), regression via `tests/test_smc_sharded.py`

**Interfaces:**
- Consumes: `backend.get_namespace`, `weights.kish_ess`.
- Produces: `systematic_resample(w, rng) -> int64 ancestor indices` (numpy or cupy, matching `w`); `should_resample(log_w, ess_frac=0.5) -> bool` (host bool, unchanged). **The `rng` stays a host numpy Generator** — one scalar draw per call — preserving the shared-seed byte-identical invariant.

- [ ] **Step 1: Write the failing test** (namespace routed; determinism under a shared seed preserved — the sharded invariant depends on it)

```python
# append to tests/test_smc_resample.py
import numpy as np
from causal_bench.sampling.resample import systematic_resample


def test_systematic_resample_uses_inferred_namespace(monkeypatch):
    import causal_bench.sampling.resample as R
    calls = {"n": 0}
    real = R.get_namespace
    def spy(*a):
        calls["n"] += 1
        return real(*a)
    monkeypatch.setattr(R, "get_namespace", spy)
    w = np.full(8, 1 / 8)
    idx = systematic_resample(w, np.random.default_rng(0))
    assert calls["n"] >= 1
    assert idx.dtype == np.int64
    assert len(idx) == 8


def test_systematic_resample_shared_seed_is_deterministic():
    w = np.random.default_rng(4).random(32); w /= w.sum()
    a = systematic_resample(w, np.random.default_rng(99))
    b = systematic_resample(w, np.random.default_rng(99))
    assert np.array_equal(a, b)                # shared seed => identical indices
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_smc_resample.py::test_systematic_resample_uses_inferred_namespace -v`
Expected: FAIL — `resample` has no `get_namespace` (AttributeError in `monkeypatch.setattr`).

- [ ] **Step 3: Port the implementation**

Edit `causal_bench/sampling/resample.py` — add the import and route `systematic_resample` through `xp` (keep `rng` host; `should_resample` needs no change):

```python
"""Systematic resampling (cumsum + searchsorted — the GPU-parallel primitive)
and the adaptive-resampling trigger. Ancestor indices are the raw material for
the localization diagnostic's lineage-collapse component; callers persist them."""
from __future__ import annotations

import numpy as np

from .backend import get_namespace
from .weights import kish_ess, normalize_log_weights


def systematic_resample(w, rng: np.random.Generator):
    """Return ancestor indices. One HOST uniform draw, N evenly-spaced
    positions, searchsorted into the CDF. O(N), vectorized. numpy or cupy
    in (matches `w`); the rng stays host so a shared seed yields byte-identical
    indices across ranks (the sharded invariant)."""
    xp = get_namespace(w)
    n = len(w)
    positions = (rng.random() + xp.arange(n)) / n    # host scalar broadcasts onto xp
    cdf = xp.cumsum(w)
    cdf[-1] = 1.0                                     # guard fp drift at the top
    return xp.searchsorted(cdf, positions).astype(xp.int64)


def should_resample(log_w, ess_frac: float = 0.5) -> bool:
    """Adaptive resampling: only trigger the barrier when ESS < ess_frac * N.
    Most steps then have no global sync at all. Returns a host bool."""
    return kish_ess(log_w) < ess_frac * len(log_w)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_smc_resample.py tests/test_smc_sharded.py -v`
Expected: PASS. The sharded byte-for-byte test staying green confirms the host-RNG invariant survived the port.

- [ ] **Step 5: Commit**

```bash
git add causal_bench/sampling/resample.py tests/test_smc_resample.py
git commit -m "feat(sampling): port systematic_resample to array namespace, keep host RNG (§1a)"
```

---

### Task 5: Port `smc_step` and validate cuda==cpu parity

**Files:**
- Modify: `causal_bench/sampling/smc.py:44-60` (`smc_step`)
- Test: `tests/test_smc_loop.py` (regression), `tests/test_smc_cuda_parity.py` (new, on-box)

**Interfaces:**
- Consumes: `backend.get_namespace`, ported `weights`/`resample`.
- Produces: `smc_step` fully namespace-generic; `run_smc(device="cuda")` numerically matches `run_smc(device="cpu")` on the same seed (on-box acceptance).

- [ ] **Step 1: Write the on-box parity test** (skips in CI, runs on the box — mirrors the repo's `bayes`-extra skip pattern)

```python
# tests/test_smc_cuda_parity.py
import numpy as np
import pytest


def _run(xp, device, rng):
    from causal_bench.sampling.smc import run_smc
    betas = np.linspace(0, 1, 12)
    mu = xp.asarray([2.0])
    x0 = xp.asarray(np.random.default_rng(0).standard_normal((64, 1)))

    def propagate(x, s):
        step_xp = type(x).__module__.split(".")[0]
        noise = np.random.default_rng(s).standard_normal(x.shape)
        return x + 0.3 * (xp.asarray(noise) if step_xp == "cupy" else noise)

    def log_weight_fn(x, s):
        return (betas[s] - betas[s - 1]) * (
            -0.5 * ((x - mu) ** 2).sum(1) + 0.5 * (x ** 2).sum(1))

    return run_smc(x0, propagate, log_weight_fn, len(betas), rng, device=device)


def test_cuda_matches_cpu_on_same_seed():
    cp = pytest.importorskip("cupy")            # skips off-box; runs on the A100 box
    cpu = _run(np, "cpu", np.random.default_rng(7))
    cuda = _run(cp, "cuda", np.random.default_rng(7))
    assert np.allclose(cpu.state.particles, cuda.state.particles, atol=1e-6)
    assert np.allclose(cpu.ess_trajectory, cuda.ess_trajectory, atol=1e-6)
    assert cpu.resample_steps == cuda.resample_steps
```

- [ ] **Step 2: Run test to verify it fails (or skips off-box)**

Run: `pytest tests/test_smc_cuda_parity.py -v`
Expected off-box: SKIPPED ("could not import 'cupy'"). On-box **before** Step 3: FAIL — `smc_step` still calls `np.zeros`, so the cuda path mixes numpy into cupy state.

- [ ] **Step 3: Port `smc_step`**

In `causal_bench/sampling/smc.py`, edit `smc_step` to infer `xp` and reset weights on-device:

```python
def smc_step(state: SMCState, log_incr, rng, ess_frac: float = 0.5):
    """One reweight → (adaptive) resample. Returns (new_state, resampled?)."""
    from .backend import get_namespace
    xp = get_namespace(state.particles)
    log_w = state.log_weights + log_incr
    if should_resample(log_w, ess_frac):
        w, _ = normalize_log_weights(log_w)
        idx = systematic_resample(w, rng)
        new = SMCState(
            particles=state.particles[idx],
            log_weights=xp.zeros(len(idx)),          # reset to uniform post-resample
            ancestry=idx,
        )
        return new, True
    return SMCState(state.particles, log_w, state.ancestry), False
```

- [ ] **Step 4: Verify regression (CI) and parity (on-box)**

Run (always): `pytest tests/test_smc_loop.py tests/test_smc_diagnostics.py tests/test_smc_ipcw.py -q`
Expected: PASS — the CPU path is unchanged.
Run (on-box, on the A100 box): `pytest tests/test_smc_cuda_parity.py -v`
Expected on-box: PASS — `run_smc(device="cuda")` matches `device="cpu")` to tolerance. **This is the §1a acceptance gate; no `device="cuda"` claim is valid until it passes on the box.**

- [ ] **Step 5: Commit**

```bash
git add causal_bench/sampling/smc.py tests/test_smc_cuda_parity.py
git commit -m "feat(sampling): port smc_step to array namespace; cuda==cpu parity test (§1a)"
```

---

### Task 6: Device resolution and multi-GPU gating

**Files:**
- Create: `causal_bench/sampling/device.py`
- Test: `tests/test_device_gating.py`

**Interfaces:**
- Produces:
  - `resolve_device(prefer="auto") -> str` — single-device torch resolution, `cuda → mps → cpu`; passes a non-`"auto"` value through unchanged.
  - `cuda_available_devices() -> list[int]` — ordered visible CUDA ids; honors `CUDA_VISIBLE_DEVICES`, else `torch.cuda.device_count()`, `[]` on a CPU box. An **empty** list is the signal for Task 8 to fall back to single-process CPU `run_smc`.

- [ ] **Step 1: Write the failing tests** (env-driven paths need no torch, so they run in CI)

```python
# tests/test_device_gating.py
import sys
import types
import pytest


def test_cuda_visible_devices_parsed_in_order(monkeypatch):
    from causal_bench.sampling.device import cuda_available_devices
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,1,2")
    assert cuda_available_devices() == [3, 1, 2]


def test_empty_cuda_visible_devices_means_cpu_box(monkeypatch):
    from causal_bench.sampling.device import cuda_available_devices
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert cuda_available_devices() == []


def test_falls_back_to_torch_device_count(monkeypatch):
    from causal_bench.sampling.device import cuda_available_devices
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    fake = types.ModuleType("torch")
    fake.cuda = types.SimpleNamespace(device_count=lambda: 4)
    monkeypatch.setitem(sys.modules, "torch", fake)
    assert cuda_available_devices() == [0, 1, 2, 3]


def test_resolve_device_passthrough():
    from causal_bench.sampling.device import resolve_device
    assert resolve_device("cpu") == "cpu"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_device_gating.py -v`
Expected: FAIL — `ModuleNotFoundError: causal_bench.sampling.device`.

- [ ] **Step 3: Implement `device.py`**

```python
# causal_bench/sampling/device.py
"""Device resolution (single-device torch) and multi-GPU gating (which CUDA
ids are visible). Both lazy-import torch so CPU-only installs/CI stay torch-free.
An empty cuda_available_devices() is the signal to fall back to CPU run_smc."""
from __future__ import annotations

import os


def resolve_device(prefer: str = "auto") -> str:
    """Single-device torch resolution: cuda → mps → cpu. A non-'auto' value
    passes through unchanged so callers can pin explicitly."""
    if prefer != "auto":
        return prefer
    import torch                                   # lazy
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def cuda_available_devices() -> list[int]:
    """Ordered visible CUDA device ids, or [] on a CPU box. Honors
    CUDA_VISIBLE_DEVICES (respecting its order); otherwise torch.cuda.device_count().
    Never raises on a CPU box — returns []."""
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env is not None:
        env = env.strip()
        if env == "":
            return []
        return [int(x) for x in env.split(",") if x.strip() != ""]
    try:
        import torch                               # lazy
        return list(range(torch.cuda.device_count()))
    except Exception:
        return []
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_device_gating.py -v`
Expected: PASS (all four).

- [ ] **Step 5: Commit**

```bash
git add causal_bench/sampling/device.py tests/test_device_gating.py
git commit -m "feat(sampling): resolve_device + cuda_available_devices gating (§1c)"
```

---

### Task 7: Island (local) resampling oracle — the low-communication default

**Files:**
- Modify: `causal_bench/sampling/sharded.py`
- Test: `tests/test_smc_sharded.py` (extend)

**Interfaces:**
- Consumes: `resample.systematic_resample`.
- Produces: `island_resample(w, k, seed) -> global ancestor indices` — each of `k` ranks resamples **only its own sub-population** from renormalized local weights (no all-to-all). The distinguishing invariant vs `sharded_systematic_resample`: **no index ever leaves its island's owned range** (§1d — trades a small bias/variance cost to remove the cross-rank particle exchange). Per-island RNGs are independent (`seed + rank`) so islands don't correlate.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_smc_sharded.py
from causal_bench.sampling.sharded import island_resample


def test_island_resample_conserves_count_and_stays_local():
    w = np.random.default_rng(2).random(60); w /= w.sum()
    k = 3
    idx = island_resample(w, k=k, seed=7)
    assert len(idx) == 60                                  # N in == N out
    bounds = np.array_split(np.arange(60), k)
    off = 0
    for b in bounds:
        seg = idx[off:off + len(b)]
        off += len(b)
        assert seg.min() >= b[0] and seg.max() <= b[-1]    # never left the island


def test_islands_are_independent_not_shared_seed():
    # two islands with identical local weights must NOT produce identical local
    # draws (independent per-rank RNG), else islands are correlated.
    w = np.concatenate([np.full(10, 0.05), np.full(10, 0.05)]); w /= w.sum()
    idx = island_resample(w, k=2, seed=0)
    left, right = idx[:10], idx[10:] - 10
    assert not np.array_equal(left, right)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_smc_sharded.py::test_island_resample_conserves_count_and_stays_local -v`
Expected: FAIL — `ImportError: cannot import name 'island_resample'`.

- [ ] **Step 3: Implement `island_resample`**

Append to `causal_bench/sampling/sharded.py`:

```python
def island_resample(w: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Island / local resampling (spec §1d): each of k ranks resamples ONLY its
    own contiguous sub-population from its locally renormalized weights — no
    all-to-all particle exchange. Returns GLOBAL ancestor indices (each mapped
    back into the full array), so no index ever crosses an island boundary.
    Per-island RNGs are seeded independently (seed + rank) to avoid correlating
    islands. This is the low-communication default; the small bias/variance cost
    is characterized on-box against the global sharded oracle."""
    n = len(w)
    bounds = np.array_split(np.arange(n), k)
    out = []
    for rank, b in enumerate(bounds):
        local_w = w[b]
        local_w = local_w / local_w.sum()               # renormalize within island
        rng = np.random.default_rng(seed + rank)        # independent per island
        local_idx = systematic_resample(local_w, rng)   # indices into the local slice
        out.append(b[local_idx])                         # map local -> global
    return np.concatenate(out)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_smc_sharded.py -v`
Expected: PASS (existing global-oracle tests plus the two island tests).

- [ ] **Step 5: Commit**

```bash
git add causal_bench/sampling/sharded.py tests/test_smc_sharded.py
git commit -m "feat(sampling): island (local) resampling oracle for low-comm path (§1d)"
```

---

### Task 8: On-box NCCL collectives — distributed==serial validation (on-box only)

**Files:**
- Create: `scripts/smc_distributed_validate.py`
- Create: `docs/superpowers/runbooks/smc-onbox-validation.md`

**Interfaces:**
- Consumes: `sampling/sharded.py` (the numpy oracle — the reference the real all-to-all must match), `device.cuda_available_devices`, ported `run_smc`.
- Produces: a `torchrun` entrypoint that (a) all-gathers weights, (b) computes systematic indices from the shared seed on every rank, (c) all-to-all redistributes particles, and asserts the result equals the single-GPU reference SMC on the same seed. **No CI pytest** — needs GPUs + NCCL; validated on the A100 box.

> This task has no fast unit-test cycle (it requires the multi-GPU fabric). Its deliverable is the validated runner plus a runbook; acceptance is the on-box assertions, not a green CI run. Keep all collectives at the `torch.distributed` layer — no custom kernels (Global Constraints).

- [ ] **Step 1: Write the distributed runner**

Create `scripts/smc_distributed_validate.py`:

```python
"""On-box multi-GPU SMC validation. Launch with:
    CUDA_VISIBLE_DEVICES=<free,NVLink-adjacent ids> \
    torchrun --nproc_per_node=<N> scripts/smc_distributed_validate.py --seed 7
Asserts distributed indices == the single-rank numpy oracle byte-for-byte, then
that the distributed SMC result matches the single-GPU reference to tolerance.
Collective layer only (all_reduce / all_gather / all_to_all) — no custom kernels."""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import torch.distributed as dist

from causal_bench.sampling.resample import systematic_resample
from causal_bench.sampling.sharded import sharded_systematic_resample


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n", type=int, default=1 << 16)
    args = ap.parse_args()

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.cuda.set_device(rank)

    # Every rank holds the same all-gathered weights (simulated here from a
    # shared seed; in the full loop this is an all_gather of local weights).
    w = np.random.default_rng(args.seed).random(args.n)
    w = w / w.sum()

    # Shared-seed systematic indices — must be byte-identical across ranks and
    # equal to the single-rank oracle. This is the decisive invariant.
    idx = systematic_resample(w, np.random.default_rng(args.seed))
    oracle = sharded_systematic_resample(w, k=world, seed=args.seed)
    # Each rank owns a contiguous slice; concatenation across ranks == oracle.
    my_slice = np.array_split(idx, world)[rank]
    gathered = [torch.empty(len(my_slice), dtype=torch.int64, device="cuda")
                for _ in range(world)]
    dist.all_gather(gathered, torch.as_tensor(my_slice, device="cuda"))
    if rank == 0:
        full = torch.cat(gathered).cpu().numpy()
        assert np.array_equal(full, oracle), "distributed indices != numpy oracle"
        # weight-finiteness fused into the ESS reduce (spec §1b)
        ess_num = torch.tensor(float(w.sum() ** 2), device="cuda")
        ess_den = torch.tensor(float((w ** 2).sum()), device="cuda")
        dist.all_reduce(ess_num); dist.all_reduce(ess_den)
        assert torch.isfinite(ess_num) and torch.isfinite(ess_den)
        print(f"[rank0] world={world} n={args.n}: distributed==oracle OK")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write the runbook**

Create `docs/superpowers/runbooks/smc-onbox-validation.md` capturing the shared-box discipline from spec §0 and the validation ladder:

```markdown
# SMC on-box multi-GPU validation runbook

Prereqs (spec §0): `nvidia-smi` (note idle GPUs), `nvidia-smi topo -m`
(confirm NVLink adjacency — `NV#` good; `PHB`/`PXB`/`SYS` = PCIe), run inside
`tmux`, keep data on the box's local NVMe (Tailscale is control-plane only).

## Ladder (stop at the first failure)
1. **2 ranks — decisive.** Exercises all_reduce + all_gather + (next) all_to_all.
   ```
   CUDA_VISIBLE_DEVICES=<2 free NVLink-adjacent ids> \
   torchrun --nproc_per_node=2 scripts/smc_distributed_validate.py --seed 7
   ```
   Expect: `distributed==oracle OK`.
2. **cuda==cpu parity** (Task 5): `pytest tests/test_smc_cuda_parity.py -v` → PASS.
3. **Throughput sweep** at 2/4/8 ranks: rerun step 1 with `--nproc_per_node` 4 then 8,
   record wall-clock and all-to-all comm cost (the O(N·dim) NVLink transfer — the
   only thing off-box measurement cannot tell us, spec §3).

Do NOT re-derive on-box: resample-trigger rate, O(N) per-particle scaling,
ESS/weight-degeneracy health, distributed==serial index invariant — all
CPU-settled and hardware-independent (spec §3).
```

- [ ] **Step 3: Syntax-check the runner off-box (no GPU needed)**

Run: `python -c "import ast; ast.parse(open('scripts/smc_distributed_validate.py').read()); print('parse OK')"`
Expected: `parse OK`. (Full execution requires the box; do not attempt NCCL init off-box.)

- [ ] **Step 4: On-box validation**

On the A100 box, run the runbook ladder step 1 (2 ranks).
Expected: `[rank0] world=2 ... distributed==oracle OK`. Then steps 2–3.
**This is the §1b/§1e acceptance gate.**

- [ ] **Step 5: Commit**

```bash
git add scripts/smc_distributed_validate.py docs/superpowers/runbooks/smc-onbox-validation.md
git commit -m "feat(sampling): on-box NCCL distributed==serial validation runner + runbook (§1b/1e)"
```

---

## Self-Review

**Spec coverage (checklist items 1–5):**
- Item 1 (`[gpu]` extra, lazy imports) → Task 1. ✔
- Item 2 (§1a xp port + cuda==cpu parity) → Tasks 2–5 (namespace helper, weights, resample, smc_step + parity). ✔
- Item 3 (§1c gating + CPU fallback + NVLink pinning) → Task 6 (`resolve_device`/`cuda_available_devices`; empty-list→CPU-fallback signal), pinning captured in the Task 8 runbook. ✔
- Item 4 (§1b/1e NCCL collectives; distributed==serial at 2 ranks, then 2/4/8 sweep) → Task 8. ✔
- Item 5 (§1d island resampling + ancestor-index indirection) → Task 7 (island oracle; ancestor indices already int32 in `run_smc`, mapped-to-global here). ✔

**Placeholder scan:** No TBD/TODO; every code step shows complete code; on-box steps give exact `torchrun`/`pytest` commands and expected output.

**Type consistency:** `get_namespace` (Task 2) is the single mechanism used identically in Tasks 3–5; `systematic_resample(w, rng)` signature unchanged across Tasks 4/7/8; `cuda_available_devices() -> list[int]` empty-list contract (Task 6) is what Task 8's fallback consumes.

**Deliberately deferred (not gaps):** true `cuda==cpu` parity and distributed==serial are on-box acceptance (Tasks 5/8), consistent with the spec's "device=cuda not yet end-to-end validated." Ancestor-index *indirection* as a standalone gather-deferral optimization is left as an on-box profiling item — the lineage vector it needs is already produced; building a separate indirection layer now would be YAGNI until the box shows the redundant-copy cost.
