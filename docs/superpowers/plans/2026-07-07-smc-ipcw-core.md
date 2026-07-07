# SMC / IPCW Core Implementation Plan (Step 3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A numpy-first twisted-diffusion SMC core with first-class IPCW survival-weight bookkeeping, whose algorithmic risks (resample-trigger rate, per-particle scaling, ESS/weight-degeneracy health, distributed==serial correctness) are fully de-risked on CPU, leaving only absolute wall-clock/communication cost for the A100 box.

**Architecture:** New additive package `causal_bench/sampling/`. Pure-numpy SMC primitives (Kish ESS, systematic resampling via cumsum+searchsorted, adaptive resampling, ancestor-lineage tracking) behind small functions; an IPCW layer that turns every out-of-band particle kill into either a weight update or a recorded survival probability ("no third option"); a diagnostics module (ESS trajectory, trigger rate, per-particle scaling, lineage persistence for the localization diagnostic); and a CPU-simulated sharded-resample correctness harness that pins the invariant the real `torch.distributed` all-to-all must satisfy. Torch/multi-GPU is a separate, deferred sub-plan.

**Tech Stack:** Python 3.11, numpy, scipy (already deps). Torch + `torch.distributed` are GPU-deferred (own plan); do NOT add torch as a hard dependency.

## Global Constraints

- Python `>=3.11`; **numpy/scipy only** for this plan — no torch, no new hard deps.
- All weight math in **log space**; normalize once per step; never exponentiate raw log-weights before subtracting the max.
- Additive package `causal_bench/sampling/`; do not touch `estimators/`, `dgp/`, `diagnostics/` except the one documented lineage hook in Task 7.
- **The design rule, enforced by the API:** every particle kill either (a) enters the SMC weights or (b) gets a recorded survival probability `G`. There is no third option — functions that drop particles without one of these must not exist.
- Keep `N` small in tests (N≤200): this validates *shape and algorithmic risk*, not speed. Absolute latency/comms are out of scope (A100 only).
- **Device handling (backend seam):** array ops go through an array-namespace `xp` from `backend.array_namespace(device)` — numpy for `device="cpu"` (default, and what every test uses), cupy for `device="cuda"` on the A100 box. Public entry points (`run_smc`) accept `device="cpu"`; inputs are moved via `backend.asarray(x, device)` and results returned as numpy via `backend.to_numpy(...)`. The CPU path stays the reference; GPU is a namespace swap, not a rewrite.
- TDD: failing test → minimal impl → green → commit. Frequent commits.

---

## File Structure

- `causal_bench/sampling/__init__.py` — public exports
- `causal_bench/sampling/weights.py` — log-weight normalization, Kish ESS
- `causal_bench/sampling/resample.py` — systematic resampling, adaptive trigger, ancestor lineage
- `causal_bench/sampling/smc.py` — `SMCState`, `smc_step`, `run_smc` driver
- `causal_bench/sampling/ipcw.py` — survival-weight bookkeeping (kills → weights or recorded G)
- `causal_bench/sampling/diagnostics.py` — ESS trajectory, trigger rate, per-particle scaling, lineage export
- `causal_bench/sampling/sharded.py` — CPU-simulated sharded resample (pins the distributed invariant)
- `tests/test_smc_weights.py`, `test_smc_resample.py`, `test_smc_loop.py`, `test_smc_ipcw.py`, `test_smc_diagnostics.py`, `test_smc_sharded.py`
- `experiments/demo_smc_ipcw.py` — end-to-end CPU demo

---

### Task 1: package skeleton + log-weight utilities + Kish ESS

**Files:**
- Create: `causal_bench/sampling/__init__.py`, `causal_bench/sampling/weights.py`
- Test: `tests/test_smc_weights.py`

**Interfaces:**
- Produces: `normalize_log_weights(log_w) -> tuple[np.ndarray, float]` (normalized weights summing to 1, log-normalizer); `kish_ess(log_w) -> float`.

- [ ] **Step 1: Write the failing test.**

```python
import numpy as np
from causal_bench.sampling.weights import normalize_log_weights, kish_ess

def test_normalize_sums_to_one_and_is_stable():
    log_w = np.array([-1000.0, -1000.0, -1000.0])   # underflow-prone
    w, log_norm = normalize_log_weights(log_w)
    assert np.isclose(w.sum(), 1.0)
    assert np.allclose(w, 1/3)

def test_kish_ess_uniform_is_n_and_degenerate_is_one():
    n = 8
    assert np.isclose(kish_ess(np.zeros(n)), n)              # uniform -> ESS = N
    spike = np.full(n, -1e9); spike[0] = 0.0
    assert np.isclose(kish_ess(spike), 1.0, atol=1e-6)      # one survivor -> ESS = 1
```

- [ ] **Step 2: Run to verify it fails.** Run: `pytest tests/test_smc_weights.py -q` — Expected: FAIL.
- [ ] **Step 3: Implement.**

```python
"""Log-space weight normalization and Kish effective sample size."""
from __future__ import annotations

import numpy as np


def normalize_log_weights(log_w: np.ndarray) -> tuple[np.ndarray, float]:
    """Return (normalized weights, log normalizer). Subtract the max before
    exponentiating so weights never under/overflow."""
    m = np.max(log_w)
    shifted = np.exp(log_w - m)
    total = shifted.sum()
    log_norm = m + np.log(total)
    return shifted / total, float(log_norm)


def kish_ess(log_w: np.ndarray) -> float:
    """Kish ESS = (sum w)^2 / sum(w^2) = 1 / sum(w_norm^2)."""
    w, _ = normalize_log_weights(log_w)
    return float(1.0 / np.sum(w ** 2))
```

- [ ] **Step 4: Create `__init__.py`** exporting these (add later names as tasks land):

```python
"""Twisted-diffusion SMC core with IPCW bookkeeping (numpy, CPU-first)."""
from .weights import normalize_log_weights, kish_ess

__all__ = ["normalize_log_weights", "kish_ess"]
```

- [ ] **Step 5: Run to verify it passes + commit.** Run: `pytest tests/test_smc_weights.py -q` — Expected: PASS.

```bash
git add causal_bench/sampling/__init__.py causal_bench/sampling/weights.py tests/test_smc_weights.py
git commit -m "feat(sampling): log-weight normalization + Kish ESS"
```

---

### Task 2: systematic resampling + adaptive trigger + ancestor lineage

**Files:**
- Create: `causal_bench/sampling/resample.py`
- Test: `tests/test_smc_resample.py`

**Interfaces:**
- Produces: `systematic_resample(w, rng) -> np.ndarray` (int ancestor indices); `should_resample(log_w, ess_frac=0.5) -> bool`.

- [ ] **Step 1: Write the failing test.**

```python
import numpy as np
from causal_bench.sampling.resample import systematic_resample, should_resample

def test_systematic_resample_duplicates_the_dominant_particle():
    w = np.array([0.001, 0.001, 0.997, 0.001])
    idx = systematic_resample(w, np.random.default_rng(0))
    assert len(idx) == 4
    assert (idx == 2).sum() >= 3                     # dominant survivor fans out
    assert idx.dtype.kind == "i"

def test_systematic_resample_is_deterministic_under_shared_seed():
    w = np.array([0.25, 0.25, 0.25, 0.25])
    a = systematic_resample(w, np.random.default_rng(7))
    b = systematic_resample(w, np.random.default_rng(7))
    assert np.array_equal(a, b)                       # shared seed -> identical

def test_should_resample_triggers_only_on_degeneracy():
    assert should_resample(np.zeros(10), ess_frac=0.5) is False   # ESS=10 > 5
    spike = np.full(10, -1e9); spike[0] = 0.0
    assert should_resample(spike, ess_frac=0.5) is True           # ESS~1 < 5
```

- [ ] **Step 2: Run to verify it fails.** Run: `pytest tests/test_smc_resample.py -q` — Expected: FAIL.
- [ ] **Step 3: Implement** (systematic resampling = cumsum + searchsorted, the GPU-parallel form).

```python
"""Systematic resampling (cumsum + searchsorted — the GPU-parallel primitive)
and the adaptive-resampling trigger. Ancestor indices are the raw material for
the localization diagnostic's lineage-collapse component; callers persist them."""
from __future__ import annotations

import numpy as np

from .weights import kish_ess, normalize_log_weights


def systematic_resample(w: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Return ancestor indices. One uniform draw, N evenly-spaced positions,
    searchsorted into the CDF. O(N) and vectorized."""
    n = len(w)
    positions = (rng.random() + np.arange(n)) / n
    cdf = np.cumsum(w)
    cdf[-1] = 1.0                                    # guard fp drift at the top
    return np.searchsorted(cdf, positions).astype(np.int64)


def should_resample(log_w: np.ndarray, ess_frac: float = 0.5) -> bool:
    """Adaptive resampling: only trigger the barrier when ESS < ess_frac * N.
    Most steps then have no global sync at all."""
    return kish_ess(log_w) < ess_frac * len(log_w)
```

- [ ] **Step 4: Run to verify it passes + commit.** Run: `pytest tests/test_smc_resample.py -q` — Expected: PASS.

```bash
git add causal_bench/sampling/resample.py tests/test_smc_resample.py
git commit -m "feat(sampling): systematic resampling + adaptive trigger"
```

---

### Task 3: SMC state + step + driver, with lineage tracking

**Files:**
- Create: `causal_bench/sampling/smc.py`
- Test: `tests/test_smc_loop.py`

**Interfaces:**
- Consumes: Tasks 1–2.
- Produces: `SMCState(particles, log_weights, ancestry)`; `smc_step(state, log_incr, rng, ess_frac=0.5) -> tuple[SMCState, bool]`; `run_smc(x0, propagate, log_weight_fn, n_steps, rng, ess_frac=0.5) -> SMCResult`.

- [ ] **Step 1: Write the failing test.** Anneal base N(0,I) → target N(μ,I) with μ far from mass; the weighted particle mean must recover μ, and resampling must fire at least once.

```python
import numpy as np
from causal_bench.sampling.smc import run_smc

def test_smc_recovers_a_far_target_mean():
    rng = np.random.default_rng(0)
    d, mu = 2, np.array([4.0, 0.0])        # rare region: 4 sigma out
    betas = np.linspace(0.0, 1.0, 20)      # annealing schedule
    x0 = rng.standard_normal((300, d))     # base samples

    def propagate(x, step):                # random-walk move (keeps support alive)
        return x + 0.3 * np.random.default_rng(step).standard_normal(x.shape)

    def log_weight_fn(x, step):            # incremental tilt toward N(mu, I)
        db = betas[step] - betas[step - 1]
        return db * (-0.5 * np.sum((x - mu) ** 2, axis=1) + 0.5 * np.sum(x ** 2, axis=1))

    res = run_smc(x0, propagate, log_weight_fn, n_steps=len(betas), rng=rng)
    est = np.average(res.state.particles, axis=0,
                     weights=np.exp(res.state.log_weights - res.state.log_weights.max()))
    assert np.linalg.norm(est - mu) < 0.6           # recovers the far mean
    assert res.n_resamples >= 1                      # degeneracy forced a resample
    assert res.ess_trajectory[0] >= res.ess_trajectory.min()
```

- [ ] **Step 2: Run to verify it fails.** Run: `pytest tests/test_smc_loop.py -q` — Expected: FAIL.
- [ ] **Step 3: Implement.**

```python
"""Twisted-diffusion SMC loop. Propagation and twist evaluation are per-particle
(embarrassingly parallel); the only synchronization is the ESS reduction inside
the adaptive resample. Ancestry is tracked so lineage collapse is observable."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .resample import should_resample, systematic_resample
from .weights import normalize_log_weights


@dataclass
class SMCState:
    particles: np.ndarray        # (N, d)
    log_weights: np.ndarray      # (N,)
    ancestry: np.ndarray         # (N,) ancestor index at the last resample


@dataclass
class SMCResult:
    state: SMCState
    ess_trajectory: np.ndarray   # ESS after each step
    resample_steps: list         # step indices where the barrier fired
    lineage: list                # ancestor-index vectors (int32) per resample

    @property
    def n_resamples(self) -> int:
        return len(self.resample_steps)


def smc_step(state: SMCState, log_incr: np.ndarray, rng, ess_frac: float = 0.5):
    """One reweight → (adaptive) resample. Returns (new_state, resampled?)."""
    log_w = state.log_weights + log_incr
    if should_resample(log_w, ess_frac):
        w, _ = normalize_log_weights(log_w)
        idx = systematic_resample(w, rng)
        new = SMCState(
            particles=state.particles[idx],
            log_weights=np.zeros(len(idx)),          # reset to uniform post-resample
            ancestry=idx,
        )
        return new, True
    return SMCState(state.particles, log_w, state.ancestry), False


def run_smc(x0, propagate, log_weight_fn, n_steps, rng, ess_frac: float = 0.5):
    from .weights import kish_ess
    state = SMCState(np.asarray(x0, float), np.zeros(len(x0)),
                     np.arange(len(x0)))
    ess, resample_steps, lineage = [], [], []
    for step in range(1, n_steps):
        state = SMCState(propagate(state.particles, step),
                         state.log_weights, state.ancestry)
        state, did = smc_step(state, log_weight_fn(state.particles, step),
                              rng, ess_frac)
        ess.append(kish_ess(state.log_weights))
        if did:
            resample_steps.append(step)
            lineage.append(state.ancestry.astype(np.int32))
    return SMCResult(state, np.asarray(ess), resample_steps, lineage)
```

- [ ] **Step 4: Export in `__init__.py`** (`run_smc`, `SMCState`, `SMCResult`, `smc_step`).
- [ ] **Step 5: Run to verify it passes + commit.** Run: `pytest tests/test_smc_loop.py -q` — Expected: PASS.

```bash
git add causal_bench/sampling/smc.py causal_bench/sampling/__init__.py tests/test_smc_loop.py
git commit -m "feat(sampling): SMC state/step/driver with lineage tracking"
```

---

### Task 4: IPCW survival-weight bookkeeping (kills → weights or recorded G)

**Files:**
- Create: `causal_bench/sampling/ipcw.py`
- Test: `tests/test_smc_ipcw.py`

**Interfaces:**
- Produces: `ipcw_weights(survival_probs, *, stabilize_by=None) -> np.ndarray`; `positivity_floor(survival_probs, floor) -> tuple[np.ndarray, np.ndarray]` (clipped G, mask of violations).

- [ ] **Step 1: Write the failing test.** A mid-trajectory filter kills half of a region; weighting survivors by 1/G restores the unbiased mean; a near-zero G is a flagged positivity violation, not silently reweighted.

```python
import numpy as np
from causal_bench.sampling.ipcw import ipcw_weights, positivity_floor

def test_ipcw_restores_unbiased_mean_after_informative_kill():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(10000)
    # informative filter: keep-prob depends on x (censoring on the covariate)
    G = 1.0 / (1.0 + np.exp(-(x + 1.0)))            # survival prob per sample
    kept = rng.random(len(x)) < G
    naive = x[kept].mean()                           # biased (survivors skew high)
    w = ipcw_weights(G[kept])
    corrected = np.average(x[kept], weights=w)
    assert abs(corrected) < abs(naive)               # bias reduced toward 0
    assert abs(corrected) < 0.05

def test_positivity_floor_flags_near_zero_survival():
    G = np.array([0.5, 0.4, 1e-6])
    clipped, violations = positivity_floor(G, floor=1e-3)
    assert violations.tolist() == [False, False, True]
    assert clipped[2] == 1e-3                         # clipped, and flagged
```

- [ ] **Step 2: Run to verify it fails.** Run: `pytest tests/test_smc_ipcw.py -q` — Expected: FAIL.
- [ ] **Step 3: Implement.**

```python
"""IPCW for out-of-band particle kills. Any kill that does NOT go through the
SMC weight bookkeeping (validity filters, heuristic pruning) is informative
censoring: model its survival probability G and weight survivors by 1/G. For
multi-step filters, G is the product of per-step survival probabilities
(discrete-time IPCW); the stabilized form multiplies by a marginal survival in
the numerator. Positivity: where G -> 0, 1/G explodes and no reweighting
recovers lost support — clip and FLAG rather than silently trust the weight."""
from __future__ import annotations

from typing import Optional

import numpy as np


def ipcw_weights(survival_probs: np.ndarray,
                 *, stabilize_by: Optional[np.ndarray] = None) -> np.ndarray:
    """Inverse-probability-of-selection weights 1/G (optionally stabilized by a
    marginal survival numerator). `survival_probs` may be a product of per-step
    survivals already."""
    G = np.asarray(survival_probs, float)
    w = 1.0 / G
    if stabilize_by is not None:
        w = np.asarray(stabilize_by, float) * w
    return w


def positivity_floor(survival_probs: np.ndarray,
                     floor: float) -> tuple[np.ndarray, np.ndarray]:
    """Clip G to `floor` and return (clipped_G, violation_mask). A violation is
    a structural positivity failure — the honest fix is upstream (twist earlier),
    not in the weights."""
    G = np.asarray(survival_probs, float)
    violations = G < floor
    return np.where(violations, floor, G), violations
```

- [ ] **Step 4: Export + commit.** Run: `pytest tests/test_smc_ipcw.py -q` — Expected: PASS.

```bash
git add causal_bench/sampling/ipcw.py causal_bench/sampling/__init__.py tests/test_smc_ipcw.py
git commit -m "feat(sampling): IPCW survival-weight bookkeeping + positivity floor"
```

---

### Task 5: SMC diagnostics — trigger rate, scaling, lineage export

**Files:**
- Create: `causal_bench/sampling/diagnostics.py`
- Test: `tests/test_smc_diagnostics.py`

**Interfaces:**
- Consumes: `SMCResult` (Task 3).
- Produces: `resample_trigger_rate(result) -> float`; `per_particle_scaling(run_fn, ns) -> dict[int, float]`; `lineage_multiplicity(result) -> np.ndarray` (survivors' fan-out histogram for the localization diagnostic).

- [ ] **Step 1: Write the failing test.**

```python
import numpy as np
from causal_bench.sampling.smc import run_smc
from causal_bench.sampling.diagnostics import (
    resample_trigger_rate, per_particle_scaling, lineage_multiplicity)

def _toy_run(n, seed=0):
    rng = np.random.default_rng(seed)
    mu = np.array([4.0]); betas = np.linspace(0, 1, 15)
    x0 = rng.standard_normal((n, 1))
    prop = lambda x, s: x + 0.3 * np.random.default_rng(s).standard_normal(x.shape)
    lw = lambda x, s: (betas[s]-betas[s-1]) * (-0.5*((x-mu)**2).sum(1) + 0.5*(x**2).sum(1))
    return run_smc(x0, prop, lw, len(betas), rng)

def test_trigger_rate_in_unit_interval():
    r = resample_trigger_rate(_toy_run(200))
    assert 0.0 <= r <= 1.0

def test_per_particle_cost_scales_roughly_linearly():
    scaling = per_particle_scaling(lambda n: _toy_run(n), ns=[50, 100, 200])
    # cost/particle should be roughly flat (linear total), not growing with N
    per = [scaling[n] / n for n in (50, 100, 200)]
    assert max(per) / min(per) < 3.0

def test_lineage_multiplicity_sums_to_n():
    r = _toy_run(100)
    if r.n_resamples:
        mult = lineage_multiplicity(r)
        assert int(mult.sum()) == 100
```

- [ ] **Step 2: Run to verify it fails.** Run: `pytest tests/test_smc_diagnostics.py -q` — Expected: FAIL.
- [ ] **Step 3: Implement.**

```python
"""CPU-observable SMC diagnostics — the hardware-independent risks. Resample
trigger rate (dominant algorithmic risk; a property of how far R sits from base
mass, NOT the GPU), per-particle scaling (verifies batching is real), and the
lineage-multiplicity histogram (survivors' fan-out — the raw material for the
localization diagnostic's lineage-collapse component)."""
from __future__ import annotations

import time

import numpy as np


def resample_trigger_rate(result) -> float:
    steps = len(result.ess_trajectory)
    return result.n_resamples / steps if steps else 0.0


def per_particle_scaling(run_fn, ns) -> dict:
    out = {}
    for n in ns:
        t0 = time.perf_counter()
        run_fn(n)
        out[n] = time.perf_counter() - t0
    return out


def lineage_multiplicity(result) -> np.ndarray:
    """Histogram of how many descendants each particle index has at the last
    resample. Highly skewed under rare-event degeneracy (few survivors, huge
    multiplicity) — that skew IS the diagnostic signal."""
    last = result.lineage[-1]
    n = len(last)
    return np.bincount(last, minlength=n)
```

- [ ] **Step 4: Export + commit.** Run: `pytest tests/test_smc_diagnostics.py -q` — Expected: PASS.

```bash
git add causal_bench/sampling/diagnostics.py causal_bench/sampling/__init__.py tests/test_smc_diagnostics.py
git commit -m "feat(sampling): trigger-rate/scaling/lineage diagnostics"
```

---

### Task 6: CPU-simulated sharded resample — pin the distributed invariant

**Files:**
- Create: `causal_bench/sampling/sharded.py`
- Test: `tests/test_smc_sharded.py`

**Interfaces:**
- Produces: `sharded_systematic_resample(w, k, seed) -> np.ndarray` — simulates `k` ranks each computing indices from all-gathered weights + a shared seed; must equal the single-rank result byte-for-byte.

This is the correctness contract the real `torch.distributed` all-to-all must satisfy. It is CPU-checkable now; only absolute comms cost needs the A100 fabric.

- [ ] **Step 1: Write the failing test.**

```python
import numpy as np
from causal_bench.sampling.resample import systematic_resample
from causal_bench.sampling.sharded import sharded_systematic_resample

def test_sharded_indices_match_single_rank_bit_for_bit():
    rng = np.random.default_rng(3)
    w = rng.random(64); w /= w.sum()
    serial = systematic_resample(w, np.random.default_rng(123))
    for k in (2, 4, 8):
        distributed = sharded_systematic_resample(w, k=k, seed=123)
        assert np.array_equal(serial, distributed)     # the decisive invariant

def test_particle_count_conserved_across_shards():
    rng = np.random.default_rng(1)
    w = rng.random(60); w /= w.sum()
    idx = sharded_systematic_resample(w, k=3, seed=9)
    assert len(idx) == 60                               # N in == N out
```

- [ ] **Step 2: Run to verify it fails.** Run: `pytest tests/test_smc_sharded.py -q` — Expected: FAIL.
- [ ] **Step 3: Implement.**

```python
"""CPU simulation of the multi-GPU resample. In production each rank all-gathers
weights so every rank holds w[1..N], then computes IDENTICAL systematic indices
from identical weights + a shared seed; an all-to-all redistributes particles
whose owner changed. Here we simulate the k ranks in-process and assert they
reproduce the single-rank indices exactly — the invariant that makes the real
all-to-all correct. Only absolute communication cost needs the A100 fabric."""
from __future__ import annotations

import numpy as np

from .resample import systematic_resample


def sharded_systematic_resample(w: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Each of k simulated ranks computes the full index vector from the same
    all-gathered weights and the same seed. Identical inputs + identical seed =>
    byte-identical indices on every rank (asserted equal to the serial run)."""
    n = len(w)
    shard_bounds = np.array_split(np.arange(n), k)
    full = None
    for _rank in range(k):
        rng = np.random.default_rng(seed)               # SHARED seed across ranks
        idx = systematic_resample(w, rng)               # full index vector per rank
        if full is None:
            full = idx
        else:
            assert np.array_equal(full, idx), "ranks disagree — seed not shared"
    # each rank keeps the slice it owns; concatenation reconstructs the whole
    return np.concatenate([full[b] for b in shard_bounds])
```

- [ ] **Step 4: Export + commit.** Run: `pytest tests/test_smc_sharded.py -q` — Expected: PASS.

```bash
git add causal_bench/sampling/sharded.py causal_bench/sampling/__init__.py tests/test_smc_sharded.py
git commit -m "feat(sampling): CPU-simulated sharded resample invariant"
```

---

### Task 7: end-to-end demo + lineage hook into the localization diagnostic

**Files:**
- Create: `experiments/demo_smc_ipcw.py`
- Modify: `causal_bench/diagnostics/localization.py` (add a thin optional consumer of a lineage-multiplicity vector — non-breaking)

- [ ] **Step 1: Write the demo** — run SMC to a far target, show ESS trajectory + trigger rate, apply an informative validity filter, IPCW-correct it, and show positivity flagging.

```python
"""python experiments/demo_smc_ipcw.py — SMC + IPCW on a far target, CPU."""
import numpy as np
from causal_bench.sampling import run_smc, kish_ess
from causal_bench.sampling.diagnostics import resample_trigger_rate, lineage_multiplicity
from causal_bench.sampling.ipcw import ipcw_weights, positivity_floor

def main():
    rng = np.random.default_rng(0)
    mu = np.array([4.0, 0.0]); betas = np.linspace(0, 1, 20)
    x0 = rng.standard_normal((300, 2))
    prop = lambda x, s: x + 0.3 * np.random.default_rng(s).standard_normal(x.shape)
    lw = lambda x, s: (betas[s]-betas[s-1]) * (-0.5*((x-mu)**2).sum(1) + 0.5*(x**2).sum(1))
    res = run_smc(x0, prop, lw, len(betas), rng)
    print("trigger rate:", round(resample_trigger_rate(res), 3))
    print("final ESS:", round(kish_ess(res.state.log_weights), 1))
    if res.n_resamples:
        print("lineage multiplicity (top 5):", sorted(lineage_multiplicity(res))[-5:])
    G = 1.0 / (1.0 + np.exp(-(res.state.particles[:, 0])))   # informative filter
    Gc, viol = positivity_floor(G, floor=1e-3)
    print("positivity violations:", int(viol.sum()))
    print("mean IPCW weight:", round(ipcw_weights(Gc).mean(), 3))

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it.** Run: `PYTHONPATH=. python experiments/demo_smc_ipcw.py` — Expected: trigger rate in (0,1), a resample fired, lineage multiplicity skewed, positivity violations counted.
- [ ] **Step 3: Lineage hook (non-breaking).** In `localization.py`, add a small function `lineage_collapse_score(multiplicity: np.ndarray) -> float` returning the normalized Gini of the multiplicity histogram (0 = uniform survival, 1 = total collapse), with a unit test in `tests/test_localization.py`. This is the documented consumer of Task 5's `lineage_multiplicity`.

```python
def lineage_collapse_score(multiplicity):
    """Normalized Gini of ancestor multiplicity: 0 = every particle survives
    equally, ->1 = a handful of survivors dominate (rare-event degeneracy)."""
    import numpy as np
    m = np.sort(np.asarray(multiplicity, float))
    n = len(m)
    if n == 0 or m.sum() == 0:
        return 0.0
    cum = np.cumsum(m)
    return float((n + 1 - 2 * (cum / cum[-1]).sum()) / n)
```

- [ ] **Step 4: Run localization tests + commit.** Run: `pytest tests/test_localization.py tests/test_smc_*.py -q` — Expected: all PASS.

```bash
git add experiments/demo_smc_ipcw.py causal_bench/diagnostics/localization.py tests/test_localization.py
git commit -m "feat(sampling): end-to-end demo + lineage-collapse hook into localization"
```

---

### Task 8: device backend seam (CPU numpy / GPU cupy) + `run_smc(device=...)`

**Files:**
- Create: `causal_bench/sampling/backend.py`
- Modify: `causal_bench/sampling/smc.py` (add `device="cpu"` to `run_smc`)
- Test: `tests/test_smc_backend.py`

**Interfaces:**
- Produces: `array_namespace(device="cpu")` (numpy for cpu, cupy for cuda); `asarray(x, device)`; `to_numpy(x)`.
- Modifies: `run_smc(..., device="cpu")` — converts `x0` via `asarray`, returns particles as numpy via `to_numpy`. CPU path (default) byte-for-byte unchanged.

- [ ] **Step 1: Write the failing test.**

```python
import numpy as np
import pytest
from causal_bench.sampling.backend import array_namespace, asarray, to_numpy

def test_cpu_namespace_is_numpy_and_roundtrips():
    assert array_namespace("cpu") is np
    x = asarray([1.0, 2.0, 3.0], "cpu")
    assert isinstance(x, np.ndarray)
    assert np.array_equal(to_numpy(x), np.array([1.0, 2.0, 3.0]))

def test_unknown_device_raises():
    with pytest.raises(ValueError):
        array_namespace("tpu")

def test_run_smc_cpu_device_returns_numpy():
    from causal_bench.sampling.smc import run_smc
    rng = np.random.default_rng(0)
    betas = np.linspace(0, 1, 10); mu = np.array([2.0])
    x0 = rng.standard_normal((50, 1))
    prop = lambda x, s: x + 0.3 * np.random.default_rng(s).standard_normal(x.shape)
    lw = lambda x, s: (betas[s]-betas[s-1]) * (-0.5*((x-mu)**2).sum(1) + 0.5*(x**2).sum(1))
    res = run_smc(x0, prop, lw, len(betas), rng, device="cpu")
    assert isinstance(res.state.particles, np.ndarray)
```

- [ ] **Step 2: Run to verify it fails.** Run: `pytest tests/test_smc_backend.py -q` — Expected: FAIL.
- [ ] **Step 3: Implement `backend.py`.**

```python
"""Array-namespace backend seam. numpy on CPU (the reference path every test
uses), cupy on CUDA for the A100 box. The SMC hot path can run on either by
selecting `xp = array_namespace(device)`; GPU is a namespace swap, not a rewrite.
Absolute GPU throughput is validated on the box — correctness is device-agnostic."""
from __future__ import annotations

import numpy as np


def array_namespace(device: str = "cpu"):
    if device == "cpu":
        return np
    if device.startswith("cuda"):
        import cupy as cp                # lazy: only needed on the GPU box
        return cp
    raise ValueError(f"unknown device: {device!r} (use 'cpu' or 'cuda')")


def asarray(x, device: str = "cpu"):
    return array_namespace(device).asarray(x)


def to_numpy(x):
    """Move any array (numpy or cupy) back to host numpy."""
    if type(x).__module__.startswith("cupy"):
        return x.get()
    return np.asarray(x)
```

- [ ] **Step 4: Add `device` to `run_smc`.** In `causal_bench/sampling/smc.py`, change the signature to `run_smc(x0, propagate, log_weight_fn, n_steps, rng, ess_frac=0.5, device="cpu")`; at the top convert `x0 = asarray(x0, device)` (import from `.backend`); before returning, set `state = SMCState(to_numpy(state.particles), to_numpy(state.log_weights), to_numpy(state.ancestry))`. The per-step math is unchanged (numpy or cupy both satisfy it).

- [ ] **Step 5: Run to verify it passes + commit.** Run: `pytest tests/test_smc_backend.py tests/test_smc_loop.py -q` — Expected: PASS (the loop test still green — CPU path unchanged).

```bash
git add causal_bench/sampling/backend.py causal_bench/sampling/smc.py causal_bench/sampling/__init__.py tests/test_smc_backend.py
git commit -m "feat(sampling): CPU/GPU array-namespace backend seam + run_smc(device=)"
```

---

## Deferred sub-plan: multi-GPU SMC (torch.distributed) — own plan, A100-gated

Structured but not built here (no torch in this env, and absolute comms cost is uninformative off-A100). Its scope, from the deployment diagram: `all_reduce(Σw, Σw²)` for Kish ESS; `all_gather` weights; identical systematic indices from shared seed (Task 6 is the numpy oracle it must match); `all_to_all` particle redistribution. **Correctness** (distributed == serial at 2 ranks) transfers from Task 6; only **timing** (all-reduce O(1), all-gather O(N), all-to-all O(N·dim) over NVLink) needs the box. Do this plan on the Lambda 8×A100.

**Stay at the collective layer — no custom kernels.** The synchronization barrier is `dist.all_reduce`; NCCL owns all device-side sync, stream ordering, and memory fencing. Do NOT hand-write CUDA kernels, cooperative-groups grid syncs, or CPU memory fences (membarrier): systematic resampling is `cupy.cumsum` + `cupy.searchsorted` (CUB `DeviceScan`, already optimal and self-synchronizing), and ranks communicate only through NCCL collectives — never lock-free shared CPU memory. Reaching below this layer means reimplementing NCCL/CUB.

**`cuda_available_devices` gating flag (required).** The distributed path activates only when GPUs are actually present and pinned:
- A module-level gate `cuda_available_devices() -> list[int]` reads `CUDA_VISIBLE_DEVICES` (falls back to `torch.cuda.device_count()`); returns the visible device ids or `[]`.
- `run_smc_distributed(..., device_ids=None)`: if `device_ids is None`, default to `cuda_available_devices()`. If the result is empty → **fall back to the single-process CPU `run_smc`** (never crash on a CPU box; the correctness path stays runnable everywhere). If non-empty → initialize the process group over exactly those devices, `nproc_per_node = len(device_ids)`.
- Respect the shared-box discipline from the GPU build spec: pin to free, **NVLink-adjacent** ids (`nvidia-smi topo -m`), set `CUDA_VISIBLE_DEVICES` before `torchrun`. The flag makes "how many ranks" a function of what's actually free, not a hardcoded 8.

**Minimize data movement at the barrier (roofline: the gather is memory-BW-bound, ≈0 arithmetic intensity):**
- **Island / local resampling** — each rank resamples its own sub-population, with only occasional global exchange; removes the `all_to_all` at a small bias/variance cost. Make it the default variant; global exact resampling is opt-in.
- **Ancestor-index indirection** — keep the int32 ancestor vector (Task 3's `lineage`) and defer the physical state gather until states are next written, avoiding redundant copies when one survivor is duplicated massively (the rare-event degeneracy regime). The index vector is cheap and is already the localization diagnostic's input.

## Self-Review

**Spec coverage:** log-weights+ESS (T1), systematic resample + adaptive trigger (T2), SMC loop with lineage (T3), IPCW kills→weights/G + positivity (T4), CPU diagnostics: trigger rate/scaling/lineage (T5), sharded-resample invariant (T6), demo + localization hook (T7), multi-GPU deferred. Every hardware-independent risk from the profiling analysis has a task. ✅

**Placeholder scan:** every code step has real numpy; no TBD/"handle edge cases". ✅

**Type consistency:** `run_smc → SMCResult(state: SMCState, ess_trajectory, resample_steps, lineage)`; `lineage_multiplicity(result)` reads `result.lineage[-1]`; `systematic_resample(w, rng)` used identically in T2, T3, T6; `ipcw_weights(G)` / `positivity_floor(G, floor)` consistent. ✅
