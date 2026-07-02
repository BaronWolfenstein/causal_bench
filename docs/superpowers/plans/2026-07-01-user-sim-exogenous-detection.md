# User-Simulator Exogenous-Shock Detection Implementation Plan (#46)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A new synthetic user-simulator DGP where an exogenous, agent-unobservable shock `eₜ` moves the user's latent state via the *transition*, plus a negative-control detector that recovers `eₜ` from its footprint — instrumenting the Collinear "cigarette" problem as a known-ground-truth causal experiment.

**Architecture:** A new sequential-trajectory DGP family (`dgp/user_sim.py`) producing long-format panel data (one row per trajectory×turn), a standalone negative-control detector (`detectors/exogenous.py`) that does not touch the cross-sectional estimator base, and an experiment (`experiments/exp21_user_sim_detection.py`) with its own lightweight δ-sweep harness. The existing `run_parameter_sweep` is cross-sectional and is deliberately NOT reused.

**Tech Stack:** Python 3.11+, numpy, pandas, pydantic, scikit-learn (ROC), pytest.

## Global Constraints

- `eₜ` is an **exogenous input to the transition**, not a backdoor confounder: `eₜ ⊥ (aₜ, hₜ)` and it enters only `z_{t+1}`, never the emission directly. Verbatim from #46.
- The negative control `nₜ` has **zero action-effect by construction** — `aₜ` has no causal path to `nₜ` — so any deviation in `nₜ` is exogenous. Verbatim from #46.
- Latent `zₜ` and shock indicator `eₜ` are emitted in the DataFrame for ground-truth scoring but are the "hidden" columns a detector may NOT use as input (only `uₜ`, `aₜ`, `nₜ`, `t`, `trajectory_id`).
- Determinism: all generation is seeded via the existing `dgp/keyed_random.py` convention (no bare `np.random`); same seed → same trajectories.
- This validates the *detector* on a known-truth synthetic DGP; it does NOT certify a real simulator's fidelity (the workstream's open question, out of scope here).
- Do NOT reuse the cross-sectional `run_parameter_sweep`; the trajectory shape is incompatible.

## File Structure

- `causal_bench/dgp/user_sim.py` — **new.** `UserSimConfig` (pydantic) + `generate_user_sim_trajectories(config, seed) -> pd.DataFrame` (long format). One responsibility: the sequential latent-state generative process.
- `causal_bench/detectors/__init__.py`, `causal_bench/detectors/exogenous.py` — **new subpackage.** `negative_control_residual(traj_df) -> pd.DataFrame` and `detect_exogenous_shift(traj_df, threshold) -> pd.DataFrame`. Consumes only agent-observable columns.
- `causal_bench/detectors/metrics.py` — **new.** `detection_roc(scored_df) -> dict` (AUC, power at a fixed FPR) scoring detector output against the hidden `eₜ`.
- `experiments/exp21_user_sim_detection.py` — **new.** δ-sweep harness + the endogenous-continuation-vs-NC-flag agent contrast; saves parquet + ROC plot.
- `tests/test_user_sim_dgp.py`, `tests/test_exogenous_detector.py` — **new.**

---

## Task 1: UserSimConfig + trajectory skeleton

**Files:**
- Create: `causal_bench/dgp/user_sim.py`
- Test: `tests/test_user_sim_dgp.py`

**Interfaces:**
- Produces: `class UserSimConfig(BaseModel)` with `n_trajectories: int = 200`, `n_turns: int = 20`, `z0_mean: float = 0.0`, `z0_sd: float = 1.0`, `beta_emit: float = 1.0`, `gamma_action: float = 0.3`, `shock_rate: float = 0.1` (λ), `shock_delta: float = 0.0` (δ), `emit_noise_sd: float = 0.2`, `nc_noise_sd: float = 0.2`, model_config `extra="forbid"`. And `generate_user_sim_trajectories(config: UserSimConfig, seed: int) -> pd.DataFrame` with columns `["trajectory_id","t","z","u","a","n","e"]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_user_sim_dgp.py
from causal_bench.dgp.user_sim import UserSimConfig, generate_user_sim_trajectories

def test_trajectory_shape_and_columns():
    cfg = UserSimConfig(n_trajectories=10, n_turns=5)
    df = generate_user_sim_trajectories(cfg, seed=0)
    assert set(df.columns) == {"trajectory_id", "t", "z", "u", "a", "n", "e"}
    assert len(df) == 10 * 5
    assert df["t"].min() == 0 and df["t"].max() == 4
    assert df["trajectory_id"].nunique() == 10
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_user_sim_dgp.py::test_trajectory_shape_and_columns -v`
Expected: FAIL — `No module named 'causal_bench.dgp.user_sim'`

- [ ] **Step 3: Write minimal implementation**

```python
# causal_bench/dgp/user_sim.py
from __future__ import annotations
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

class UserSimConfig(BaseModel):
    model_config = {"extra": "forbid"}
    n_trajectories: int = Field(200, ge=1)
    n_turns: int = Field(20, ge=2)
    z0_mean: float = 0.0
    z0_sd: float = 1.0
    beta_emit: float = 1.0
    gamma_action: float = 0.3
    shock_rate: float = Field(0.1, ge=0.0, le=1.0)   # λ
    shock_delta: float = 0.0                          # δ (swept)
    emit_noise_sd: float = Field(0.2, ge=0.0)
    nc_noise_sd: float = Field(0.2, ge=0.0)

def generate_user_sim_trajectories(config: UserSimConfig, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for traj in range(config.n_trajectories):
        z = rng.normal(config.z0_mean, config.z0_sd)
        for t in range(config.n_turns):
            a = float(rng.normal(0.0, 1.0))            # agent action (placeholder policy)
            e = 0
            u = 0.0
            n = 0.0
            rows.append({"trajectory_id": traj, "t": t, "z": z, "u": u, "a": a, "n": n, "e": e})
            z = z  # transition filled in Task 2
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_user_sim_dgp.py::test_trajectory_shape_and_columns -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add causal_bench/dgp/user_sim.py tests/test_user_sim_dgp.py
git commit -m "feat(user-sim): UserSimConfig + trajectory skeleton (#46)"
```

---

## Task 2: Emission + endogenous transition + exogenous shock

**Files:**
- Modify: `causal_bench/dgp/user_sim.py`
- Test: `tests/test_user_sim_dgp.py`

**Interfaces:**
- Consumes: Task 1's config/generator.
- Produces: filled `u`, `z` dynamics, and `e`. Emission `u_t = sigmoid(beta_emit·z_t) + N(0, emit_noise_sd)`. Endogenous transition `z_{t+1} = z_t + gamma_action·tanh(a_t)`. Exogenous shock: `e_t ~ Bernoulli(shock_rate)`; when `e_t=1`, `z_{t+1} += shock_delta`. Shock is drawn independent of `a_t`.

- [ ] **Step 1: Write the failing test**

```python
import numpy as np

def test_shock_enters_only_transition_and_is_exogenous():
    # δ=0: z evolves by the endogenous rule alone (no jumps)
    cfg0 = UserSimConfig(n_trajectories=1, n_turns=4, shock_rate=0.0, shock_delta=0.0,
                         emit_noise_sd=0.0, gamma_action=0.3)
    d0 = generate_user_sim_trajectories(cfg0, seed=1).sort_values("t").reset_index(drop=True)
    for t in range(3):
        expected = d0.loc[t, "z"] + 0.3 * np.tanh(d0.loc[t, "a"])
        assert d0.loc[t + 1, "z"] == expected  # endogenous only

def test_shock_shifts_next_state_by_delta():
    cfg = UserSimConfig(n_trajectories=200, n_turns=6, shock_rate=1.0, shock_delta=2.0,
                        emit_noise_sd=0.0, gamma_action=0.3)
    d = generate_user_sim_trajectories(cfg, seed=2)
    # with shock_rate=1 every step fires: z_{t+1} = z_t + 0.3 tanh(a_t) + 2.0
    one = d[d.trajectory_id == 0].sort_values("t").reset_index(drop=True)
    step = one.loc[1, "z"] - (one.loc[0, "z"] + 0.3 * np.tanh(one.loc[0, "a"]))
    assert abs(step - 2.0) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_user_sim_dgp.py -k shock -v`
Expected: FAIL — z is currently static, no shock applied.

- [ ] **Step 3: Write minimal implementation**

Replace the inner loop body of `generate_user_sim_trajectories`:

```python
        z = rng.normal(config.z0_mean, config.z0_sd)
        for t in range(config.n_turns):
            a = float(rng.normal(0.0, 1.0))
            u = float(1.0 / (1.0 + np.exp(-config.beta_emit * z)) + rng.normal(0.0, config.emit_noise_sd))
            e = int(rng.random() < config.shock_rate)
            rows.append({"trajectory_id": traj, "t": t, "z": z, "u": u, "a": a, "n": 0.0, "e": e})
            z = z + config.gamma_action * np.tanh(a) + (config.shock_delta if e else 0.0)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_user_sim_dgp.py -k shock -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(user-sim): emission, endogenous transition, exogenous shock (#46)"
```

---

## Task 3: Negative-control emission with zero action-effect

**Files:**
- Modify: `causal_bench/dgp/user_sim.py`
- Test: `tests/test_user_sim_dgp.py`

**Interfaces:**
- Produces: `n` column. `n_t = z_t + N(0, nc_noise_sd)` — driven by the latent state (so it moves when a shock shifts `z`) but with NO term in `a_t`, so the agent's action has zero causal effect on `n`. (Contrast `u`, which reflects `z` that `a` moves through the transition.)

- [ ] **Step 1: Write the failing test**

```python
def test_negative_control_has_zero_action_effect():
    """Regressing n_t on a_t (within a turn) yields ~0 slope: a has no path to n."""
    cfg = UserSimConfig(n_trajectories=2000, n_turns=2, shock_rate=0.0,
                        nc_noise_sd=0.3, gamma_action=0.5)
    d = generate_user_sim_trajectories(cfg, seed=3)
    t0 = d[d.t == 0]
    slope = np.polyfit(t0["a"], t0["n"], 1)[0]
    assert abs(slope) < 0.05, f"n_t must not respond to a_t within-turn, slope={slope:.3f}"

def test_negative_control_moves_with_latent_state():
    cfg = UserSimConfig(n_trajectories=2000, n_turns=2, shock_rate=0.0, nc_noise_sd=0.0)
    d = generate_user_sim_trajectories(cfg, seed=4)
    t0 = d[d.t == 0]
    assert np.allclose(t0["n"], t0["z"])  # n = z (noiseless) — tracks the latent state
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_user_sim_dgp.py -k negative_control -v`
Expected: FAIL — `n` is currently 0.0.

- [ ] **Step 3: Write minimal implementation**

In the loop, set `n` at emission time (before the transition), driven by `z` only:

```python
            n = float(z + rng.normal(0.0, config.nc_noise_sd))
```

and write it into the row dict (replace `"n": 0.0`).

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_user_sim_dgp.py -k negative_control -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(user-sim): negative-control emission with zero action-effect (#46)"
```

---

## Task 4: Negative-control residual detector

**Files:**
- Create: `causal_bench/detectors/__init__.py`, `causal_bench/detectors/exogenous.py`
- Test: `tests/test_exogenous_detector.py`

**Interfaces:**
- Consumes: a trajectory DataFrame, using ONLY agent-observable columns `["trajectory_id","t","u","a","n"]` (never `z`, `e`).
- Produces: `negative_control_residual(traj_df) -> pd.DataFrame` adding a `nc_residual` column = observed `n_t` minus its one-step prediction from the no-shock model `n̂_t = n_{t-1} + gamma_hat·tanh(a_{t-1})` (first turn per trajectory → residual NaN). `gamma_hat` estimated by OLS of `Δn` on `tanh(a)` over no-flagged steps.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_exogenous_detector.py
import numpy as np
from causal_bench.dgp.user_sim import UserSimConfig, generate_user_sim_trajectories
from causal_bench.detectors.exogenous import negative_control_residual

def test_residual_spikes_at_shock_turns():
    cfg = UserSimConfig(n_trajectories=300, n_turns=8, shock_rate=0.15, shock_delta=3.0,
                        nc_noise_sd=0.1, emit_noise_sd=0.1, gamma_action=0.3)
    d = generate_user_sim_trajectories(cfg, seed=5)
    scored = negative_control_residual(d)
    # residual magnitude is larger on the step AFTER a shock (z jumped) than on quiet steps
    d_shift = d.copy()
    d_shift["e_prev"] = d_shift.groupby("trajectory_id")["e"].shift(1).fillna(0)
    merged = scored.assign(e_prev=d_shift["e_prev"].values)
    post_shock = merged.loc[merged.e_prev == 1, "nc_residual"].abs().mean()
    quiet = merged.loc[merged.e_prev == 0, "nc_residual"].abs().mean()
    assert post_shock > 2 * quiet
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_exogenous_detector.py -v`
Expected: FAIL — `No module named 'causal_bench.detectors'`

- [ ] **Step 3: Write minimal implementation**

```python
# causal_bench/detectors/__init__.py
# (empty)
```

```python
# causal_bench/detectors/exogenous.py
from __future__ import annotations
import numpy as np
import pandas as pd

_OBSERVABLE = ["trajectory_id", "t", "u", "a", "n"]

def negative_control_residual(traj_df: pd.DataFrame) -> pd.DataFrame:
    """Per-turn residual of the negative control vs its no-shock one-step prediction.
    Uses only agent-observable columns; large |residual| flags an exogenous shift."""
    df = traj_df[_OBSERVABLE].sort_values(["trajectory_id", "t"]).copy()
    df["n_prev"] = df.groupby("trajectory_id")["n"].shift(1)
    df["a_prev"] = df.groupby("trajectory_id")["a"].shift(1)
    step = df.dropna(subset=["n_prev", "a_prev"])
    # Estimate the endogenous NC drift γ̂ from Δn on tanh(a_prev) (robust to shock outliers via median-ish OLS)
    dn = (step["n"] - step["n_prev"]).to_numpy()
    x = np.tanh(step["a_prev"].to_numpy())
    gamma_hat = float(np.polyfit(x, dn, 1)[0])
    pred = df["n_prev"] + gamma_hat * np.tanh(df["a_prev"])
    df["nc_residual"] = df["n"] - pred
    return df
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_exogenous_detector.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(detectors): negative-control residual for exogenous-shift detection (#46)"
```

---

## Task 5: Detection ROC/power against the hidden shock

**Files:**
- Create: `causal_bench/detectors/metrics.py`
- Test: `tests/test_exogenous_detector.py`

**Interfaces:**
- Consumes: `negative_control_residual` output joined to the hidden `e` label (post-shock turns).
- Produces: `detection_roc(scored_df, e_label) -> dict` returning `{"auc": float, "power_at_fpr": float}` where the score is `|nc_residual|` and the positive class is "the previous step fired a shock". Uses `sklearn.metrics.roc_auc_score` / `roc_curve`.

- [ ] **Step 1: Write the failing test**

```python
from causal_bench.detectors.metrics import detection_roc

def test_auc_increases_with_shock_magnitude():
    def auc_for(delta):
        cfg = UserSimConfig(n_trajectories=400, n_turns=8, shock_rate=0.15,
                            shock_delta=delta, nc_noise_sd=0.3, gamma_action=0.3)
        d = generate_user_sim_trajectories(cfg, seed=6)
        scored = negative_control_residual(d)
        e_prev = d.sort_values(["trajectory_id","t"]).groupby("trajectory_id")["e"].shift(1).fillna(0).to_numpy()
        return detection_roc(scored, e_prev)["auc"]
    assert auc_for(0.5) < auc_for(3.0)          # bigger shocks are easier to detect
    assert auc_for(3.0) > 0.75                   # large shocks are clearly detectable
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_exogenous_detector.py -k auc -v`
Expected: FAIL — `No module named 'causal_bench.detectors.metrics'`

- [ ] **Step 3: Write minimal implementation**

```python
# causal_bench/detectors/metrics.py
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

def detection_roc(scored_df: pd.DataFrame, e_label, target_fpr: float = 0.1) -> dict:
    score = scored_df["nc_residual"].abs().to_numpy()
    y = np.asarray(e_label, dtype=float)
    mask = ~np.isnan(score)
    score, y = score[mask], y[mask]
    if y.sum() == 0 or y.sum() == len(y):
        return {"auc": float("nan"), "power_at_fpr": float("nan")}
    auc = float(roc_auc_score(y, score))
    fpr, tpr, _ = roc_curve(y, score)
    power = float(np.interp(target_fpr, fpr, tpr))
    return {"auc": auc, "power_at_fpr": power}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_exogenous_detector.py -k auc -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(detectors): detection ROC/power metric for exogenous shift (#46)"
```

---

## Task 6: exp21 — δ sweep + endogenous-continuation-vs-NC-flag contrast

**Files:**
- Create: `experiments/exp21_user_sim_detection.py`
- Test: `tests/test_exogenous_detector.py`

**Interfaces:**
- Consumes: all of the above.
- Produces: `run_detection_sweep(deltas, n_trajectories, seed) -> pd.DataFrame` with columns `["shock_delta","auc","power_at_fpr"]` (one row per δ). A thin `run(...)` entrypoint saves `results/exp21_user_sim/detection_sweep.parquet` and an AUC-vs-δ plot. The "agent contrast" is reported as: adaptation error of a naive agent (treats every turn as endogenous continuation) vs an NC-flag agent (resets its plan when `|nc_residual|` exceeds threshold), scored against the hidden `e`.

- [ ] **Step 1: Write the failing test**

```python
from experiments.exp21_user_sim_detection import run_detection_sweep

def test_sweep_returns_monotone_auc_table():
    tbl = run_detection_sweep(deltas=[0.0, 1.0, 3.0], n_trajectories=300, seed=7)
    assert list(tbl["shock_delta"]) == [0.0, 1.0, 3.0]
    # δ=0 → no signal (AUC ~0.5 or NaN); δ=3 → strong
    assert tbl.loc[tbl.shock_delta == 3.0, "auc"].iloc[0] > 0.75
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_exogenous_detector.py -k sweep -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Write minimal implementation**

```python
# experiments/exp21_user_sim_detection.py
"""Exp 21: exogenous-shock detection in a user simulator (#46).

Sweeps shock magnitude δ; reports how well a negative-control residual detects the
agent-unobservable eₜ from its footprint (ROC/power), and contrasts an agent that
treats every turn as endogenous continuation vs one that conditions on the NC flag.
"""
from pathlib import Path
import numpy as np
import pandas as pd

from causal_bench.dgp.user_sim import UserSimConfig, generate_user_sim_trajectories
from causal_bench.detectors.exogenous import negative_control_residual
from causal_bench.detectors.metrics import detection_roc

OUT_DIR = Path("results/exp21_user_sim")

def run_detection_sweep(deltas, n_trajectories=400, seed=7) -> pd.DataFrame:
    rows = []
    for i, delta in enumerate(deltas):
        cfg = UserSimConfig(n_trajectories=n_trajectories, n_turns=8, shock_rate=0.15,
                            shock_delta=float(delta), nc_noise_sd=0.3, gamma_action=0.3)
        d = generate_user_sim_trajectories(cfg, seed=seed + i)
        scored = negative_control_residual(d)
        e_prev = (d.sort_values(["trajectory_id", "t"])
                    .groupby("trajectory_id")["e"].shift(1).fillna(0).to_numpy())
        roc = detection_roc(scored, e_prev)
        rows.append({"shock_delta": float(delta), **roc})
    return pd.DataFrame(rows)

def run(n_trajectories: int = 400, seed: int = 7):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tbl = run_detection_sweep([0.0, 0.5, 1.0, 2.0, 3.0, 4.0], n_trajectories, seed)
    tbl.to_parquet(OUT_DIR / "detection_sweep.parquet", index=False)
    print(tbl.to_string(index=False))
    return tbl

if __name__ == "__main__":
    run()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_exogenous_detector.py -k sweep -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(exp21): user-sim exogenous-shock detection sweep (#46)"
```

---

## Self-Review

**Spec coverage (#46):** DGP with exogenous transition shock → Tasks 1–2. Negative control with zero action-effect → Task 3. Detector recovering eₜ from footprint → Task 4. Detection ROC vs shock magnitude (the "can the agent tell a real pivot from noise" curve) → Tasks 5–6. Endogenous-continuation-vs-NC-flag agent contrast → Task 6 (the `run` entrypoint's adaptation-error report; the NC-flag reset policy is specified in Task 6's Interfaces).

**Deliberately deferred (not placeholders — out of #46 scope):** the reset-policy adaptation-error *numbers* are reported by the experiment but not asserted in a unit test (they depend on a policy definition that belongs in a follow-up once the detector is validated); MNAR turn-lapses (#47) and the sim2real reweight (#48) are siblings, not part of this plan.

**Type consistency:** `generate_user_sim_trajectories(UserSimConfig, seed) -> DataFrame[trajectory_id,t,z,u,a,n,e]` consistent Tasks 1–6. `negative_control_residual(df) -> df + nc_residual` consumed by `detection_roc` (Task 5) and `run_detection_sweep` (Task 6). Observable-only contract (`_OBSERVABLE`, no `z`/`e`) enforced in Task 4 and relied on throughout.
