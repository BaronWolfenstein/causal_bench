# MNAR Turn-Missingness for Dialogue Implementation Plan (#47)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Specialize exp13's censoring/missingness grid to multi-turn dialogue: user turns go missing under MCAR/MAR/MNAR mechanisms (the "went for a cigarette" lapse), with a turn-lapse severity axis, and measure how badly a trajectory-reward estimand is biased under each — plus which observable proxies partially recover the MNAR case.

**Architecture:** A missingness layer (`dgp/dialogue_missingness.py`) that annotates any long-format trajectory DataFrame with an `observed` mask and a turn-lapse `dt`, three mechanisms; a reward-estimand module (`estimators/reward_missingness.py`) with naive / IPW / proxy-corrected estimators; and `exp27` sweeping mechanism × severity. Core layer and estimators are tested on small hand-built frames (self-contained); only `exp27` consumes #46's `generate_user_sim_trajectories` as the trajectory source.

**Tech Stack:** Python 3.11+, numpy, pandas, scikit-learn (logistic propensity), pytest.

## Global Constraints

- Three mechanisms mirror exp13 exactly: **MCAR** (`R⊥` everything), **MAR** (`R | h_t`, observable-driven, IPW-correctable), **MNAR** (`R | z_t`, latent-driven, NOT correctable from observables alone). Verbatim from #47 / exp13.
- The missingness layer consumes ONLY a long-format trajectory frame with columns `["trajectory_id","t","z","u","a"]` (and optionally a proxy); it never assumes the #46 generator, so it works on any trajectory source.
- `z` is the hidden latent state: mechanisms and ground-truth reward may read it, but the naive/IPW estimators may NOT (only `observed`, `u`, `a`, and any declared proxy).
- Determinism: seeded via `np.random.default_rng(seed)`; same seed → same mask.
- Honest endpoint: report the residual MNAR bias that no proxy removes — do not claim full correction.
- `exp27` is the free experiment number (exp21/24/25 taken, exp22 reserved for immortal-time, exp26 is #46).

## File Structure

- `causal_bench/dgp/dialogue_missingness.py` — **new.** `apply_turn_missingness(traj_df, mechanism, severity, seed, proxy_noise_sd=0.0) -> pd.DataFrame` adding `observed` (bool), `dt` (turn-lapse), and `z_proxy` (noisy proxy of `z`).
- `causal_bench/estimators/reward_missingness.py` — **new.** `true_reward`, `naive_reward`, `ipw_reward`, `proxy_reward` — all `-> float`.
- `experiments/exp27_dialogue_mnar.py` — **new.** mechanism × severity sweep over #46 trajectories; reward-bias table per estimator.
- `tests/test_dialogue_missingness.py`, `tests/test_reward_missingness.py` — **new.**

---

## Task 1: Missingness layer skeleton + MCAR + turn-lapse

**Files:**
- Create: `causal_bench/dgp/dialogue_missingness.py`
- Test: `tests/test_dialogue_missingness.py`

**Interfaces:**
- Produces: `apply_turn_missingness(traj_df, mechanism, severity, seed, proxy_noise_sd=0.0) -> pd.DataFrame` adding `observed: bool`, `dt: int` (turns since previous observed turn, ≥1), `z_proxy: float` (`z + N(0, proxy_noise_sd)`). MCAR: `P(miss) = severity`, i.i.d.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_dialogue_missingness.py
import numpy as np, pandas as pd
from causal_bench.dgp.dialogue_missingness import apply_turn_missingness

def _traj(n_traj=200, n_turns=10, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for j in range(n_traj):
        z = rng.normal(0, 1)
        for t in range(n_turns):
            a = rng.normal(0, 1); z = z + 0.2*np.tanh(a)
            rows.append({"trajectory_id": j, "t": t, "z": z, "u": 1/(1+np.exp(-z)), "a": a})
    return pd.DataFrame(rows)

def test_mcar_drops_independent_fraction_and_sets_dt():
    df = apply_turn_missingness(_traj(), mechanism="mcar", severity=0.3, seed=1)
    assert set(["observed", "dt", "z_proxy"]).issubset(df.columns)
    frac = 1 - df["observed"].mean()
    assert abs(frac - 0.3) < 0.03                      # ~severity dropped
    # dt ≥ 1 and larger when more turns are skipped
    assert df["dt"].min() >= 1
    corr = np.corrcoef(df["z"], df["observed"].astype(float))[0, 1]
    assert abs(corr) < 0.05                            # MCAR: drop ⊥ latent state
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_dialogue_missingness.py -k mcar -v`
Expected: FAIL — `No module named 'causal_bench.dgp.dialogue_missingness'`

- [ ] **Step 3: Write minimal implementation**

```python
# causal_bench/dgp/dialogue_missingness.py
from __future__ import annotations
import numpy as np
import pandas as pd

def _miss_prob(df: pd.DataFrame, mechanism: str, severity: float) -> np.ndarray:
    if mechanism == "mcar":
        return np.full(len(df), severity)
    raise ValueError(f"unknown mechanism {mechanism!r}")

def apply_turn_missingness(traj_df: pd.DataFrame, mechanism: str, severity: float,
                           seed: int, proxy_noise_sd: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = traj_df.sort_values(["trajectory_id", "t"]).copy()
    p = np.clip(_miss_prob(df, mechanism, severity), 0.0, 1.0)
    df["observed"] = rng.random(len(df)) >= p
    df["z_proxy"] = df["z"] + rng.normal(0.0, proxy_noise_sd, len(df))
    # dt = turns since previous observed turn within a trajectory
    dt = []
    for _, g in df.groupby("trajectory_id"):
        gap, out = 0, []
        for obs in g["observed"]:
            gap += 1
            out.append(gap)
            if obs:
                gap = 0
        dt.extend(out)
    df["dt"] = dt
    return df
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_dialogue_missingness.py -k mcar -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add causal_bench/dgp/dialogue_missingness.py tests/test_dialogue_missingness.py
git commit -m "feat(dialogue-mnar): missingness layer + MCAR + turn-lapse dt (#47)"
```

---

## Task 2: MAR mechanism (observable-driven)

**Files:**
- Modify: `causal_bench/dgp/dialogue_missingness.py`
- Test: `tests/test_dialogue_missingness.py`

**Interfaces:**
- Produces: `mechanism="mar"` → `P(miss)_t = sigmoid(severity·(|a_{t-1}| − mean|a|))` — driven by the observable prior action magnitude, independent of `z` given the observable.

- [ ] **Step 1: Write the failing test**

```python
def test_mar_depends_on_observable_not_latent_given_it():
    df = apply_turn_missingness(_traj(seed=2), mechanism="mar", severity=2.0, seed=3)
    df["a_prev_abs"] = df.groupby("trajectory_id")["a"].shift(1).abs()
    sub = df.dropna(subset=["a_prev_abs"])
    # missingness tracks the observable |a_prev|
    c_obs = np.corrcoef(sub["a_prev_abs"], (~sub["observed"]).astype(float))[0, 1]
    assert c_obs > 0.1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_dialogue_missingness.py -k mar -v`
Expected: FAIL — `unknown mechanism 'mar'`

- [ ] **Step 3: Write minimal implementation**

Extend `_miss_prob`:

```python
    if mechanism == "mar":
        a_prev = df.groupby("trajectory_id")["a"].shift(1).abs()
        x = a_prev.fillna(a_prev.mean()).to_numpy()
        return 1.0 / (1.0 + np.exp(-severity * (x - np.nanmean(x))))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_dialogue_missingness.py -k mar -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(dialogue-mnar): MAR mechanism on observable a_prev (#47)"
```

---

## Task 3: MNAR mechanism (latent-driven)

**Files:**
- Modify: `causal_bench/dgp/dialogue_missingness.py`
- Test: `tests/test_dialogue_missingness.py`

**Interfaces:**
- Produces: `mechanism="mnar"` → `P(miss)_t = sigmoid(−severity·(z_t − mean z))` — low latent state (frustrated) drops more; not a function of observables.

- [ ] **Step 1: Write the failing test**

```python
def test_mnar_low_latent_state_drops_more():
    df = apply_turn_missingness(_traj(seed=4), mechanism="mnar", severity=2.0, seed=5)
    c = np.corrcoef(df["z"], (~df["observed"]).astype(float))[0, 1]
    assert c < -0.1                # lower z → more likely missing
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_dialogue_missingness.py -k mnar -v`
Expected: FAIL — `unknown mechanism 'mnar'`

- [ ] **Step 3: Write minimal implementation**

Extend `_miss_prob`:

```python
    if mechanism == "mnar":
        z = df["z"].to_numpy()
        return 1.0 / (1.0 + np.exp(-severity * (-(z - z.mean()))))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_dialogue_missingness.py -k mnar -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(dialogue-mnar): MNAR mechanism on latent z (#47)"
```

---

## Task 4: Reward estimand — true vs naive-observed

**Files:**
- Create: `causal_bench/estimators/reward_missingness.py`
- Test: `tests/test_reward_missingness.py`

**Interfaces:**
- Produces: `true_reward(df) -> float` = mean of `u` over ALL turns; `naive_reward(df) -> float` = mean of `u` over `observed` turns only. Bias = naive − true.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_reward_missingness.py
import numpy as np
from causal_bench.dgp.dialogue_missingness import apply_turn_missingness
from causal_bench.estimators.reward_missingness import true_reward, naive_reward
from tests.test_dialogue_missingness import _traj

def test_mcar_naive_unbiased_mnar_biased():
    base = _traj(n_traj=600, seed=6)
    mcar = apply_turn_missingness(base, "mcar", severity=0.4, seed=7)
    mnar = apply_turn_missingness(base, "mnar", severity=2.5, seed=7)
    assert abs(naive_reward(mcar) - true_reward(mcar)) < 0.01      # MCAR: unbiased
    assert naive_reward(mnar) - true_reward(mnar) > 0.03           # MNAR: dropping low-u turns inflates reward
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_reward_missingness.py -k unbiased -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Write minimal implementation**

```python
# causal_bench/estimators/reward_missingness.py
from __future__ import annotations
import numpy as np
import pandas as pd

def true_reward(df: pd.DataFrame) -> float:
    return float(df["u"].mean())

def naive_reward(df: pd.DataFrame) -> float:
    return float(df.loc[df["observed"], "u"].mean())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_reward_missingness.py -k unbiased -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add causal_bench/estimators/reward_missingness.py tests/test_reward_missingness.py
git commit -m "feat(dialogue-mnar): true vs naive-observed reward estimand (#47)"
```

---

## Task 5: IPW estimator — corrects MAR, not MNAR

**Files:**
- Modify: `causal_bench/estimators/reward_missingness.py`
- Test: `tests/test_reward_missingness.py`

**Interfaces:**
- Produces: `ipw_reward(df, feature_cols) -> float` — Hájek-normalized inverse-probability-of-observation weighted mean of `u` over observed turns, with `P(observed | features)` from logistic regression on `feature_cols`. With observable features it corrects MAR; it cannot correct MNAR (features don't include `z`).

- [ ] **Step 1: Write the failing test**

```python
from causal_bench.estimators.reward_missingness import ipw_reward

def _with_obs_features(df):
    df = df.copy()
    df["a_prev_abs"] = df.groupby("trajectory_id")["a"].shift(1).abs().fillna(0.0)
    return df

def test_ipw_corrects_mar_but_not_mnar():
    base = _traj(n_traj=800, seed=8)
    mar = _with_obs_features(apply_turn_missingness(base, "mar", severity=2.0, seed=9))
    mnar = _with_obs_features(apply_turn_missingness(base, "mnar", severity=2.5, seed=9))
    t_mar, t_mnar = true_reward(mar), true_reward(mnar)
    # MAR: IPW on the observable closes most of the naive bias
    assert abs(ipw_reward(mar, ["a_prev_abs"]) - t_mar) < abs(naive_reward(mar) - t_mar) * 0.5
    # MNAR: IPW on observables does NOT close the bias (still substantial)
    assert abs(ipw_reward(mnar, ["a_prev_abs"]) - t_mnar) > 0.02
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_reward_missingness.py -k ipw -v`
Expected: FAIL — `cannot import name 'ipw_reward'`

- [ ] **Step 3: Write minimal implementation**

```python
from sklearn.linear_model import LogisticRegression

def ipw_reward(df: pd.DataFrame, feature_cols: list[str]) -> float:
    X = df[feature_cols].to_numpy()
    y = df["observed"].astype(int).to_numpy()
    p = LogisticRegression(max_iter=1000).fit(X, y).predict_proba(X)[:, 1]
    obs = df["observed"].to_numpy()
    w = np.where(obs, 1.0 / np.clip(p, 1e-3, 1.0), 0.0)
    u = df["u"].to_numpy()
    return float(np.sum(w * u) / np.sum(w))   # Hájek
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_reward_missingness.py -k ipw -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(dialogue-mnar): IPW reward — corrects MAR not MNAR (#47)"
```

---

## Task 6: Proxy-corrected estimator — partial MNAR recovery + residual

**Files:**
- Modify: `causal_bench/estimators/reward_missingness.py`
- Test: `tests/test_reward_missingness.py`

**Interfaces:**
- Produces: `proxy_reward(df, proxy_col="z_proxy") -> float` — IPW using a noisy proxy for `z` in the propensity model. Recovers *part* of the MNAR bias in proportion to proxy quality; a nonzero residual remains (the honest endpoint).

- [ ] **Step 1: Write the failing test**

```python
from causal_bench.estimators.reward_missingness import proxy_reward

def test_proxy_partially_recovers_mnar_with_residual():
    base = _traj(n_traj=800, seed=10)
    # good proxy (low noise) recovers more of the MNAR bias than naive, but not all
    mnar = apply_turn_missingness(base, "mnar", severity=2.5, seed=11, proxy_noise_sd=0.3)
    t = true_reward(mnar)
    naive_bias = abs(naive_reward(mnar) - t)
    proxy_bias = abs(proxy_reward(mnar, "z_proxy") - t)
    assert proxy_bias < naive_bias        # partial recovery
    assert proxy_bias > 0.005             # residual remains — no full correction
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_reward_missingness.py -k proxy -v`
Expected: FAIL — `cannot import name 'proxy_reward'`

- [ ] **Step 3: Write minimal implementation**

```python
def proxy_reward(df: pd.DataFrame, proxy_col: str = "z_proxy") -> float:
    return ipw_reward(df, [proxy_col])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_reward_missingness.py -k proxy -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(dialogue-mnar): proxy-corrected reward, partial MNAR recovery (#47)"
```

---

## Task 7: exp27 — mechanism × severity sweep

**Files:**
- Create: `experiments/exp27_dialogue_mnar.py`
- Test: `tests/test_reward_missingness.py`

**Interfaces:**
- Consumes: #46's `generate_user_sim_trajectories` (trajectory source), the missingness layer, and the four reward estimators.
- Produces: `run_missingness_sweep(mechanisms, severities, seed) -> pd.DataFrame` with columns `["mechanism","severity","naive_bias","ipw_bias","proxy_bias"]` (bias = estimate − true). A `run(...)` entrypoint saves `results/exp27_dialogue_mnar/reward_bias.parquet`.

- [ ] **Step 1: Write the failing test**

```python
from experiments.exp27_dialogue_mnar import run_missingness_sweep

def test_sweep_shows_mnar_ipw_gap():
    tbl = run_missingness_sweep(mechanisms=["mcar", "mnar"], severities=[2.0], seed=12)
    mnar = tbl[tbl.mechanism == "mnar"].iloc[0]
    # under MNAR, IPW-on-observables leaves more bias than the proxy correction
    assert abs(mnar["ipw_bias"]) > abs(mnar["proxy_bias"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_reward_missingness.py -k sweep -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Write minimal implementation**

```python
# experiments/exp27_dialogue_mnar.py
"""Exp 27: MNAR turn-missingness in dialogue (#47), exp13 sibling.

Sweeps missingness mechanism × severity over user-simulator trajectories; reports
how biased a trajectory-reward estimand is under naive / IPW-on-observables /
proxy-corrected estimators. MNAR is uncorrectable by IPW-on-observables and only
partially recovered by an observable proxy for the latent state.
"""
from pathlib import Path
import numpy as np
import pandas as pd

from causal_bench.dgp.user_sim import UserSimConfig, generate_user_sim_trajectories
from causal_bench.dgp.dialogue_missingness import apply_turn_missingness
from causal_bench.estimators.reward_missingness import (
    true_reward, naive_reward, ipw_reward, proxy_reward,
)

OUT_DIR = Path("results/exp27_dialogue_mnar")

def _base(seed):
    cfg = UserSimConfig(n_trajectories=800, n_turns=10, shock_rate=0.0, emit_noise_sd=0.1)
    return generate_user_sim_trajectories(cfg, seed=seed)

def run_missingness_sweep(mechanisms, severities, seed=12) -> pd.DataFrame:
    base = _base(seed)
    rows = []
    for mech in mechanisms:
        for sev in severities:
            d = apply_turn_missingness(base, mech, float(sev), seed=seed + 1, proxy_noise_sd=0.3)
            d["a_prev_abs"] = d.groupby("trajectory_id")["a"].shift(1).abs().fillna(0.0)
            t = true_reward(d)
            rows.append({
                "mechanism": mech, "severity": float(sev),
                "naive_bias": naive_reward(d) - t,
                "ipw_bias": ipw_reward(d, ["a_prev_abs"]) - t,
                "proxy_bias": proxy_reward(d, "z_proxy") - t,
            })
    return pd.DataFrame(rows)

def run(seed: int = 12):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tbl = run_missingness_sweep(["mcar", "mar", "mnar"], [1.0, 2.0, 3.0], seed)
    tbl.to_parquet(OUT_DIR / "reward_bias.parquet", index=False)
    print(tbl.to_string(index=False))
    return tbl

if __name__ == "__main__":
    run()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_reward_missingness.py -k sweep -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(exp27): dialogue MNAR mechanism × severity reward-bias sweep (#47)"
```

---

## Self-Review

**Spec coverage (#47):** MCAR/MAR/MNAR grid → Tasks 1–3. Turn-lapse Δt severity axis → `dt` (Task 1). Reward-as-estimand, missingness-as-censoring → Task 4. IPW corrects MAR not MNAR (the exp13 story) → Task 5. Proxy partial-recovery with residual (the "L1 proxy" move + honest endpoint) → Task 6. Mechanism × severity sweep → Task 7.

**Dependency:** Tasks 1–6 are self-contained (tested on hand-built frames via `_traj`); only Task 7 (`exp27`) imports #46's `generate_user_sim_trajectories`, so this plan is blocked on #46 Tasks 1–2 landing *only for the experiment*, not for the core layer. If #46 slips, Tasks 1–6 still proceed.

**Type consistency:** `apply_turn_missingness(...) -> df + [observed, dt, z_proxy]` consumed by all four estimators. `true/naive_reward(df) -> float`, `ipw_reward(df, feature_cols) -> float`, `proxy_reward(df, proxy_col) -> float` consistent Tasks 4–7. The hidden-`z` contract (mechanisms/true_reward may read `z`; naive/ipw may not) is honored: `ipw_reward` takes explicit `feature_cols` and never defaults to `z`.
