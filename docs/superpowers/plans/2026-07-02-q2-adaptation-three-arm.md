# Q2 Three-Arm Adaptation v1 (A1+B1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the three-arm (naive / NC-flag / oracle) belief-tracking experiment from `docs/superpowers/specs/2026-07-02-q2-adaptation-design.md` — measure the fraction of achievable adaptation the imperfect detector captures, and how that capture degrades with observability.

**Architecture:** A shared EKF-style belief filter over the existing user-sim DGP (`causal_bench/dgp/user_sim.py`) that consumes only the agent-observable footprint (`u`, `a`) plus an optional per-turn shock flag; the three arms differ ONLY in the flag channel (none / detector / ground truth). New `causal_bench/adaptation/` package (filter + tracking metrics), one new detector helper (`threshold_at_fpr`), one new experiment (`exp28`). No DGP changes.

**Tech Stack:** numpy, pandas, existing `causal_bench.detectors` modules, pytest.

## Global Constraints

- **Information structure (spec §1):** the belief filter may read only `["trajectory_id", "t", "u", "a"]` plus the flag array. It must never read `n`, `z`, or `e`. (`oracle_flags`/`nc_flags` construct flags outside the filter; the oracle helper is the only place ground-truth `e` is read, and only for the ceiling arm.)
- **Naive is not a strawman (spec §4):** the naive arm both predicts under the endogenous model AND measurement-updates on `u_t`. Do not build a prediction-only baseline.
- **Threshold `c` comes from detection calibration at a target FPR** (spec §4) — computed on a separate calibration draw (different seed), not a free parameter.
- **Headline metric is marginal capture** `(naive − NC-flag) / (naive − oracle)`, not "flag beats naive" (spec §4).
- No new DGP primitives; reuse `z`, `u`, `a`, `nc_residual`, `nc_coupling` (spec §7).
- The filter is given the true model parameters (`gamma`, `beta_emit`, `emit_noise_sd`) — these are *model* knowledge, not shift information; document this in the filter docstring.
- Commit style: `feat: <what> (#46)` / `test: ...`. NEVER write "closes/fixes #N" in any commit or PR body, even negated — use "part of #46".
- If a statistical test assertion fails, do NOT silently retune seeds/thresholds — stop and investigate whether the filter/metric is wrong first.

---

### Task 1: `threshold_at_fpr` detector helper

**Files:**
- Modify: `causal_bench/detectors/metrics.py`
- Test: `tests/test_exogenous_detector.py`

**Interfaces:**
- Consumes: `negative_control_residual` output (`nc_residual` column, NaN on first turns).
- Produces: `threshold_at_fpr(scored_df: pd.DataFrame, e_label, target_fpr: float = 0.1) -> float` — the detection cutoff `c` such that quiet steps exceed it at ≈`target_fpr` rate. Used by Task 3's `nc_flags` and Task 5's calibration.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_exogenous_detector.py`:

```python
def test_threshold_at_fpr_hits_target_on_quiet_steps():
    from causal_bench.detectors.metrics import threshold_at_fpr
    cfg = UserSimConfig(n_trajectories=400, n_turns=8, shock_rate=0.15,
                        shock_delta=2.0, nc_noise_sd=0.3, gamma_action=0.3)
    d = generate_user_sim_trajectories(cfg, seed=6)
    scored = negative_control_residual(d)
    e_prev = (d.sort_values(["trajectory_id", "t"])
                .groupby("trajectory_id")["e"].shift(1).fillna(0).to_numpy())
    c = threshold_at_fpr(scored, e_prev, target_fpr=0.1)
    score = scored["nc_residual"].abs().to_numpy()
    mask = ~np.isnan(score)
    quiet = score[mask][np.asarray(e_prev, dtype=float)[mask] == 0]
    fpr = float((quiet > c).mean())
    assert abs(fpr - 0.1) < 0.02          # empirical FPR ≈ target
    # and the threshold has power: post-shock steps exceed it more often than quiet ones
    hot = score[mask][np.asarray(e_prev, dtype=float)[mask] == 1]
    assert (hot > c).mean() > 0.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_exogenous_detector.py::test_threshold_at_fpr_hits_target_on_quiet_steps -v`
Expected: FAIL with `ImportError: cannot import name 'threshold_at_fpr'`

- [ ] **Step 3: Write minimal implementation**

Append to `causal_bench/detectors/metrics.py`:

```python
def threshold_at_fpr(scored_df: pd.DataFrame, e_label, target_fpr: float = 0.1) -> float:
    """Detection cutoff c for the |nc_residual| detector at a target false-positive rate.

    c is the (1 − target_fpr) quantile of |nc_residual| over quiet steps (previous
    step fired no shock). Flagging when |nc_residual| > c then false-alarms on
    quiet steps at ≈ target_fpr. NaN residuals (first turns) are dropped.
    """
    score = scored_df["nc_residual"].abs().to_numpy()
    y = np.asarray(e_label, dtype=float)
    mask = ~np.isnan(score)
    quiet = score[mask][y[mask] == 0]
    return float(np.quantile(quiet, 1.0 - target_fpr))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_exogenous_detector.py -v`
Expected: all PASS (existing 4 tests + the new one)

- [ ] **Step 5: Commit**

```bash
git add causal_bench/detectors/metrics.py tests/test_exogenous_detector.py
git commit -m "feat: threshold_at_fpr detection cutoff for the NC residual detector (#46)"
```

---

### Task 2: naive belief filter (EKF on the emission)

**Files:**
- Create: `causal_bench/adaptation/__init__.py` (empty)
- Create: `causal_bench/adaptation/filters.py`
- Test: `tests/test_adaptation.py`

**Interfaces:**
- Consumes: DGP long-format DataFrame with columns `[trajectory_id, t, z, u, a, n, e]` (only `trajectory_id, t, u, a` are read).
- Produces: `run_belief_filter(traj_df, *, gamma: float, beta_emit: float = 1.0, emit_noise_sd: float = 0.2, z0_mean: float = 0.0, z0_sd: float = 1.0, q_process: float = 0.05, inflate_var: float = 4.0, flag=None) -> pd.DataFrame` — input rows sorted by `(trajectory_id, t)` with added float columns `z_hat`, `z_hat_var`. `flag` is a bool array aligned to the sorted rows (Task 3 exercises it; this task implements the full signature).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_adaptation.py`:

```python
"""Tests for the Q2 three-arm belief filter and tracking metrics (#46).

Arms differ only in the shock-flag channel: naive (no flag), NC-flag (detector),
oracle (true e). The filter reads only the agent-observable footprint (u, a).
"""
import numpy as np

from causal_bench.dgp.user_sim import UserSimConfig, generate_user_sim_trajectories
from causal_bench.adaptation.filters import run_belief_filter


def _filter_kwargs(cfg):
    return dict(gamma=cfg.gamma_action, beta_emit=cfg.beta_emit,
                emit_noise_sd=cfg.emit_noise_sd, z0_mean=cfg.z0_mean, z0_sd=cfg.z0_sd)


def test_filter_output_shape_and_columns():
    cfg = UserSimConfig(n_trajectories=5, n_turns=6)
    d = generate_user_sim_trajectories(cfg, seed=0)
    f = run_belief_filter(d, **_filter_kwargs(cfg))
    assert len(f) == len(d)
    assert {"z_hat", "z_hat_var"} <= set(f.columns)
    assert (f["z_hat_var"] > 0).all()
    assert f.equals(f.sort_values(["trajectory_id", "t"]).reset_index(drop=True))


def test_filter_tracks_latent_state_without_shocks():
    cfg = UserSimConfig(n_trajectories=300, n_turns=12, shock_rate=0.0,
                        emit_noise_sd=0.2, gamma_action=0.3)
    d = generate_user_sim_trajectories(cfg, seed=1)
    f = run_belief_filter(d, **_filter_kwargs(cfg))
    late = f[f["t"] >= 4]
    err = (late["z_hat"] - late["z"]).abs().mean()
    prior_err = (late["z"] - cfg.z0_mean).abs().mean()
    assert err < 0.6 * prior_err    # far better than never updating the prior


def test_naive_filter_partially_self_corrects_after_shock():
    """Anti-strawman check (spec §4): naive measurement-updates on u, so its
    post-shock error DECREASES over the turns after a shock."""
    cfg = UserSimConfig(n_trajectories=500, n_turns=12, shock_rate=0.08,
                        shock_delta=2.0, emit_noise_sd=0.2, gamma_action=0.3)
    d = generate_user_sim_trajectories(cfg, seed=2)
    f = run_belief_filter(d, **_filter_kwargs(cfg))
    f = f.copy()
    f["abs_err"] = (f["z_hat"] - f["z"]).abs()
    # turns since the most recent shock, per trajectory
    errs_by_k = {}
    for _, g in f.groupby("trajectory_id", sort=False):
        e = g["e"].to_numpy()
        ae = g["abs_err"].to_numpy()
        last = None
        for i in range(len(e)):
            if last is not None:
                k = i - last
                errs_by_k.setdefault(k, []).append(ae[i])
            if e[i] == 1:
                last = i
    assert np.mean(errs_by_k[4]) < 0.8 * np.mean(errs_by_k[1])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_adaptation.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'causal_bench.adaptation'`

- [ ] **Step 3: Write the implementation**

Create empty `causal_bench/adaptation/__init__.py`, then create `causal_bench/adaptation/filters.py`:

```python
"""Belief filters for Q2 adaptation (#46) — the agent-side model the DGP lacks.

Information structure (design spec §1): the filter consumes ONLY the
agent-observable footprint (u, a) plus an optional per-turn shock flag. The
negative control n and the ground truth z, e are never read here — shift
information reaches the belief only via the flag channel. The three arms share
this one filter and differ only in what the flag is: nothing (naive), the NC
detector's flag, or the true shock indicator (oracle ceiling).

The filter is an EKF on the sigmoid emission u = σ(β·z) + ε. It is given the
true model parameters (gamma, beta_emit, emit_noise_sd) — model knowledge, not
shift information: it has no shock term, so an exogenous jump in z is exactly
what it cannot predict.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Columns the belief filter may read — the agent-observable footprint only.
_OBSERVABLE = ["trajectory_id", "t", "u", "a"]


def run_belief_filter(
    traj_df: pd.DataFrame,
    *,
    gamma: float,
    beta_emit: float = 1.0,
    emit_noise_sd: float = 0.2,
    z0_mean: float = 0.0,
    z0_sd: float = 1.0,
    q_process: float = 0.05,
    inflate_var: float = 4.0,
    flag=None,
) -> pd.DataFrame:
    """Filter ẑ_t from the footprint; returns rows sorted by (trajectory_id, t)
    with added columns ``z_hat`` and ``z_hat_var``.

    Per turn: predict ẑ⁻ = ẑ + γ·tanh(a_prev), P⁻ = P + q_process; if the
    aligned ``flag`` entry is truthy, admit an exogenous jump by adding
    ``inflate_var`` to P⁻ so the next emission dominates the update; then an
    EKF measurement update on u_t under h(z) = σ(β·z).
    """
    df = traj_df.sort_values(["trajectory_id", "t"]).reset_index(drop=True)
    obs = df[_OBSERVABLE]
    flag_arr = (np.zeros(len(df), dtype=bool) if flag is None
                else np.asarray(flag, dtype=bool))
    if len(flag_arr) != len(df):
        raise ValueError("flag must align with traj_df rows")
    R = emit_noise_sd**2
    z_hats = np.empty(len(df))
    p_vars = np.empty(len(df))
    i = 0
    for _, g in obs.groupby("trajectory_id", sort=False):
        z_hat, P = z0_mean, z0_sd**2
        a_prev = None
        for u_t, a_t in zip(g["u"], g["a"]):
            if a_prev is not None:
                z_hat = z_hat + gamma * np.tanh(a_prev)
                P = P + q_process
            if flag_arr[i]:
                P = P + inflate_var
            s = 1.0 / (1.0 + np.exp(-beta_emit * z_hat))
            H = beta_emit * s * (1.0 - s)
            S = H * H * P + R
            K = P * H / S
            z_hat = z_hat + K * (u_t - s)
            P = (1.0 - K * H) * P
            z_hats[i], p_vars[i] = z_hat, P
            a_prev = a_t
            i += 1
    out = df.copy()
    out["z_hat"] = z_hats
    out["z_hat_var"] = p_vars
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_adaptation.py -v`
Expected: 3 PASS. If `test_naive_filter_partially_self_corrects_after_shock` fails, the filter's measurement update is broken (e.g. K applied in the wrong space) — debug the filter, do not loosen the assertion.

- [ ] **Step 5: Commit**

```bash
git add causal_bench/adaptation/ tests/test_adaptation.py
git commit -m "feat: EKF belief filter over the user-sim footprint (#46)"
```

---

### Task 3: flag arms — `oracle_flags` and `nc_flags`

**Files:**
- Modify: `causal_bench/adaptation/filters.py`
- Test: `tests/test_adaptation.py`

**Interfaces:**
- Consumes: `run_belief_filter` (Task 2), `negative_control_residual` and `threshold_at_fpr` (Task 1).
- Produces:
  - `oracle_flags(traj_df: pd.DataFrame) -> np.ndarray` — bool array aligned to `(trajectory_id, t)`-sorted rows; True where the *previous* step fired the true shock `e`.
  - `nc_flags(traj_df: pd.DataFrame, threshold: float) -> np.ndarray` — same alignment; True where `|nc_residual| > threshold` (NaN residual → False).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_adaptation.py`:

```python
def test_oracle_and_nc_flags_align_and_fire():
    from causal_bench.adaptation.filters import oracle_flags, nc_flags
    from causal_bench.detectors.exogenous import negative_control_residual
    from causal_bench.detectors.metrics import threshold_at_fpr
    cfg = UserSimConfig(n_trajectories=200, n_turns=8, shock_rate=0.15,
                        shock_delta=2.0, nc_noise_sd=0.3, gamma_action=0.3)
    d = generate_user_sim_trajectories(cfg, seed=3)
    ds = d.sort_values(["trajectory_id", "t"]).reset_index(drop=True)

    ofl = oracle_flags(d)
    expected = (ds.groupby("trajectory_id")["e"].shift(1).fillna(0) == 1).to_numpy()
    assert ofl.dtype == bool and (ofl == expected).all()

    scored = negative_control_residual(d)
    e_prev = ds.groupby("trajectory_id")["e"].shift(1).fillna(0).to_numpy()
    c = threshold_at_fpr(scored, e_prev, target_fpr=0.1)
    nfl = nc_flags(d, threshold=c)
    assert nfl.dtype == bool and len(nfl) == len(d)
    assert not nfl[ds["t"] == 0].any()          # NaN residual on first turns → no flag
    # detector flags fire mostly where the oracle does (δ=2 is well-detectable)
    assert nfl[ofl].mean() > 0.5
    assert nfl[~ofl].mean() < 0.15


def test_oracle_arm_beats_naive_post_shock():
    from causal_bench.adaptation.filters import oracle_flags
    cfg = UserSimConfig(n_trajectories=500, n_turns=12, shock_rate=0.08,
                        shock_delta=2.0, emit_noise_sd=0.2, gamma_action=0.3)
    d = generate_user_sim_trajectories(cfg, seed=4)
    kw = _filter_kwargs(cfg)
    f_naive = run_belief_filter(d, **kw)
    f_oracle = run_belief_filter(d, flag=oracle_flags(d), **kw)

    def post_shock_err(f):
        f = f.copy()
        f["abs_err"] = (f["z_hat"] - f["z"]).abs()
        e_prev = f.groupby("trajectory_id")["e"].shift(1).fillna(0)
        return f.loc[e_prev == 1, "abs_err"].mean()

    assert post_shock_err(f_oracle) < 0.75 * post_shock_err(f_naive)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_adaptation.py -v`
Expected: the two new tests FAIL with `ImportError: cannot import name 'oracle_flags'`

- [ ] **Step 3: Write the implementation**

Append to `causal_bench/adaptation/filters.py`:

```python
def oracle_flags(traj_df: pd.DataFrame) -> np.ndarray:
    """Ceiling arm's flag: the TRUE shock indicator, shifted to the turn where the
    jumped latent state is first observable. Reads ground-truth ``e`` — permitted
    only here, for the oracle ceiling (design spec §4)."""
    d = traj_df.sort_values(["trajectory_id", "t"])
    return (d.groupby("trajectory_id")["e"].shift(1).fillna(0) == 1).to_numpy()


def nc_flags(traj_df: pd.DataFrame, threshold: float) -> np.ndarray:
    """Detector arm's flag: |nc_residual| > threshold, aligned to sorted rows.
    NaN residuals (first turns) never flag."""
    from causal_bench.detectors.exogenous import negative_control_residual

    scored = negative_control_residual(traj_df)   # sorted by (trajectory_id, t)
    resid = scored["nc_residual"].abs().to_numpy()
    return np.nan_to_num(resid, nan=-np.inf) > threshold
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_adaptation.py -v`
Expected: 5 PASS

- [ ] **Step 5: Commit**

```bash
git add causal_bench/adaptation/filters.py tests/test_adaptation.py
git commit -m "feat: NC-flag and oracle flag channels for the three-arm filter (#46)"
```

---

### Task 4: tracking metrics — post-shock error, recovery time, marginal capture

**Files:**
- Create: `causal_bench/adaptation/metrics.py`
- Test: `tests/test_adaptation.py`

**Interfaces:**
- Consumes: `run_belief_filter` output (columns `z`, `e`, `z_hat` at minimum).
- Produces:
  - `tracking_metrics(filtered_df: pd.DataFrame, window: int = 4) -> dict` with keys `post_shock_err` (mean |ẑ−z| over turns 1..window after each shock), `quiet_err` (mean |ẑ−z| on all other turns), `time_to_recover` (mean turns after a shock until |ẑ−z| ≤ 1.5·quiet_err, censored at `window`), `n_shock_turns` (int).
  - `marginal_capture(err_naive: float, err_flag: float, err_oracle: float) -> float` — `(naive − flag)/(naive − oracle)`, NaN if the denominator is ≤ 1e-12 or non-finite.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_adaptation.py`:

```python
def test_tracking_metrics_on_constructed_trajectory():
    import pandas as pd
    from causal_bench.adaptation.metrics import tracking_metrics
    # one trajectory, shock fires at t=2; error is 0 on quiet turns, then 2, 1, 0.05
    df = pd.DataFrame({
        "trajectory_id": [0] * 8,
        "t": range(8),
        "z":     [0.0] * 8,
        "e":     [0, 0, 1, 0, 0, 0, 0, 0],
        "z_hat": [0.0, 0.0, 0.0, 2.0, 1.0, 0.05, 0.0, 0.0],
    })
    m = tracking_metrics(df, window=4)
    # post-shock turns are t=3..6 with errors 2, 1, 0.05, 0
    assert abs(m["post_shock_err"] - np.mean([2.0, 1.0, 0.05, 0.0])) < 1e-12
    assert m["n_shock_turns"] == 4
    # quiet turns t=0,1,2,7 have error 0
    assert m["quiet_err"] == 0.0
    # quiet_err = 0 → tolerance 0 → recovery at first exactly-zero error: k=4 (t=6)
    assert m["time_to_recover"] == 4.0


def test_marginal_capture_bounds_and_degeneracy():
    from causal_bench.adaptation.metrics import marginal_capture
    assert marginal_capture(1.0, 0.4, 0.2) == 0.75
    assert marginal_capture(1.0, 1.0, 0.2) == 0.0
    assert marginal_capture(1.0, 0.2, 0.2) == 1.0
    assert np.isnan(marginal_capture(1.0, 0.5, 1.0))     # no achievable gap
    assert np.isnan(marginal_capture(1.0, 0.5, float("nan")))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_adaptation.py -v`
Expected: the two new tests FAIL with `ModuleNotFoundError`/`ImportError` on `causal_bench.adaptation.metrics`

- [ ] **Step 3: Write the implementation**

Create `causal_bench/adaptation/metrics.py`:

```python
"""Adaptation metrics for the Q2 three-arm contrast (#46).

Post-shock tracking error and time-to-recover per arm, and the headline
marginal-capture ratio (design spec §4): the fraction of achievable adaptation
(naive → oracle) that the imperfect detector's flag captures. "Flag beats
naive" alone is near-tautological and is NOT the reported result.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _turns_since_last_shock(e: np.ndarray) -> np.ndarray:
    """For each turn, turns elapsed since the most recent shock fired (the shock
    at turn i moves z entering turn i+1). NaN before any shock."""
    out = np.full(len(e), np.nan)
    last = None
    for i, e_i in enumerate(e):
        if last is not None:
            out[i] = i - last
        if e_i == 1:
            last = i
    return out


def tracking_metrics(filtered_df: pd.DataFrame, window: int = 4) -> dict:
    """Post-shock tracking error, quiet-step error, and mean time-to-recover.

    post_shock_err: mean |ẑ−z| over turns 1..window after each shock.
    quiet_err: mean |ẑ−z| over all other turns.
    time_to_recover: per shock, first k in 1..window with |ẑ−z| ≤ 1.5·quiet_err
    (censored at window if never, or if the trajectory ends first).
    """
    d = filtered_df.sort_values(["trajectory_id", "t"]).copy()
    d["abs_err"] = (d["z_hat"] - d["z"]).abs()

    post_errs, quiet_errs = [], []
    for _, g in d.groupby("trajectory_id", sort=False):
        e = g["e"].to_numpy()
        ae = g["abs_err"].to_numpy()
        since = _turns_since_last_shock(e)
        in_window = (since >= 1) & (since <= window)
        post_errs.append(ae[in_window])
        quiet_errs.append(ae[~in_window])
    post = np.concatenate(post_errs) if post_errs else np.array([])
    quiet = np.concatenate(quiet_errs) if quiet_errs else np.array([])
    quiet_err = float(quiet.mean()) if len(quiet) else float("nan")

    tol = 1.5 * quiet_err
    recoveries = []
    for _, g in d.groupby("trajectory_id", sort=False):
        e = g["e"].to_numpy()
        ae = g["abs_err"].to_numpy()
        for i in np.flatnonzero(e == 1):
            rec = window
            for k in range(1, window + 1):
                if i + k < len(ae) and ae[i + k] <= tol:
                    rec = k
                    break
            recoveries.append(rec)

    return {
        "post_shock_err": float(post.mean()) if len(post) else float("nan"),
        "quiet_err": quiet_err,
        "time_to_recover": float(np.mean(recoveries)) if recoveries else float("nan"),
        "n_shock_turns": int(len(post)),
    }


def marginal_capture(err_naive: float, err_flag: float, err_oracle: float) -> float:
    """Headline (spec §4): fraction of achievable adaptation the detector captures,
    (naive − flag) / (naive − oracle). NaN when there is no achievable gap."""
    denom = err_naive - err_oracle
    if not np.isfinite(denom) or denom <= 1e-12:
        return float("nan")
    return float((err_naive - err_flag) / denom)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_adaptation.py -v`
Expected: 7 PASS

- [ ] **Step 5: Commit**

```bash
git add causal_bench/adaptation/metrics.py tests/test_adaptation.py
git commit -m "feat: post-shock tracking error, recovery time, marginal capture (#46)"
```

---

### Task 5: exp28 — three-arm experiment and capture-vs-observability sweep

**Files:**
- Create: `experiments/exp28_q2_adaptation.py`
- Test: `tests/test_exp28_adaptation.py`

**Interfaces:**
- Consumes: everything above plus `UserSimConfig`, `generate_user_sim_trajectories`, `negative_control_residual`.
- Produces:
  - `run_three_arm(shock_delta=2.0, nc_coupling=1.0, n_trajectories=400, n_turns=12, seed=11, target_fpr=0.1, window=4) -> pd.DataFrame` — 3 rows (arm ∈ naive/nc_flag/oracle), columns `arm, shock_delta, nc_coupling, threshold, post_shock_err, quiet_err, time_to_recover, n_shock_turns, capture` (capture repeated on all rows).
  - `run_capture_vs_observability(couplings, shock_delta=2.0, ...) -> pd.DataFrame` — concatenated `run_three_arm` tables, one per coupling.
  - `run()` — writes `results/exp28_q2_adaptation/{three_arm,capture_vs_observability}.parquet`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_exp28_adaptation.py`:

```python
"""Tests for exp28 — the Q2 three-arm adaptation contrast (#46).

Headline is the marginal-capture ratio (naive − NC-flag)/(naive − oracle), and
its predicted degradation as the negative control weakens (spec §4)."""
import numpy as np

from experiments.exp28_q2_adaptation import run_three_arm, run_capture_vs_observability


def test_three_arm_ordering_at_high_observability():
    tbl = run_three_arm(shock_delta=2.0, nc_coupling=1.0, n_trajectories=400,
                        n_turns=12, seed=11)
    assert set(tbl["arm"]) == {"naive", "nc_flag", "oracle"}
    err = tbl.set_index("arm")["post_shock_err"]
    # oracle is the ceiling; a good detector puts nc_flag strictly between
    assert err["oracle"] < err["nc_flag"] < err["naive"]
    cap = tbl["capture"].iloc[0]
    assert (tbl["capture"] == cap).all()
    assert cap > 0.3               # near-direct control captures a real fraction
    rec = tbl.set_index("arm")["time_to_recover"]
    assert rec["oracle"] <= rec["naive"]


def test_capture_degrades_as_control_weakens():
    tbl = run_capture_vs_observability(couplings=[1.0, 0.05], shock_delta=2.0,
                                       n_trajectories=400, n_turns=12, seed=11)
    caps = tbl.groupby("nc_coupling")["capture"].first()
    # as the control degrades the NC-flag arm slides from oracle toward naive
    assert caps[0.05] < caps[1.0]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_exp28_adaptation.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'experiments.exp28_q2_adaptation'`

- [ ] **Step 3: Write the implementation**

Create `experiments/exp28_q2_adaptation.py`:

```python
"""Exp 28: Q2 three-arm adaptation contrast (#46; design spec 2026-07-02).

Given the eₜ detector (exp26/Q1), how much does acting on its flag help the
agent's belief re-track the user's latent state after an unobserved shock?
Three arms sharing one EKF belief filter over the footprint (u, a):

- naive: self-correcting filter, no flag (NOT a strawman — it measurement-
  updates on u every turn, so it partially recovers on its own);
- nc_flag: same filter; on |nc_residual| > c it inflates belief variance so
  the next emission dominates (c calibrated at a target FPR on a separate
  draw — the tie-back to exp26's ROC);
- oracle: same filter conditioned on the true shock indicator — the ceiling.

Headline: marginal capture (naive − nc_flag)/(naive − oracle), and its
degradation as the negative control weakens (nc_coupling ↓). "Flag beats
naive" alone is near-tautological and is not the reported result.

v1 limit (spec §5): this measures belief-tracking (A1+B1), a proxy for
adaptation, not task outcome; the act-on-belief reward loop (B2) is deferred.
"""
from pathlib import Path

import pandas as pd

from causal_bench.adaptation.filters import nc_flags, oracle_flags, run_belief_filter
from causal_bench.adaptation.metrics import marginal_capture, tracking_metrics
from causal_bench.detectors.exogenous import negative_control_residual
from causal_bench.detectors.metrics import threshold_at_fpr
from causal_bench.dgp.user_sim import UserSimConfig, generate_user_sim_trajectories

OUT_DIR = Path("results/exp28_q2_adaptation")


def _make_cfg(shock_delta, nc_coupling, n_trajectories, n_turns):
    return UserSimConfig(n_trajectories=n_trajectories, n_turns=n_turns,
                         shock_rate=0.15, shock_delta=float(shock_delta),
                         nc_noise_sd=0.3, nc_coupling=float(nc_coupling),
                         gamma_action=0.3)


def calibrate_threshold(cfg: UserSimConfig, seed: int, target_fpr: float = 0.1) -> float:
    """Detection cutoff at target FPR from a separate calibration draw (Q1 tie-back)."""
    d = generate_user_sim_trajectories(cfg, seed=seed)
    scored = negative_control_residual(d)
    e_prev = (d.sort_values(["trajectory_id", "t"])
                .groupby("trajectory_id")["e"].shift(1).fillna(0).to_numpy())
    return threshold_at_fpr(scored, e_prev, target_fpr=target_fpr)


def run_three_arm(shock_delta: float = 2.0, nc_coupling: float = 1.0,
                  n_trajectories: int = 400, n_turns: int = 12, seed: int = 11,
                  target_fpr: float = 0.1, window: int = 4) -> pd.DataFrame:
    cfg = _make_cfg(shock_delta, nc_coupling, n_trajectories, n_turns)
    c = calibrate_threshold(cfg, seed=seed + 1000, target_fpr=target_fpr)
    d = generate_user_sim_trajectories(cfg, seed=seed)
    kw = dict(gamma=cfg.gamma_action, beta_emit=cfg.beta_emit,
              emit_noise_sd=cfg.emit_noise_sd, z0_mean=cfg.z0_mean, z0_sd=cfg.z0_sd)
    arms = {"naive": None, "nc_flag": nc_flags(d, threshold=c), "oracle": oracle_flags(d)}
    rows = []
    for arm, fl in arms.items():
        m = tracking_metrics(run_belief_filter(d, flag=fl, **kw), window=window)
        rows.append({"arm": arm, "shock_delta": float(shock_delta),
                     "nc_coupling": float(nc_coupling), "threshold": c, **m})
    tbl = pd.DataFrame(rows)
    err = tbl.set_index("arm")["post_shock_err"]
    tbl["capture"] = marginal_capture(err["naive"], err["nc_flag"], err["oracle"])
    return tbl


def run_capture_vs_observability(couplings, shock_delta: float = 2.0,
                                 n_trajectories: int = 400, n_turns: int = 12,
                                 seed: int = 11, target_fpr: float = 0.1,
                                 window: int = 4) -> pd.DataFrame:
    tables = [run_three_arm(shock_delta, coupling, n_trajectories, n_turns,
                            seed=seed + i, target_fpr=target_fpr, window=window)
              for i, coupling in enumerate(couplings)]
    return pd.concat(tables, ignore_index=True)


def run(n_trajectories: int = 400, seed: int = 11):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    three = run_three_arm(n_trajectories=n_trajectories, seed=seed)
    three.to_parquet(OUT_DIR / "three_arm.parquet", index=False)
    print(three.to_string(index=False))
    sweep = run_capture_vs_observability([1.0, 0.7, 0.5, 0.3, 0.1],
                                         n_trajectories=n_trajectories, seed=seed)
    sweep.to_parquet(OUT_DIR / "capture_vs_observability.parquet", index=False)
    print(sweep[["nc_coupling", "arm", "post_shock_err", "time_to_recover", "capture"]]
          .to_string(index=False))
    return three, sweep


if __name__ == "__main__":
    run()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_exp28_adaptation.py -v`
Expected: 2 PASS (each takes tens of seconds — the filter loop is per-row Python). If the ordering assertion fails, inspect the per-arm table (`run_three_arm(...)` in a REPL) before touching thresholds: check the detector's flag rate at coupling 1.0 (should be ≈ FPR on quiet turns, >0.5 on shock turns) and that `inflate_var` actually moves `z_hat` post-shock.

- [ ] **Step 5: Run the experiment end-to-end once**

Run: `python -m experiments.exp28_q2_adaptation`
Expected: prints the two tables; capture at coupling 1.0 clearly above capture at 0.1; `results/exp28_q2_adaptation/*.parquet` written. Sanity-read the numbers against spec §4's prediction before committing.

- [ ] **Step 6: Commit**

```bash
git add experiments/exp28_q2_adaptation.py tests/test_exp28_adaptation.py
git commit -m "feat: exp28 three-arm Q2 adaptation contrast with marginal-capture headline (#46)"
```

---

### Task 6: full-suite verification and PR

**Files:**
- Modify: none expected (fix regressions if the suite finds any)

- [ ] **Step 1: Run the full test suite**

Run: `pytest tests/ -x -q`
Expected: all tests pass (some existing tests are slow / need optional R deps — if a pre-existing failure is unrelated to this work, confirm it also fails on `main` before ignoring it).

- [ ] **Step 2: Push branch and open PR**

```bash
git push -u origin feat/q2-three-arm-adaptation
gh pr create --title "feat: three-arm Q2 adaptation contrast (exp28) (#46)" --body "$(cat <<'EOF'
Implements the v1 (A1+B1) three-arm design from the Q2 design pass (#55, spec docs/superpowers/specs/2026-07-02-q2-adaptation-design.md). Part of #46.

- Shared EKF belief filter over the agent-observable footprint (u, a); shift information reaches the belief ONLY via the flag channel (spec §1).
- Three arms: self-correcting naive (no flag, not a strawman), NC-flag (variance inflation on |nc_residual| > c, with c calibrated at target FPR on a separate draw — the exp26/Q1 tie-back), and an oracle ceiling on the true shock indicator.
- Headline: marginal capture (naive − NC-flag)/(naive − oracle) and its degradation as nc_coupling weakens; post-shock tracking error and time-to-recover per arm.
- Stops at the Guga seam (spec §6): belief-tracking proxy only; A2 (discrete goals) and B2 (act-on-belief reward loop) deferred.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

Expected: PR created. (Reminder: no closing keywords anywhere in the body.)

---

## Deviation log (recorded during execution)

- **exp28 uses `shock_rate=0.08`, not exp26's 0.15.** Shocks are same-sign, so at rate 0.15 over 12 turns `z` accumulates into the sigmoid emission's saturated range (53% of post-shock turns at |z|>3), where `u` is uninformative — no arm can re-track, the achievable gap collapses, and the NC arm's false-positive inflations even nudged it past the "ceiling" (capture 1.017). At 0.08 the regime is informative and the predicted curve is clean. Detection itself is unaffected (the NC is linear in `z`).
- **Task 5's ordering test was split in two.** At coupling 1.0 the detector is near-perfect, so `nc_flag` is statistically AT the oracle ceiling and strict `oracle < nc_flag` tests noise. The strict betweenness `oracle < nc_flag < naive` is asserted at coupling 0.3 (imperfect detector); the coupling-1.0 test asserts near-ceiling capture (> 0.7) instead.
- **Task 4's exact float equalities** in `test_marginal_capture_bounds_and_degeneracy` were switched to `pytest.approx` (0.6/0.8 is not exactly representable).
