# causal_bench Phase 2: LTMLE, Collider Trap, concrete Bridge

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extend the MVP with LTMLE (the correct estimator for time-varying confounders), the collider trap experiment (Exp 5), keyed event hashing for paired comparisons, and a reticulate-based R bridge to McCoy's `concrete` package.

**Why now:** McCoy is actively adding post-randomization predictors of censoring (G(C>t | W_baseline, L1_time-varying)) to `concrete`. Our Python LTMLE + Exp 5 directly benchmarks the same problem from the other direction. The reticulate bridge lets us run identical datasets through both systems while McCoy is still developing.

**Architecture:** All new estimators follow the existing `BaseEstimator` interface. LTMLE lives at `causal_bench/estimators/ltmle.py`. The R bridge lives at `causal_bench/estimators/concrete_rmst.py` and uses `rpy2` (already in requirements) rather than subprocess. Experiments are standalone scripts in `experiments/`.

**Tech Stack:** Same as MVP plus `rpy2>=3.5` (already declared), R + `concrete` package (user installs separately).

---

## Revised priority order

```
Phase 2a (do now):
  Task 17  Keyed event hashing
  Task 18  Exp 1 — censoring gradient (first public demo)
  Task 19  Cox+L1 collider estimator (stub already exists)
  Task 20  LTMLE estimator
  Task 21  Exp 5 — collider trap

Phase 2b (after Exp 5):
  Task 22  IPW + overlap weighting estimators
  Task 23  AIPW estimator
  Task 24  Exp 7 — edwards_realistic combined (money experiment)

Phase 2c (R bridge — coordinate with McCoy):
  Task 25  reticulate bridge to concrete
  Task 26  Exp 8 — McCoy experiment (RMST vs pointwise)
```

---

## Task 17: Keyed event hashing

**Why:** Before running multi-scenario experiments (Exp 7, Edwards variants), we need the same patient to appear in edwards_realistic, edwards_optimistic, and edwards_pessimistic. Without this, cross-scenario differences reflect both scenario parameters AND random patient sampling. Keyed hashing eliminates the sampling variance.

**Files:**
- Create: `causal_bench/causal_bench/dgp/keyed_random.py`
- Modify: `causal_bench/causal_bench/dgp/survival.py` — update `compute_true_effects` to use keyed hashing for the reference population

**Step 1: Write failing test**

```python
# tests/test_dgp.py (add)
from causal_bench.dgp.keyed_random import keyed_uniform

def test_keyed_uniform_deterministic():
    v1 = keyed_uniform(patient_id=5, event_type="treatment", scenario="clean", seed=42)
    v2 = keyed_uniform(patient_id=5, event_type="treatment", scenario="clean", seed=42)
    assert v1 == v2

def test_keyed_uniform_range():
    vals = [keyed_uniform(i, "survival", "clean", 0) for i in range(1000)]
    assert all(0.0 <= v < 1.0 for v in vals)

def test_keyed_uniform_scenario_independence():
    """Same patient, different scenarios → different values."""
    v1 = keyed_uniform(5, "survival", "edwards_realistic", 0)
    v2 = keyed_uniform(5, "survival", "edwards_pessimistic", 0)
    assert v1 != v2

def test_keyed_uniform_patient_independence():
    v1 = keyed_uniform(1, "survival", "clean", 0)
    v2 = keyed_uniform(2, "survival", "clean", 0)
    assert v1 != v2
```

**Step 2: Implement keyed_random.py**

```python
# causal_bench/dgp/keyed_random.py
import hashlib


def keyed_uniform(patient_id: int, event_type: str,
                  scenario: str, seed: int) -> float:
    """Deterministic uniform(0,1) draw keyed by (patient_id, event_type, scenario, seed).

    Enables paired counterfactual comparisons: the same patient appears in
    multiple scenarios with identical baseline covariates.
    Reference: Buffalo et al. (2026).
    """
    key = f"{seed}:{patient_id}:{event_type}:{scenario}"
    h = hashlib.sha256(key.encode()).digest()
    return int.from_bytes(h[:8], "big") / (2**64)


def keyed_normal(patient_id: int, event_type: str,
                 scenario: str, seed: int) -> float:
    """Standard normal draw via inverse CDF of keyed_uniform."""
    from scipy.stats import norm
    u = keyed_uniform(patient_id, event_type, scenario, seed)
    # Clamp away from 0/1 to avoid ±inf
    u = max(1e-10, min(1 - 1e-10, u))
    return float(norm.ppf(u))
```

**Step 3: Run tests, commit**

```bash
pytest tests/test_dgp.py -v
git commit -m "feat: keyed event hashing for paired counterfactual comparisons"
```

---

## Task 18: Exp 1 — Censoring gradient

**Files:**
- Create: `experiments/exp1_censoring.py`

**What it does:** Sweeps `censoring_informativeness` from 0 to 1.0 in 6 steps. 200 sims (fast) or 500 sims (publication). All 5 MVP estimators. Saves 5-row panel plot.

**Expected result:** Naive and KM degrade monotonically. TMLE+IPCW stays flat. TMLE+IPCW+Comply is best at high informativeness. This is the core story.

```python
# experiments/exp1_censoring.py
"""Exp 1: Censoring informativeness gradient."""
from pathlib import Path
import numpy as np
from causal_bench.dgp.scenarios import get_scenario
from causal_bench.estimators import MVP_ESTIMATORS
from causal_bench.runner import run_parameter_sweep
from causal_bench.viz import plot_panel, generate_summary_table

PARAM_VALUES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
N_SIMS = 200  # increase to 500 for publication
OUT_DIR = Path("results/exp1_censoring")

if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = get_scenario("censor_moderate")  # censoring_rate=0.30

    results = run_parameter_sweep(
        base_config=base,
        param_name="censoring_informativeness",
        param_values=PARAM_VALUES,
        estimator_names=MVP_ESTIMATORS,
        n_sim=N_SIMS,
        n_jobs=-1,
        seed=42,
    )

    fig = plot_panel(
        results, PARAM_VALUES, "censoring_informativeness",
        title="Exp 1: Censoring informativeness gradient",
        save_path=str(OUT_DIR / "panel.png"),
    )
    print(f"Saved panel plot → {OUT_DIR}/panel.png")

    # Save per-value summary tables
    for i, val in enumerate(PARAM_VALUES):
        slice_results = {name: sr_list[i]
                         for name, sr_list in results.items()
                         if sr_list[i] is not None}
        tbl = generate_summary_table(slice_results)
        (OUT_DIR / f"summary_inf{val:.1f}.md").write_text(tbl)

    print("Done.")
```

**Run it:**
```bash
python experiments/exp1_censoring.py
```

**Commit:**
```bash
git commit -m "feat: Exp 1 censoring gradient experiment"
```

---

## Task 19: Cox+L1 collider estimator

**Files:**
- Modify: `causal_bench/causal_bench/estimators/cox.py` — `CoxEstimator(include_L1=True)` already exists; verify it works with the DGP's L1 column
- Modify: `causal_bench/causal_bench/estimators/__init__.py` — add `cox_l1` to registry
- Modify: `causal_bench/causal_bench/viz.py` — add color/label for `cox_l1`

**The DGP already generates L1 when `collider_strength > 0`** (it's in the spec but the MVP survival.py skips it). For the collider experiment, we need to add L1 generation to `survival.py`.

**Add L1 to survival.py generate_data:**

When `config.collider_strength > 0`, generate L1 and add it to the returned DataFrame:
```python
# After T_true is computed, before censoring:
L1_obs = None
if config.collider_strength > 0:
    # L1 is caused by A (mediator) and U (confounder), measured at t_L1
    L1_raw = (0.5 * A + 0.4 * W3 + 0.3 * U * config.collider_strength
              + rng.standard_normal(n) * config.sigma_L)
    # Only observed if patient still in study at t_L1
    alive_at_L1 = T_true > config.t_L1
    L1_obs = np.where(alive_at_L1, L1_raw, np.nan)
```

Add `"L1": L1_obs` to the returned DataFrame (will be NaN when `collider_strength=0` or patient dropped out).

**Update registry:**
```python
# estimators/__init__.py — add:
from causal_bench.estimators.cox import CoxEstimator
ESTIMATOR_REGISTRY["cox_l1"] = CoxEstimator(include_L1=True)
```

**Update viz.py colors/labels:**
```python
COLORS["cox_l1"] = "#B30000"   # dark red (danger — collider bias)
LABELS["cox_l1"] = "Cox+L1 (collider)"
```

**Tests:**
```python
def test_l1_generated_when_collider_strength_positive():
    cfg = DGPConfig(n=300, collider_strength=0.5, seed=0)
    df = generate_data(cfg)
    assert "L1" in df.columns
    assert df["L1"].notna().any()

def test_l1_absent_when_collider_strength_zero():
    cfg = DGPConfig(n=300, collider_strength=0.0, seed=0)
    df = generate_data(cfg)
    # L1 column may exist but should be all NaN or absent
    if "L1" in df.columns:
        assert df["L1"].isna().all()

def test_cox_l1_returns_result():
    from causal_bench.estimators.cox import CoxEstimator
    cfg = DGPConfig(n=300, collider_strength=0.5, seed=0)
    df = generate_data(cfg)
    results = CoxEstimator(include_L1=True, n_bootstrap=5).estimate(df)
    assert results[0].name == "Cox+L1"
```

**Commit:**
```bash
git commit -m "feat: L1 time-varying confounder in DGP, Cox+L1 collider estimator"
```

---

## Task 20: LTMLE estimator

**Files:**
- Create: `causal_bench/causal_bench/estimators/ltmle.py`

**Algorithm:** Sequential regression marginalizing over L1 so no collider bias is introduced.

```
Step 1. Fit E[Y | A, W, L1] using SuperLearner on patients with observed L1
Step 2. Target with IPCW clever covariate at t_L1 (censoring between 0 and t_L1)
Step 3. Pseudo-outcome: for each patient, predict under observed L1 → marginalize over L1 distribution
Step 4. Regress pseudo-outcome on (A, W) — no L1
Step 5. Target with treatment clever covariate at baseline
Step 6. EIF-based SE
```

**Key property:** L1 enters Step 1 but is *marginalized out* in Step 3 so it never enters the final estimand. No collider bias because we never condition on L1 in the treatment model.

**Implementation sketch:**

```python
class LTMLEEstimator(BaseEstimator):
    name = "LTMLE"

    def __init__(self, n_folds=5, random_state=42):
        self.n_folds = n_folds
        self.random_state = random_state

    def estimate(self, df, horizon=1.0, estimand="ATE"):
        # Requires L1 column; falls back to TMLE+IPCW if absent
        if "L1" not in df.columns or df["L1"].isna().all():
            from causal_bench.estimators.tmle_ipcw import TMLEIPCWEstimator
            return TMLEIPCWEstimator().estimate(df, horizon, estimand)

        W_cols = ["W1", "W2", "W3", "W4"]
        # ... (full implementation in task)
```

**Tests:**
```python
def test_ltmle_returns_result():
    cfg = DGPConfig(n=400, collider_strength=0.5, seed=0)
    df = generate_data(cfg)
    from causal_bench.estimators.ltmle import LTMLEEstimator
    results = LTMLEEstimator(n_folds=3).estimate(df, horizon=1.0)
    assert results[0].name == "LTMLE"
    assert not np.isnan(results[0].point_estimate)

def test_ltmle_falls_back_without_l1():
    cfg = DGPConfig(n=300, collider_strength=0.0, seed=0)
    df = generate_data(cfg)
    from causal_bench.estimators.ltmle import LTMLEEstimator
    results = LTMLEEstimator(n_folds=3).estimate(df)
    # Falls back to TMLE+IPCW, still returns a result
    assert len(results) >= 1
```

**Add to registry:**
```python
ESTIMATOR_REGISTRY["ltmle"] = LTMLEEstimator()
```

**Commit:**
```bash
git commit -m "feat: LTMLE estimator for time-varying confounders"
```

---

## Task 21: Exp 5 — Collider trap

**Files:**
- Create: `experiments/exp5_collider.py`

**What it does:** Sweeps `collider_strength` 0→1. Compares:
- Cox (no L1) — confounding bias from omitting L1
- Cox+L1 — collider bias from conditioning on L1
- LTMLE — correct answer, marginalizes over L1

**The key visual:** At high `collider_strength`, Cox and Cox+L1 both show bias in *opposite directions*. LTMLE stays near zero. This is the "impossible choice" — naive methods are wrong either way. Only LTMLE is right.

```python
# experiments/exp5_collider.py
"""Exp 5: Collider trap — time-varying confounder gradient."""
from pathlib import Path
from causal_bench.dgp.config import DGPConfig
from causal_bench.runner import run_parameter_sweep
from causal_bench.viz import plot_panel, plot_collider_panel

PARAM_VALUES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
ESTIMATORS = ["cox", "cox_l1", "ltmle", "tmle_ipcw"]
OUT_DIR = Path("results/exp5_collider")

if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = DGPConfig(n=500, censoring_informativeness=0.3, true_tau=-0.5)

    results = run_parameter_sweep(
        base_config=base,
        param_name="collider_strength",
        param_values=PARAM_VALUES,
        estimator_names=ESTIMATORS,
        n_sim=200,
        n_jobs=-1,
        seed=42,
    )

    # Standard panel plot
    plot_panel(results, PARAM_VALUES, "collider_strength",
               title="Exp 5: Collider trap",
               save_path=str(OUT_DIR / "panel.png"))

    print(f"Saved → {OUT_DIR}/panel.png")
```

**Add `plot_collider_panel` to viz.py** — 3-column subplot showing Cox / Cox+L1 / LTMLE bias separately, making the opposite-direction bias visually clear.

**Commit:**
```bash
git commit -m "feat: Exp 5 collider trap experiment"
```

---

## Task 22: IPW + Overlap weighting estimators

**Files:**
- Create: `causal_bench/causal_bench/estimators/ipw.py`
- Create: `causal_bench/causal_bench/estimators/overlap.py`

**IPW:** Horvitz-Thompson estimator with weight truncation at 1st/99th percentile. Sandwich SE.
**Overlap:** Li, Morgan & Zaslavsky (2018) weights h(x) = g(x)(1-g(x)). Targets ATO (different estimand — stable under positivity violations).

**Add to registry:** `"ipw"`, `"overlap"`
**Add to viz:** colors from spec (`#E34A33` for IPW, `#9E9AC8` for Overlap)

---

## Task 23: AIPW estimator

**Files:**
- Create: `causal_bench/causal_bench/estimators/aipw.py`

**AIPW:** Doubly-robust EIF estimator. SuperLearner for Q and g. No IPCW. Sandwich SE. ATE only.
This is simpler than TMLE (no targeting step) but still doubly robust.

---

## Task 24: Exp 7 — Edwards combined (money experiment)

Run `edwards_realistic`, `edwards_optimistic`, `edwards_pessimistic` with all estimators.
1000 sims each. Forest plot + bias-variance scatter + summary table.
**Requires keyed event hashing (Task 17) to be done first.**

---

## Task 25: rpy2 bridge to concrete (reticulate-compatible)

**Bridge direction:** `rpy2` calls R *from* Python. `reticulate` calls Python *from* R (what McCoy uses in RStudio). These are complementary — our bridge is rpy2 on the Python side; `r_scripts/concrete_bridge.R` is structured so McCoy can also source it directly in RStudio via reticulate.

**Serialization note:** DataFrames are passed in-memory via `pandas2ri` (Arrow-backed). No CSV, no pickle, no subprocess. `joblib` pickling in the MC runner is for config dicts (tiny) — not DataFrames — and is fine at current scale (n≤700). Spark/Arrow persistence is post-Phase-2 if run counts scale to 10k+.

**Files:**
- Create: `causal_bench/estimators/concrete_rmst.py`
- Create: `r_scripts/concrete_bridge.R`
- Create: `tests/test_concrete_bridge.py` — DataFrame conversion edge cases (runs without R installed)

**Python side (calls R via rpy2):**
```python
def _concrete_available() -> bool:
    try:
        import rpy2.robjects as ro
        ro.packages.importr("concrete")
        return True
    except Exception:
        return False

class ConcreteRMSTEstimator(BaseEstimator):
    name = "concrete_RMST"

    def estimate(self, df, horizon=1.0, estimand="ATE"):
        if not _concrete_available():
            import warnings
            warnings.warn("concrete R package not available — skipping")
            return []
        import rpy2.robjects as ro
        from rpy2.robjects import pandas2ri
        pandas2ri.activate()
        concrete = ro.packages.importr("concrete")
        r_df = pandas2ri.py2rpy(df)
        args = concrete.formatArguments(
            DataTable=r_df,
            EventTime=ro.StrVector(["T_obs"]),
            EventType=ro.StrVector(["event_type"]),
            Treatment=ro.StrVector(["A"]),
            Intervention=ro.IntVector([0, 1]),
            TargetTime=ro.FloatVector([horizon]),
        )
        est = concrete.doConcrete(args)
        rmst = concrete.targetRMST(est)
        # parse result...
```

**R-side script (reticulate-compatible):** `r_scripts/concrete_bridge.R` can be sourced by McCoy in RStudio:
```r
library(reticulate)
library(concrete)
# source_python("causal_bench/dgp/survival.py")  # McCoy can call generate_data() directly
run_concrete_bridge <- function(py_df, horizon = 1.0) { ... }
```

**DataFrame conversion edge cases to test (no R needed):**
- NaN in L1 column (most patients have NaN L1 — must not crash pandas2ri)
- Float64 vs float32 columns
- Boolean columns (Delta) that should convert to int
- Integer treatment A that should stay integer
- Zero-variance columns
- n=0 edge case

**Graceful fallback:** returns `[]` (empty list) if rpy2 not installed or concrete unavailable — the MC runner treats missing estimator results as N/A in summary tables.

---

## Task 26: Exp 8 — McCoy experiment (RMST)

**Requires:** competing risks DGP (not yet implemented), concrete bridge (Task 25).
Compare:
- `concrete` direct RMST (estimator #12)
- Pointwise-then-integrate at K=2, 5, 10, 20 (estimator #13)

Reproduces McCoy's finding: +37 days bias at K=2, eliminated at K=20, eliminated entirely by direct RMST targeting.

---

## Post-Phase-2 checklist

- [ ] Diagnostics module (positivity, SMD, SE calibration)
- [ ] Competing risks DGP (needed for Exp 8)
- [ ] Pointwise RMST estimator (#13)
- [ ] Tipping-point sensitivity (`--tipping-point` flag)
- [ ] ESS diagnostic (`--ess` flag)
- [ ] Jupyter walkthrough notebook
- [ ] Full README update with actual results

---

## References added in this phase

- van der Laan & Gruber (2012). LTMLE. *Int J Biostatistics*.
- Li, Morgan & Zaslavsky (2018). Balancing covariates via propensity score weighting.
- McCoy (2026). Direct RMST targeting for competing-risks TMLE. `concrete` R package.
- Buffalo et al. (2026). Event-keyed hashing for causally valid simulations.
