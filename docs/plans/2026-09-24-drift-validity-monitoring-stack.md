# Drift & validity monitoring stack — composition spec

**Motivation.** R.D. Montgomery's calibration loop ("A Probability You Can Count On") monitors a *predictive*
model in production: a PSI drift alarm on raw scores + a labeled audit queue + periodic refit. causal_bench
already has the pieces to (a) make that loop **statistically rigorous** and (b) extend it from *prediction*
drift to **causal-validity** drift. This spec lays out the layered stack and what to build.

The through-line: monitors 1–4 below are all **prediction-layer** — they detect that the *predictor* is
degrading. Layer 5 is the **causal layer** — whether the drift has broken the *estimate's validity*. A
non-drifting, well-calibrated predictor can still be causally invalid after a shift that breaks overlap or
opens a back-door. Prediction-drift monitoring is **necessary but not sufficient** for causal validity — the
same refrain as ESS (positivity) and calibration (exp52/exp57).

## The stack (by label-cost × what it detects × rigor)

| # | Layer | Detects | Label cost | Built? |
|---|-------|---------|-----------|--------|
| 1 | **PSI / covariate-shift alarm** (Montgomery) | input/score *distribution* shift — NO accuracy signal | none | ext. (trivial) |
| 2 | **AGL — agreement-on-the-line** | *accuracy degradation* under shift (label-free) | none, needs ≥2 decorrelated models | **NO (concept only)** |
| 3 | **Streaming error detectors — DDM / EDDM / ADWIN / Page–Hinkley** | error-rate *change-point* on the labeled stream; ADWIN also picks the refit window | labeled stream | NO (classic, thin) |
| 4 | **Anytime-valid monitor** (`sequential.py`) | error/calibration drift with *valid* continuous peeking | labeled stream | **YES** |
| 5 | **Causal-validity impact** (exp38 / exp17 / exp37) | did the shift break positivity / transport / confounding | labeled or design | **YES** |

### 1. PSI (Montgomery) — the cheap trigger
Population-stability index on the raw-score histogram. Label-free, fires on *any* input/score move, but a move
need not hurt accuracy — high false-alarm rate, no validity signal. The outermost, cheapest tripwire.

### 2. AGL (Agreement-on-the-Line) — the label-free accuracy estimate
Under distribution shift, two models' *agreement* is linearly correlated with their *accuracy* (same line as
in-distribution), so inter-model agreement **estimates OOD accuracy without labels**. Sits between PSI (input
shift, no accuracy) and the audit queue (labeled): it tells you whether the shift PSI flagged actually *hurts*.
**Load-bearing requirement (from the AGL memory): independent failures ⇒ cross-BACKBONE models, not cross-seed.**
- *Prediction use:* run two decorrelated churn predictors; agreement drop ⇒ accuracy drop ⇒ trigger the audit.
- *Causal extension (novel, worth flagging):* the AGL template applied to the **estimate**, not the predictor —
  do two DR estimators with **different nuisance backbones** agree on the effect? Cross-backbone disagreement of
  the causal contrast is a label-free drift/validity signal for the *causal* quantity (ties to the metric-hacking
  guard, causal_bench #126). Honest caveat: AGL estimates *predictive* accuracy; two agreeing models can both be
  causally wrong (both confounded) — so even the causal extension is necessary-not-sufficient re unmeasured U.

### 3. DDM / EDDM / ADWIN / Page–Hinkley — the classic streaming detectors
Supervised concept-drift detectors on the audit-labeled error stream:
- **DDM** (Gama 2004): thresholds on online error mean+std (warning/drift). Heuristic; assumes error only rises on drift.
- **EDDM**: monitors the *distance between* errors — better for *gradual* drift.
- **ADWIN** (Bifet–Gavaldà): adaptive variable-length window with a Hoeffding change bound; **also decides when to
  forget old data** — i.e. it answers Montgomery's "when to refit the calibration table" *adaptively* instead of
  on a fixed cadence. This is the most useful of the four for the calibration loop.
- **Page–Hinkley**: CUSUM change-point on a running mean.
These are the historical bridge to layer 4: ADWIN's Hoeffding window and PH's CUSUM are precursors of
anytime-valid / e-process monitoring. Keep DDM/ADWIN as cheap baselines; ADWIN's *windowing* is the reusable idea.

### 4. Anytime-valid monitor (`sequential.py`) — the rigorous core (BUILD THIS)
`sequential.py` already has `confidence_sequence` (anytime-valid CI) + Lan–DeMets/OBF boundaries, built for trial
efficacy monitoring. **Repurpose it to the production audit stream:** monitor the labeled calibration error /
accuracy with a confidence sequence and trigger refit on boundary crossing — continuous peeking with *no*
alpha-spending inflation, subsuming DDM/ADWIN/PH with a valid guarantee. This is the smallest high-value build:
a thin `CalibrationMonitor` wrapping `confidence_sequence` over a streaming (p̂, y) audit sample, emitting
warning/refit signals, with ADWIN-style adaptive forgetting as the window policy.

### 5. Causal-validity impact (exp38 / exp17 / exp37) — the layer Montgomery lacks
Any alarm from 1–4 says "the *predictor* moved." Whether the *causal estimate* is still valid is a different
question, and it's where causal_bench is differentiated:
- **exp38** — positivity/propensity under train-vs-deploy shift: did the shift break overlap (ESS)?
- **exp17** — transport: reweight/re-identify to the new population.
- **exp37** — compounding shift × unmeasured confounding: drift can degrade the causal estimate *faster* than
  calibration decay alone, so "just recalibrate" can be insufficient.
Compose: alarm (1–4) → positivity/ESS check (exp38) → transport or re-estimate (exp17); exp37 is the warning that
the causal gap may exceed the predictive gap.

## Build plan
1. **`CalibrationMonitor`** (validation or a new `monitoring.py`): `confidence_sequence`-backed anytime-valid
   calibration/accuracy monitor over a streaming audit sample; ADWIN-style adaptive window; warning/refit signals.
   Self-validate: no-drift stream → no false refit at the nominal rate; injected drift → detects, with the
   anytime-valid CI never under-covering under continuous peeking (vs a naive repeated-test baseline that inflates).
2. **AGL layer** (candidate, gated on a 2-backbone setup): cross-backbone agreement → OOD-accuracy estimate; and
   the causal extension (cross-nuisance-backbone agreement of the DR contrast). Needs decorrelated backbones.
3. **DDM/ADWIN baselines** (optional, thin) — as cheap comparators to (1), showing the anytime-valid monitor's
   validity advantage under peeking.
4. **Wire alarm → causal-impact** (exp38/exp17): a demo where a drift alarm triggers the positivity-under-shift
   check and the estimate is re-validated (or flagged) — the prediction-layer→causal-layer handoff.

## Honest scope
Montgomery is not in the repo; this is composition *opportunity*, not existing integration. The one concrete,
high-value, buildable-now piece is (1) — `sequential.py` already exists and the repurposing is thin. AGL is a
concept (not code) and needs the 2-backbone infrastructure. Everything here is prediction-layer monitoring made
rigorous + connected to the causal layer; none of it closes the unmeasured-confounding gap (that stays the
E-value's job, exp57).
