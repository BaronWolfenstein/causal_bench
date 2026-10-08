# Page-Hinkley replay lessons from security-triage-demo

The security demo's Page-Hinkley threshold 5 was too sensitive to noisy Brier-loss
streams. Constant/zero-loss tests passed, but independent stationary simulations
revealed frequent first alarms. Development-only selection plus a frozen fresh
check qualified a higher-sensitivity threshold for that demo's finite horizon.
This does not establish an anytime-valid false-alarm bound. Its threshold 120
must not be copied into causal_bench's different loss scale, slack or horizon.

The reusable changes here are:

- `StreamingPageHinkley` maintains explicit warm-up, a latched first alarm and
  JSON-restorable state. The existing `page_hinkley` wrapper keeps its original
  default warm-up of one and existing delta/threshold defaults.
- Finite real scalar input is required. Arrays, booleans and nonfinite values
  raise before state changes; overflow also leaves state unchanged.
- Experiment 58 counts early alarms, genuine post-change first alarms and misses
  as disjoint outcomes for all three detectors. An early sticky alarm is not a
  successful detection when a change later occurs.
- The change index is zero-based and alarm indices are one-based: alarm 500 is
  still on the baseline for `change_at=500`; alarm 501 is the first changed
  observation, with a one-observation detection delay.
- Reports show measured rates and actual counts, rather than hardcoded outcomes
  or a claim that one detector universally dominates another. The mixture betting
  guarantee requires a fixed valid baseline and a conditional-mean null. An
  estimated baseline or arbitrary serial dependence cannot silently inherit it.

The isolated replay in `results/page_hinkley_replay_review_v1` used the existing
fixed PH settings (slack 0.05, threshold 6), 2000 observations, a change at 500,
64 stationary streams and 64 changed streams. PH had zero stationary alarms and
64 genuine detections; its median delay was 63.5 observations. DDM had 43 genuine
detections and 21 early alarms. The mixture monitor had 61 genuine detections and
three early alarms; these are no longer credited as successes. This small replay
does not select parameters or replace historical results.

Tests cover historical batch parity, exact equality boundaries, warm-up, sticky
first indices, JSON interruption recovery before/after alarms, malformed scalar
and restored-state inputs, and adversarial experiment accounting. Further classic
detector tuning should use separate development and check streams, including
noise/dependence, intended horizon and delayed/selective labels where relevant.
