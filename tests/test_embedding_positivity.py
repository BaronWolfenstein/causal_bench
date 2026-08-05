"""Self-validating checks for exp50 (embedding as causal adjustment set).

These pin the claims that make the experiment interpretable:
  1. no confounding (conf=0) => every method unbiased;
  2. a LINEAR outcome model shows NO attenuation even at severe positivity;
  3. a FLEXIBLE outcome model attenuates toward null, monotonically in positivity.

Plus the SDR-spec frontier (#206) reductions on the 2-D effect-modification x positivity
sweep:
  4. the prognostic score fails under effect modification, the double-score does not;
  5. SDR must be ARM-STRATIFIED (pooling Y contaminates the subspace) -> it recovers at
     gamma=0;
  6. at severe positivity the outcome-relevant subspace still contains the positivity
     direction, so the ATO-on-phi response is needed on top of the reduction.
"""
from causal_bench.validation.embedding_positivity import report_rows, report_rows_2d


def _rows(confs, flex, n=1500, n_reps=5):
    return report_rows(n=n, n_reps=n_reps, confs=confs, flex=flex, seed=0)


def _cell(gamma, conf, n=1200, n_reps=6):
    return report_rows_2d(n=n, n_reps=n_reps, gammas=(gamma,), confs=(conf,), seed=0)[0]


def test_no_confounding_unbiased():
    r = _rows((0.0,), flex=True)[0]
    assert abs(r["oracle_Ustar"]) < 0.08
    assert abs(r["naive_fullW"]) < 0.08


def test_linear_outcome_has_no_pathology():
    # linear ridge outcome recovers the effect even at severe positivity (conf=5)
    r = _rows((5.0,), flex=False)[0]
    assert abs(r["oracle_Ustar"]) < 0.08
    assert abs(r["naive_fullW"]) < 0.08


def test_flexible_outcome_attenuates_with_positivity():
    low, high = _rows((1.0, 5.0), flex=True)
    b_low, b_high = low["naive_fullW"], high["naive_fullW"]
    assert b_high < -0.10           # clearly attenuated toward null at severe positivity
    assert b_high < b_low - 0.05     # attenuation grows with positivity severity


def test_prognostic_reduction_recovers_at_severe_positivity():
    # tier-2: the prognostic-score reduction largely recovers where naive full-W attenuates
    r = _rows((5.0,), flex=True)[0]
    assert r["naive_fullW"] < -0.12            # naive is badly attenuated
    assert abs(r["prog_score_W"]) < 0.09       # prognostic-score reduction ~recovers


# ---- SDR-spec frontier: 2-D (effect modification x positivity) ----

def test_double_score_beats_prognostic_under_effect_modification():
    r = _cell(gamma=4.0, conf=1.0)
    assert abs(r["prog_score_bias"]) > 0.3                       # prognostic fails under EM
    assert abs(r["double_score_bias"]) < 0.5 * abs(r["prog_score_bias"])  # double-score robust


def test_sdr_must_be_arm_stratified_recovers_at_gamma0():
    # arm-stratified SDR recovers at gamma=0 (pooled-Y SDR would be badly biased, ~-0.29)
    r = _cell(gamma=0.0, conf=1.0)
    assert abs(r["sdr_bias"]) < 0.15


def test_ato_on_phi_is_needed_at_severe_positivity():
    # Constant-effect severe-positivity cell (gamma=0 -> ATO == ATE, so the comparison is
    # clean): the reduction alone inherits the positivity direction (posv_sdr stays high),
    # and the ATO-on-phi response recovers where both naive and plain SDR-AIPW attenuate.
    r = _cell(gamma=0.0, conf=3.0)
    assert r["posv_sdr"] > 0.3                                   # subspace still near-deterministic
    assert abs(r["sdr_ato_bias"]) < abs(r["naive_fullW_bias"])   # ATO response beats naive
    assert abs(r["sdr_ato_bias"]) < abs(r["sdr_bias"])           # ... and beats plain SDR-AIPW


def test_conf0_reductions_all_unbiased():
    r = _cell(gamma=0.0, conf=0.0)
    for m in ("double_score_bias", "sdr_bias", "prog_score_bias"):
        assert abs(r[m]) < 0.15
