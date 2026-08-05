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
from causal_bench.validation.embedding_positivity import (
    report_rows, report_rows_2d, role_stress_rows,
)


def _rows(confs, flex, n=1500, n_reps=5):
    return report_rows(n=n, n_reps=n_reps, confs=confs, flex=flex, seed=0)


def _cell(gamma, conf, n=1200, n_reps=6):
    # bias-ORDERING checks: the fast in-sample IF is sufficient (relationships hold either way)
    return report_rows_2d(n=n, n_reps=n_reps, gammas=(gamma,), confs=(conf,),
                          crossfit=False, seed=0)[0]


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


def test_crossfit_gives_near_nominal_coverage_and_low_bias():
    # DML cross-fitting (nuisances AND the reduction map fit out-of-fold) brings the
    # composition's interval coverage toward nominal AND removes the in-sample plug-in bias,
    # at the hardest corner (strong effect modification x severe positivity).
    cf = report_rows_2d(n=1500, n_reps=10, gammas=(4.0,), confs=(3.0,),
                        crossfit=True, n_folds=5, seed=0)[0]
    assert cf["sdr_ato_cov"] >= 0.8            # ~nominal 95% coverage
    assert abs(cf["sdr_ato_bias"]) < 0.12      # composition near-unbiased under cross-fit


# ---- causal-role stress-test: does the outcome-targeted reduction handle each role? ----

def test_causal_role_stress():
    """Instrument vs collider behaviour of the reductions (bias excess over the oracle, which
    adjusts the true confounder only). Robust relationships, checked in-sample for speed."""
    rows = {r["scenario"]: r for r in role_stress_rows(n=2000, n_reps=6, crossfit=False, seed=0)}
    base, inst, col = rows["base"], rows["+instrument"], rows["+collider"]

    # base: every reduction tracks the oracle (small excess)
    for m in ("prog_excess", "double_excess", "sdr_excess"):
        assert abs(base[m]) < 0.2

    # instrument: the moment-based SDR is more instrument-robust than the arm-conditional
    # double-score (conditioning on A opens the instrument->treatment collider path)
    assert abs(inst["sdr_excess"]) < abs(inst["double_excess"])

    # collider (Y-predictive M-bias): NO reduction protects -- all fooled with large excess,
    # while the oracle (never adjusts the collider) stays clean. The honest limit.
    for m in ("naive_fullW_excess", "prog_excess", "double_excess", "sdr_excess"):
        assert col[m] < -0.3
    assert abs(col["oracle_U"]) < 0.25


def test_oracle_double_score_is_ate_sufficient():
    """Grounds the sufficiency theory (spec Validity section, Props 1-2): adjusting for the TRUE
    potential-outcome surfaces (b0, b1) recovers the ATE under effect modification, while the true
    prognostic b0 ALONE fails -- the pair is the minimal ATE-sufficient reduction."""
    import numpy as np
    from causal_bench.validation.embedding_positivity import simulate, _EM_DIR, _aipw
    beta = np.array([1.5, 1.0])                       # the DGP's outcome loading (Y = tau_i*A + U@beta)
    dbl, prog, tru = [], [], []
    for r in range(15):
        s = simulate(2500, conf=2.0, tau=1.0, gamma=4.0, seed=r)
        U, A, Y = s["Ustar"], s["A"], s["Y"]
        b0 = U @ beta                                # true E[Y(0)|W]
        b1 = 1.0 * (1.0 + 4.0 * (U @ _EM_DIR)) + U @ beta   # true E[Y(1)|W]
        tru.append(s["true_ate"])
        dbl.append(_aipw(np.column_stack([b0, b1]), A, Y))
        prog.append(_aipw(b0[:, None], A, Y))
    t = float(np.mean(tru))
    assert abs(np.mean(dbl) - t) < 0.15              # double-score: ATE-sufficient at strong EM
    assert np.mean(prog) - t < -0.4                  # prognostic alone: fails under EM (Prop 2)
