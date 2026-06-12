from dataclasses import asdict

import numpy as np
import pandas as pd
from causal_bench.dgp.config import DGPConfig
from causal_bench.dgp.survival import generate_data


def test_dgp_config_defaults():
    cfg = DGPConfig()
    assert cfg.n == 500
    assert cfg.true_tau == -0.5
    assert cfg.censoring_informativeness == 0.0
    assert cfg.compliance_available is True
    assert cfg.seed == 42


def test_dgp_config_override():
    cfg = DGPConfig(n=200, true_tau=-0.3, censoring_informativeness=0.6)
    assert cfg.n == 200
    assert cfg.true_tau == -0.3
    assert cfg.censoring_informativeness == 0.6


def test_dgp_config_is_dataclass():
    cfg = DGPConfig()
    d = asdict(cfg)
    assert "n" in d
    assert "true_tau" in d
    assert "compliance_available" in d


# --- Survival DGP tests ---

from causal_bench.dgp.survival import generate_data, compute_true_effects, compute_true_rmst


def test_generate_data_shape():
    cfg = DGPConfig(n=200, seed=0)
    df = generate_data(cfg)
    assert len(df) == 200
    required = {"T_obs", "Delta", "event_type", "A", "W1", "W2", "W3", "W4",
                "compliance", "Y_neg", "enrollment_time"}
    assert required.issubset(df.columns)


def test_generate_data_u_not_observed():
    cfg = DGPConfig(n=200, seed=0)
    df = generate_data(cfg)
    assert "U" not in df.columns


def test_generate_data_delta_binary():
    cfg = DGPConfig(n=500, seed=1)
    df = generate_data(cfg)
    assert set(df["Delta"].unique()).issubset({0.0, 1.0})


def test_generate_data_treatment_prevalence():
    cfg = DGPConfig(n=2000, treatment_prevalence=0.5,
                   unmeasured_confounding_strength=0.0,
                   positivity_severity=0.0, seed=2)
    df = generate_data(cfg)
    assert 0.35 <= df["A"].mean() <= 0.65


def test_generate_data_censoring_rate():
    cfg = DGPConfig(n=2000, censoring_rate=0.25,
                   censoring_informativeness=0.0, seed=3)
    df = generate_data(cfg)
    # Check the pre-horizon dropout rate (what censoring_rate actually calibrates)
    dropout_rate = ((df["Delta"] == 0) & (df["T_obs"] < cfg.horizon - 1e-9)).mean()
    assert 0.15 <= dropout_rate <= 0.35, f"dropout_rate={dropout_rate:.3f} not near target 0.25"


def test_generate_data_compliance_in_01():
    cfg = DGPConfig(n=500, seed=4)
    df = generate_data(cfg)
    assert df["compliance"].between(0, 1).all()


def test_generate_data_negative_control():
    """Y_neg should have near-zero raw correlation with A (no treatment in DGP)."""
    cfg = DGPConfig(n=3000, unmeasured_confounding_strength=0.0, seed=5)
    df = generate_data(cfg)
    corr = df[["A", "Y_neg"]].corr().loc["A", "Y_neg"]
    assert abs(corr) < 0.15


def test_compute_true_effects_clean():
    cfg = DGPConfig(n=500, unmeasured_confounding_strength=0.0,
                   positivity_severity=0.0, true_tau=-0.5, seed=0)
    effects = compute_true_effects(cfg, n_ref=10_000)
    assert "ATE" in effects
    assert "ATT" in effects
    # Both should be negative (treatment reduces risk probability? or increases?)
    # true_tau=-0.5 reduces log survival time → increases risk → positive risk difference
    # Actually: lower T means earlier event → higher risk → treatment with tau=-0.5
    # means shorter survival for treated → higher event rate → positive ATE
    assert isinstance(effects["ATE"], float)
    assert isinstance(effects["ATT"], float)
    assert -1.0 < effects["ATE"] < 1.0
    # true_tau=-0.5 shortens survival → treated have higher event rate → ATE > 0
    assert effects["ATE"] > 0
    assert effects["ATT"] > 0


# --- Scenario registry tests ---

from causal_bench.dgp.scenarios import get_scenario, list_scenarios


def test_get_scenario_clean():
    cfg = get_scenario("clean")
    assert cfg.censoring_informativeness == 0.0
    assert cfg.positivity_severity == 0.0
    assert cfg.true_tau == -0.5


def test_get_scenario_edwards_realistic():
    cfg = get_scenario("edwards_realistic")
    assert cfg.n == 700
    assert cfg.censoring_informativeness == 0.6
    assert cfg.positivity_severity == 1.5


def test_get_scenario_unknown_raises():
    import pytest
    with pytest.raises(ValueError, match="Unknown scenario"):
        get_scenario("nonexistent_scenario")


def test_list_scenarios_includes_expected():
    names = list_scenarios()
    for expected in ["clean", "edwards_realistic", "edwards_optimistic",
                     "edwards_pessimistic", "censor_mild", "censor_severe"]:
        assert expected in names


def test_get_scenario_returns_dgpconfig():
    cfg = get_scenario("clean")
    from causal_bench.dgp.config import DGPConfig
    assert isinstance(cfg, DGPConfig)


def test_get_scenario_each_valid():
    """Every scenario in the registry should produce a valid DGPConfig."""
    for name in list_scenarios():
        cfg = get_scenario(name)
        assert cfg.n > 0
        assert 0.0 <= cfg.censoring_rate <= 1.0


# --- Keyed random tests ---

from causal_bench.dgp.keyed_random import keyed_uniform, keyed_normal


def test_keyed_uniform_deterministic():
    v1 = keyed_uniform(patient_id=5, event_type="treatment", scenario="clean", seed=42)
    v2 = keyed_uniform(patient_id=5, event_type="treatment", scenario="clean", seed=42)
    assert v1 == v2


def test_keyed_uniform_range():
    vals = [keyed_uniform(i, "survival", "clean", 0) for i in range(1000)]
    assert all(0.0 <= v < 1.0 for v in vals)


def test_keyed_uniform_scenario_independence():
    v1 = keyed_uniform(5, "survival", "edwards_realistic", 0)
    v2 = keyed_uniform(5, "survival", "edwards_pessimistic", 0)
    assert v1 != v2


def test_keyed_uniform_patient_independence():
    v1 = keyed_uniform(1, "survival", "clean", 0)
    v2 = keyed_uniform(2, "survival", "clean", 0)
    assert v1 != v2


def test_keyed_normal_is_normal():
    import numpy as np
    vals = [keyed_normal(i, "covariate", "clean", 0) for i in range(500)]
    # Should be roughly N(0,1): mean near 0, std near 1
    assert abs(np.mean(vals)) < 0.15
    assert abs(np.std(vals) - 1.0) < 0.15


def test_keyed_uniform_seed_independence():
    """Different seeds → different values for same patient."""
    v1 = keyed_uniform(5, "survival", "clean", 0)
    v2 = keyed_uniform(5, "survival", "clean", 1)
    assert v1 != v2


# --- L1 time-varying confounder tests ---

def test_l1_always_in_dataframe():
    """L1 column always present regardless of collider_strength."""
    cfg = DGPConfig(n=300, collider_strength=0.5, seed=0)
    df = generate_data(cfg)
    assert "L1" in df.columns
    assert df["L1"].notna().any()


def test_l1_nan_for_early_deaths():
    """L1 is NaN for patients who die before t_L1."""
    cfg = DGPConfig(n=500, collider_strength=0.5, t_L1=0.5, seed=0)
    df = generate_data(cfg)
    assert df["L1"].isna().sum() > 0


def test_l1_present_with_zero_collider_strength():
    """L1 column exists even when collider_strength=0 (no U-driven component)."""
    cfg = DGPConfig(n=300, collider_strength=0.0, seed=0)
    df = generate_data(cfg)
    assert "L1" in df.columns
    # Some patients should still have observed L1 (those alive past t_L1)
    assert df["L1"].notna().any()


# --- Competing risks DGP tests ---

def test_event_type_always_present():
    """event_type column present in single-cause and competing-risks modes."""
    for cr in [False, True]:
        cfg = DGPConfig(n=300, competing_risks=cr, seed=0)
        df = generate_data(cfg)
        assert "event_type" in df.columns

def test_event_type_values_single_cause():
    """Without competing risks, event_type ∈ {0, 1} and equals Delta."""
    cfg = DGPConfig(n=500, competing_risks=False, seed=1)
    df = generate_data(cfg)
    assert set(df["event_type"].unique()).issubset({0, 1})
    assert (df["event_type"] == df["Delta"].astype(int)).all()

def test_event_type_values_competing():
    """With competing risks, event_type ∈ {0, 1, 2}."""
    cfg = DGPConfig(n=500, competing_risks=True, seed=2)
    df = generate_data(cfg)
    assert set(df["event_type"].unique()).issubset({0, 1, 2})
    # All three causes should appear in n=500
    assert 1 in df["event_type"].values
    assert 2 in df["event_type"].values

def test_competing_risks_cause1_matches_delta():
    """Delta==1 iff event_type==1 (cause-1 event)."""
    cfg = DGPConfig(n=500, competing_risks=True, seed=3)
    df = generate_data(cfg)
    assert (df["Delta"] == (df["event_type"] == 1).astype(float)).all()

def test_competing_risks_no_negative_times():
    cfg = DGPConfig(n=300, competing_risks=True, seed=4)
    df = generate_data(cfg)
    assert (df["T_obs"] >= 0).all()

def test_competing_risks_t_obs_within_horizon():
    cfg = DGPConfig(n=300, competing_risks=True, horizon=1.0, seed=5)
    df = generate_data(cfg)
    assert (df["T_obs"] <= 1.0 + 1e-9).all()

def test_competing_risks_cause_fractions_nonzero():
    """Both causes should account for a reasonable share of events."""
    cfg = DGPConfig(n=2000, competing_risks=True, seed=6)
    df = generate_data(cfg)
    n_cause1 = (df["event_type"] == 1).sum()
    n_cause2 = (df["event_type"] == 2).sum()
    n_total = len(df)
    # Each cause should be at least 5% of all patients
    assert n_cause1 / n_total > 0.05, f"cause-1 fraction too low: {n_cause1/n_total:.3f}"
    assert n_cause2 / n_total > 0.05, f"cause-2 fraction too low: {n_cause2/n_total:.3f}"


# --- compute_true_rmst tests ---

def test_compute_true_rmst_returns_dict():
    cfg = DGPConfig(n=500, seed=0)
    result = compute_true_rmst(cfg)
    assert isinstance(result, dict)
    for key in ("ATE", "ATT", "rmst_treated", "rmst_control"):
        assert key in result, f"missing key: {key}"


def test_compute_true_rmst_finite():
    cfg = DGPConfig(n=500, seed=1)
    result = compute_true_rmst(cfg)
    for k, v in result.items():
        assert np.isfinite(v), f"{k} is not finite: {v}"


def test_compute_true_rmst_bounded_by_horizon():
    """Per-arm RMST must be in (0, horizon]."""
    cfg = DGPConfig(n=500, horizon=1.0, seed=2)
    result = compute_true_rmst(cfg)
    assert 0 < result["rmst_treated"] <= cfg.horizon
    assert 0 < result["rmst_control"] <= cfg.horizon


def test_compute_true_rmst_ate_equals_arm_difference():
    cfg = DGPConfig(n=500, seed=3)
    result = compute_true_rmst(cfg)
    assert abs(result["ATE"] - (result["rmst_treated"] - result["rmst_control"])) < 1e-10


def test_compute_true_rmst_sign_matches_treatment_direction():
    """true_tau=-0.5 shortens log T → treated die sooner → RMST(A=1) < RMST(A=0) → ATE < 0.
    (Mirrors compute_true_effects where RD > 0: treated have higher event rate.)"""
    cfg = DGPConfig(n=500, true_tau=-0.5, seed=4)
    assert cfg.true_tau < 0
    result = compute_true_rmst(cfg)
    assert result["ATE"] < 0, f"Expected ATE < 0 for treatment that shortens survival, got {result['ATE']:.4f}"


def test_compute_true_rmst_vs_risk_difference_ordering():
    """RMST diff and RD should have opposite signs: treatment that shortens survival
    (true_tau < 0) increases event risk (RD > 0) and decreases time lived (RMST diff < 0)."""
    cfg = DGPConfig(n=500, seed=5)
    rmst = compute_true_rmst(cfg)
    rd   = compute_true_effects(cfg)
    assert np.sign(rmst["ATE"]) != np.sign(rd["ATE"]), (
        f"RMST ATE ({rmst['ATE']:.4f}) and RD ATE ({rd['ATE']:.4f}) should have opposite signs"
    )


def test_compute_true_rmst_stable_across_seeds():
    """Two large n_ref runs with different seeds should agree within Monte Carlo noise."""
    cfg = DGPConfig(seed=0)
    r1 = compute_true_rmst(cfg, n_ref=20_000)
    r2 = compute_true_rmst(DGPConfig(seed=1), n_ref=20_000)
    # With n_ref=20k and true_tau=-0.5, RMST diff is ~0.1; allow 0.05 tolerance
    assert abs(r1["ATE"] - r2["ATE"]) < 0.05, (
        f"RMST ATE estimates too far apart: {r1['ATE']:.4f} vs {r2['ATE']:.4f}"
    )


def test_run_simulation_true_value_override():
    """Passing true_value to run_simulation() skips compute_true_effects()
    and uses the provided value in SimResult."""
    from causal_bench.runner import run_simulation
    cfg = DGPConfig(n=200, seed=0)
    sentinel = 42.0
    results = run_simulation(
        cfg, estimator_names=["naive"], n_sim=5, n_jobs=1, true_value=sentinel
    )
    assert "naive" in results
    assert results["naive"].true_value == sentinel


# --- Win ratio true-value tests ---

from causal_bench.dgp.survival import compute_true_win_ratio


def test_compute_true_win_ratio_keys():
    cfg = DGPConfig(seed=0)
    result = compute_true_win_ratio(cfg, n_ref=5_000)
    for key in ("ATE", "ATT", "p_win", "p_loss", "net_benefit"):
        assert key in result, f"missing key: {key}"


def test_compute_true_win_ratio_probabilities_valid():
    cfg = DGPConfig(seed=0)
    result = compute_true_win_ratio(cfg, n_ref=5_000)
    assert 0.0 <= result["p_win"] <= 1.0
    assert 0.0 <= result["p_loss"] <= 1.0
    assert result["p_win"] + result["p_loss"] <= 1.0 + 1e-9


def test_compute_true_win_ratio_wr_positive():
    cfg = DGPConfig(seed=0)
    result = compute_true_win_ratio(cfg, n_ref=5_000)
    assert result["ATE"] > 0.0


def test_compute_true_win_ratio_sign_matches_treatment_direction():
    # true_tau=-0.5 shortens T → T1 < T0 → p_win < p_loss → WR < 1
    cfg = DGPConfig(true_tau=-0.5, seed=0)
    result = compute_true_win_ratio(cfg, n_ref=10_000)
    assert result["ATE"] < 1.0, f"WR should be <1 for true_tau=-0.5, got {result['ATE']:.3f}"


def test_compute_true_win_ratio_net_benefit_consistent():
    cfg = DGPConfig(seed=1)
    result = compute_true_win_ratio(cfg, n_ref=5_000)
    expected_nb = result["p_win"] - result["p_loss"]
    assert abs(result["net_benefit"] - expected_nb) < 1e-9


def test_compute_true_win_ratio_deterministic():
    cfg = DGPConfig(seed=42)
    r1 = compute_true_win_ratio(cfg, n_ref=5_000)
    r2 = compute_true_win_ratio(cfg, n_ref=5_000)
    assert r1["ATE"] == r2["ATE"]


# ── Stratified block randomization ──────────────────────────────────────────

def test_stratified_block_randomize_balance():
    cfg = DGPConfig(n=400, strata_cols=("W2", "W4"), strata_block_size=4, seed=7)
    df = generate_data(cfg)
    # Overall balance should be close to 0.5
    frac = df["A"].mean()
    assert 0.40 <= frac <= 0.60, f"Treatment fraction {frac:.3f} outside [0.40, 0.60]"


def test_stratified_block_randomize_within_strata_balance():
    cfg = DGPConfig(n=800, strata_cols=("W2", "W4"), strata_block_size=4, seed=99)
    df = generate_data(cfg)
    for w2 in [0, 1]:
        for w4 in [0, 1]:
            sub = df[(df["W2"] == w2) & (df["W4"] == w4)]
            if len(sub) < 10:
                continue
            frac = sub["A"].mean()
            assert 0.35 <= frac <= 0.65, (
                f"Stratum W2={w2} W4={w4}: treatment fraction {frac:.3f} out of range"
            )


def test_stratified_block_randomize_strata_attrs():
    cfg = DGPConfig(n=200, strata_cols=("W2", "W4"), strata_block_size=4, seed=3)
    df = generate_data(cfg)
    assert df.attrs.get("strata_cols") == ["W2", "W4"]


def test_stratified_base_scenario():
    from causal_bench.dgp.scenarios import get_scenario
    cfg = get_scenario("stratified_base")
    assert cfg.strata_cols is not None
    df = generate_data(cfg)
    assert 0.40 <= df["A"].mean() <= 0.60


def test_concrete_rmst_strata_in_registry():
    from causal_bench.estimators import ESTIMATOR_REGISTRY
    assert "concrete_RMST_strata" in ESTIMATOR_REGISTRY
