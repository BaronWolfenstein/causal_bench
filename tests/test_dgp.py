from dataclasses import asdict

import numpy as np
import pandas as pd
from causal_bench.dgp.config import DGPConfig


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

from causal_bench.dgp.survival import generate_data, compute_true_effects


def test_generate_data_shape():
    cfg = DGPConfig(n=200, seed=0)
    df = generate_data(cfg)
    assert len(df) == 200
    required = {"T_obs", "Delta", "A", "W1", "W2", "W3", "W4",
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
    observed_censor = 1 - df["Delta"].mean()
    # Allow loose tolerance since calibration is approximate
    assert 0.05 <= observed_censor <= 0.60


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
