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
    from dataclasses import asdict
    cfg = DGPConfig()
    d = asdict(cfg)
    assert "n" in d
    assert "true_tau" in d
    assert "compliance_available" in d
