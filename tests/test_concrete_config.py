"""Unit tests for ConcreteConfig — pure Python (no rpy2 / R). The validated analysis-plan object that all
concrete_*.py estimators funnel through; these tests pin the #223 invariants (grid ≥ 2, optional MinNuisance,
positivity-neutral default) at the rpy2 seam."""
import pytest

from causal_bench.estimators.concrete_config import ConcreteConfig


class TestConcreteConfigDefaults:
    def test_default_is_positivity_neutral(self):
        # min_nuisance defaults to None so the structural fix ships no truncation change; to_r_kwargs
        # must OMIT the key entirely (the R helper then uses concrete's own default).
        kw = ConcreteConfig().to_r_kwargs()
        assert "min_nuisance" not in kw
        assert kw["covars"] == ["W1", "W2", "W3", "W4"]
        assert kw["cv_folds"] == 5
        assert kw["rmst_grid_n"] == 12

    def test_min_nuisance_included_when_set(self):
        kw = ConcreteConfig(min_nuisance=0.025).to_r_kwargs()
        assert kw["min_nuisance"] == pytest.approx(0.025)

    def test_learners_omitted_when_none(self):
        kw = ConcreteConfig().to_r_kwargs()
        assert "propensity_learners" not in kw and "hazard_learners" not in kw

    def test_frozen(self):
        c = ConcreteConfig()
        with pytest.raises(Exception):
            c.cv_folds = 9  # frozen=True


class TestConcreteConfigValidation:
    @pytest.mark.parametrize("bad", [
        {"rmst_grid_n": 1},      # RMST needs >= 2 grid points to integrate
        {"cv_folds": 1},         # >= 2 folds
        {"min_nuisance": 0.0},   # gt 0
        {"min_nuisance": 0.9},   # lt 0.5
        {"covariates": ()},      # non-empty
        {"bogus": 1},            # extra="forbid"
    ])
    def test_rejects_invalid(self, bad):
        with pytest.raises(Exception):
            ConcreteConfig(**bad)

    def test_grid_two_ok(self):
        assert ConcreteConfig(rmst_grid_n=2).rmst_grid_n == 2
