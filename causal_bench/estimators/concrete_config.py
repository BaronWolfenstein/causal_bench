"""ConcreteConfig — one validated analysis-plan object for the concrete rpy2 bridge.

Six estimators (concrete_rmst, concrete_psnb, concrete_clinical_rmtif, concrete_simultaneous,
concrete_win_ratio, concrete_pro_win_ratio) funnel into r_scripts/concrete_bridge.R, and until now
every one silently inherited concrete's R-side defaults. That boundary is exactly where a validated
schema pays off: R errors at the rpy2 seam are opaque (we hit `"RHS of == is length 0 which is not 1
or nrow(16000)"` and `"RMST integrated over fewer than two target times"` — neither legible), so the
knobs that actually matter are validated **here, in Python**, before crossing into R.

The load-bearing fields encode the #223 findings (see docs/plans/2026-09-23-concrete-*.md):
  • `rmst_grid_n`   — RMST is ∫₀^τ S dt; concrete needs a DENSE TargetTime grid to integrate it. A single
                     time gives a crude/erroring integral. Must be ≥ 2. (This was half the "attenuation".)
  • `min_nuisance`  — the propensity g-bound (`MinNuisance`) that caps the TMLE clever covariate 1/g. The
                     residual bias GREW with n (overlap tails: A~expit(0.7·W1)); truncation is the fix.
                     Default 0.025 (clever-covariate cap 40); VALUE pending the positivity confirmation.
  • learners        — left None = concrete's defaults ON PURPOSE: the hazards are already unregularized
                     `Lrnr.Cox`, and swapping the propensity library (SL.glm) is a no-op. The original
                     "de-regularize the learners" hypothesis (#223) is FALSIFIED; do not set these to
                     "fix attenuation" — the levers are the grid and `min_nuisance`, not the library.

`to_r_kwargs()` serialises to a plain dict the bridge functions accept; the Intervention slot convention
(contrast slots 1 vs 2, NOT the bugged c(1,0)) lives in the R helper `.concrete_args`, not here.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ConcreteConfig(BaseModel):
    """Validated concrete analysis plan shared across the concrete_*.py estimators."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    covariates: tuple[str, ...] = Field(
        default=("W1", "W2", "W3", "W4"),
        description="Outcome-model covariates. L1 is routed to CensoringTV by the bridge, never here.",
    )
    cv_folds: int = Field(default=5, ge=2, description="CV folds (concrete CVArg V). ≥2.")
    min_nuisance: float | None = Field(
        default=None, gt=0.0, lt=0.5,
        description="Propensity g lower bound (concrete MinNuisance); caps clever covariate at 1/min_nuisance. "
                    "None = concrete's own default (no extra truncation). The positivity fix flips this to a "
                    "small bound (~0.025) once verified — kept None here so the structural fix is positivity-"
                    "neutral (#223; see docs/plans/2026-09-23-concrete-positivity-fix.md).",
    )
    rmst_grid_n: int = Field(
        default=12, ge=2,
        description="Number of TargetTime grid points for RMST integration. ≥2 (a single time cannot integrate).",
    )
    propensity_learners: tuple[str, ...] | None = Field(
        default=None,
        description="Propensity (treatment) SL library. None = concrete default. NOT an attenuation lever (#223).",
    )
    hazard_learners: tuple[str, ...] | None = Field(
        default=None,
        description="Hazard learner library. None = concrete default (already unregularized Lrnr.Cox).",
    )

    @field_validator("covariates")
    @classmethod
    def _nonempty_covariates(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        if len(v) == 0:
            raise ValueError("covariates must be non-empty")
        if any((not isinstance(c, str)) or c == "" for c in v):
            raise ValueError("covariates must be non-empty strings")
        return v

    def to_r_kwargs(self) -> dict:
        """Serialise to a plain dict of bridge kwargs (rpy2-marshalable Python types).

        Keys map onto `.concrete_args` parameters in concrete_bridge.R. `None` learner libraries are
        omitted so the bridge falls through to concrete's defaults (the intended behaviour)."""
        out: dict = {
            "covars": list(self.covariates),
            "cv_folds": int(self.cv_folds),
            "rmst_grid_n": int(self.rmst_grid_n),
        }
        if self.min_nuisance is not None:
            out["min_nuisance"] = float(self.min_nuisance)
        if self.propensity_learners is not None:
            out["propensity_learners"] = list(self.propensity_learners)
        if self.hazard_learners is not None:
            out["hazard_learners"] = list(self.hazard_learners)
        return out
