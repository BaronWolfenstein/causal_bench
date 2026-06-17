from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

_STRATA_ELIGIBLE_COLS = frozenset({"W1", "W2", "W3", "W4"})


class DGPConfig(BaseModel):
    # frozen: configs are never mutated in place anywhere in the codebase
    # (verified via grep) — making that structural, not just conventional,
    # means a future `cfg.n = 600` typo raises instead of silently no-op-ing.
    # extra="forbid": a misspelled override (e.g. censor_info= instead of
    # censoring_informativeness=) currently sails through as an ignored
    # kwarg and silently runs the *wrong* DGP; this makes it a ValidationError
    # at construction time instead.
    model_config = {"frozen": True, "extra": "forbid"}

    # Sample
    n: int = Field(500, ge=1, le=100_000)

    # Treatment
    true_tau: float = -0.5
    treatment_prevalence: float = Field(0.5, ge=0.0, le=1.0)  # base-rate used in propensity model logit intercept
    # Soft ceilings, not hard: README documents positivity_severity in
    # [0, 3] and unmeasured_confounding_strength in [0, 0.8], but sweeps
    # (e.g. exp2/exp3) deliberately probe near/at those edges, so only the
    # physically-required floor (>=0) is enforced here.
    positivity_severity: float = Field(0.0, ge=0.0)
    unmeasured_confounding_strength: float = Field(0.0, ge=0.0)

    # Outcome
    outcome_nonlinearity: float = 0.0
    effect_heterogeneity: float = 0.0
    baseline_hazard: str = "weibull"  # valid: "weibull" (only in MVP)
    horizon: float = Field(1.0, gt=0.0)

    # Censoring
    censoring_rate: float = Field(0.25, ge=0.0, lt=1.0)
    censoring_informativeness: float = Field(0.0, ge=0.0, le=1.0)
    censoring_mechanism: Literal["independent", "covariate_dependent", "informative"] = "covariate_dependent"
    censoring_beta_T: float = 0.0                     # coefficient on T_true for "informative" mechanism

    # Crossover (unused in MVP, present for scenario compatibility)
    crossover_rate: float = Field(0.0, ge=0.0, le=1.0)
    crossover_informativeness: float = 0.0

    # Time-varying confounder (unused in MVP)
    collider_strength: float = Field(0.0, ge=0.0, le=1.0)
    sigma_L: float = 0.5
    t_L1: float = 0.5

    # Competing risks
    # cause-1 (primary event): treatment effect is true_tau, same as single-event case.
    # cause-2 (competing event): cause2_treatment_effect controls treatment's effect on
    # the competing cause's hazard (0.0 = no treatment effect on competing cause,
    # the most common assumption).
    competing_risks: bool = False
    cause2_treatment_effect: float = 0.0

    # Enrollment drift
    enrollment_drift: float = Field(0.0, ge=0.0, le=1.0)
    enrollment_period: float = Field(1.0, gt=0.0)

    # Compliance covariate
    # Used unconditionally as rho = sqrt(compliance_censoring_r2) (survival.py)
    # — rho is a correlation coefficient, so this must stay in [0, 1] or the
    # sqrt is mathematically meaningless (previously: silent NaN propagation).
    compliance_censoring_r2: float = Field(0.3, ge=0.0, le=1.0)

    # Stratified block randomization
    # When set, treatment is assigned via permuted blocks within the named strata
    # rather than Bernoulli. Columns must be in ["W1", "W2", "W3", "W4"].
    strata_cols: Optional[tuple[str, ...]] = None
    strata_block_size: int = Field(4, ge=2)       # must be even — enforced below

    seed: int = 42

    @model_validator(mode="after")
    def _check_couplings(self) -> "DGPConfig":
        # strata_block_size must be even: _stratified_block_randomize does
        # `half = block_size // 2` (survival.py) — an odd block_size silently
        # floors into an unbalanced block (e.g. block_size=5 -> 2 treated /
        # 3 control) rather than raising, which would quietly bias the
        # treatment-prevalence within strata.
        if self.strata_block_size % 2 != 0:
            raise ValueError(
                f"strata_block_size must be even (got {self.strata_block_size}) — "
                "an odd value silently produces unbalanced treat/control blocks"
            )
        if self.strata_cols is not None:
            unknown = set(self.strata_cols) - _STRATA_ELIGIBLE_COLS
            if unknown:
                raise ValueError(
                    f"strata_cols contains {sorted(unknown)}, not in "
                    f"{sorted(_STRATA_ELIGIBLE_COLS)} — survival.py's "
                    "_col_map lookup would otherwise raise a bare KeyError "
                    "deep inside generate_data() instead of failing here"
                )
        # censoring_beta_T only has any effect when censoring_mechanism ==
        # "informative" (survival.py's other two branches never reference
        # it) — setting it under another mechanism is a silent no-op that
        # almost certainly means the wrong mechanism was selected.
        if self.censoring_beta_T != 0.0 and self.censoring_mechanism != "informative":
            raise ValueError(
                f"censoring_beta_T={self.censoring_beta_T} has no effect under "
                f"censoring_mechanism='{self.censoring_mechanism}' (only the "
                "'informative' mechanism uses it) — likely a misconfiguration"
            )
        # cause2_treatment_effect only has any effect when competing_risks=True
        # (survival.py's competing-risks branch is skipped entirely otherwise)
        # — setting a non-zero value without competing_risks=True is a silent no-op.
        if self.cause2_treatment_effect != 0.0 and not self.competing_risks:
            raise ValueError(
                f"cause2_treatment_effect={self.cause2_treatment_effect} has no "
                "effect when competing_risks=False — likely a misconfiguration"
            )
        return self

    def with_overrides(self, **overrides) -> "DGPConfig":
        """Return a new, fully validated config with the given fields overridden.

        Unlike model_copy(update=...), which writes the override values directly
        into __dict__ without running them through pydantic validation, this
        re-runs the constructor — so an invalid override (e.g. a typo'd
        censoring_mechanism) raises pydantic.ValidationError here, in the
        caller's process, rather than silently producing an invalid config.
        """
        return DGPConfig(**{**self.model_dump(), **overrides})
