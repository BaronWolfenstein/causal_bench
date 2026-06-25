from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator

_STRATA_ELIGIBLE_COLS = frozenset({"W1", "W2", "W3", "W4"})


class IndependentCensoringConfig(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}
    kind: Literal["independent"] = "independent"


class CovariateDependentCensoringConfig(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}
    kind: Literal["covariate_dependent"] = "covariate_dependent"
    informativeness: float = Field(0.0, ge=0.0, le=1.0)


class InformativeCensoringConfig(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}
    kind: Literal["informative"] = "informative"
    beta_T: float = 0.0


class LatentConfounderCensoringConfig(BaseModel):
    """MNAR via U (the latent unmeasured confounder).

    Censoring hazard depends on U through the informativeness weight, making
    dropout informative even after conditioning on all observed covariates.
    Use for scenarios (like ENCIRCLE) calibrated to published marginals under
    this mechanism — switching to CovariateDependentCensoringConfig would
    break those calibrations.

    Distinct from InformativeCensoringConfig (which conditions on T_true) and
    CovariateDependentCensoringConfig (which is pure MAR given W, A).
    """
    model_config = {"frozen": True, "extra": "forbid"}
    kind: Literal["latent_confounder"] = "latent_confounder"
    informativeness: float = Field(0.25, ge=0.0, le=1.0)


CensoringConfig = Annotated[
    Union[
        IndependentCensoringConfig,
        CovariateDependentCensoringConfig,
        InformativeCensoringConfig,
        LatentConfounderCensoringConfig,
    ],
    Field(discriminator="kind"),
]


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
    censoring: CensoringConfig = Field(default_factory=CovariateDependentCensoringConfig)

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
    # Shared-frailty coupling between cause-1 (HFH) propensity and cause-2 (death)
    # propensity. Patients with higher HFH risk also die sooner when > 0, approximating
    # McCoy's 1.2^popcount state-dependent hazard escalation in a first-event model.
    # 0.0 = independent causes; 0.5 = moderate; 1.0 = strong coupling.
    hfh_death_escalation: float = Field(0.0, ge=0.0, le=2.0)

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

    # Explicit binary subgroup structure for HTE benchmarking (issue #20).
    # When set, survival.py replaces `true_tau + effect_heterogeneity*W1` with a
    # step function: cate_high for subgroup_col > median, cate_low otherwise.
    # A scalar "Y" column and a "subgroup_label" column are added to the DataFrame.
    # All three fields must be set together; effect_heterogeneity must be 0.0.
    subgroup_col: Optional[str] = None
    cate_high: Optional[float] = None
    cate_low: Optional[float] = None

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
        # cause2_treatment_effect only has any effect when competing_risks=True
        # (survival.py's competing-risks branch is skipped entirely otherwise)
        # — setting a non-zero value without competing_risks=True is a silent no-op.
        if self.cause2_treatment_effect != 0.0 and not self.competing_risks:
            raise ValueError(
                f"cause2_treatment_effect={self.cause2_treatment_effect} has no "
                "effect when competing_risks=False — likely a misconfiguration"
            )
        if self.hfh_death_escalation != 0.0 and not self.competing_risks:
            raise ValueError(
                f"hfh_death_escalation={self.hfh_death_escalation} has no effect "
                "when competing_risks=False — likely a misconfiguration"
            )
        # subgroup_col / cate_high / cate_low must be set together (all or none).
        # effect_heterogeneity must be 0 when subgroup_col is active — both define
        # individual-level CATEs and combining them silently produces a hybrid that
        # satisfies neither design.
        subgroup_fields = [self.subgroup_col, self.cate_high, self.cate_low]
        n_set = sum(x is not None for x in subgroup_fields)
        if 0 < n_set < 3:
            raise ValueError(
                "subgroup_col, cate_high, and cate_low must all be set together "
                f"(got subgroup_col={self.subgroup_col!r}, "
                f"cate_high={self.cate_high!r}, cate_low={self.cate_low!r})"
            )
        if self.subgroup_col is not None:
            if self.subgroup_col not in _STRATA_ELIGIBLE_COLS:
                raise ValueError(
                    f"subgroup_col={self.subgroup_col!r} not in "
                    f"{sorted(_STRATA_ELIGIBLE_COLS)}"
                )
            if self.effect_heterogeneity != 0.0:
                raise ValueError(
                    "effect_heterogeneity must be 0.0 when subgroup_col is set — "
                    "the step CATE (cate_high / cate_low) already defines individual "
                    "treatment effects; mixing in a linear effect_heterogeneity term "
                    "produces an unintended hybrid DGP"
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
