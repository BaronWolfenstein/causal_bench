from dataclasses import dataclass


@dataclass
class DGPConfig:
    # Sample
    n: int = 500
    n_treated_fraction: float = 0.5   # target fraction for post-stratification (unused in MVP)

    # Treatment
    true_tau: float = -0.5
    treatment_prevalence: float = 0.5  # base-rate used in propensity model logit intercept
    positivity_severity: float = 0.0
    unmeasured_confounding_strength: float = 0.0

    # Outcome
    outcome_nonlinearity: float = 0.0
    effect_heterogeneity: float = 0.0
    baseline_hazard: str = "weibull"  # valid: "weibull" (only in MVP)
    horizon: float = 1.0

    # Censoring
    censoring_rate: float = 0.25
    censoring_informativeness: float = 0.0

    # Crossover (unused in MVP, present for scenario compatibility)
    crossover_rate: float = 0.0
    crossover_informativeness: float = 0.0

    # Time-varying confounder (unused in MVP)
    collider_strength: float = 0.0
    sigma_L: float = 0.5
    t_L1: float = 0.5

    # Competing risks (unused in MVP)
    competing_risks: bool = False
    cause1_fraction: float = 0.4
    cause1_treatment_effect: float = -0.3
    cause2_treatment_effect: float = -0.6

    # Enrollment drift
    enrollment_drift: float = 0.0
    enrollment_period: float = 1.0

    # Compliance covariate
    compliance_available: bool = True
    compliance_censoring_r2: float = 0.3

    # Stratified block randomization
    # When set, treatment is assigned via permuted blocks within the named strata
    # rather than Bernoulli. Columns must be in ["W1", "W2", "W3", "W4"].
    strata_cols: object = None       # list[str] | None; object avoids mutable-default issue
    strata_block_size: int = 4       # must be even

    seed: int = 42
