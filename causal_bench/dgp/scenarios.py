from causal_bench.dgp.config import DGPConfig, CovariateDependentCensoringConfig

_CENS = CovariateDependentCensoringConfig  # local alias

_CLEAN = dict(
    n=500, censoring=_CENS(informativeness=0.0), censoring_rate=0.25,
    positivity_severity=0.0, unmeasured_confounding_strength=0.0,
    collider_strength=0.0, crossover_rate=0.0, enrollment_drift=0.0,
    true_tau=-0.5,
)

_REGISTRY: dict[str, dict] = {
    "clean": _CLEAN,
    # Censoring gradient
    "censor_mild":     {**_CLEAN, "censoring": _CENS(informativeness=0.3), "censoring_rate": 0.25},
    "censor_moderate": {**_CLEAN, "censoring": _CENS(informativeness=0.6), "censoring_rate": 0.30},
    "censor_severe":   {**_CLEAN, "censoring": _CENS(informativeness=1.0), "censoring_rate": 0.40},
    # Positivity gradient
    "positivity_mild":     {**_CLEAN, "positivity_severity": 1.0},
    "positivity_moderate": {**_CLEAN, "positivity_severity": 2.0},
    "positivity_severe":   {**_CLEAN, "positivity_severity": 3.0},
    # Unmeasured confounding gradient
    "unmeasured_mild":   {**_CLEAN, "unmeasured_confounding_strength": 0.2},
    "unmeasured_mod":    {**_CLEAN, "unmeasured_confounding_strength": 0.5},
    "unmeasured_strong": {**_CLEAN, "unmeasured_confounding_strength": 0.8},
    # Edwards variants
    "edwards_realistic": dict(
        n=700,
        censoring=_CENS(informativeness=0.6), censoring_rate=0.25,
        positivity_severity=1.5, crossover_rate=0.05,
        unmeasured_confounding_strength=0.2,
        collider_strength=0.4, enrollment_drift=0.15,
        outcome_nonlinearity=0.5, effect_heterogeneity=0.3,
        true_tau=-0.5,
    ),
    "edwards_optimistic": dict(
        n=700,
        censoring=_CENS(informativeness=0.3), censoring_rate=0.15,
        positivity_severity=0.5, unmeasured_confounding_strength=0.1,
        collider_strength=0.2, enrollment_drift=0.05,
        true_tau=-0.5,
    ),
    "edwards_pessimistic": dict(
        n=700,
        censoring=_CENS(informativeness=0.9), censoring_rate=0.40,
        positivity_severity=2.5, crossover_rate=0.10,
        unmeasured_confounding_strength=0.4,
        collider_strength=0.7, enrollment_drift=0.3,
        outcome_nonlinearity=0.7, effect_heterogeneity=0.5,
        true_tau=-0.5,
    ),
    # Stratified block randomization — for Exp 11 / SE correction benchmark
    # W2 (Bern 0.5) × W4 (Bern 0.3) → 4 strata; block size 4.
    # Strata account for ~20% of outcome variance via their W2/W4 prognostic effects.
    "stratified_base": {
        **_CLEAN,
        "strata_cols": ("W2", "W4"),
        "strata_block_size": 4,
        "censoring": _CENS(informativeness=0.0),
        "censoring_rate": 0.20,
    },
    # Competing risks — for Exp 8 / McCoy experiment
    # cause-1 (primary event): treatment effect is true_tau, same as single-event case.
    # cause-2 (competing event): cause2_treatment_effect controls treatment's effect on
    # the competing cause's hazard.
    "competing_risks_base": {
        **_CLEAN,
        "n": 600, "competing_risks": True,
        "censoring": _CENS(informativeness=0.3), "censoring_rate": 0.20,
        "true_tau": -0.3,
    },
    # ENCIRCLE-calibrated — for Exp 16 / calibrated replication
    # Calibrated to published 1-year ENCIRCLE marginals (device arm, n=299):
    #   composite 25.2% (device) / ~45% (historical performance goal)
    #   mortality 13.9%, HF hospitalization 16.7%, ~5.4% overlap
    #   ~19% missing at 1-year visit
    # DGP validation (n=100k): device comp≈0.257, HFH≈0.166, death≈0.090
    #                           control comp≈0.465, ATE≈−0.144
    # NOTE: true_tau > 0 means device EXTENDS survival (fewer bad events).
    # The reverse sign convention from other scenarios where true_tau < 0 means
    # treatment causes the event sooner (those scenarios use a beneficial-event framing).
    "encircle_calibrated": {
        "n": 700,
        "treatment_prevalence": 0.43,       # ~299 treated in n=700
        "true_tau": 0.48,                   # device reduces HFH risk (extends T1)
        "competing_risks": True,
        "cause2_treatment_effect": 0.32,    # device also reduces mortality (extends T2)
        "hfh_death_escalation": 0.55,       # HFH-prone patients die sooner (shared frailty)
        "censoring_rate": 0.19,             # ~19% missing at 1-year visit
        "censoring": _CENS(informativeness=0.25),  # mild informative: sicker patients miss more
        "horizon": 0.77,
        "positivity_severity": 0.5,         # mild enrollment heterogeneity
        "unmeasured_confounding_strength": 0.1,
    },
}


def get_scenario(name: str) -> DGPConfig:
    if name not in _REGISTRY:
        raise ValueError(f"Unknown scenario '{name}'. Known: {list(_REGISTRY)}")
    return DGPConfig(**_REGISTRY[name])


def list_scenarios() -> list[str]:
    return list(_REGISTRY.keys())
