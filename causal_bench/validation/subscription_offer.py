"""Known-truth baseline retention offers; oracle arrays never enter estimators."""
from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd
from scipy.special import expit

from causal_bench.estimators.monthly_offer import estimate_offer


def counterfactual_hazard(df, arm, horizon, effect):
    age = np.arange(horizon)
    month = (df["cohort"].to_numpy()[:, None] + age) % 12
    eta = (-2.7 - 0.065 * age + 0.35 * df["channel"].to_numpy()[:, None]
           - 0.45 * df["plan"].to_numpy()[:, None] - 0.4 * df["engaged"].to_numpy()[:, None]
           + 0.4 * np.sin(2 * np.pi * month / 12) + 0.2 * np.cos(2 * np.pi * month / 12)
           + 0.06 * df["plan"].to_numpy()[:, None] * age
           + arm * effect * (1 - 0.04 * age))
    return expit(eta)


def population_truth(horizon=12, effect=-0.55):
    """Exact integration over 8 baseline types and 24 acquisition cohorts."""
    values, weights = [], []
    for channel, plan, engaged, cohort in product((0, 1), (0, 1), (0, 1), range(24)):
        values.append((channel, plan, engaged, cohort))
        weights.append(np.prod([p if x else 1 - p
                                for x, p in zip((channel, plan, engaged), (.45, .5, .6))]) / 24)
    df = pd.DataFrame(values, columns=["channel", "plan", "engaged", "cohort"])
    arms = []
    for a in (0, 1):
        h = counterfactual_hazard(df, a, horizon, effect)
        s = np.column_stack([np.ones(len(df)), np.cumprod(1 - h, axis=1)])
        arms.append(float(np.dot(weights, s[:, :-1].sum(axis=1))))
    return {"arm_rmst_months": arms, "effect_months": arms[1] - arms[0]}


def simulate_offers(n=6000, *, horizon=12, seed=0, assignment="confounded", effect=-0.55):
    if n < 1 or not isinstance(horizon, int) or not 1 <= horizon <= 12:
        raise ValueError("Positive n and horizon in 1..12 required")
    if assignment not in ("randomized", "confounded"):
        raise ValueError("Unknown offer assignment")
    rng = np.random.default_rng(seed)
    channel, plan, engaged = rng.binomial(1, [.45, .5, .6], (n, 3)).T
    cohort = rng.integers(0, 24, n)
    p = (np.full(n, .5) if assignment == "randomized" else
         expit(-.3 + channel - plan + .8 * engaged + .3 * np.sin(2 * np.pi * cohort / 12)))
    a = rng.binomial(1, p)
    df = pd.DataFrame({"subscription_id": np.arange(n), "cohort": cohort, "channel": channel,
                       "plan": plan, "engaged": engaged, "A": a})
    hazards = np.stack([counterfactual_hazard(df, arm, horizon, effect) for arm in (0, 1)], axis=1)
    factual = hazards[np.arange(n), a]
    failure = rng.random((n, horizon)) < factual
    event_time = np.where(failure.any(axis=1), failure.argmax(axis=1) + 1, horizon + 1)
    # Independent per-month observation dropout, conditional on baseline and A.
    # It occurs halfway through the month, before that month's cancellation.
    q = expit(-3 + .4 * channel - .3 * plan + .2 * engaged + .35 * a)
    dropout = rng.random((n, horizon)) < q[:, None]
    censor_time = np.where(dropout.any(axis=1), dropout.argmax(axis=1) + .5, horizon)
    limit = np.minimum(censor_time, horizon)
    df["event"] = (event_time <= limit).astype(int)
    df["observed_time"] = np.minimum(event_time, limit)
    s = np.concatenate([np.ones((n, 2, 1)), np.cumprod(1 - hazards, axis=2)], axis=2)
    return df, {"hazard": hazards, "survival": s, "propensity": p,
                "censor_survival_factual": (1 - q[:, None]) ** np.arange(1, horizon + 1),
                "event_time": event_time,
                "sample_effect_months": float(np.mean(s[:, 1, :-1].sum(axis=1) - s[:, 0, :-1].sum(axis=1))),
                "population": population_truth(horizon, effect)}


def concrete_companion(n, horizon, seed, assignment, effect):
    """Optional CONTINUOUS-time comparator with its own matching oracle.

    concrete's Cox/grid-integrated RMST is not silently substituted for the exact
    monthly sum. Observe beyond the estimand horizon; reject legacy horizon caps.
    The separate comparator uses no seasonality or time-dependent hazard effects.
    """
    from causal_bench.estimators.concrete_config import ConcreteConfig
    from causal_bench.estimators.concrete_rmst import ConcreteRMSTEstimator, _concrete_available
    if not _concrete_available():
        return {"status": "unavailable", "reason": "R/rpy2/concrete not installed; no fallback result"}
    rng = np.random.default_rng(seed)
    x = rng.binomial(1, [.45, .5, .6], (n, 3))
    w4 = rng.normal(size=n)
    g = np.full(n, .5) if assignment == "randomized" else expit(-.3 + x[:, 0] - x[:, 1] + .8 * x[:, 2])
    a = rng.binomial(1, g)
    rate0 = np.exp(-2.5 + .45 * x[:, 0] - .6 * x[:, 1] - .3 * x[:, 2] + .15 * w4)
    t = rng.exponential(1 / (rate0 * np.exp(effect * a)))
    c = rng.exponential(40 * np.exp(.2 * x[:, 0] - .2 * a))
    observed = np.minimum(np.minimum(t, c), 2 * horizon)
    df = pd.DataFrame({"W1": x[:, 0].astype(float), "W2": x[:, 1].astype(float),
                       "W3": x[:, 2].astype(float), "W4": w4, "A": a,
                       "T_obs": observed, "Delta": (t <= np.minimum(c, 2 * horizon)).astype(int)})
    rates = np.column_stack([rate0, rate0 * np.exp(effect)])
    sample_truth = float(np.mean(np.diff(-np.expm1(-rates * horizon) / rates, axis=1)))
    est = ConcreteRMSTEstimator(rmst_contrast=True, horizon_policy="strict",
                               config=ConcreteConfig(rmst_grid_n=24, cv_folds=3),
                               subprocess_timeout=120.)
    results = est.estimate(df, horizon=float(horizon))
    if not results:
        return {"status": "failed", "reason": "concrete produced no valid strict-horizon RMST result"}
    result = results[0]
    return {"status": "executed", "time_convention": "continuous exponential survival; separate companion",
            "horizon_months": horizon, "estimate_months": result.point_estimate,
            "integration_grid_n": 24,
            "standard_error": result.standard_error, "ci95": [result.ci_lower, result.ci_upper],
            "sample_oracle_effect_months": sample_truth,
            "scope": "optional numerical/estimand comparator; not monthly estimator coverage validation"}


def run(*, n=6000, seeds=20, horizon=12, folds=3, include_concrete=False):
    if seeds < 1:
        raise ValueError("At least one Monte Carlo seed required")
    records, summaries, companions = [], {}, {}
    for assignment in ("randomized", "confounded"):
        for effect in (-.55, 0.):
            case = f"{assignment}_{'benefit' if effect else 'null'}"
            truth = population_truth(horizon, effect)
            for seed in range(seeds):
                df, oracle = simulate_offers(n, horizon=horizon, seed=seed, assignment=assignment, effect=effect)
                kwargs = dict(horizon=horizon, folds=folds, seed=seed + 10_000)
                adjusted = estimate_offer(df, **kwargs, randomized_probability=.5 if assignment == "randomized" else None)
                outcome_only = estimate_offer(df, **kwargs, propensity_adjusted=False, censor_adjusted=False)
                weighting_only = estimate_offer(df, **kwargs, outcome_adjusted=False,
                                                randomized_probability=.5 if assignment == "randomized" else None)
                naive = estimate_offer(df, **kwargs, outcome_adjusted=False,
                                       propensity_adjusted=False, censor_adjusted=False)
                methods = {"aipw_adjusted": adjusted, "aipw_correct_outcome": outcome_only,
                           "aipw_correct_weights": weighting_only,
                           "naive_unadjusted_hazard": {"estimate_months": naive["g_computation_months"]}}
                records.append({"case": case, "seed": seed, "truth_months": truth["effect_months"],
                                "sample_oracle_months": oracle["sample_effect_months"], "methods": methods})
                print(f"OFFER {case} seed={seed} truth={truth['effect_months']:.4f} "
                      f"AIPW={adjusted['estimate_months']:.4f} naive={naive['g_computation_months']:.4f}", flush=True)
            summaries[case] = {}
            for method in methods:
                values = [r["methods"][method] for r in records if r["case"] == case]
                errors = np.array([v["estimate_months"] - truth["effect_months"] for v in values])
                item = {"truth_months": truth["effect_months"], "bias_months": float(errors.mean()),
                        "rmse_months": float(np.sqrt(np.mean(errors**2))), "replicates": len(values)}
                if values[0].get("ci95") is not None:
                    coverage = np.mean([v["ci95"][0] <= truth["effect_months"] <= v["ci95"][1] for v in values])
                    count, z = len(values), 1.96
                    center = (coverage + z*z/(2*count)) / (1 + z*z/count)
                    radius = z*np.sqrt(coverage*(1-coverage)/count + z*z/(4*count*count)) / (1+z*z/count)
                    item.update({"wald_coverage": float(coverage),
                                 "coverage_mcse": float(np.sqrt(coverage * (1 - coverage) / len(values))),
                                 "coverage_wilson95": [float(center-radius), float(center+radius)],
                                 "mean_standard_error": float(np.mean([v["standard_error"] for v in values]))})
                summaries[case][method] = item
            if include_concrete:
                companions[case] = concrete_companion(n, horizon, 0, assignment, effect)
    return {"configuration": {"n": n, "seeds": seeds, "horizon": horizon, "folds": folds},
            "summaries": summaries, "replicates": records, "concrete_continuous_companions": companions,
            "identification": "baseline exchangeability, positivity, consistency, conditionally independent censoring",
            "split_scope": "cross-fitting within eligible population for causal inference; exp59 future-cohort prediction splits unchanged",
            "units": "extra active months per eligible subscriber assigned an offer; month-end cancellation; sum S(0)..S(H-1)"}
