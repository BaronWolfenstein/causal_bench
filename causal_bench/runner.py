import numpy as np
import pandas as pd
from typing import Optional
from joblib import Parallel, delayed
from tqdm import tqdm
from numpy.random import SeedSequence

from causal_bench.dgp.config import DGPConfig
from causal_bench.dgp.survival import generate_data, compute_true_effects
from causal_bench.estimators import get_estimator
from causal_bench.metrics import SimResult


def _run_one_sim(
    config_dict: dict,
    estimator_names: list[str],
    child_entropy: int,
    estimand: str,
    horizon: float,
) -> dict:
    """Run one simulation replicate. Accepts a plain dict (not DGPConfig) for pickling."""
    rng = np.random.default_rng(child_entropy)
    sim_seed = int(rng.integers(0, 2**31))
    config = DGPConfig(**{**config_dict, "seed": sim_seed})
    df = generate_data(config, rng=rng)

    out = {}
    for name in estimator_names:
        est = get_estimator(name)  # fresh lookup per worker — no shared state
        try:
            results = est.estimate(df, horizon=horizon, estimand=estimand)
            nc_val = est.estimate_negative_control(df, horizon=horizon)
        except Exception:
            results = []
            nc_val = float("nan")

        match = next((r for r in results if r.estimand == estimand), None)
        if match is None and results:
            match = results[0]
        out[name] = (match, nc_val)
    return out


def run_simulation(
    dgp_config: DGPConfig,
    estimator_names: list[str],
    n_sim: int = 500,
    n_jobs: int = -1,
    seed: int = 42,
    estimand: str = "ATE",
    horizon: Optional[float] = None,
    true_value: Optional[float] = None,
) -> dict[str, SimResult]:
    """Run Monte Carlo simulations for a set of estimators.

    Parameters
    ----------
    true_value:
        Pre-computed true effect value.  When provided, skips the internal
        call to compute_true_effects() — use this when the estimators target
        a different estimand than the risk difference (e.g. RMST difference).
    """
    if horizon is None:
        horizon = dgp_config.horizon

    if true_value is None:
        print(f"Computing true {estimand}...", flush=True)
        true_effects = compute_true_effects(dgp_config)
        true_value = true_effects.get(estimand, true_effects["ATE"])

    # Pass config as plain dict for safe pickling across joblib workers
    config_dict = dgp_config.__dict__.copy()

    child_entropies = [int(e) for e in SeedSequence(seed).generate_state(n_sim)]

    sim_outputs = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_run_one_sim)(config_dict, estimator_names, entropy, estimand, horizon)
        for entropy in tqdm(child_entropies, desc="Simulations", total=n_sim)
    )

    results: dict[str, SimResult] = {}
    for name in estimator_names:
        estimates, ses, ci_lows, ci_highs, nc_vals = [], [], [], [], []
        for sim_out in sim_outputs:
            res, nc = sim_out.get(name, (None, float("nan")))
            if res is not None and not np.isnan(res.point_estimate):
                estimates.append(res.point_estimate)
                ses.append(res.standard_error)
                ci_lows.append(res.ci_lower)
                ci_highs.append(res.ci_upper)
                nc_vals.append(float(nc) if not np.isnan(float(nc)) else 0.0)

        if not estimates:
            continue

        results[name] = SimResult(
            estimator_name=name,
            estimand=estimand,
            true_value=true_value,
            n_sim=len(estimates),
            estimates=np.array(estimates),
            se_estimates=np.array(ses),
            ci_lowers=np.array(ci_lows),
            ci_uppers=np.array(ci_highs),
            nc_estimates=np.array(nc_vals),
        )
    return results


def run_parameter_sweep(
    base_config: DGPConfig,
    param_name: str,
    param_values: list,
    estimator_names: list[str],
    n_sim: int = 500,
    **kwargs,
) -> dict[str, list[SimResult]]:
    """Sweep one DGP parameter. Returns {estimator_name: [SimResult per value]}."""
    from dataclasses import asdict
    all_results: dict[str, list[SimResult]] = {name: [] for name in estimator_names}
    for val in param_values:
        config = DGPConfig(**{**asdict(base_config), param_name: val})
        sim_results = run_simulation(config, estimator_names, n_sim=n_sim, **kwargs)
        for name in estimator_names:
            all_results[name].append(sim_results.get(name))
    return all_results
