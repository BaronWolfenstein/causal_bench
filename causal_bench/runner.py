import numpy as np
import pandas as pd
from typing import Optional
from joblib import Parallel, delayed
from pydantic import ValidationError
from tqdm import tqdm
from numpy.random import SeedSequence

from causal_bench.dgp.config import DGPConfig
from causal_bench.dgp.survival import generate_data, compute_true_effects
from causal_bench.estimators import get_estimator
from causal_bench.metrics import ComparisonSpec, SimResult

# Generous per-task bound for joblib.Parallel(timeout=...). The slowest known
# single-replicate call (tmle_ipcw_boot at its registry-default 200 bootstrap
# reps) took ~100s; this leaves ample margin so a genuinely hung worker fails
# fast and visibly instead of silently pegging a core indefinitely.
SIM_TASK_TIMEOUT_SECONDS = 180.0


def classify_error(exc: Exception) -> str:
    """Label a sim-replicate failure for diagnostics, not just a generic None.

    Distinguishes config/result validation errors (pydantic.ValidationError),
    R-bridge failures (rpy2 exceptions), and everything else (numerical
    non-convergence and other estimator-internal failures).
    """
    if isinstance(exc, ValidationError):
        return "validation_error"
    if type(exc).__module__.startswith("rpy2"):
        return "r_bridge_error"
    return "estimation_error"


def build_validated_replicate_configs(config_dict: dict, child_entropies: list[int]) -> list[dict]:
    """Construct & validate one DGPConfig per replicate in the parent process.

    Each replicate gets its own seed (drawn deterministically from its
    child_entropy, exactly as the worker used to do internally). Building and
    validating all of them here means an invalid config_dict raises
    pydantic.ValidationError immediately, in the parent's own stack — visible
    and undamaged — rather than inside a loky worker mid-sweep, where it would
    otherwise be caught by the per-estimator try/except and collapsed into a
    silent None for that replicate.

    Returns plain dicts (already proven valid) so workers can reconstruct the
    DGPConfig via model_construct(), skipping redundant re-validation.
    """
    replicate_dicts = []
    for entropy in child_entropies:
        seed_rng = np.random.default_rng(entropy)
        sim_seed = int(seed_rng.integers(0, 2**31))
        config = DGPConfig(**{**config_dict, "seed": sim_seed})
        replicate_dicts.append(config.__dict__.copy())
    return replicate_dicts


def _run_one_sim(
    replicate_config_dict: dict,
    estimator_names: list[str],
    child_entropy: int,
    estimand: str,
    horizon: float,
) -> dict:
    """Run one simulation replicate.

    replicate_config_dict was already constructed and validated in the parent
    process (see build_validated_replicate_configs) — model_construct() here
    skips re-validation rather than re-running it per worker.
    """
    rng = np.random.default_rng(child_entropy)
    rng.integers(0, 2**31)  # discard: matches the draw the parent consumed to pick this replicate's seed, keeping the RNG stream identical
    config = DGPConfig.model_construct(**replicate_config_dict)
    df = generate_data(config, rng=rng)

    out = {}
    errors: dict[str, str] = {}
    for name in estimator_names:
        est = get_estimator(name)  # fresh lookup per worker — no shared state
        try:
            results = est.estimate(df, horizon=horizon, estimand=estimand)
            nc_val = est.estimate_negative_control(df, horizon=horizon)
        except Exception as e:
            results = []
            nc_val = float("nan")
            errors[name] = classify_error(e)

        match = next((r for r in results if r.estimand == estimand), None)
        if match is None and results:
            match = results[0]
        out[name] = (match, nc_val)
    out["__errors__"] = errors
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
    debug_first_replicate: bool = True,
    task_timeout: float = SIM_TASK_TIMEOUT_SECONDS,
    comparison_specs: Optional[dict[str, ComparisonSpec]] = None,
) -> dict[str, SimResult]:
    """Run Monte Carlo simulations for a set of estimators.

    Parameters
    ----------
    true_value:
        Pre-computed true effect value.  When provided, skips the internal
        call to compute_true_effects() — use this when the estimators target
        a different estimand than the risk difference (e.g. RMST difference).
    debug_first_replicate:
        Run the first replicate synchronously, in-process, before dispatching
        the full joblib sweep. Any exception not already swallowed by the
        per-estimator try/except (e.g. a bad config or a generate_data crash)
        then surfaces with a full, unmangled traceback immediately, instead of
        whatever loky's IPC manages to propagate from inside a worker process
        mid-sweep.
    task_timeout:
        Per-replicate timeout (seconds) passed to joblib.Parallel — a hung
        worker raises TimeoutError and fails the run fast and visibly, rather
        than silently pegging a core for an extended period.
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
    # Construct & validate every replicate's DGPConfig here, in the parent —
    # see build_validated_replicate_configs for why.
    replicate_configs = build_validated_replicate_configs(config_dict, child_entropies)

    if debug_first_replicate and replicate_configs:
        _run_one_sim(replicate_configs[0], estimator_names, child_entropies[0], estimand, horizon)

    sim_outputs = Parallel(n_jobs=n_jobs, backend="loky", timeout=task_timeout)(
        delayed(_run_one_sim)(replicate_configs[i], estimator_names, child_entropies[i], estimand, horizon)
        for i in tqdm(range(n_sim), desc="Simulations", total=n_sim)
    )

    error_counts: dict[str, dict[str, int]] = {}
    for sim_out in sim_outputs:
        for name, label in sim_out.get("__errors__", {}).items():
            error_counts.setdefault(name, {}).setdefault(label, 0)
            error_counts[name][label] += 1
    for name, counts in error_counts.items():
        breakdown = ", ".join(f"{k}={v}" for k, v in counts.items())
        print(f"  {name}: {sum(counts.values())}/{n_sim} replicates failed ({breakdown})")

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
            comparison_spec=comparison_specs.get(name) if comparison_specs else None,
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
    all_results: dict[str, list[SimResult]] = {name: [] for name in estimator_names}
    for val in param_values:
        config = base_config.with_overrides(**{param_name: val})
        sim_results = run_simulation(config, estimator_names, n_sim=n_sim, **kwargs)
        for name in estimator_names:
            all_results[name].append(sim_results.get(name))
    return all_results
