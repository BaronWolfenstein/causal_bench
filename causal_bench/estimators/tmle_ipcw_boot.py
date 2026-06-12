import numpy as np
import pandas as pd
from scipy import stats
from causal_bench.estimators.base import BaseEstimator
from causal_bench.estimators.tmle_ipcw import TMLEIPCWEstimator
from causal_bench.metrics import EstimatorResult


class TMLEIPCWBootEstimator(BaseEstimator):
    """TMLE+IPCW with bootstrap SE.

    Runs the standard TMLE+IPCW on B bootstrap resamples and uses
    std(bootstrap point estimates) as the SE.  The point estimate and
    CI come from the original (non-resampled) fit.  The IC-based SE
    from the original fit is also stored in convergence_info for
    comparison.

    Intended as a diagnostic reference: shows what calibrated coverage
    looks like and quantifies the IC SE underestimate.
    """

    def __init__(self, n_bootstrap: int = 200, use_compliance: bool = False,
                 n_folds: int = 5, random_state: int = 42):
        self.n_bootstrap = n_bootstrap
        self.use_compliance = use_compliance
        self.n_folds = n_folds
        self.random_state = random_state
        self._base = TMLEIPCWEstimator(use_compliance=use_compliance,
                                       n_folds=n_folds,
                                       random_state=random_state)

    @property
    def name(self) -> str:
        suffix = "+Comply" if self.use_compliance else ""
        return f"TMLE+IPCW+Boot{suffix}"

    def estimate(self, df: pd.DataFrame, horizon: float = 1.0,
                 estimand: str = "ATE") -> list[EstimatorResult]:
        n = len(df)
        rng = np.random.default_rng(self.random_state)

        # Original fit — point estimate + IC-based SE for comparison
        orig_results = {r.estimand: r for r in self._base.estimate(df, horizon, estimand)}

        # Bootstrap
        boot_points: dict[str, list[float]] = {}
        for _ in range(self.n_bootstrap):
            idx = rng.integers(0, n, size=n)
            df_boot = df.iloc[idx].reset_index(drop=True)
            try:
                for r in self._base.estimate(df_boot, horizon, estimand):
                    boot_points.setdefault(r.estimand, []).append(r.point_estimate)
            except Exception:
                pass

        results = []
        z = stats.norm.ppf(0.975)
        for est, orig in orig_results.items():
            bps = boot_points.get(est, [])
            if len(bps) < 10:
                results.append(orig)
                continue
            boot_se = float(np.std(bps, ddof=1))
            point = orig.point_estimate
            results.append(EstimatorResult(
                name=self.name,
                estimand=est,
                point_estimate=point,
                standard_error=boot_se,
                ci_lower=point - z * boot_se,
                ci_upper=point + z * boot_se,
                convergence_info={
                    "ic_se": orig.standard_error,
                    "boot_se": boot_se,
                    "ic_boot_ratio": orig.standard_error / boot_se if boot_se > 1e-10 else None,
                    "n_boot_success": len(bps),
                },
            ))
        return results
