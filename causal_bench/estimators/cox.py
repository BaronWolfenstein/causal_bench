# causal_bench/estimators/cox.py
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from scipy import stats
from causal_bench.estimators.base import BaseEstimator
from causal_bench.metrics import EstimatorResult


class CoxEstimator(BaseEstimator):
    name = "Cox"

    def __init__(self, include_L1: bool = False, n_bootstrap: int = 100):
        self.include_L1 = include_L1
        self.n_bootstrap = n_bootstrap

    def estimate(self, df: pd.DataFrame, horizon: float = 1.0,
                 estimand: str = "ATE") -> list[EstimatorResult]:
        covs = ["A", "W1", "W2", "W3", "W4"]
        if self.include_L1 and "L1" in df.columns:
            covs.append("L1")

        fit_df = df[covs + ["T_obs", "Delta"]].dropna()
        name = "Cox+L1" if self.include_L1 else self.name

        def _fit_and_predict(data):
            cph = CoxPHFitter(penalizer=0.1)
            cph.fit(data, duration_col="T_obs", event_col="Delta",
                    fit_options={"max_steps": 50})
            d1 = data.copy(); d1["A"] = 1.0
            d0 = data.copy(); d0["A"] = 0.0
            s1 = cph.predict_survival_function(d1, times=[horizon]).iloc[0].mean()
            s0 = cph.predict_survival_function(d0, times=[horizon]).iloc[0].mean()
            return (1 - s1) - (1 - s0)

        point = _fit_and_predict(fit_df)

        # Bootstrap SE
        rng = np.random.default_rng(42)
        boot_ests = []
        for _ in range(self.n_bootstrap):
            idx = rng.choice(len(fit_df), size=len(fit_df), replace=True)
            try:
                boot_ests.append(_fit_and_predict(fit_df.iloc[idx].reset_index(drop=True)))
            except Exception:
                pass
        se = float(np.std(boot_ests, ddof=1)) if len(boot_ests) > 1 else 0.05
        se = max(se, 1e-6)
        z = stats.norm.ppf(0.975)
        return [EstimatorResult(
            name=name, estimand="ATE",
            point_estimate=float(point), standard_error=se,
            ci_lower=float(point) - z * se,
            ci_upper=float(point) + z * se,
        )]
