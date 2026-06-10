# causal_bench/estimators/kaplan_meier.py
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from scipy import stats
from causal_bench.estimators.base import BaseEstimator
from causal_bench.metrics import EstimatorResult


class KaplanMeierEstimator(BaseEstimator):
    name = "KM"

    def estimate(self, df: pd.DataFrame, horizon: float = 1.0,
                 estimand: str = "ATE") -> list[EstimatorResult]:
        arm1 = df[df["A"] == 1]
        arm0 = df[df["A"] == 0]

        km1 = KaplanMeierFitter().fit(arm1["T_obs"], arm1["Delta"])
        km0 = KaplanMeierFitter().fit(arm0["T_obs"], arm0["Delta"])

        s1 = float(km1.survival_function_at_times([horizon]).iloc[0])
        s0 = float(km0.survival_function_at_times([horizon]).iloc[0])
        point = (1 - s1) - (1 - s0)

        var1 = _greenwood_var(km1, horizon)
        var0 = _greenwood_var(km0, horizon)
        se = float(np.sqrt(var1 + var0))
        se = max(se, 1e-6)
        z = stats.norm.ppf(0.975)
        return [EstimatorResult(
            name=self.name, estimand="ATE",
            point_estimate=point, standard_error=se,
            ci_lower=point - z * se, ci_upper=point + z * se,
        )]


def _greenwood_var(km: KaplanMeierFitter, t: float) -> float:
    """Greenwood variance of 1-S(t) at time t."""
    tbl = km.event_table[km.event_table.index <= t]
    s_t = float(km.survival_function_at_times([t]).iloc[0])
    gw = 0.0
    for row in tbl.itertuples():
        d = row.observed
        n_risk = row.at_risk
        if n_risk > d > 0:
            gw += d / (n_risk * (n_risk - d))
    return (s_t ** 2) * gw
