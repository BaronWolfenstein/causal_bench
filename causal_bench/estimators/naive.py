# causal_bench/estimators/naive.py
import numpy as np
import pandas as pd
from scipy import stats
from causal_bench.estimators.base import BaseEstimator
from causal_bench.metrics import EstimatorResult


class NaiveEstimator(BaseEstimator):
    name = "Naive"

    def estimate(self, df: pd.DataFrame, horizon: float = 1.0,
                 estimand: str = "ATE") -> list[EstimatorResult]:
        y1 = df.loc[df["A"] == 1, "Delta"].astype(float)
        y0 = df.loc[df["A"] == 0, "Delta"].astype(float)

        if len(y1) == 0 or len(y0) == 0:
            return [EstimatorResult(self.name, estimand, float("nan"),
                                    float("nan"), float("nan"), float("nan"))]

        point = float(y1.mean() - y0.mean())
        se = float(np.sqrt(y1.var(ddof=1) / len(y1) + y0.var(ddof=1) / len(y0)))
        se = max(se, 1e-6)
        z = stats.norm.ppf(0.975)
        return [EstimatorResult(
            name=self.name, estimand=estimand,
            point_estimate=point, standard_error=se,
            ci_lower=point - z * se, ci_upper=point + z * se,
        )]
