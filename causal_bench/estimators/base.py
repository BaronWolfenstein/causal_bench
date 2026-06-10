# causal_bench/estimators/base.py
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from causal_bench.metrics import EstimatorResult


class BaseEstimator(ABC):
    name: str = "base"

    @abstractmethod
    def estimate(
        self,
        df: pd.DataFrame,
        horizon: float = 1.0,
        estimand: str = "ATE",
    ) -> list[EstimatorResult]:
        """
        Args:
            df: DataFrame with T_obs, Delta, A, W1-W4, compliance, Y_neg, enrollment_time
            horizon: time point for risk difference
            estimand: "ATE" or "ATT"
        Returns:
            list of EstimatorResult
        """
        ...

    def estimate_negative_control(
        self, df: pd.DataFrame, horizon: float = 1.0
    ) -> float:
        """Naive difference in Y_neg (above-median) by treatment arm.
        Should be ~0 in expectation since Y_neg has no treatment effect."""
        threshold = df["Y_neg"].median()
        y_nc = (df["Y_neg"] > threshold).astype(float)
        treated = df["A"] == 1
        if treated.sum() == 0 or (~treated).sum() == 0:
            return float("nan")
        return float(y_nc[treated].mean() - y_nc[~treated].mean())
