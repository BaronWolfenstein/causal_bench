from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class EstimatorResult:
    name: str
    estimand: str
    point_estimate: float
    standard_error: float
    ci_lower: float
    ci_upper: float
    ess: Optional[float] = None
    convergence_info: Optional[dict] = None


@dataclass
class SimResult:
    estimator_name: str
    estimand: str
    true_value: float
    n_sim: int
    estimates: np.ndarray = field(repr=False)
    se_estimates: np.ndarray = field(repr=False)
    ci_lowers: np.ndarray = field(repr=False)
    ci_uppers: np.ndarray = field(repr=False)
    nc_estimates: np.ndarray = field(repr=False)

    @property
    def bias(self) -> float:
        return float(np.mean(self.estimates) - self.true_value)

    @property
    def rmse(self) -> float:
        return float(np.sqrt(np.mean((self.estimates - self.true_value) ** 2)))

    @property
    def coverage(self) -> float:
        covered = (self.ci_lowers <= self.true_value) & (self.true_value <= self.ci_uppers)
        return float(np.mean(covered))

    @property
    def ci_width(self) -> float:
        return float(np.mean(self.ci_uppers - self.ci_lowers))

    @property
    def se_ratio(self) -> float:
        empirical_se = np.std(self.estimates, ddof=1)
        if empirical_se < 1e-10:
            return float("nan")
        return float(np.median(self.se_estimates) / empirical_se)

    @property
    def nc_bias(self) -> float:
        return float(np.mean(self.nc_estimates))

    def summary(self) -> dict:
        return {
            "estimator": self.estimator_name,
            "estimand": self.estimand,
            "true": round(self.true_value, 4),
            "bias": round(self.bias, 4),
            "rmse": round(self.rmse, 4),
            "coverage": round(self.coverage, 3),
            "ci_width": round(self.ci_width, 4),
            "se_ratio": round(self.se_ratio, 3),
            "nc_bias": round(self.nc_bias, 4),
        }
