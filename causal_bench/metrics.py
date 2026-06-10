from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
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

    def to_parquet(self, path: str | Path) -> None:
        """Persist SimResult arrays to a Parquet file.

        Schema: one row per simulation replicate, array columns plus scalar
        fields stored as Parquet metadata so from_parquet() reconstructs exactly.

        Requires pyarrow (pip install "causal_bench[storage]").
        """
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.table({
            "estimates":    self.estimates.astype(np.float64),
            "se_estimates": self.se_estimates.astype(np.float64),
            "ci_lowers":    self.ci_lowers.astype(np.float64),
            "ci_uppers":    self.ci_uppers.astype(np.float64),
            "nc_estimates": self.nc_estimates.astype(np.float64),
        })
        # Scalar fields go into Parquet metadata (bytes, so encode as str)
        meta = {
            b"estimator_name": self.estimator_name.encode(),
            b"estimand":       self.estimand.encode(),
            b"true_value":     str(self.true_value).encode(),
            b"n_sim":          str(self.n_sim).encode(),
        }
        table = table.replace_schema_metadata({**table.schema.metadata, **meta}
                                               if table.schema.metadata else meta)
        pq.write_table(table, path, compression="snappy")

    @classmethod
    def from_parquet(cls, path: str | Path) -> "SimResult":
        """Reconstruct a SimResult from a Parquet file written by to_parquet().

        Requires pyarrow (pip install "causal_bench[storage]").
        """
        import pyarrow.parquet as pq

        table = pq.read_table(path)
        meta  = table.schema.metadata or {}
        return cls(
            estimator_name = meta[b"estimator_name"].decode(),
            estimand       = meta[b"estimand"].decode(),
            true_value     = float(meta[b"true_value"].decode()),
            n_sim          = int(meta[b"n_sim"].decode()),
            estimates      = table["estimates"].to_numpy(),
            se_estimates   = table["se_estimates"].to_numpy(),
            ci_lowers      = table["ci_lowers"].to_numpy(),
            ci_uppers      = table["ci_uppers"].to_numpy(),
            nc_estimates   = table["nc_estimates"].to_numpy(),
        )
