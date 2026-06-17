from __future__ import annotations
from pathlib import Path
from typing import Optional
import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator, model_validator, Field


def gaussian_multiplier_quantile(
    ic_matrix: np.ndarray,
    se_vec: np.ndarray,
    alpha: float = 0.05,
    n_draws: int = 10_000,
    rng: np.random.Generator | None = None,
) -> float:
    """(1-alpha) quantile of max_j |ε'IC_j / (√n·se_j)| for ε ~ N(0,I_n).

    Implements the Gaussian multiplier bootstrap for simultaneous inference
    across q estimands.  IC columns must be pre-centered (mean ≈ 0).

    Parameters
    ----------
    ic_matrix : (n, q) array — per-subject influence functions, one column per estimand
    se_vec    : (q,) array — standard errors (sqrt(Var(IC_j)/n)) of each estimand
    alpha     : significance level (default 0.05 → 95% simultaneous bands)
    n_draws   : number of Gaussian multiplier bootstrap draws (default 10 000)
    rng       : numpy Generator; seeded at 0 if None

    Returns
    -------
    float — (1-alpha) quantile; used as the joint critical value q̂
    """
    if rng is None:
        rng = np.random.default_rng(0)
    n, q = ic_matrix.shape
    eps = rng.standard_normal((n_draws, n))              # (B, n)
    standardized = (eps @ ic_matrix) / (np.sqrt(n) * se_vec)  # (B, q)
    return float(np.quantile(np.max(np.abs(standardized), axis=1), 1.0 - alpha))


class ComparisonSpec(BaseModel):
    """Declares the estimand coherence between an estimator's output and the
    truth it is compared against.

    The Exp 8 case is the canonical example: compute_true_effects() returns
    an *all-cause* counterfactual risk difference, but concrete_RMST correctly
    estimates the *cause-specific* CIF difference in competing-risk scenarios —
    the comparison is estimand-mismatched.  Both sides carry the label "ATE"
    (since estimand labels are user-supplied strings, not semantic types), so a
    simple string-equality check wouldn't catch this.  The mismatch must be
    *declared* explicitly by the experiment author.

    Usage
    -----
    Well-matched comparison (most experiments):
        No ComparisonSpec needed — comparison is implicitly coherent.

    Known-mismatched comparison (e.g. Exp 8 concrete_RMST):
        ComparisonSpec(
            estimator_estimand="cause_specific_CIF_diff",
            truth_estimand="all_cause_RD",
            allow_known_mismatch=True,
        )
        Constructing this without allow_known_mismatch=True raises immediately,
        forcing the author to consciously opt in.

    The spec is stored on SimResult.comparison_spec so the known mismatch
    annotation travels with the result and is visible in summaries/audits.
    """
    model_config = ConfigDict(frozen=True)

    estimator_estimand: str
    truth_estimand: str
    allow_known_mismatch: bool = False

    @model_validator(mode="after")
    def _check_coherence(self) -> "ComparisonSpec":
        if self.estimator_estimand != self.truth_estimand and not self.allow_known_mismatch:
            raise ValueError(
                f"estimator targets '{self.estimator_estimand}' but truth was "
                f"computed as '{self.truth_estimand}' — these are different "
                "estimands and the comparison will be biased.  Set "
                "allow_known_mismatch=True to suppress this if the mismatch "
                "is intentional and documented (e.g. Exp 8: concrete_RMST "
                "computes cause-specific CIF while compute_true_effects() "
                "returns the all-cause risk difference)."
            )
        return self


class EstimatorResult(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    name: str
    estimand: str
    point_estimate: float
    standard_error: float
    ci_lower: float
    ci_upper: float
    ess: Optional[float] = None
    convergence_info: Optional[dict] = None
    ic: Optional[np.ndarray] = None

    @field_validator("ic")
    @classmethod
    def _validate_ic(cls, v: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if v is None:
            return None
        return np.asarray(v)

    # NOTE: estimand is intentionally left as plain `str`, not a Literal —
    # concrete_simultaneous.py emits dynamically-built labels like
    # "RD_t0.4"/"RMST_t0.7" (one per evaluation time, see its line ~142
    # `est_lbl = str(row.get("estimand", ...))`), and overlap.py uses "ATO".
    # A fixed Literal set would reject those real, currently-passing results.

    @model_validator(mode="after")
    def _check_coherence(self) -> "EstimatorResult":
        # NaN is the codebase's established "did not converge" sentinel
        # (see e.g. naive.py's empty-arm branch, tmle_ipcw.py:179,
        # concrete_simultaneous.py's `row.get(..., np.nan)` fallback) — it
        # must pass through untouched, not be treated as a coherence
        # violation. Only finite values are checked.
        se = self.standard_error
        if np.isfinite(se) and se <= 0:
            raise ValueError(
                f"standard_error must be > 0 (got {se}); use NaN to signal "
                "a non-converged replicate instead of 0 or a negative value"
            )
        pt, lo, hi = self.point_estimate, self.ci_lower, self.ci_upper
        if np.isfinite(pt) and np.isfinite(lo) and np.isfinite(hi):
            tol = 1e-9 * max(1.0, abs(pt))
            if not (lo - tol <= pt <= hi + tol):
                raise ValueError(
                    f"CI must bracket the point estimate: ci_lower={lo}, "
                    f"point_estimate={pt}, ci_upper={hi}"
                )
        return self

    def __repr__(self) -> str:
        fields = ", ".join(
            f"{name}={getattr(self, name)!r}"
            for name in self.model_fields
            if name != "ic"
        )
        return f"{self.__class__.__name__}({fields})"


class SimResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    estimator_name: str
    estimand: str
    true_value: float
    n_sim: int
    estimates: np.ndarray
    se_estimates: np.ndarray
    ci_lowers: np.ndarray
    ci_uppers: np.ndarray
    nc_estimates: np.ndarray
    # If the estimator's estimand and the truth's estimand are semantically
    # different (e.g. cause-specific CIF vs all-cause RD in Exp 8), the
    # experiment author must attach a ComparisonSpec with allow_known_mismatch=True
    # to document the deliberate mismatch.  None means the comparison is
    # implicitly coherent (the default for all non-competing-risk experiments).
    comparison_spec: Optional[ComparisonSpec] = None

    def __repr__(self) -> str:
        fields = ", ".join(
            f"{name}={getattr(self, name)!r}"
            for name in ("estimator_name", "estimand", "true_value", "n_sim")
        )
        return f"{self.__class__.__name__}({fields})"

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


class SimResultFamily(BaseModel):
    """Simultaneous-inference results across a joint estimand family.

    Holds one SimResult per family member (estimand) together with the
    simultaneous CIs and critical values so joint coverage can be computed.

    Attributes
    ----------
    members       : list of SimResult — one per estimand in the family
    sim_ci_lowers : dict[estimand → (n_sim,) array] — simultaneous CI lower bounds
    sim_ci_uppers : dict[estimand → (n_sim,) array] — simultaneous CI upper bounds
    crit_values   : (n_sim,) array — per-simulation joint critical value q̂
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    members: list[SimResult]
    sim_ci_lowers: dict[str, np.ndarray]
    sim_ci_uppers: dict[str, np.ndarray]
    crit_values:   np.ndarray

    @property
    def joint_pointwise_coverage(self) -> float:
        """Fraction of sims where ALL estimands are covered by their pointwise CI."""
        if not self.members:
            return float("nan")
        n = self.members[0].n_sim
        per_sim = np.ones(n, dtype=bool)
        for sr in self.members:
            if len(sr.ci_lowers) != n:
                continue
            per_sim &= (sr.ci_lowers <= sr.true_value) & (sr.true_value <= sr.ci_uppers)
        return float(np.mean(per_sim))

    @property
    def simultaneous_coverage(self) -> float:
        """Fraction of sims where ALL estimands are covered by their simultaneous CI."""
        if not self.members or not self.sim_ci_lowers:
            return float("nan")
        n = self.members[0].n_sim
        per_sim = np.ones(n, dtype=bool)
        for sr in self.members:
            lo = self.sim_ci_lowers.get(sr.estimand)
            hi = self.sim_ci_uppers.get(sr.estimand)
            if lo is None or hi is None or len(lo) != n:
                continue
            per_sim &= (lo <= sr.true_value) & (sr.true_value <= hi)
        return float(np.mean(per_sim))

    @property
    def mean_crit_value(self) -> float:
        return float(np.mean(self.crit_values)) if len(self.crit_values) else float("nan")

    def summary(self) -> dict:
        rows = []
        for sr in self.members:
            row = sr.summary()
            lo = self.sim_ci_lowers.get(sr.estimand)
            hi = self.sim_ci_uppers.get(sr.estimand)
            if lo is not None and hi is not None and len(lo) == sr.n_sim:
                sim_cov = float(np.mean((lo <= sr.true_value) & (sr.true_value <= hi)))
                row["sim_coverage"] = round(sim_cov, 3)
            rows.append(row)
        return {
            "per_estimand":              rows,
            "joint_pointwise_coverage":  round(self.joint_pointwise_coverage, 3),
            "simultaneous_coverage":     round(self.simultaneous_coverage, 3),
            "mean_crit_value":           round(self.mean_crit_value, 4),
        }
