"""Tests for Arrow/Parquet SimResult persistence."""
import tempfile
from pathlib import Path

import numpy as np
import pytest

from causal_bench.metrics import SimResult


def _make_sim_result(n_sim=50, seed=0) -> SimResult:
    rng = np.random.default_rng(seed)
    return SimResult(
        estimator_name="tmle_ipcw",
        estimand="ATE",
        true_value=0.12,
        n_sim=n_sim,
        estimates=rng.normal(0.12, 0.05, n_sim),
        se_estimates=np.abs(rng.normal(0.05, 0.01, n_sim)),
        ci_lowers=rng.normal(0.02, 0.05, n_sim),
        ci_uppers=rng.normal(0.22, 0.05, n_sim),
        nc_estimates=rng.normal(0, 0.02, n_sim),
    )


class TestSimResultParquet:

    def test_roundtrip_scalar_fields(self, tmp_path):
        sr = _make_sim_result()
        path = tmp_path / "sr.parquet"
        sr.to_parquet(path)
        loaded = SimResult.from_parquet(path)
        assert loaded.estimator_name == sr.estimator_name
        assert loaded.estimand == sr.estimand
        assert loaded.true_value == sr.true_value
        assert loaded.n_sim == sr.n_sim

    def test_roundtrip_array_values(self, tmp_path):
        sr = _make_sim_result()
        path = tmp_path / "sr.parquet"
        sr.to_parquet(path)
        loaded = SimResult.from_parquet(path)
        np.testing.assert_array_almost_equal(loaded.estimates,    sr.estimates)
        np.testing.assert_array_almost_equal(loaded.se_estimates, sr.se_estimates)
        np.testing.assert_array_almost_equal(loaded.ci_lowers,    sr.ci_lowers)
        np.testing.assert_array_almost_equal(loaded.ci_uppers,    sr.ci_uppers)
        np.testing.assert_array_almost_equal(loaded.nc_estimates, sr.nc_estimates)

    def test_roundtrip_preserves_metrics(self, tmp_path):
        """Derived properties (bias, coverage, rmse) survive the roundtrip."""
        sr = _make_sim_result()
        path = tmp_path / "sr.parquet"
        sr.to_parquet(path)
        loaded = SimResult.from_parquet(path)
        assert abs(loaded.bias     - sr.bias)     < 1e-9
        assert abs(loaded.rmse     - sr.rmse)     < 1e-9
        assert abs(loaded.coverage - sr.coverage) < 1e-9

    def test_file_is_parquet(self, tmp_path):
        """File written by to_parquet() is a valid Parquet file."""
        import pyarrow.parquet as pq
        sr = _make_sim_result()
        path = tmp_path / "sr.parquet"
        sr.to_parquet(path)
        meta = pq.read_metadata(path)
        assert meta.num_rows == sr.n_sim
        assert meta.num_columns == 5  # estimates, se_estimates, ci_lowers, ci_uppers, nc_estimates

    def test_compression_snappy(self, tmp_path):
        """Output uses Snappy compression (fast, small)."""
        import pyarrow.parquet as pq
        sr = _make_sim_result(n_sim=200)
        path = tmp_path / "sr.parquet"
        sr.to_parquet(path)
        pf = pq.ParquetFile(path)
        col_meta = pf.metadata.row_group(0).column(0)
        assert col_meta.compression.lower() == "snappy"

    def test_float32_arrays_upcast(self, tmp_path):
        """float32 arrays are stored as float64 for precision."""
        import pyarrow.parquet as pq
        sr = _make_sim_result()
        sr.estimates = sr.estimates.astype(np.float32)
        path = tmp_path / "sr.parquet"
        sr.to_parquet(path)
        table = pq.read_table(path)
        assert table.schema.field("estimates").type == __import__("pyarrow").float64()

    def test_large_n_sim(self, tmp_path):
        """n_sim=2000 roundtrips without precision loss."""
        sr = _make_sim_result(n_sim=2000)
        path = tmp_path / "sr.parquet"
        sr.to_parquet(path)
        loaded = SimResult.from_parquet(path)
        assert loaded.n_sim == 2000
        np.testing.assert_array_almost_equal(loaded.estimates, sr.estimates)

    def test_negative_true_value(self, tmp_path):
        """Negative true_value (common in our DGP) roundtrips correctly."""
        sr = _make_sim_result()
        sr.true_value = -0.5
        path = tmp_path / "sr.parquet"
        sr.to_parquet(path)
        loaded = SimResult.from_parquet(path)
        assert loaded.true_value == -0.5

    def test_path_as_string(self, tmp_path):
        """to_parquet / from_parquet accept str paths, not just Path objects."""
        sr = _make_sim_result()
        path = str(tmp_path / "sr.parquet")
        sr.to_parquet(path)
        loaded = SimResult.from_parquet(path)
        assert loaded.estimator_name == sr.estimator_name

    def test_overwrite(self, tmp_path):
        """Writing to an existing path overwrites cleanly."""
        sr1 = _make_sim_result(seed=0)
        sr2 = _make_sim_result(seed=99)
        path = tmp_path / "sr.parquet"
        sr1.to_parquet(path)
        sr2.to_parquet(path)
        loaded = SimResult.from_parquet(path)
        np.testing.assert_array_almost_equal(loaded.estimates, sr2.estimates)
