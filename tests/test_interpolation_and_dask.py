"""Tests for interpolation.py and init_dask.py coverage.

Exercises:
- interpolate_scipy (grid-to-grid, pairwise, inmemory, various modes)
- apply_over_time_depth (via interpolate_scipy pairwise path)
- scipy_bilinear (grid-to-grid and pairwise)
- interpolate_dataset (unified interface, scipy backend)
- rename_to_standard_pyinterp / rename_back (coordinate renaming)
- configure_dask_logging (log level + dask config)
- configure_dask_workers_env (with mock client)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import xarray as xr

# ---------------------------------------------------------------------------
# Helpers – synthetic datasets
# ---------------------------------------------------------------------------

_LAT = np.linspace(-10, 10, 11)  # 11 points
_LON = np.linspace(20, 40, 11)


def _gridded_ds(
    lat: np.ndarray = _LAT,
    lon: np.ndarray = _LON,
    var_name: str = "ssh",
    n_time: int = 2,
    with_depth: bool = False,
    seed: int = 0,
) -> xr.Dataset:
    """Create a small gridded xr.Dataset for interpolation tests."""
    rng = np.random.default_rng(seed)
    times = pd.date_range("2025-01-01", periods=n_time, freq="1D")

    shape: List[int] = [n_time, len(lat), len(lon)]
    coords: Dict[str, Any] = {"time": times, "latitude": lat, "longitude": lon}
    dims = ["time", "latitude", "longitude"]

    if with_depth:
        depths = np.array([0.0, 10.0])
        shape.insert(1, len(depths))
        coords["depth"] = depths
        dims.insert(1, "depth")

    data = rng.standard_normal(shape).astype(np.float32)
    ds = xr.Dataset({var_name: (dims, data)}, coords=coords)
    ds[var_name].attrs["standard_name"] = var_name
    return ds


# ===================================================================
# Section 1 – scipy_bilinear
# ===================================================================


class TestScipyBilinear:
    """Low-level scipy_bilinear function."""

    def test_grid_to_grid(self):
        """Interpolate on a regular sub-grid (Cartesian product)."""
        from dctools.processing.interpolation import scipy_bilinear

        # Source: 11×11 identity-like surface
        lat_src = _LAT
        lon_src = _LON
        data = np.outer(lat_src, lon_src).astype(np.float64)

        tgt_lat = np.array([-5.0, 0.0, 5.0])
        tgt_lon = np.array([25.0, 30.0, 35.0])

        out = scipy_bilinear(data, lat_src, lon_src, tgt_lat, tgt_lon, pairwise=False)
        assert out.shape == (3, 3)
        # At (0, 30): value should be 0*30 = 0
        assert abs(out[1, 1] - 0.0) < 1e-6

    def test_pairwise(self):
        """Pairwise (track) interpolation."""
        from dctools.processing.interpolation import scipy_bilinear

        lat_src = _LAT
        lon_src = _LON
        data = np.ones((len(lat_src), len(lon_src)), dtype=np.float64)

        tgt_lat = np.array([0.0, 5.0])
        tgt_lon = np.array([30.0, 35.0])

        out = scipy_bilinear(data, lat_src, lon_src, tgt_lat, tgt_lon, pairwise=True)
        assert out.shape == (2,)
        np.testing.assert_allclose(out, 1.0)

    def test_transposed_shape(self):
        """Data with (lon, lat) shape is auto-transposed."""
        from dctools.processing.interpolation import scipy_bilinear

        lat_src = _LAT
        lon_src = _LON
        # Provide data as (lon, lat) – shape (11, 11)
        data = np.ones((len(lon_src), len(lat_src)), dtype=np.float64)

        tgt_lat = np.array([0.0])
        tgt_lon = np.array([30.0])

        out = scipy_bilinear(data, lat_src, lon_src, tgt_lat, tgt_lon)
        assert out.shape == (1, 1)
        np.testing.assert_allclose(out, 1.0)


# ===================================================================
# Section 2 – rename helpers
# ===================================================================


class TestRenameHelpers:
    """Test rename_to_standard_pyinterp and rename_back."""

    def test_rename_round_trip(self):
        """Rename lat/lon → latitude/longitude then back."""
        from dctools.processing.interpolation import rename_back, rename_to_standard_pyinterp

        ds = _gridded_ds()
        ds_renamed, mapping = rename_to_standard_pyinterp(ds, "lat", "lon")
        # Already named latitude/longitude → mapping should be empty or identity
        assert isinstance(ds_renamed, xr.Dataset)

        ds_back = rename_back(ds, ds_renamed, mapping)
        assert set(ds_back.data_vars) == set(ds.data_vars)

    def test_rename_with_different_names(self):
        """Dataset with non-standard names gets renamed."""
        from dctools.processing.interpolation import rename_back, rename_to_standard_pyinterp

        ds = _gridded_ds().rename({"latitude": "lat", "longitude": "lon"})
        ds_renamed, mapping = rename_to_standard_pyinterp(ds, "lat", "lon")
        assert "latitude" in ds_renamed.coords or "lat" in ds_renamed.coords
        ds_back = rename_back(ds, ds_renamed, mapping)
        assert isinstance(ds_back, xr.Dataset)


# ===================================================================
# Section 3 – interpolate_scipy (full function)
# ===================================================================


class TestInterpolateScipy:
    """Tests for the top-level interpolate_scipy function."""

    def test_grid_to_grid_lazy(self):
        """Grid-to-grid interpolation, lazy output mode."""
        from dctools.processing.interpolation import interpolate_scipy

        ds = _gridded_ds(seed=1)
        target = {"lat": np.array([-5.0, 0.0, 5.0]), "lon": np.array([25.0, 30.0, 35.0])}

        result = interpolate_scipy(ds, target, output_mode="lazy")
        assert isinstance(result, xr.Dataset)
        assert "ssh" in result.data_vars

    def test_grid_to_grid_inmemory(self):
        """Grid-to-grid, in-memory mode triggers .compute()."""
        from dctools.processing.interpolation import interpolate_scipy

        ds = _gridded_ds(seed=2)
        target = {"lat": np.array([0.0]), "lon": np.array([30.0])}

        result = interpolate_scipy(ds, target, output_mode="inmemory")
        assert isinstance(result, xr.Dataset)
        # inmemory result should be loaded (numpy backed)
        assert result["ssh"].values is not None

    def test_pairwise_mode(self):
        """Pairwise (Grid-to-Track) interpolation."""
        from dctools.processing.interpolation import interpolate_scipy

        ds = _gridded_ds(seed=3)
        target = {
            "lat": np.array([-5.0, 0.0, 5.0]),
            "lon": np.array([25.0, 30.0, 35.0]),
        }

        result = interpolate_scipy(ds, target, output_mode="lazy", pairwise=True)
        assert isinstance(result, xr.Dataset)
        assert "ssh" in result.data_vars

    def test_pairwise_length_mismatch(self):
        """Pairwise with unequal lat/lon raises ValueError."""
        from dctools.processing.interpolation import interpolate_scipy

        ds = _gridded_ds(seed=4)
        target = {"lat": np.array([0.0, 1.0]), "lon": np.array([30.0])}

        with pytest.raises(ValueError, match="same length"):
            interpolate_scipy(ds, target, pairwise=True)

    def test_with_depth(self):
        """Interpolation on a dataset that has a depth dimension."""
        from dctools.processing.interpolation import interpolate_scipy

        ds = _gridded_ds(seed=5, with_depth=True)
        target = {"lat": np.array([0.0]), "lon": np.array([30.0])}

        result = interpolate_scipy(ds, target, output_mode="lazy")
        assert isinstance(result, xr.Dataset)
        assert "ssh" in result.data_vars

    def test_zarr_output(self, tmp_path):
        """Zarr output mode writes and re-opens from disk."""
        from dctools.processing.interpolation import interpolate_scipy

        ds = _gridded_ds(seed=6, n_time=1)
        target = {"lat": np.array([0.0, 5.0]), "lon": np.array([25.0, 30.0])}
        zarr_path = str(tmp_path / "interp_out.zarr")

        result = interpolate_scipy(
            ds, target, output_mode="zarr", output_path=zarr_path
        )
        assert isinstance(result, xr.Dataset)
        assert "ssh" in result.data_vars

    def test_unknown_mode_raises(self):
        """Unknown output_mode raises ValueError."""
        from dctools.processing.interpolation import interpolate_scipy

        ds = _gridded_ds(seed=7)
        target = {"lat": np.array([0.0]), "lon": np.array([30.0])}

        with pytest.raises(ValueError, match="Unknown mode"):
            interpolate_scipy(ds, target, output_mode="foobar")


# ===================================================================
# Section 4 – interpolate_dataset (unified interface, scipy backend)
# ===================================================================


class TestInterpolateDataset:
    """Tests for the unified interpolate_dataset dispatcher."""

    def _ds_with_standard_names(self, seed: int = 10) -> xr.Dataset:
        """Return a dataset with lat/lon named 'latitude'/'longitude'."""
        return _gridded_ds(seed=seed)

    def test_scipy_grid_to_grid(self):
        """Scipy backend, grid-to-grid."""
        from dctools.processing.interpolation import interpolate_dataset

        ds = self._ds_with_standard_names(seed=11)
        target = {"lat": np.array([-5.0, 0.0, 5.0]), "lon": np.array([25.0, 30.0, 35.0])}

        result = interpolate_dataset(ds, target, interpolation_lib="scipy")
        assert isinstance(result, xr.Dataset)
        assert "ssh" in result.data_vars

    def test_scipy_pairwise(self):
        """Scipy backend, pairwise interpolation."""
        from dctools.processing.interpolation import interpolate_dataset

        ds = self._ds_with_standard_names(seed=12)
        target = {
            "lat": np.array([-5.0, 0.0, 5.0]),
            "lon": np.array([25.0, 30.0, 35.0]),
        }

        result = interpolate_dataset(ds, target, interpolation_lib="scipy", pairwise=True)
        assert isinstance(result, xr.Dataset)
        assert "ssh" in result.data_vars

    def test_unknown_lib_raises(self):
        """Unknown interpolation_lib raises ValueError."""
        from dctools.processing.interpolation import interpolate_dataset

        ds = self._ds_with_standard_names(seed=13)
        target = {"lat": np.array([0.0]), "lon": np.array([30.0])}

        with pytest.raises(ValueError, match="Unknown interpolation library"):
            interpolate_dataset(ds, target, interpolation_lib="nonexistent")

    def test_attrs_preserved(self):
        """Variable and coordinate attributes are preserved after interpolation."""
        from dctools.processing.interpolation import interpolate_dataset

        ds = self._ds_with_standard_names(seed=14)
        ds["ssh"].attrs["units"] = "m"
        ds["ssh"].attrs["long_name"] = "Sea Surface Height"
        target = {"lat": np.array([0.0, 5.0]), "lon": np.array([25.0, 30.0])}

        result = interpolate_dataset(ds, target, interpolation_lib="scipy")
        assert result["ssh"].attrs.get("units") == "m"
        assert result["ssh"].attrs.get("long_name") == "Sea Surface Height"

    def test_missing_standard_name_gets_added(self):
        """Variables without standard_name get one assigned automatically."""
        from dctools.processing.interpolation import interpolate_dataset

        ds = self._ds_with_standard_names(seed=15)
        # Remove standard_name
        ds["ssh"].attrs.pop("standard_name", None)
        target = {"lat": np.array([0.0]), "lon": np.array([30.0])}

        result = interpolate_dataset(ds, target, interpolation_lib="scipy")
        # Should have standard_name auto-assigned
        assert result["ssh"].attrs.get("standard_name") == "ssh"


# ===================================================================
# Section 5 – apply_over_time_depth
# ===================================================================


class TestApplyOverTimeDepth:
    """Direct tests for apply_over_time_depth."""

    def test_basic_grid_to_grid(self):
        """Basic apply_over_time_depth for grid-to-grid."""
        from dctools.processing.interpolation import apply_over_time_depth

        ds = _gridded_ds(seed=20)
        lat_src = ds["latitude"].values
        lon_src = ds["longitude"].values
        tgt_lat = np.array([-5.0, 0.0, 5.0])
        tgt_lon = np.array([25.0, 30.0, 35.0])

        out = apply_over_time_depth(
            ds,
            var_names=["ssh"],
            depth_name="depth",
            lat_name="latitude",
            lon_name="longitude",
            lat_src=lat_src,
            lon_src=lon_src,
            tgt_lat=tgt_lat,
            tgt_lon=tgt_lon,
        )
        assert "ssh" in out
        assert isinstance(out["ssh"], xr.DataArray)

    def test_pairwise(self):
        """apply_over_time_depth in pairwise mode."""
        from dctools.processing.interpolation import apply_over_time_depth

        ds = _gridded_ds(seed=21)
        lat_src = ds["latitude"].values
        lon_src = ds["longitude"].values
        tgt_lat = np.array([-5.0, 0.0, 5.0])
        tgt_lon = np.array([25.0, 30.0, 35.0])

        out = apply_over_time_depth(
            ds,
            var_names=["ssh"],
            depth_name="depth",
            lat_name="latitude",
            lon_name="longitude",
            lat_src=lat_src,
            lon_src=lon_src,
            tgt_lat=tgt_lat,
            tgt_lon=tgt_lon,
            pairwise=True,
        )
        assert "ssh" in out

    def test_missing_var_skipped(self):
        """Variable not in dataset is silently skipped."""
        from dctools.processing.interpolation import apply_over_time_depth

        ds = _gridded_ds(seed=22)
        lat_src = ds["latitude"].values
        lon_src = ds["longitude"].values

        out = apply_over_time_depth(
            ds,
            var_names=["nonexistent"],
            depth_name="depth",
            lat_name="latitude",
            lon_name="longitude",
            lat_src=lat_src,
            lon_src=lon_src,
            tgt_lat=np.array([0.0]),
            tgt_lon=np.array([30.0]),
        )
        assert len(out) == 0


# ===================================================================
# Section 6 – configure_dask_logging
# ===================================================================


class TestConfigureDaskLogging:
    """Test configure_dask_logging coverage."""

    def test_sets_log_levels(self):
        """After configure_dask_logging, distributed loggers should be ERROR."""
        from dctools.utilities.init_dask import configure_dask_logging

        configure_dask_logging()

        # Check that distributed loggers are set to ERROR
        for name in ("distributed", "distributed.worker", "distributed.scheduler"):
            assert logging.getLogger(name).level == logging.ERROR

    def test_dask_config_values(self):
        """Dask config values are set correctly."""
        import dask

        from dctools.utilities.init_dask import configure_dask_logging

        configure_dask_logging()

        cfg = dask.config.config
        worker_cfg = cfg.get("distributed", {}).get("worker", {})
        mem_cfg = worker_cfg.get("memory", {})
        assert float(mem_cfg.get("target", 0)) == pytest.approx(0.8)
        assert float(mem_cfg.get("spill", 0)) == pytest.approx(0.9)
        assert float(mem_cfg.get("pause", 0)) == pytest.approx(0.95)


# ===================================================================
# Section 7 – configure_dask_workers_env (with mock client)
# ===================================================================


class TestConfigureDaskWorkersEnv:
    """Test configure_dask_workers_env with a mocked distributed.Client."""

    def test_success_path(self):
        """Successful run on mock client returns True."""
        from dctools.utilities.init_dask import configure_dask_workers_env

        mock_client = MagicMock()
        mock_client.run.return_value = {"worker-1": True, "worker-2": True}

        result = configure_dask_workers_env(mock_client)
        assert result is True
        mock_client.run.assert_called_once()

    def test_failure_path(self):
        """Client.run exception returns False."""
        from dctools.utilities.init_dask import configure_dask_workers_env

        mock_client = MagicMock()
        mock_client.run.side_effect = RuntimeError("connection lost")

        result = configure_dask_workers_env(mock_client)
        assert result is False

    def test_worker_function_is_callable(self):
        """The function passed to client.run is callable and returns True."""
        from dctools.utilities.init_dask import configure_dask_workers_env

        captured_fn = None

        def capture_run(fn):
            nonlocal captured_fn
            captured_fn = fn
            return {"w1": fn()}

        mock_client = MagicMock()
        mock_client.run.side_effect = capture_run

        result = configure_dask_workers_env(mock_client)
        assert result is True
        # The worker function itself should return True
        assert captured_fn is not None
        assert captured_fn() is True
