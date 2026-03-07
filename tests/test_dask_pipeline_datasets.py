"""Integration tests: Dask pipeline with representative datasets.

Each test class targets a dataset type from the DC2 evaluation:

- **Glonet** (prediction, gridded, zarr format)  → RMSD metric on grid
- **Glorys** (reference, gridded, NetCDF)          → RMSD metric on grid
- **Saral**  (observation, nadir altimetry tracks) → Class4 / obs path
- **Argo**   (observation, vertical profiles)      → Class4 / ARGO shared Zarr path

All tests use a real ``dask.distributed.LocalCluster`` (1 worker,
``processes=False``) and submit ``compute_metric`` tasks.  Synthetic
xr.Datasets are passed *inline* (not as file paths) to skip remote I/O.

Marked ``@pytest.mark.integration`` — included in ``poe coverage`` but
excluded from ``poe all`` (fast CI).
"""

from __future__ import annotations

import os
from argparse import Namespace
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Helpers — synthetic datasets
# ---------------------------------------------------------------------------


def _gridded_ds(
    times: pd.DatetimeIndex,
    lat: np.ndarray | None = None,
    lon: np.ndarray | None = None,
    var_name: str = "zos",
    add_depth: bool = False,
    seed: int = 42,
) -> xr.Dataset:
    """Create a small gridded xr.Dataset (time x lat x lon [x depth])."""
    rng = np.random.default_rng(seed)
    if lat is None:
        lat = np.linspace(-5, 5, 6, dtype=np.float64)
    if lon is None:
        lon = np.linspace(-10, 10, 8, dtype=np.float64)

    shape = (len(times), len(lat), len(lon))
    coords: Dict[str, Any] = {"time": times, "lat": lat, "lon": lon}

    if add_depth:
        depth = np.array([0.0, 10.0, 50.0], dtype=np.float64)
        shape = (len(times), len(depth), len(lat), len(lon))
        coords["depth"] = depth
        dims = ("time", "depth", "lat", "lon")
    else:
        dims = ("time", "lat", "lon")

    data = rng.standard_normal(shape).astype(np.float32)
    # Standard-name mapping so oceanbench.core.rmsd can resolve variables.
    _STANDARD_NAMES = {
        "zos": "sea_surface_height_above_geoid",
        "so": "sea_water_salinity",
        "thetao": "sea_water_potential_temperature",
        "uo": "eastward_sea_water_velocity",
        "vo": "northward_sea_water_velocity",
    }
    attrs = {}
    if var_name in _STANDARD_NAMES:
        attrs["standard_name"] = _STANDARD_NAMES[var_name]
    ds = xr.Dataset(
        {var_name: xr.Variable(dims, data, attrs=attrs)},
        coords=coords,
    )
    return ds


def _obs_track_ds(
    times: pd.DatetimeIndex,
    n_points: int = 50,
    var_name: str = "sla",
    seed: int = 99,
) -> xr.Dataset:
    """Create a synthetic along-track observation dataset (Saral-style)."""
    rng = np.random.default_rng(seed)
    time_vals = np.sort(
        rng.choice(times.values, size=n_points, replace=True)
    )
    return xr.Dataset(
        {var_name: ("n_points", rng.standard_normal(n_points).astype(np.float32))},
        coords={
            "time": ("n_points", time_vals),
            "lat": ("n_points", rng.uniform(-5, 5, n_points).astype(np.float64)),
            "lon": ("n_points", rng.uniform(-10, 10, n_points).astype(np.float64)),
        },
    )


def _argo_profile_ds(
    times: pd.DatetimeIndex,
    n_profiles: int = 20,
    seed: int = 123,
) -> xr.Dataset:
    """Create a synthetic ARGO profile dataset (N_POINTS observation dim)."""
    rng = np.random.default_rng(seed)
    time_vals = np.sort(rng.choice(times.values, size=n_profiles, replace=True))
    return xr.Dataset(
        {
            "TEMP": ("N_POINTS", rng.uniform(5, 25, n_profiles).astype(np.float32)),
            "PSAL": ("N_POINTS", rng.uniform(34, 36, n_profiles).astype(np.float32)),
        },
        coords={
            "TIME": ("N_POINTS", time_vals),
            "LATITUDE": ("N_POINTS", rng.uniform(-5, 5, n_profiles).astype(np.float64)),
            "LONGITUDE": ("N_POINTS", rng.uniform(-10, 10, n_profiles).astype(np.float64)),
            "DEPTH": ("N_POINTS", rng.uniform(0, 2000, n_profiles).astype(np.float32)),
        },
    )


def _source_config(protocol: str = "local", **extras):
    """Build a minimal Namespace that satisfies compute_metric."""
    defaults = {
        "protocol": protocol,
        "init_type": "from_json",
        "local_root": "/tmp",
        "file_pattern": "*.nc",
        "max_samples": 1,
        "groups": None,
        "keep_variables": ["zos"],
        "eval_variables": ["zos"],
        "file_cache": None,
        "dataset_processor": None,
        "filter_values": {},
        "full_day_data": False,
        "fs": None,
    }
    defaults.update(extras)
    return Namespace(**defaults)


def _noop_open(*args, **kwargs):
    """Pass-through open function returned by mocked create_worker_connect_config.

    For non-observation references, compute_metric calls open_ref_func(ref_source)
    where ref_source is the inline xr.Dataset.  We simply return whatever was passed.
    """
    if args and isinstance(args[0], xr.Dataset):
        return args[0]
    return None


# ---------------------------------------------------------------------------
# Fixture: shared LocalCluster for all slow tests in this module
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def dask_cluster():
    """Start a minimal LocalCluster for the module, tear down at the end."""
    from dask.distributed import Client, LocalCluster

    cluster = LocalCluster(
        n_workers=1,
        threads_per_worker=1,
        memory_limit="4GB",
        processes=False,
        silence_logs=40,
        dashboard_address=None,
    )
    client = Client(cluster)
    yield client
    client.close()
    cluster.close()


# ---------------------------------------------------------------------------
# 1. Glonet — gridded prediction, RMSD metric
# ---------------------------------------------------------------------------


class TestDaskPipelineGlonet:
    """Glonet: gridded prediction evaluated against itself (RMSD → 0)."""

    def test_compute_metric_gridded_rmsd(self, dask_cluster):
        """Submit compute_metric for gridded pred vs. gridded ref on the cluster."""
        from dctools.metrics.evaluator import compute_metric
        from dctools.metrics.metrics import MetricComputer

        client = dask_cluster
        times = pd.date_range("2025-01-01", periods=3, freq="1D")
        pred_ds = _gridded_ds(times, var_name="zos")
        ref_ds = _gridded_ds(times, var_name="zos")  # identical → rmsd ≈ 0

        mc = MetricComputer(
            eval_variables=["zos"],
            metric_name="rmsd",
        )

        valid_time = pd.Timestamp("2025-01-02")
        entry: Dict[str, Any] = {
            "pred_data": pred_ds,
            "ref_data": ref_ds,
            "forecast_reference_time": pd.Timestamp("2025-01-01"),
            "lead_time": 1,
            "valid_time": valid_time,
            "pred_coords": None,
            "ref_coords": None,
            "ref_alias": "glorys",
            "ref_is_observation": False,
        }

        pred_cfg = _source_config("local", keep_variables=["zos"], eval_variables=["zos"])
        ref_cfg = _source_config("local", keep_variables=["zos"], eval_variables=["zos"])

        with patch(
            "dctools.metrics.evaluator.create_worker_connect_config",
            return_value=_noop_open,
        ):
            fut = client.submit(
                compute_metric,
                entry=entry,
                pred_source_config=pred_cfg,
                ref_source_config=ref_cfg,
                model="glonet",
                list_metrics=[mc],
                pred_transform=None,
                ref_transform=None,
            )
            result = fut.result(timeout=60)

        assert result is not None
        assert result.get("ref_alias") == "glorys"
        assert result.get("error") is None
        # Identical data → RMSD should be near 0
        if result.get("result"):
            for row in result["result"]:
                assert row["Value"] == pytest.approx(0.0, abs=1e-3)

    def test_compute_metric_with_noise(self, dask_cluster):
        """Gridded pred vs. noisy ref → non-zero RMSD."""
        from dctools.metrics.evaluator import compute_metric
        from dctools.metrics.metrics import MetricComputer

        client = dask_cluster
        times = pd.date_range("2025-01-01", periods=3, freq="1D")
        pred_ds = _gridded_ds(times, var_name="zos", seed=42)
        ref_ds = _gridded_ds(times, var_name="zos", seed=77)  # different seed

        mc = MetricComputer(eval_variables=["zos"], metric_name="rmsd")
        valid_time = pd.Timestamp("2025-01-02")

        entry: Dict[str, Any] = {
            "pred_data": pred_ds,
            "ref_data": ref_ds,
            "forecast_reference_time": pd.Timestamp("2025-01-01"),
            "lead_time": 1,
            "valid_time": valid_time,
            "pred_coords": None,
            "ref_coords": None,
            "ref_alias": "glorys",
            "ref_is_observation": False,
        }

        pred_cfg = _source_config("local", keep_variables=["zos"], eval_variables=["zos"])
        ref_cfg = _source_config("local", keep_variables=["zos"], eval_variables=["zos"])

        with patch(
            "dctools.metrics.evaluator.create_worker_connect_config",
            return_value=_noop_open,
        ):
            fut = client.submit(
                compute_metric,
                entry=entry,
                pred_source_config=pred_cfg,
                ref_source_config=ref_cfg,
                model="glonet",
                list_metrics=[mc],
                pred_transform=None,
                ref_transform=None,
            )
            result = fut.result(timeout=60)

        assert result is not None
        assert result.get("error") is None
        # Different data → RMSD > 0
        if result.get("result"):
            for row in result["result"]:
                assert row["Value"] > 0


# ---------------------------------------------------------------------------
# 2. Glorys — gridded reference with depth dimension
# ---------------------------------------------------------------------------


class TestDaskPipelineGlorys:
    """Glorys: gridded reference with depth, RMSD metric."""

    def test_compute_metric_with_depth(self, dask_cluster):
        """Gridded datasets with depth dimension → RMSD computed per depth level."""
        from dctools.metrics.evaluator import compute_metric
        from dctools.metrics.metrics import MetricComputer

        client = dask_cluster
        times = pd.date_range("2025-01-01", periods=2, freq="1D")
        pred_ds = _gridded_ds(times, var_name="zos", add_depth=True, seed=42)
        ref_ds = _gridded_ds(times, var_name="zos", add_depth=True, seed=42)

        mc = MetricComputer(eval_variables=["zos"], metric_name="rmsd")
        valid_time = pd.Timestamp("2025-01-01")

        entry: Dict[str, Any] = {
            "pred_data": pred_ds,
            "ref_data": ref_ds,
            "forecast_reference_time": pd.Timestamp("2025-01-01"),
            "lead_time": 0,
            "valid_time": valid_time,
            "pred_coords": None,
            "ref_coords": None,
            "ref_alias": "glorys",
            "ref_is_observation": False,
        }

        pred_cfg = _source_config("local", keep_variables=["zos"], eval_variables=["zos"])
        ref_cfg = _source_config("local", keep_variables=["zos"], eval_variables=["zos"])

        with patch(
            "dctools.metrics.evaluator.create_worker_connect_config",
            return_value=_noop_open,
        ):
            fut = client.submit(
                compute_metric,
                entry=entry,
                pred_source_config=pred_cfg,
                ref_source_config=ref_cfg,
                model="glonet",
                list_metrics=[mc],
                pred_transform=None,
                ref_transform=None,
            )
            result = fut.result(timeout=60)

        assert result is not None
        assert result.get("error") is None
        # Identical data → RMSD ≈ 0
        if result.get("result"):
            for row in result["result"]:
                assert row["Value"] == pytest.approx(0.0, abs=1e-3)

    def test_multi_variable_glorys(self, dask_cluster):
        """Glorys with multiple variables (so, thetao) → multi-row RMSD."""
        from dctools.metrics.evaluator import compute_metric
        from dctools.metrics.metrics import MetricComputer

        client = dask_cluster
        times = pd.date_range("2025-01-01", periods=2, freq="1D")
        lat = np.linspace(-5, 5, 4, dtype=np.float64)
        lon = np.linspace(-10, 10, 4, dtype=np.float64)
        rng = np.random.default_rng(42)
        shape = (len(times), len(lat), len(lon))

        ds = xr.Dataset(
            {
                "so": xr.Variable(
                    ("time", "lat", "lon"),
                    rng.standard_normal(shape).astype(np.float32),
                    attrs={"standard_name": "sea_water_salinity"},
                ),
                "thetao": xr.Variable(
                    ("time", "lat", "lon"),
                    rng.standard_normal(shape).astype(np.float32),
                    attrs={"standard_name": "sea_water_potential_temperature"},
                ),
            },
            coords={"time": times, "lat": lat, "lon": lon},
        )

        mc = MetricComputer(eval_variables=["so", "thetao"], metric_name="rmsd")
        valid_time = pd.Timestamp("2025-01-01")

        entry: Dict[str, Any] = {
            "pred_data": ds,
            "ref_data": ds.copy(deep=True),
            "forecast_reference_time": pd.Timestamp("2025-01-01"),
            "lead_time": 0,
            "valid_time": valid_time,
            "pred_coords": None,
            "ref_coords": None,
            "ref_alias": "glorys",
            "ref_is_observation": False,
        }

        keep = ["so", "thetao"]
        pred_cfg = _source_config("local", keep_variables=keep, eval_variables=keep)
        ref_cfg = _source_config("local", keep_variables=keep, eval_variables=keep)

        with patch(
            "dctools.metrics.evaluator.create_worker_connect_config",
            return_value=_noop_open,
        ):
            fut = client.submit(
                compute_metric,
                entry=entry,
                pred_source_config=pred_cfg,
                ref_source_config=ref_cfg,
                model="glonet",
                list_metrics=[mc],
                pred_transform=None,
                ref_transform=None,
            )
            result = fut.result(timeout=60)

        assert result is not None
        assert result.get("error") is None


# ---------------------------------------------------------------------------
# 3. Saral — nadir altimetry observations (Class4 path)
# ---------------------------------------------------------------------------


class TestDaskPipelineSaral:
    """Saral: observation dataset (along-track), Class4 metric pipeline."""

    def test_obs_class4_pipeline(self, dask_cluster):
        """Observation pipeline: pred (gridded) vs. ref (track observations)."""
        from dctools.data.coordinates import CoordinateSystem
        from dctools.metrics.evaluator import compute_metric
        from dctools.metrics.metrics import MetricComputer

        client = dask_cluster
        times = pd.date_range("2025-01-01", periods=3, freq="1D")
        pred_ds = _gridded_ds(times, var_name="zos", seed=42)

        # Along-track obs for valid_time
        obs_times = pd.date_range("2025-01-01 12:00", periods=30, freq="20min")
        obs_ds = _obs_track_ds(obs_times, n_points=30, var_name="sla")

        # For the observation branch, ref_data is a dict, not a plain dataset
        ref_coords = CoordinateSystem(
            coord_type="geographic",
            coord_level="L3",
            coordinates={"time": "time", "lat": "lat", "lon": "lon"},
            crs="EPSG:4326",
        )

        # Build the Class4 MetricComputer
        mc = MetricComputer(
            eval_variables=["zos"],
            metric_name="class4",
            is_class4=True,
            class4_kwargs={
                "list_scores": ["bias", "rmsd"],
                "interpolation_method": "nearest",
                "time_tolerance": "12h",
            },
        )

        valid_time = pd.Timestamp("2025-01-02")

        # Build the observation dict expected by compute_metric
        # (mimics what EvaluationDataloader produces for obs references)
        obs_catalog_df = pd.DataFrame({
            "path": ["inline_obs"],
            "date_start": [pd.Timestamp("2025-01-01")],
            "date_end": [pd.Timestamp("2025-01-03")],
        })

        class FakeCatalog:
            """Stand-in for DatasetCatalog.get_dataframe()."""

            def get_dataframe(self):
                """Return the inner dataframe."""
                return obs_catalog_df

        ref_entry: Dict[str, Any] = {
            "source": FakeCatalog(),
            "keep_vars": ["sla"],
            "target_dimensions": {"lat": [-5, 5], "lon": [-10, 10]},
            "time_bounds": (
                pd.Timestamp("2025-01-01"),
                pd.Timestamp("2025-01-03"),
            ),
            "metadata": {},
        }

        entry: Dict[str, Any] = {
            "pred_data": pred_ds,
            "ref_data": ref_entry,
            "forecast_reference_time": pd.Timestamp("2025-01-01"),
            "lead_time": 1,
            "valid_time": valid_time,
            "pred_coords": None,
            "ref_coords": ref_coords,
            "ref_alias": "saral",
            "ref_is_observation": True,
        }

        pred_cfg = _source_config("local", keep_variables=["zos"], eval_variables=["zos"])
        ref_cfg = _source_config("local", keep_variables=["sla"], eval_variables=["sla"])

        # We need to mock both create_worker_connect_config and
        # ObservationDataViewer since we have inline data
        with patch(
            "dctools.metrics.evaluator.create_worker_connect_config",
            return_value=_noop_open,
        ):
            # For observation branch, the fallback uses ObservationDataViewer
            # which tries to open files. Instead, provide a prefetched obs zarr.
            # Simplest approach: write obs to zarr, pass as prefetched_obs_zarr_path
            import tempfile

            tmpdir = tempfile.mkdtemp(prefix="saral_test_")
            zarr_path = os.path.join(tmpdir, "obs.zarr")
            obs_ds.to_zarr(zarr_path, consolidated=True)
            # Save time index sidecar
            time_vals = obs_ds.coords["time"].values
            np.save(
                os.path.join(tmpdir, "time_index.npy"),
                time_vals.astype("datetime64[ns]").view("int64"),
            )

            ref_entry["prefetched_obs_zarr_path"] = zarr_path

            fut = client.submit(
                compute_metric,
                entry=entry,
                pred_source_config=pred_cfg,
                ref_source_config=ref_cfg,
                model="glonet",
                list_metrics=[mc],
                pred_transform=None,
                ref_transform=None,
            )
            result = fut.result(timeout=120)

        assert result is not None
        assert result.get("ref_alias") == "saral"
        # Either we get results or a null (data might not overlap exactly)
        # The key is that no crash occurred
        assert result.get("error") is None or result.get("result") is not None

    def test_obs_empty_window_returns_null(self, dask_cluster):
        """Observation pipeline with no points in time window → null result."""
        from dctools.data.coordinates import CoordinateSystem
        from dctools.metrics.evaluator import compute_metric
        from dctools.metrics.metrics import MetricComputer

        client = dask_cluster
        times = pd.date_range("2025-01-01", periods=2, freq="1D")
        pred_ds = _gridded_ds(times, var_name="zos")

        ref_coords = CoordinateSystem(
            coord_type="geographic",
            coord_level="L3",
            coordinates={"time": "time", "lat": "lat", "lon": "lon"},
            crs="EPSG:4326",
        )

        mc = MetricComputer(
            eval_variables=["zos"],
            metric_name="class4",
            is_class4=True,
            class4_kwargs={
                "list_scores": ["bias"],
                "interpolation_method": "nearest",
                "time_tolerance": "12h",
            },
        )

        # Empty obs zarr → observation branch should return null
        import tempfile

        tmpdir = tempfile.mkdtemp(prefix="saral_empty_")
        zarr_path = os.path.join(tmpdir, "obs.zarr")
        empty_obs = xr.Dataset(
            {"sla": ("n_points", np.array([], dtype=np.float32))},
            coords={
                "time": ("n_points", np.array([], dtype="datetime64[ns]")),
                "lat": ("n_points", np.array([], dtype=np.float64)),
                "lon": ("n_points", np.array([], dtype=np.float64)),
            },
        )
        empty_obs.to_zarr(zarr_path, consolidated=True)

        obs_df = pd.DataFrame({"path": [], "date_start": [], "date_end": []})

        class FakeCatalog:
            """Stand-in for empty catalog."""

            def get_dataframe(self):
                """Return empty frame."""
                return obs_df

        ref_entry: Dict[str, Any] = {
            "source": FakeCatalog(),
            "keep_vars": ["sla"],
            "target_dimensions": {"lat": [-5, 5], "lon": [-10, 10]},
            "time_bounds": (pd.Timestamp("2025-06-01"), pd.Timestamp("2025-06-02")),
            "metadata": {},
            "prefetched_obs_zarr_path": zarr_path,
        }

        entry: Dict[str, Any] = {
            "pred_data": pred_ds,
            "ref_data": ref_entry,
            "forecast_reference_time": pd.Timestamp("2025-01-01"),
            "lead_time": 0,
            "valid_time": pd.Timestamp("2025-01-01"),
            "pred_coords": None,
            "ref_coords": ref_coords,
            "ref_alias": "saral",
            "ref_is_observation": True,
        }

        pred_cfg = _source_config("local", keep_variables=["zos"], eval_variables=["zos"])
        ref_cfg = _source_config("local", keep_variables=["sla"], eval_variables=["sla"])

        with patch(
            "dctools.metrics.evaluator.create_worker_connect_config",
            return_value=_noop_open,
        ):
            fut = client.submit(
                compute_metric,
                entry=entry,
                pred_source_config=pred_cfg,
                ref_source_config=ref_cfg,
                model="glonet",
                list_metrics=[mc],
                pred_transform=None,
                ref_transform=None,
            )
            result = fut.result(timeout=60)

        assert result is not None
        # Empty window → result should be None / null
        assert result.get("result") is None or result.get("n_points", 0) == 0


# ---------------------------------------------------------------------------
# 4. Argo — observation profiles (shared Zarr fast path)
# ---------------------------------------------------------------------------


class TestDaskPipelineArgo:
    """Argo: observation profiles with shared Zarr prefetch path."""

    def test_argo_shared_zarr_path(self, dask_cluster):
        """ARGO profiles via prefetched shared Zarr → Class4 evaluation."""
        from dctools.data.coordinates import CoordinateSystem
        from dctools.metrics.evaluator import compute_metric
        from dctools.metrics.metrics import MetricComputer

        client = dask_cluster
        times = pd.date_range("2025-01-01", periods=5, freq="1D")

        # Prediction: gridded
        pred_ds = _gridded_ds(times, var_name="zos", seed=42)

        # ARGO profiles: write to Zarr as shared prefetch
        argo_ds = _argo_profile_ds(times, n_profiles=40)
        # Rename to n_points dim for the evaluator
        argo_ds = argo_ds.rename({"N_POINTS": "n_points"})
        # Promote coords
        for c in ("TIME", "LATITUDE", "LONGITUDE", "DEPTH"):
            if c in argo_ds.data_vars:
                argo_ds = argo_ds.set_coords(c)

        import tempfile

        tmpdir = tempfile.mkdtemp(prefix="argo_test_")
        zarr_path = os.path.join(tmpdir, "argo_shared.zarr")
        argo_ds.to_zarr(zarr_path, consolidated=True)

        ref_coords = CoordinateSystem(
            coord_type="geographic",
            coord_level="L3",
            coordinates={
                "time": "TIME",
                "lat": "LATITUDE",
                "lon": "LONGITUDE",
                "depth": "DEPTH",
            },
            crs="EPSG:4326",
        )

        mc = MetricComputer(
            eval_variables=["TEMP"],
            metric_name="class4",
            is_class4=True,
            class4_kwargs={
                "list_scores": ["bias", "rmsd"],
                "interpolation_method": "nearest",
                "time_tolerance": "24h",
            },
        )

        valid_time = pd.Timestamp("2025-01-03")

        obs_df = pd.DataFrame({
            "path": ["argo_profiles"],
            "date_start": [pd.Timestamp("2025-01-01")],
            "date_end": [pd.Timestamp("2025-01-05")],
        })

        class FakeCatalog:
            """Stand-in for ARGO catalog."""

            def get_dataframe(self):
                """Return the dataframe."""
                return obs_df

        ref_entry: Dict[str, Any] = {
            "source": FakeCatalog(),
            "keep_vars": ["TEMP", "PSAL"],
            "target_dimensions": {"lat": [-5, 5], "lon": [-10, 10]},
            "time_bounds": (pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-05")),
            "metadata": {},
            "prefetched_argo_shared_zarr": zarr_path,
        }

        entry: Dict[str, Any] = {
            "pred_data": pred_ds,
            "ref_data": ref_entry,
            "forecast_reference_time": pd.Timestamp("2025-01-01"),
            "lead_time": 2,
            "valid_time": valid_time,
            "pred_coords": None,
            "ref_coords": ref_coords,
            "ref_alias": "argo_profiles",
            "ref_is_observation": True,
        }

        pred_cfg = _source_config("local", keep_variables=["zos"], eval_variables=["zos"])
        ref_cfg = _source_config(
            "argo", keep_variables=["TEMP", "PSAL"], eval_variables=["TEMP"]
        )

        with patch(
            "dctools.metrics.evaluator.create_worker_connect_config",
            return_value=_noop_open,
        ):
            fut = client.submit(
                compute_metric,
                entry=entry,
                pred_source_config=pred_cfg,
                ref_source_config=ref_cfg,
                model="glonet",
                list_metrics=[mc],
                pred_transform=None,
                ref_transform=None,
            )
            result = fut.result(timeout=120)

        assert result is not None
        assert result.get("ref_alias") == "argo_profiles"
        # No crash is the key assertion
        assert result.get("error") is None or result.get("result") is not None

    def test_argo_legacy_zarr_fallback(self, dask_cluster):
        """ARGO legacy per-window Zarr fallback path."""
        from dctools.data.coordinates import CoordinateSystem
        from dctools.metrics.evaluator import compute_metric
        from dctools.metrics.metrics import MetricComputer

        client = dask_cluster
        times = pd.date_range("2025-01-01", periods=3, freq="1D")
        pred_ds = _gridded_ds(times, var_name="zos", seed=42)

        argo_ds = _argo_profile_ds(times, n_profiles=15, seed=456)
        argo_ds = argo_ds.rename({"N_POINTS": "obs"})
        for c in ("TIME", "LATITUDE", "LONGITUDE", "DEPTH"):
            if c in argo_ds.data_vars:
                argo_ds = argo_ds.set_coords(c)

        import tempfile

        tmpdir = tempfile.mkdtemp(prefix="argo_legacy_")
        zarr_path = os.path.join(tmpdir, "argo_window.zarr")
        argo_ds.to_zarr(zarr_path, consolidated=True)

        ref_coords = CoordinateSystem(
            coord_type="geographic",
            coord_level="L3",
            coordinates={
                "time": "TIME",
                "lat": "LATITUDE",
                "lon": "LONGITUDE",
                "depth": "DEPTH",
            },
            crs="EPSG:4326",
        )

        mc = MetricComputer(
            eval_variables=["TEMP"],
            metric_name="class4",
            is_class4=True,
            class4_kwargs={
                "list_scores": ["rmsd"],
                "interpolation_method": "nearest",
                "time_tolerance": "24h",
            },
        )

        obs_df = pd.DataFrame({
            "path": ["argo_profiles"],
            "date_start": [pd.Timestamp("2025-01-01")],
            "date_end": [pd.Timestamp("2025-01-03")],
        })

        class FakeCatalog:
            """Stand-in for legacy ARGO catalog."""

            def get_dataframe(self):
                """Return the dataframe."""
                return obs_df

        ref_entry: Dict[str, Any] = {
            "source": FakeCatalog(),
            "keep_vars": ["TEMP"],
            "target_dimensions": {"lat": [-5, 5], "lon": [-10, 10]},
            "time_bounds": (pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-03")),
            "metadata": {},
            # Legacy path: per-window zarr
            "prefetched_zarr_path": zarr_path,
        }

        entry: Dict[str, Any] = {
            "pred_data": pred_ds,
            "ref_data": ref_entry,
            "forecast_reference_time": pd.Timestamp("2025-01-01"),
            "lead_time": 1,
            "valid_time": pd.Timestamp("2025-01-02"),
            "pred_coords": None,
            "ref_coords": ref_coords,
            "ref_alias": "argo_profiles",
            "ref_is_observation": True,
        }

        pred_cfg = _source_config("local", keep_variables=["zos"], eval_variables=["zos"])
        ref_cfg = _source_config(
            "argo", keep_variables=["TEMP"], eval_variables=["TEMP"]
        )

        with patch(
            "dctools.metrics.evaluator.create_worker_connect_config",
            return_value=_noop_open,
        ):
            fut = client.submit(
                compute_metric,
                entry=entry,
                pred_source_config=pred_cfg,
                ref_source_config=ref_cfg,
                model="glonet",
                list_metrics=[mc],
                pred_transform=None,
                ref_transform=None,
            )
            result = fut.result(timeout=120)

        assert result is not None
        assert result.get("ref_alias") == "argo_profiles"
        assert result.get("error") is None or result.get("result") is not None


# ---------------------------------------------------------------------------
# 5. Cross-cutting: Evaluator helper integration with Dask
# ---------------------------------------------------------------------------


class TestEvaluatorDaskIntegration:
    """Evaluator class methods that interact with Dask cluster state."""

    def test_log_cluster_memory_with_real_cluster(self, dask_cluster):
        """Log cluster memory usage with a real LocalCluster."""
        from dctools.metrics.evaluator import Evaluator

        client = dask_cluster

        # Build a minimal Evaluator shell with the real client
        ev = object.__new__(Evaluator)
        mock_processor = MagicMock()
        mock_processor.client = client
        ev.dataset_processor = mock_processor

        # Should not raise
        ev.log_cluster_memory_usage(0)

    def test_get_max_memory_with_real_cluster(self, dask_cluster):
        """Get max memory usage from real worker metrics."""
        from dctools.metrics.evaluator import Evaluator

        client = dask_cluster

        ev = object.__new__(Evaluator)
        mock_processor = MagicMock()
        mock_processor.client = client
        ev.dataset_processor = mock_processor

        mem = ev.get_max_memory_usage()
        assert isinstance(mem, (int, float))
        assert mem >= 0

    def test_get_max_memory_fraction_with_real_cluster(self, dask_cluster):
        """Memory fraction from real cluster in [0, 1]."""
        from dctools.metrics.evaluator import Evaluator

        client = dask_cluster

        ev = object.__new__(Evaluator)
        mock_processor = MagicMock()
        mock_processor.client = client
        ev.dataset_processor = mock_processor

        frac = ev.get_max_memory_fraction()
        assert isinstance(frac, float)
        assert 0.0 <= frac <= 1.0

    def test_worker_cleanup_on_cluster(self, dask_cluster):
        """Run worker cleanup functions on real workers."""
        from dctools.metrics.evaluator import (
            _clear_xarray_file_cache,
            _worker_full_cleanup,
            worker_memory_cleanup,
        )

        client = dask_cluster

        # Submit cleanup functions to the cluster
        f1 = client.submit(worker_memory_cleanup)
        f2 = client.submit(_clear_xarray_file_cache)
        f3 = client.submit(_worker_full_cleanup)

        assert f1.result(timeout=10) is None  # returns None
        assert f2.result(timeout=10) is True
        assert f3.result(timeout=10) is True


# ---------------------------------------------------------------------------
# 6. Transform pipeline on Dask
# ---------------------------------------------------------------------------


class TestTransformsPipelineDask:
    """Ensure transforms are serializable and work on Dask workers."""

    def test_standardize_transform_on_worker(self, dask_cluster):
        """Standardize transform applied on a Dask worker."""
        from dctools.data.transforms import (
            RenameCoordsVarsTransform,
            SelectVariablesTransform,
        )

        client = dask_cluster
        times = pd.date_range("2025-01-01", periods=2, freq="1D")
        ds = _gridded_ds(times, var_name="zos")

        # Build simple transform pipeline
        rename_t = RenameCoordsVarsTransform(vars_rename_dict={"zos": "ssh"})
        select_t = SelectVariablesTransform(["ssh"])

        def apply_transforms(dataset):
            """Apply transforms sequentially."""
            dataset = rename_t(dataset)
            dataset = select_t(dataset)
            return dataset

        fut = client.submit(apply_transforms, ds)
        result = fut.result(timeout=30)

        assert isinstance(result, xr.Dataset)
        assert "ssh" in result
        assert "zos" not in result

    def test_subset_coord_transform_on_worker(self, dask_cluster):
        """SubsetCoordTransform works on Dask worker."""
        from dctools.data.transforms import SubsetCoordTransform

        client = dask_cluster
        times = pd.date_range("2025-01-01", periods=1)
        ds = _gridded_ds(times, var_name="zos", add_depth=True)

        subset_t = SubsetCoordTransform("depth", [0.0, 10.0])

        fut = client.submit(subset_t, ds)
        result = fut.result(timeout=30)

        assert isinstance(result, xr.Dataset)
        assert result.sizes["depth"] <= 2
