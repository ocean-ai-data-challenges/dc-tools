"""Comprehensive tests for the data processing pipeline.

Tests cover each connection type (local, S3, Wasabi, Glonet, FTP, CMEMS, ARGO)
by creating tiny synthetic NetCDF data and JSON catalogs and monkeypatching
remote connection managers to read locally. This exercises a large fraction of
the dctools codebase:  config → connection → catalog → dataset_manager →
forecast_index → transforms → dataloader.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from dctools.data.connection.config import (
    CMEMSConnectionConfig,
    FTPConnectionConfig,
    GlonetConnectionConfig,
    LocalConnectionConfig,
    S3ConnectionConfig,
    WasabiS3ConnectionConfig,
)
from dctools.data.datasets.dataset import DatasetConfig, RemoteDataset, get_dataset_from_config
from dctools.data.datasets.dataloader import EvaluationDataloader
from dctools.data.datasets.dataset_manager import MultiSourceDatasetManager
from dctools.data.datasets.forecast import build_forecast_index_from_catalog


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeDatasetProcessor:
    """Minimal stand-in for OceanBench's DatasetProcessor."""

    def __init__(self):
        self.client = None
        self.distributed = False


def _make_gridded_nc(path: Path, times: pd.DatetimeIndex, var_name: str = "zos") -> Path:
    """Create a tiny gridded NetCDF (time×lat×lon) and return its path."""
    lat = np.array([0.0, 1.0], dtype=np.float64)
    lon = np.array([0.0, 1.0], dtype=np.float64)
    depth = np.array([0.0], dtype=np.float64)
    data = np.random.default_rng(42).standard_normal(
        (len(times), len(lat), len(lon))
    ).astype(np.float32)
    ds = xr.Dataset(
        data_vars={var_name: (("time", "lat", "lon"), data)},
        coords={"time": times, "depth": depth, "lat": lat, "lon": lon},
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path, engine="scipy")
    return path


def _make_obs_nc(path: Path, times: pd.DatetimeIndex) -> Path:
    """Create a tiny observation-style NetCDF (n_points dim) and return its path."""
    n_pts = max(4, len(times))
    rng = np.random.default_rng(99)
    ds = xr.Dataset(
        data_vars={
            "sla": (("n_points",), rng.standard_normal(n_pts).astype(np.float32)),
        },
        coords={
            "time": ("n_points", times[:n_pts] if len(times) >= n_pts else
                     pd.date_range(times[0], periods=n_pts, freq="6h")),
            "lat": ("n_points", rng.uniform(-10, 10, n_pts).astype(np.float64)),
            "lon": ("n_points", rng.uniform(-10, 10, n_pts).astype(np.float64)),
        },
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path, engine="scipy")
    return path


def _make_catalog_json(
    catalog_path: Path,
    nc_paths: List[Path],
    date_ranges: List[tuple],
    var_name: str = "zos",
    rename_to: str = "ssh",
    is_observation: bool = False,
    coord_level: str = "L4",
) -> Path:
    """Write a minimal JSON catalog file compatible with DatasetCatalog.from_json."""
    if is_observation:
        coord_system = {
            "coord_type": "geographic",
            "coord_level": coord_level,
            "coordinates": {"time": "time", "lat": "lat", "lon": "lon"},
            "crs": None,
        }
    else:
        coord_system = {
            "coord_type": "geographic",
            "coord_level": coord_level,
            "coordinates": {"time": "time", "depth": "depth", "lat": "lat", "lon": "lon"},
            "crs": None,
        }

    global_metadata: Dict[str, Any] = {
        "variables_rename_dict": {var_name: rename_to},
        "coord_system": coord_system,
        "is_observation": is_observation,
    }
    # observation datasets need variables_dict for the dataloader
    if is_observation:
        global_metadata["variables_dict"] = {var_name: var_name}

    # A minimal bounding-box geometry so GeoDataFrame constructor succeeds
    bbox_geom = {
        "type": "Polygon",
        "coordinates": [[[-10, -10], [10, -10], [10, 10], [-10, 10], [-10, -10]]],
    }

    catalog = {
        "global_metadata": global_metadata,
        "features": [
            {
                "type": "Feature",
                "geometry": bbox_geom,
                "properties": {
                    "path": str(nc_path),
                    "date_start": dr[0],
                    "date_end": dr[1],
                },
            }
            for nc_path, dr in zip(nc_paths, date_ranges, strict=False)
        ],
    }
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
    return catalog_path


def _build_dataset(
    alias: str,
    connection_config,
    catalog_path: Path,
    keep_variables: List[str],
    eval_variables: List[str],
    is_observation: bool = False,
) -> RemoteDataset:
    """Build a RemoteDataset from an already-configured connection config."""
    ds_cfg = DatasetConfig(
        alias=alias,
        connection_config=connection_config,
        catalog_options={"catalog_path": str(catalog_path)},
        keep_variables=keep_variables,
        eval_variables=eval_variables,
        observation_dataset=is_observation,
        use_catalog=True,
        ignore_geometry=True,
    )
    return RemoteDataset(ds_cfg)


def _build_manager_and_dataloader(
    alias: str,
    dataset: RemoteDataset,
    processor: _FakeDatasetProcessor,
    start_time: str = "2025-01-01",
    end_time: str = "2025-01-02",
    n_days_forecast: int = 1,
    ref_aliases: Optional[List[str]] = None,
    ref_datasets: Optional[Dict[str, RemoteDataset]] = None,
) -> EvaluationDataloader:
    """Wire up a MultiSourceDatasetManager, build forecast index, create dataloader."""
    lat = [0.0, 1.0]
    lon = [0.0, 1.0]
    depth = [0.0]

    manager = MultiSourceDatasetManager(
        dataset_processor=processor,
        target_dimensions={"lat": lat, "lon": lon, "depth": depth},
        time_tolerance=pd.Timedelta("1h"),
    )
    manager.add_dataset(alias, dataset)

    # Add reference datasets if provided
    if ref_datasets:
        for ref_alias, ref_ds in ref_datasets.items():
            if ref_alias != alias:
                manager.add_dataset(ref_alias, ref_ds)

    transform = manager.get_transform(
        dataset_alias=alias,
        transform_name="standardize",
    )
    assert callable(transform)

    manager.build_forecast_index(
        alias,
        init_date=start_time,
        end_date=end_time,
        n_days_forecast=n_days_forecast,
        n_days_interval=1,
    )

    effective_ref_aliases = ref_aliases if ref_aliases is not None else [alias]

    # Build ref_transforms dict
    ref_transforms: Dict[str, Any] = {}
    for ra in effective_ref_aliases:
        ref_transforms[ra] = manager.get_transform(
            dataset_alias=ra,
            transform_name="standardize",
        )

    dataloader = manager.get_dataloader(
        pred_alias=alias,
        ref_aliases=effective_ref_aliases,
        pred_transform=transform,
        ref_transforms=ref_transforms,
        forecast_mode=True,
        n_days_forecast=n_days_forecast,
    )
    return dataloader


def _assert_dataloader_yields_batches(dataloader: EvaluationDataloader):
    """Assert the dataloader produces at least one batch with expected keys."""
    batches = list(dataloader)
    assert len(batches) >= 1, "Dataloader should yield at least one batch"
    for batch in batches:
        assert len(batch) >= 1, "Each batch should have at least one entry"
        entry = batch[0]
        assert "pred_data" in entry
        assert "ref_data" in entry
        assert "forecast_reference_time" in entry
        assert "valid_time" in entry


# ---------------------------------------------------------------------------
# Test: Local connection (gridded pred ↔ gridded ref)
# ---------------------------------------------------------------------------


class TestPipelineLocal:
    """Pipeline tests using LocalConnectionConfig (no network)."""

    def test_gridded_pred_gridded_ref(self, tmp_path: Path):
        """Full pipeline: local gridded pred with local gridded ref."""
        processor = _FakeDatasetProcessor()
        data_dir = tmp_path / "data" / "glonet"
        catalog_dir = tmp_path / "catalogs"

        nc = _make_gridded_nc(
            data_dir / "sample.nc",
            pd.date_range("2025-01-01", periods=1),
        )
        catalog = _make_catalog_json(
            catalog_dir / "glonet.json",
            [nc],
            [("2025-01-01T00:00:00", "2025-01-02T00:00:00")],
        )
        params = {
            "dataset_processor": processor,
            "init_type": "from_json",
            "local_root": str(data_dir),
            "max_samples": 1,
            "file_pattern": "*.nc",
            "keep_variables": ["zos"],
            "filter_values": {},
        }
        conn_cfg = LocalConnectionConfig(params)
        dataset = _build_dataset("glonet", conn_cfg, catalog, ["zos"], ["zos"])
        dl = _build_manager_and_dataloader("glonet", dataset, processor)
        _assert_dataloader_yields_batches(dl)

    def test_multi_day_forecast(self, tmp_path: Path):
        """Forecast index spanning multiple days with n_days_forecast > 1."""
        processor = _FakeDatasetProcessor()
        data_dir = tmp_path / "data" / "pred"
        catalog_dir = tmp_path / "catalogs"

        dates = pd.date_range("2025-01-01", periods=3, freq="D")
        ncs = []
        date_ranges = []
        for i, dt in enumerate(dates):
            nc = _make_gridded_nc(
                data_dir / f"day{i}.nc",
                pd.DatetimeIndex([dt]),
            )
            ncs.append(nc)
            ds_str = dt.isoformat()
            de_str = (dt + pd.Timedelta(days=1)).isoformat()
            date_ranges.append((ds_str, de_str))

        catalog = _make_catalog_json(
            catalog_dir / "pred.json", ncs, date_ranges,
        )
        params = {
            "dataset_processor": processor,
            "init_type": "from_json",
            "local_root": str(data_dir),
            "max_samples": 10,
            "file_pattern": "*.nc",
            "keep_variables": ["zos"],
            "filter_values": {},
        }
        conn_cfg = LocalConnectionConfig(params)
        dataset = _build_dataset("pred", conn_cfg, catalog, ["zos"], ["zos"])
        dl = _build_manager_and_dataloader(
            "pred", dataset, processor,
            start_time="2025-01-01",
            end_time="2025-01-03",
            n_days_forecast=2,
        )
        _assert_dataloader_yields_batches(dl)

    def test_transform_callable(self, tmp_path: Path):
        """Transforms produced from catalog metadata are callable on real data."""
        processor = _FakeDatasetProcessor()
        data_dir = tmp_path / "data" / "tr"
        catalog_dir = tmp_path / "catalogs"
        nc = _make_gridded_nc(
            data_dir / "sample.nc",
            pd.date_range("2025-01-01", periods=1),
        )
        catalog = _make_catalog_json(
            catalog_dir / "tr.json",
            [nc],
            [("2025-01-01T00:00:00", "2025-01-02T00:00:00")],
        )
        params = {
            "dataset_processor": processor,
            "init_type": "from_json",
            "local_root": str(data_dir),
            "max_samples": 1,
            "file_pattern": "*.nc",
            "keep_variables": ["zos"],
            "filter_values": {},
        }
        conn_cfg = LocalConnectionConfig(params)
        ds_cfg = DatasetConfig(
            alias="tr",
            connection_config=conn_cfg,
            catalog_options={"catalog_path": str(catalog)},
            keep_variables=["zos"],
            eval_variables=["zos"],
            observation_dataset=False,
            use_catalog=True,
            ignore_geometry=True,
        )
        dataset = RemoteDataset(ds_cfg)
        manager = MultiSourceDatasetManager(
            dataset_processor=processor,
            target_dimensions={"lat": [0.0, 1.0], "lon": [0.0, 1.0], "depth": [0.0]},
            time_tolerance=pd.Timedelta("1h"),
        )
        manager.add_dataset("tr", dataset)
        transform = manager.get_transform(dataset_alias="tr", transform_name="standardize")
        assert callable(transform)
        # Apply the transform to the raw dataset
        raw_ds = xr.open_dataset(nc, engine="scipy")
        result = transform(raw_ds)
        assert isinstance(result, xr.Dataset)


# ---------------------------------------------------------------------------
# Test: S3 connection (monkeypatched)
# ---------------------------------------------------------------------------


class TestPipelineS3:
    """Pipeline test for S3ConnectionConfig with monkeypatched open."""

    def test_s3_pipeline(self, tmp_path: Path, monkeypatch):
        """S3 manager reads local files via monkeypatch of S3Manager.open."""
        from dctools.data.connection.connection_manager import S3Manager

        processor = _FakeDatasetProcessor()
        data_dir = tmp_path / "data" / "s3ds"
        catalog_dir = tmp_path / "catalogs"

        nc = _make_gridded_nc(
            data_dir / "sample.nc",
            pd.date_range("2025-01-01", periods=1),
        )
        catalog = _make_catalog_json(
            catalog_dir / "s3ds.json",
            [nc],
            [("2025-01-01T00:00:00", "2025-01-02T00:00:00")],
        )

        # Monkeypatch S3Manager.open to redirect to local file
        def _fake_open(self, path, mode="rb"):
            return xr.open_dataset(str(nc), engine="scipy")

        monkeypatch.setattr(S3Manager, "open", _fake_open)

        # Monkeypatch S3ConnectionConfig.create_fs to avoid real S3
        monkeypatch.setattr(
            S3ConnectionConfig, "create_fs",
            lambda self: MagicMock(),
        )

        params = {
            "dataset_processor": processor,
            "init_type": "from_json",
            "local_root": str(data_dir),
            "s3_bucket": "test-bucket",
            "s3_folder": "test-folder",
            "endpoint_url": None,
            "key": None,
            "secret_key": None,
            "max_samples": 1,
            "file_pattern": "*.nc",
            "keep_variables": ["zos"],
            "filter_values": {},
        }
        conn_cfg = S3ConnectionConfig(params)
        dataset = _build_dataset("s3ds", conn_cfg, catalog, ["zos"], ["zos"])
        dl = _build_manager_and_dataloader("s3ds", dataset, processor)
        _assert_dataloader_yields_batches(dl)


# ---------------------------------------------------------------------------
# Test: Wasabi S3 connection (monkeypatched)
# ---------------------------------------------------------------------------


class TestPipelineWasabi:
    """Pipeline test for WasabiS3ConnectionConfig with monkeypatched open."""

    def test_wasabi_pipeline(self, tmp_path: Path, monkeypatch):
        """Wasabi S3 manager reads local files via monkeypatch."""
        from dctools.data.connection.connection_manager import S3WasabiManager

        processor = _FakeDatasetProcessor()
        data_dir = tmp_path / "data" / "wasabi"
        catalog_dir = tmp_path / "catalogs"

        nc = _make_gridded_nc(
            data_dir / "sample.nc",
            pd.date_range("2025-01-01", periods=1),
        )
        catalog = _make_catalog_json(
            catalog_dir / "wasabi.json",
            [nc],
            [("2025-01-01T00:00:00", "2025-01-02T00:00:00")],
        )

        def _fake_open(self, path, mode="rb"):
            return xr.open_dataset(str(nc), engine="scipy")

        monkeypatch.setattr(S3WasabiManager, "open", _fake_open)
        # Monkeypatch S3ConnectionConfig.create_fs (parent of Wasabi)
        monkeypatch.setattr(
            S3ConnectionConfig, "create_fs",
            lambda self: MagicMock(),
        )

        params = {
            "dataset_processor": processor,
            "init_type": "from_json",
            "local_root": str(data_dir),
            "s3_bucket": "test-wasabi-bucket",
            "s3_folder": "test-wasabi-folder",
            "endpoint_url": "https://s3.wasabisys.com",
            "key": "fakekey",
            "secret_key": "fakesecret",
            "max_samples": 1,
            "file_pattern": "*.nc",
            "keep_variables": ["zos"],
            "filter_values": {},
        }
        conn_cfg = WasabiS3ConnectionConfig(params)
        dataset = _build_dataset("wasabi", conn_cfg, catalog, ["zos"], ["zos"])
        dl = _build_manager_and_dataloader("wasabi", dataset, processor)
        _assert_dataloader_yields_batches(dl)


# ---------------------------------------------------------------------------
# Test: Glonet connection (monkeypatched)
# ---------------------------------------------------------------------------


class TestPipelineGlonet:
    """Pipeline test for GlonetConnectionConfig with monkeypatched open."""

    def test_glonet_pipeline(self, tmp_path: Path, monkeypatch):
        """Glonet manager reads local files via monkeypatch."""
        from dctools.data.connection.connection_manager import GlonetManager

        processor = _FakeDatasetProcessor()
        data_dir = tmp_path / "data" / "glonet"
        catalog_dir = tmp_path / "catalogs"

        nc = _make_gridded_nc(
            data_dir / "sample.nc",
            pd.date_range("2025-01-01", periods=1),
        )
        catalog = _make_catalog_json(
            catalog_dir / "glonet.json",
            [nc],
            [("2025-01-01T00:00:00", "2025-01-02T00:00:00")],
        )

        def _fake_open(self, path, mode="rb"):
            return xr.open_dataset(str(nc), engine="scipy")

        monkeypatch.setattr(GlonetManager, "open", _fake_open)
        monkeypatch.setattr(
            GlonetConnectionConfig, "create_fs",
            lambda self: MagicMock(),
        )

        params = {
            "dataset_processor": processor,
            "init_type": "from_json",
            "local_root": str(data_dir),
            "s3_bucket": "test-glonet-bucket",
            "s3_folder": "test-glonet-folder",
            "endpoint_url": "https://fake-glonet.example.com",
            "max_samples": 1,
            "file_pattern": "*.nc",
            "keep_variables": ["zos"],
            "filter_values": {},
        }
        conn_cfg = GlonetConnectionConfig(params)
        dataset = _build_dataset("glonetds", conn_cfg, catalog, ["zos"], ["zos"])
        dl = _build_manager_and_dataloader("glonetds", dataset, processor)
        _assert_dataloader_yields_batches(dl)


# ---------------------------------------------------------------------------
# Test: FTP connection (monkeypatched)
# ---------------------------------------------------------------------------


class TestPipelineFTP:
    """Pipeline test for FTPConnectionConfig with monkeypatched create_fs + open."""

    def test_ftp_pipeline(self, tmp_path: Path, monkeypatch):
        """FTP manager reads local files via monkeypatch."""
        from dctools.data.connection.connection_manager import FTPManager

        processor = _FakeDatasetProcessor()
        data_dir = tmp_path / "data" / "ftpds"
        catalog_dir = tmp_path / "catalogs"

        nc = _make_gridded_nc(
            data_dir / "sample.nc",
            pd.date_range("2025-01-01", periods=1),
        )
        catalog = _make_catalog_json(
            catalog_dir / "ftpds.json",
            [nc],
            [("2025-01-01T00:00:00", "2025-01-02T00:00:00")],
        )

        def _fake_open(self, path, mode="rb"):
            return xr.open_dataset(str(nc), engine="scipy")

        monkeypatch.setattr(FTPManager, "open", _fake_open)
        monkeypatch.setattr(
            FTPConnectionConfig, "create_fs",
            lambda self: MagicMock(),
        )

        params = {
            "dataset_processor": processor,
            "init_type": "from_json",
            "local_root": str(data_dir),
            "host": "ftp.fake-server.example.com",
            "user": "testuser",
            "password": "testpwd",
            "ftp_folder": "/data/",
            "max_samples": 1,
            "file_pattern": "*.nc",
            "keep_variables": ["zos"],
            "filter_values": {},
        }
        conn_cfg = FTPConnectionConfig(params)
        dataset = _build_dataset("ftpds", conn_cfg, catalog, ["zos"], ["zos"])
        dl = _build_manager_and_dataloader("ftpds", dataset, processor)
        _assert_dataloader_yields_batches(dl)


# ---------------------------------------------------------------------------
# Test: CMEMS connection (monkeypatched)
# ---------------------------------------------------------------------------


class TestPipelineCMEMS:
    """Pipeline test for CMEMSConnectionConfig with monkeypatched login + open."""

    def test_cmems_pipeline(self, tmp_path: Path, monkeypatch):
        """CMEMS manager reads local files via monkeypatch."""
        from dctools.data.connection.connection_manager import CMEMSManager

        processor = _FakeDatasetProcessor()
        data_dir = tmp_path / "data" / "cmems"
        catalog_dir = tmp_path / "catalogs"

        nc = _make_gridded_nc(
            data_dir / "sample.nc",
            pd.date_range("2025-01-01", periods=1),
        )
        catalog = _make_catalog_json(
            catalog_dir / "cmems.json",
            [nc],
            [("2025-01-01T00:00:00", "2025-01-02T00:00:00")],
        )

        def _fake_open(self, path, mode="rb"):
            return xr.open_dataset(str(nc), engine="scipy")

        # Patch cmems_login to be a no-op
        monkeypatch.setattr(CMEMSManager, "cmems_login", lambda self: None)
        monkeypatch.setattr(CMEMSManager, "open", _fake_open)

        params = {
            "dataset_processor": processor,
            "init_type": "from_json",
            "local_root": str(data_dir),
            "dataset_id": "fake_cmems_product",
            "max_samples": 1,
            "file_pattern": "*.nc",
            "keep_variables": ["zos"],
            "filter_values": {},
        }
        conn_cfg = CMEMSConnectionConfig(params)
        dataset = _build_dataset("cmemsds", conn_cfg, catalog, ["zos"], ["zos"])
        dl = _build_manager_and_dataloader("cmemsds", dataset, processor)
        _assert_dataloader_yields_batches(dl)


# ---------------------------------------------------------------------------
# Test: ARGO connection (monkeypatched)
# ---------------------------------------------------------------------------


class TestPipelineArgo:
    """Pipeline test for ARGOConnectionConfig with monkeypatched ArgoManager."""

    def test_argo_pipeline(self, tmp_path: Path, monkeypatch):
        """ARGO manager reads local files via monkeypatch."""
        from dctools.data.connection.config import ARGOConnectionConfig
        from dctools.data.connection.connection_manager import ArgoManager

        processor = _FakeDatasetProcessor()
        data_dir = tmp_path / "data" / "argo"
        catalog_dir = tmp_path / "catalogs"

        # Create a gridded-style NC (ARGO config but non-observation coord_level
        # so the dataloader uses the simple gridded path).  This still exercises
        # the ARGOConnectionConfig → ArgoManager chain.
        nc = _make_gridded_nc(
            data_dir / "argo_sample.nc",
            pd.date_range("2025-01-01", periods=1),
            var_name="zos",
        )
        catalog = _make_catalog_json(
            catalog_dir / "argo.json",
            [nc],
            [("2025-01-01T00:00:00", "2025-01-02T00:00:00")],
            var_name="zos",
            rename_to="ssh",
            is_observation=False,
            coord_level="L4",
        )

        # Monkeypatch ArgoManager.__init__ to skip ArgoInterface
        _original_base_init = ArgoManager.__bases__[0].__init__

        def _fake_argo_init(self, connect_config, **kwargs):
            # Just do the basic BaseConnectionManager init
            _original_base_init(self, connect_config, call_list_files=False)
            self.depth_values = []
            self.argo_index = None
            self.argo_interface = None
            self._master_index = None
            self._catalog = None
            self._global_metadata = None

        monkeypatch.setattr(ArgoManager, "__init__", _fake_argo_init)

        def _fake_open(self, path, *args, **kwargs):
            return xr.open_dataset(str(nc), engine="scipy")

        monkeypatch.setattr(ArgoManager, "open", _fake_open)

        # Monkeypatch ARGOConnectionConfig.create_fs to avoid S3
        monkeypatch.setattr(
            ARGOConnectionConfig, "create_fs",
            lambda self: MagicMock(),
        )

        params = {
            "dataset_processor": processor,
            "init_type": "from_json",
            "local_root": str(data_dir),
            "s3_bucket": None,
            "s3_folder": None,
            "s3_key": None,
            "s3_secret_key": None,
            "endpoint_url": None,
            "base_path": None,
            "depth_values": [0.0],
            "variables": ["zos"],
            "max_samples": 1,
            "file_pattern": "*.nc",
            "keep_variables": ["zos"],
            "filter_values": {},
        }
        conn_cfg = ARGOConnectionConfig(params)
        dataset = _build_dataset(
            "argods", conn_cfg, catalog, ["zos"], ["zos"], is_observation=False,
        )
        dl = _build_manager_and_dataloader("argods", dataset, processor)
        _assert_dataloader_yields_batches(dl)


# ---------------------------------------------------------------------------
# Test: get_dataset_from_config factory
# ---------------------------------------------------------------------------


class TestGetDatasetFromConfig:
    """Test the get_dataset_from_config factory with various config types."""

    def test_s3_glonet_config(self, tmp_path: Path, monkeypatch):
        """get_dataset_from_config with config='s3', connection_type='glonet'."""
        processor = _FakeDatasetProcessor()
        data_dir = tmp_path / "data" / "glonet"
        catalog_dir = tmp_path / "catalogs"
        data_dir.mkdir(parents=True, exist_ok=True)
        catalog_dir.mkdir(parents=True, exist_ok=True)

        nc = _make_gridded_nc(
            data_dir / "sample.nc",
            pd.date_range("2025-01-01", periods=1),
        )
        _make_catalog_json(
            catalog_dir / "glonet.json",
            [nc],
            [("2025-01-01T00:00:00", "2025-01-02T00:00:00")],
        )

        # Monkeypatch GlonetConnectionConfig.create_fs
        monkeypatch.setattr(GlonetConnectionConfig, "create_fs", lambda self: MagicMock())

        source = {
            "dataset": "glonet",
            "config": "s3",
            "connection_type": "glonet",
            "s3_bucket": "test-bucket",
            "s3_folder": "test-folder",
            "url": "https://fake.example.com",
            "keep_variables": ["zos"],
            "eval_variables": ["zos"],
        }
        dataset = get_dataset_from_config(
            source=source,
            root_data_folder=str(tmp_path / "data"),
            root_catalog_folder=str(catalog_dir),
            dataset_processor=processor,
            max_samples=1,
            use_catalog=True,
        )
        assert dataset is not None
        assert dataset.alias == "glonet"

    def test_cmems_config(self, tmp_path: Path, monkeypatch):
        """get_dataset_from_config with config='cmems'."""
        from dctools.data.connection.connection_manager import CMEMSManager

        processor = _FakeDatasetProcessor()
        data_dir = tmp_path / "data" / "cmems_product"
        catalog_dir = tmp_path / "catalogs"
        data_dir.mkdir(parents=True, exist_ok=True)
        catalog_dir.mkdir(parents=True, exist_ok=True)

        nc = _make_gridded_nc(
            data_dir / "sample.nc",
            pd.date_range("2025-01-01", periods=1),
        )
        _make_catalog_json(
            catalog_dir / "cmems_product.json",
            [nc],
            [("2025-01-01T00:00:00", "2025-01-02T00:00:00")],
        )

        monkeypatch.setattr(CMEMSManager, "cmems_login", lambda self: None)

        source = {
            "dataset": "cmems_product",
            "config": "cmems",
            "cmems_product_name": "fake_product",
            "keep_variables": ["zos"],
            "eval_variables": ["zos"],
        }
        dataset = get_dataset_from_config(
            source=source,
            root_data_folder=str(tmp_path / "data"),
            root_catalog_folder=str(catalog_dir),
            dataset_processor=processor,
            max_samples=1,
            use_catalog=True,
        )
        assert dataset is not None
        assert dataset.alias == "cmems_product"

    def test_unknown_config_raises(self, tmp_path: Path):
        """get_dataset_from_config raises ValueError on unknown config name."""
        processor = _FakeDatasetProcessor()
        data_dir = tmp_path / "data"
        catalog_dir = tmp_path / "catalogs"
        data_dir.mkdir(parents=True, exist_ok=True)
        catalog_dir.mkdir(parents=True, exist_ok=True)
        source = {
            "dataset": "foo",
            "config": "unknown_protocol",
        }
        with pytest.raises(ValueError, match="Unknown dataset config name"):
            get_dataset_from_config(
                source=source,
                root_data_folder=str(data_dir),
                root_catalog_folder=str(catalog_dir),
                dataset_processor=processor,
            )


# ---------------------------------------------------------------------------
# Test: build_forecast_index_from_catalog standalone
# ---------------------------------------------------------------------------


class TestForecastIndex:
    """Direct tests for the forecast index builder."""

    def test_basic_forecast_index(self):
        """Build a forecast index from a minimal catalog DataFrame."""
        import geopandas as gpd

        records = []
        dates = pd.date_range("2025-01-01", periods=5, freq="D")
        for dt in dates:
            records.append({
                "path": f"/fake/{dt.date()}.nc",
                "date_start": dt,
                "date_end": dt + pd.Timedelta(days=1),
                "geometry": None,
            })
        catalog_df = gpd.GeoDataFrame(records)

        fi = build_forecast_index_from_catalog(
            catalog_df,
            init_date="2025-01-01",
            end_date="2025-01-05",
            n_days_forecast=3,
            n_days_interval=1,
        )
        assert not fi.empty
        assert "forecast_reference_time" in fi.columns
        assert "lead_time" in fi.columns
        assert "valid_time" in fi.columns
        assert "file" in fi.columns
        # Each forecast sequence should have n_days_forecast entries
        assert len(fi) % 3 == 0

    def test_empty_catalog_raises(self):
        """Empty catalog should raise ValueError."""
        import geopandas as gpd

        catalog_df = gpd.GeoDataFrame({
            "path": pd.Series(dtype=str),
            "date_start": pd.Series(dtype="datetime64[ns]"),
            "date_end": pd.Series(dtype="datetime64[ns]"),
            "geometry": pd.Series(dtype=object),
        })
        with pytest.raises((ValueError, AssertionError)):
            build_forecast_index_from_catalog(
                catalog_df,
                init_date="2025-01-01",
                end_date="2025-01-05",
                n_days_forecast=3,
                n_days_interval=1,
            )


# ---------------------------------------------------------------------------
# Test: DatasetConfig and CONNECTION_MANAGER_MAP
# ---------------------------------------------------------------------------


class TestDatasetConfig:
    """Test DatasetConfig class and connection manager mapping."""

    def test_connection_manager_map_has_all_types(self):
        """CONNECTION_MANAGER_MAP should contain all 7 connection types."""
        from dctools.data.connection.config import ARGOConnectionConfig
        from dctools.data.connection.connection_manager import (
            ArgoManager,
            CMEMSManager,
            FTPManager,
            GlonetManager,
            LocalConnectionManager,
            S3Manager,
            S3WasabiManager,
        )

        cmap = DatasetConfig.CONNECTION_MANAGER_MAP
        assert cmap[LocalConnectionConfig] == LocalConnectionManager
        assert cmap[CMEMSConnectionConfig] == CMEMSManager
        assert cmap[S3ConnectionConfig] == S3Manager
        assert cmap[WasabiS3ConnectionConfig] == S3WasabiManager
        assert cmap[FTPConnectionConfig] == FTPManager
        assert cmap[GlonetConnectionConfig] == GlonetManager
        assert cmap[ARGOConnectionConfig] == ArgoManager

    def test_dataset_config_attributes(self, tmp_path: Path):
        """DatasetConfig correctly stores all init arguments."""
        processor = _FakeDatasetProcessor()
        params = {
            "dataset_processor": processor,
            "init_type": "from_json",
            "local_root": str(tmp_path),
            "max_samples": 1,
            "file_pattern": "*.nc",
            "keep_variables": ["zos"],
            "filter_values": {},
        }
        conn = LocalConnectionConfig(params)
        cfg = DatasetConfig(
            alias="myds",
            connection_config=conn,
            catalog_options={"catalog_path": "/tmp/fake.json"},
            keep_variables=["zos"],
            eval_variables=["ssh"],
            observation_dataset=True,
            use_catalog=False,
            ignore_geometry=True,
        )
        assert cfg.alias == "myds"
        assert cfg.keep_variables == ["zos"]
        assert cfg.eval_variables == ["ssh"]
        assert cfg.observation_dataset is True
        assert cfg.use_catalog is False
        assert cfg.ignore_geometry is True


# ---------------------------------------------------------------------------
# Test: MultiSourceDatasetManager operations
# ---------------------------------------------------------------------------


class TestMultiSourceDatasetManager:
    """Tests for dataset manager methods beyond the main pipeline."""

    def test_add_duplicate_alias_raises(self, tmp_path: Path):
        """Adding a dataset with a duplicate alias should raise ValueError."""
        processor = _FakeDatasetProcessor()
        data_dir = tmp_path / "data" / "dup"
        catalog_dir = tmp_path / "catalogs"
        nc = _make_gridded_nc(
            data_dir / "sample.nc",
            pd.date_range("2025-01-01", periods=1),
        )
        catalog = _make_catalog_json(
            catalog_dir / "dup.json",
            [nc],
            [("2025-01-01T00:00:00", "2025-01-02T00:00:00")],
        )
        params = {
            "dataset_processor": processor,
            "init_type": "from_json",
            "local_root": str(data_dir),
            "max_samples": 1,
            "file_pattern": "*.nc",
            "keep_variables": ["zos"],
            "filter_values": {},
        }
        conn = LocalConnectionConfig(params)
        ds = _build_dataset("dup", conn, catalog, ["zos"], ["zos"])

        manager = MultiSourceDatasetManager(
            dataset_processor=processor,
            target_dimensions={"lat": [0.0, 1.0], "lon": [0.0, 1.0], "depth": [0.0]},
            time_tolerance=pd.Timedelta("1h"),
        )
        manager.add_dataset("dup", ds)

        # Rebuild a second dataset object for the duplicate
        ds2 = _build_dataset("dup", conn, catalog, ["zos"], ["zos"])
        with pytest.raises(ValueError, match="already exists"):
            manager.add_dataset("dup", ds2)

    def test_get_metadata_dict(self, tmp_path: Path):
        """get_metadata_dict returns valid metadata for all datasets."""
        processor = _FakeDatasetProcessor()
        data_dir = tmp_path / "data" / "metads"
        catalog_dir = tmp_path / "catalogs"
        nc = _make_gridded_nc(
            data_dir / "sample.nc",
            pd.date_range("2025-01-01", periods=1),
        )
        catalog = _make_catalog_json(
            catalog_dir / "metads.json",
            [nc],
            [("2025-01-01T00:00:00", "2025-01-02T00:00:00")],
        )
        params = {
            "dataset_processor": processor,
            "init_type": "from_json",
            "local_root": str(data_dir),
            "max_samples": 1,
            "file_pattern": "*.nc",
            "keep_variables": ["zos"],
            "filter_values": {},
        }
        conn = LocalConnectionConfig(params)
        ds = _build_dataset("metads", conn, catalog, ["zos"], ["zos"])

        manager = MultiSourceDatasetManager(
            dataset_processor=processor,
            target_dimensions={"lat": [0.0, 1.0], "lon": [0.0, 1.0], "depth": [0.0]},
            time_tolerance=pd.Timedelta("1h"),
        )
        manager.add_dataset("metads", ds)

        meta = manager.get_metadata_dict()
        assert "metads" in meta
        assert isinstance(meta["metads"], dict)

    def test_filter_all_by_date(self, tmp_path: Path):
        """filter_all_by_date should keep only entries within the date range."""
        processor = _FakeDatasetProcessor()
        data_dir = tmp_path / "data" / "filt"
        catalog_dir = tmp_path / "catalogs"

        ncs = []
        date_ranges = []
        for i, dt in enumerate(pd.date_range("2025-01-01", periods=3, freq="D")):
            nc = _make_gridded_nc(data_dir / f"day{i}.nc", pd.DatetimeIndex([dt]))
            ncs.append(nc)
            date_ranges.append((dt.isoformat(), (dt + pd.Timedelta(days=1)).isoformat()))

        catalog = _make_catalog_json(
            catalog_dir / "filt.json", ncs, date_ranges,
        )
        params = {
            "dataset_processor": processor,
            "init_type": "from_json",
            "local_root": str(data_dir),
            "max_samples": 10,
            "file_pattern": "*.nc",
            "keep_variables": ["zos"],
            "filter_values": {},
        }
        conn = LocalConnectionConfig(params)
        ds = _build_dataset("filt", conn, catalog, ["zos"], ["zos"])

        manager = MultiSourceDatasetManager(
            dataset_processor=processor,
            target_dimensions={"lat": [0.0, 1.0], "lon": [0.0, 1.0], "depth": [0.0]},
            time_tolerance=pd.Timedelta("1h"),
        )
        manager.add_dataset("filt", ds)
        # Filter to keep only the first day
        manager.filter_all_by_date("2025-01-01", "2025-01-02")
        cat = manager.datasets["filt"].get_catalog()
        assert cat is not None
        df = cat.get_dataframe()
        assert len(df) <= 3  # Should have been filtered


# ---------------------------------------------------------------------------
# Test: connection_manager utilities
# ---------------------------------------------------------------------------


class TestConnectionManagerUtils:
    """Tests for utility functions in connection_manager module."""

    def test_get_time_bound_values_datetime(self):
        """get_time_bound_values extracts min/max timestamps from a dataset."""
        from dctools.data.connection.connection_manager import get_time_bound_values

        times = pd.date_range("2025-01-01", periods=5, freq="D")
        ds = xr.Dataset({"x": (("time",), np.zeros(5))}, coords={"time": times})
        t_min, t_max = get_time_bound_values(ds)
        assert t_min == pd.Timestamp("2025-01-01")
        assert t_max == pd.Timestamp("2025-01-05")

    def test_get_time_bound_values_no_time(self):
        """get_time_bound_values returns (None, None) when no time variable."""
        from dctools.data.connection.connection_manager import get_time_bound_values

        ds = xr.Dataset({"x": (("y",), [1.0, 2.0])}, coords={"y": [0, 1]})
        t_min, t_max = get_time_bound_values(ds)
        assert t_min is None
        assert t_max is None

    def test_get_time_bound_values_empty_time(self):
        """get_time_bound_values returns (None, None) for empty time array."""
        from dctools.data.connection.connection_manager import get_time_bound_values

        ds = xr.Dataset(
            {"x": (("time",), np.array([], dtype=np.float32))},
            coords={"time": pd.DatetimeIndex([], dtype="datetime64[ns]")},
        )
        t_min, t_max = get_time_bound_values(ds)
        assert t_min is None
        assert t_max is None

    def test_adjust_full_day(self, tmp_path: Path):
        """adjust_full_day extends single-midnight timestamps to full day."""
        from dctools.data.connection.connection_manager import LocalConnectionManager

        processor = _FakeDatasetProcessor()
        data_dir = tmp_path / "adj"
        data_dir.mkdir(parents=True, exist_ok=True)
        params = {
            "dataset_processor": processor,
            "init_type": "from_json",
            "local_root": str(data_dir),
            "max_samples": 1,
            "file_pattern": "*.nc",
            "keep_variables": ["zos"],
            "filter_values": {},
        }
        conn = LocalConnectionConfig(params)
        mgr = LocalConnectionManager(conn, call_list_files=False)

        midnight = pd.Timestamp("2025-01-01 00:00:00")
        start, end = mgr.adjust_full_day(midnight, midnight)
        assert start == midnight
        assert end > midnight  # Should be 23:59:59.999999

    def test_open_local_existing(self, tmp_path: Path):
        """LocalConnectionManager supports check and open_local for missing files."""
        from dctools.data.connection.connection_manager import LocalConnectionManager

        # supports() correctly identifies local paths
        assert LocalConnectionManager.supports("/data/test.nc") is True
        assert LocalConnectionManager.supports("s3://bucket/key.nc") is False

        # open_local returns None for non-existent files
        processor = _FakeDatasetProcessor()
        data_dir = tmp_path / "openl"
        data_dir.mkdir(parents=True, exist_ok=True)
        params = {
            "dataset_processor": processor,
            "init_type": "from_json",
            "local_root": str(data_dir),
            "max_samples": 1,
            "file_pattern": "*.nc",
            "keep_variables": ["zos"],
            "filter_values": {},
        }
        conn = LocalConnectionConfig(params)
        mgr = LocalConnectionManager(conn, call_list_files=False)
        result = mgr.open_local(str(data_dir / "missing.nc"))
        assert result is None

    def test_open_local_nonexistent(self, tmp_path: Path):
        """open_local returns None for a non-existent file."""
        from dctools.data.connection.connection_manager import LocalConnectionManager

        processor = _FakeDatasetProcessor()
        data_dir = tmp_path / "nope"
        data_dir.mkdir(parents=True, exist_ok=True)
        params = {
            "dataset_processor": processor,
            "init_type": "from_json",
            "local_root": str(data_dir),
            "max_samples": 1,
            "file_pattern": "*.nc",
            "keep_variables": ["zos"],
            "filter_values": {},
        }
        conn = LocalConnectionConfig(params)
        mgr = LocalConnectionManager(conn, call_list_files=False)
        ds = mgr.open_local(str(data_dir / "missing.nc"))
        assert ds is None

    def test_local_list_files(self, tmp_path: Path):
        """LocalConnectionManager.list_files finds local NC files."""
        from dctools.data.connection.connection_manager import LocalConnectionManager

        processor = _FakeDatasetProcessor()
        data_dir = tmp_path / "listf"
        _make_gridded_nc(data_dir / "a.nc", pd.date_range("2025-01-01", periods=1))
        _make_gridded_nc(data_dir / "b.nc", pd.date_range("2025-01-02", periods=1))
        params = {
            "dataset_processor": processor,
            "init_type": "from_data",
            "local_root": str(data_dir),
            "max_samples": 10,
            "file_pattern": "*.nc",
            "keep_variables": ["zos"],
            "filter_values": {},
        }
        conn = LocalConnectionConfig(params)
        mgr = LocalConnectionManager(conn, call_list_files=False)
        files = mgr.list_files()
        assert len(files) == 2

    def test_local_supports(self):
        """LocalConnectionManager.supports checks for local paths."""
        from dctools.data.connection.connection_manager import LocalConnectionManager

        assert LocalConnectionManager.supports("/home/data/test.nc") is True
        assert LocalConnectionManager.supports("file:///home/data/test.nc") is True
        assert LocalConnectionManager.supports("s3://bucket/key.nc") is False

    def test_clean_for_serialization(self):
        """clean_for_serialization removes non-serializable attributes."""
        from dctools.data.connection.connection_manager import clean_for_serialization

        fake_params = type(
            "P", (), {"fs": "fake_fs", "dataset_processor": None},
        )()
        fake_obj = type("Obj", (), {"params": fake_params})()
        obj = clean_for_serialization(fake_obj)
        assert obj.params.fs is None


# ---------------------------------------------------------------------------
# Test: DatasetCatalog operations
# ---------------------------------------------------------------------------


class TestDatasetCatalogOps:
    """Test DatasetCatalog methods beyond basic loading."""

    def test_catalog_from_json_and_dataframe(self, tmp_path: Path):
        """Catalog loaded from JSON has valid GeoDataFrame."""
        data_dir = tmp_path / "cat"
        nc = _make_gridded_nc(
            data_dir / "sample.nc",
            pd.date_range("2025-01-01", periods=1),
        )
        catalog_path = _make_catalog_json(
            tmp_path / "catalog.json",
            [nc],
            [("2025-01-01T00:00:00", "2025-01-02T00:00:00")],
        )
        from dctools.data.datasets.dc_catalog import DatasetCatalog

        cat = DatasetCatalog.from_json(str(catalog_path), "test", limit=10)
        df = cat.get_dataframe()
        assert len(df) == 1
        assert "path" in df.columns
        assert "date_start" in df.columns

    def test_catalog_filter_by_date(self, tmp_path: Path):
        """filter_by_date keeps only entries in the given range."""
        data_dir = tmp_path / "catfilt"
        ncs = []
        date_ranges = []
        for i, dt in enumerate(pd.date_range("2025-01-01", periods=4, freq="D")):
            nc = _make_gridded_nc(data_dir / f"d{i}.nc", pd.DatetimeIndex([dt]))
            ncs.append(nc)
            date_ranges.append((dt.isoformat(), (dt + pd.Timedelta(days=1)).isoformat()))
        catalog_path = _make_catalog_json(tmp_path / "filt.json", ncs, date_ranges)

        from dctools.data.datasets.dc_catalog import DatasetCatalog

        cat = DatasetCatalog.from_json(str(catalog_path), "test", limit=10)
        assert len(cat.get_dataframe()) == 4
        cat.filter_by_date(
            pd.Timestamp("2025-01-01"),
            pd.Timestamp("2025-01-02"),
        )
        df = cat.get_dataframe()
        assert len(df) <= 2

    def test_catalog_to_json_roundtrip(self, tmp_path: Path):
        """A catalog can be saved to JSON and reloaded."""
        data_dir = tmp_path / "rt"
        nc = _make_gridded_nc(data_dir / "s.nc", pd.date_range("2025-01-01", periods=1))
        catalog_path = _make_catalog_json(
            tmp_path / "orig.json", [nc],
            [("2025-01-01T00:00:00", "2025-01-02T00:00:00")],
        )
        from dctools.data.datasets.dc_catalog import DatasetCatalog

        cat = DatasetCatalog.from_json(str(catalog_path), "test", limit=10)
        out_path = str(tmp_path / "roundtrip.json")
        cat.to_json(out_path)
        assert Path(out_path).exists()
        cat2 = DatasetCatalog.from_json(out_path, "test", limit=10)
        assert len(cat2.get_dataframe()) == len(cat.get_dataframe())

    def test_catalog_list_paths(self, tmp_path: Path):
        """list_paths returns file paths from the catalog."""
        data_dir = tmp_path / "lp"
        nc = _make_gridded_nc(data_dir / "s.nc", pd.date_range("2025-01-01", periods=1))
        catalog_path = _make_catalog_json(
            tmp_path / "lp.json", [nc],
            [("2025-01-01T00:00:00", "2025-01-02T00:00:00")],
        )
        from dctools.data.datasets.dc_catalog import DatasetCatalog

        cat = DatasetCatalog.from_json(str(catalog_path), "test", limit=10)
        paths = cat.list_paths()
        assert len(paths) == 1
        assert str(nc) in paths[0]

    def test_catalog_append(self, tmp_path: Path):
        """Append adds a new entry to the catalog."""
        from dctools.data.datasets.dc_catalog import CatalogEntry, DatasetCatalog
        from shapely.geometry import box

        # Build from entries (not from_json) so .entries is populated
        entry1 = CatalogEntry(
            path=str(tmp_path / "a.nc"),
            date_start=pd.Timestamp("2025-01-01"),
            date_end=pd.Timestamp("2025-01-02"),
            geometry=box(-10, -10, 10, 10),
        )
        cat = DatasetCatalog(
            alias="test",
            global_metadata={"variables_rename_dict": {}},
            entries=[entry1],
        )
        assert len(cat.get_dataframe()) == 1

        entry2 = CatalogEntry(
            path=str(tmp_path / "b.nc"),
            date_start=pd.Timestamp("2025-01-03"),
            date_end=pd.Timestamp("2025-01-04"),
            geometry=box(-10, -10, 10, 10),
        )
        cat.append(entry2)
        assert len(cat.get_dataframe()) == 2


# ---------------------------------------------------------------------------
# Test: CoordinateSystem
# ---------------------------------------------------------------------------


class TestCoordinateSystem:
    """Test CoordinateSystem class methods."""

    def test_coordinate_system_basics(self):
        """CoordinateSystem stores attributes correctly."""
        from dctools.utilities.coordinates import CoordinateSystem

        cs = CoordinateSystem(
            coord_type="geographic",
            coord_level="L4",
            coordinates={"time": "time", "lat": "lat", "lon": "lon", "depth": "depth"},
            crs=None,
        )
        assert cs.is_geographic()
        assert not cs.is_polar()
        assert not cs.is_observation_dataset()
        d = cs.to_dict()
        assert d["coord_type"] == "geographic"

    def test_observation_dataset_detection(self):
        """coord_level != 'L4' → is_observation_dataset returns True."""
        from dctools.utilities.coordinates import CoordinateSystem

        cs = CoordinateSystem(
            coord_type="geographic",
            coord_level="L2",
            coordinates={"time": "time", "lat": "lat", "lon": "lon"},
            crs=None,
        )
        assert cs.is_observation_dataset()


# ---------------------------------------------------------------------------
# Test: Transforms standalone
# ---------------------------------------------------------------------------


class TestTransformsStandalone:
    """Test individual transform functions."""

    def test_detect_normalize_longitude_180(self):
        """Already in [-180,180] range: dataset unchanged."""
        from dctools.processing.transforms import detect_and_normalize_longitude_system

        ds = xr.Dataset(
            {"x": (("lon",), [1.0, 2.0, 3.0])},
            coords={"lon": [-10.0, 0.0, 10.0]},
        )
        result = detect_and_normalize_longitude_system(ds, "lon")
        np.testing.assert_array_equal(result["lon"].values, ds["lon"].values)

    def test_detect_normalize_longitude_360(self):
        """[0,360] range should be converted to [-180,180]."""
        from dctools.processing.transforms import detect_and_normalize_longitude_system

        ds = xr.Dataset(
            {"x": (("lon",), [1.0, 2.0, 3.0])},
            coords={"lon": [0.0, 180.0, 359.0]},
        )
        result = detect_and_normalize_longitude_system(ds, "lon")
        assert float(result["lon"].min()) < 0

    def test_detect_longitude_system_classifications(self):
        """_detect_longitude_system returns correct system labels."""
        from dctools.processing.transforms import _detect_longitude_system

        assert _detect_longitude_system(0.0, 359.0) == "[0, 360]"
        assert _detect_longitude_system(-180.0, 180.0) == "[-180, 180]"
        assert _detect_longitude_system(-10.0, 50.0) == "[-180, 180]"

    def test_custom_transforms_standardize_pipeline(self, tmp_path: Path):
        """CustomTransforms with standardize pipeline is callable."""
        from dctools.processing.transforms import CustomTransforms

        processor = _FakeDatasetProcessor()
        pipeline = [
            {"name": "select_variables", "kwargs": {"variables": ["zos"]}},
            {
                "name": "rename_coords_vars",
                "kwargs": {
                    "coords_rename_dict": {"time": "time", "lat": "lat", "lon": "lon"},
                    "vars_rename_dict": {"zos": "ssh"},
                },
            },
            {"name": "detect_normalize_longitude", "kwargs": {}},
        ]
        ct = CustomTransforms(pipeline, dataset_processor=processor)
        assert callable(ct)

    def test_convert_longitude_to_180(self):
        """_convert_longitude_to_180 converts [0,360] to [-180,180]."""
        from dctools.processing.transforms import _convert_longitude_to_180

        ds = xr.Dataset(
            {"x": (("lon",), [1.0, 2.0, 3.0])},
            coords={"lon": [0.0, 180.0, 359.0]},
        )
        result = _convert_longitude_to_180(ds, "lon")
        assert float(result["lon"].min()) < 0
        assert float(result["lon"].max()) <= 180.0


# ---------------------------------------------------------------------------
# Test: CatalogEntry
# ---------------------------------------------------------------------------


class TestCatalogEntry:
    """Test CatalogEntry creation and serialization."""

    def test_catalog_entry_to_dict_and_back(self):
        """CatalogEntry round-trips through dict serialization."""
        from dctools.data.datasets.dc_catalog import CatalogEntry
        from shapely.geometry import box

        entry = CatalogEntry(
            path="/data/test.nc",
            date_start=pd.Timestamp("2025-01-01"),
            date_end=pd.Timestamp("2025-01-02"),
            geometry=box(-10, -10, 10, 10),
        )
        d = entry.to_dict()
        assert d["path"] == "/data/test.nc"
        assert "geometry" in d
        entry2 = CatalogEntry.from_dict(d)
        assert entry2.path == entry.path
        assert entry2.date_start == entry.date_start

    def test_catalog_entry_none_geometry(self):
        """CatalogEntry handles None geometry."""
        from dctools.data.datasets.dc_catalog import CatalogEntry

        entry = CatalogEntry(
            path="/test.nc",
            date_start=pd.Timestamp("2025-01-01"),
            date_end=pd.Timestamp("2025-01-02"),
            geometry=None,
        )
        d = entry.to_dict()
        assert d["geometry"] is None
