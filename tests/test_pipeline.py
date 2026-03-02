"""Fast, local-only tests for the data processing pipeline.

The original version of this test exercised remote S3/CMEMS access, which makes
CI and developer runs extremely slow and flaky. This module builds a tiny local
NetCDF file + a minimal JSON catalog in a tmp directory, then checks that the
dataset manager can build a forecast index and a dataloader.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from dctools.data.connection.config import LocalConnectionConfig
from dctools.data.datasets.dataset import DatasetConfig, RemoteDataset
from dctools.data.datasets.dataloader import EvaluationDataloader
from dctools.data.datasets.dataset_manager import MultiSourceDatasetManager


class _FakeDatasetProcessor:
    """Minimal stand-in for OceanBench's DatasetProcessor.

    The unit tests in this repo should not require the full OceanBench stack.
    """

    def __init__(self):
        self.client = None
        self.distributed = False


@pytest.fixture(scope="function")
def local_test_config(tmp_path: Path) -> SimpleNamespace:
    """Minimal test config that keeps everything local and tiny."""
    return SimpleNamespace(
        data_directory=str(tmp_path / "data"),
        catalog_dir=str(tmp_path / "catalogs"),
        batch_size=1,
        max_samples=1,
        start_time="2025-01-01",
        end_time="2025-01-02",
    )


def check_dataloader(
    dataloader: EvaluationDataloader,
):
    """Check that dataloader yields valid batches."""
    for batch in dataloader:
        # Check that the batch contains the expected keys
        assert "pred_data" in batch[0]
        assert "ref_data" in batch[0]
        # Check that values are strings (paths)
        assert isinstance(batch[0]["pred_data"], str)
        if batch[0]["ref_data"]:
            assert isinstance(batch[0]["ref_data"], str)


@pytest.fixture(scope="function")
def local_dataset_manager(
    local_test_config: SimpleNamespace, tmp_path: Path
) -> MultiSourceDatasetManager:
    """Create a dataset manager containing a single tiny local dataset."""
    data_root = Path(local_test_config.data_directory)
    catalog_root = Path(local_test_config.catalog_dir)
    data_root.mkdir(parents=True, exist_ok=True)
    catalog_root.mkdir(parents=True, exist_ok=True)

    # --- Create a tiny NetCDF file ---
    ds_dir = data_root / "glonet"
    ds_dir.mkdir(parents=True, exist_ok=True)
    nc_path = ds_dir / "sample.nc"

    time = pd.date_range("2025-01-01", periods=1)
    depth = np.asarray([0.0], dtype=float)
    lat = np.asarray([0.0, 1.0], dtype=float)
    lon = np.asarray([0.0, 1.0], dtype=float)
    data = np.zeros((1, 2, 2), dtype=np.float32)

    xr.Dataset(
        data_vars={"zos": (("time", "lat", "lon"), data)},
        coords={"time": time, "depth": depth, "lat": lat, "lon": lon},
    ).to_netcdf(nc_path, engine="scipy")

    # --- Minimal JSON catalog referencing that local file ---
    catalog_path = catalog_root / "glonet.json"
    catalog_json = {
        "global_metadata": {
            "variables_rename_dict": {"zos": "ssh"},
            "coord_system": {
                "coord_type": "geographic",
                "coord_level": "L4",
                "coordinates": {"time": "time", "depth": "depth", "lat": "lat", "lon": "lon"},
                "crs": None,
            },
            "is_observation": False,
        },
        "features": [
            {
                "type": "Feature",
                "geometry": None,
                "properties": {
                    "path": str(nc_path),
                    "date_start": "2025-01-01T00:00:00",
                    "date_end": "2025-01-02T00:00:00",
                },
            }
        ],
    }
    catalog_path.write_text(json.dumps(catalog_json), encoding="utf-8")

    processor = _FakeDatasetProcessor()
    target_dimensions = {"lat": lat.tolist(), "lon": lon.tolist(), "depth": depth.tolist()}

    manager = MultiSourceDatasetManager(
        dataset_processor=processor,
        target_dimensions=target_dimensions,
        time_tolerance=pd.Timedelta("1h"),
    )

    connect_config_params = {
        "dataset_processor": processor,
        "init_type": "from_json",
        "local_root": str(ds_dir),
        "max_samples": 1,
        "file_pattern": "*.nc",
        "keep_variables": ["zos"],
        "filter_values": {},
    }
    local_conn_cfg = LocalConnectionConfig(connect_config_params)
    ds_cfg = DatasetConfig(
        alias="glonet",
        connection_config=local_conn_cfg,
        catalog_options={"catalog_path": str(catalog_path)},
        keep_variables=["zos"],
        eval_variables=["zos"],
        observation_dataset=False,
        use_catalog=True,
        ignore_geometry=True,
    )
    dataset = RemoteDataset(ds_cfg)
    manager.add_dataset("glonet", dataset)
    return manager


# @pytest.mark.usefixtures("setup_datasets", "test_config")
# class TestPipeline:


def test_pipeline_local_minimal(
    local_test_config: SimpleNamespace, local_dataset_manager: MultiSourceDatasetManager
):
    """Smoke-test: can build forecast index + dataloader on a tiny local dataset."""
    alias = "glonet"
    assert alias in local_dataset_manager.datasets

    # Ensure transforms can be created from the catalog metadata.
    transform = local_dataset_manager.get_transform(
        dataset_alias=alias,
        transform_name="standardize",
    )
    assert callable(transform)

    local_dataset_manager.build_forecast_index(
        alias,
        init_date=local_test_config.start_time,
        end_date=local_test_config.end_time,
        n_days_forecast=1,
        n_days_interval=1,
    )

    dataloader = local_dataset_manager.get_dataloader(
        pred_alias=alias,
        ref_aliases=[alias],
        batch_size=local_test_config.batch_size,
        pred_transform=transform,
        ref_transforms={alias: transform},
        forecast_mode=True,
        n_days_forecast=1,
    )
    check_dataloader(dataloader)
