"""Tests for connection manager functionality."""

import pytest
import os
from types import SimpleNamespace

import pandas as pd
import xarray as xr
from dctools.data.connection.config import LocalConnectionConfig
from dctools.data.connection.connection_manager import (
    ArgoManager,
    LocalConnectionManager,
)


def test_local_connection_manager(tmp_path):
    """Test LocalConnectionManager functionality."""
    # Setup - Create some dummy files
    d = tmp_path / "data"
    d.mkdir()
    (d / "file1.nc").touch()
    (d / "file2.nc").touch()
    (d / "other.txt").touch()

    params = {
        "local_root": str(d),
        "file_pattern": "*.nc",
        "init_type": "recursive",
        "max_samples": 10,
        "groups": None,
        "keep_variables": None,
        "file_cache": None,
        "dataset_processor": None,
        "filter_values": {"dummy": "value"},
        "full_day_data": False,
    }

    # We might need to mock create_fs if it fails or use real one
    try:
        config = LocalConnectionConfig(params)
        manager = LocalConnectionManager(config)

        files = manager.list_files()

        # normalize paths for comparison
        files = [os.path.normpath(f) for f in files]
        expected1 = os.path.normpath(str(d / "file1.nc"))
        expected2 = os.path.normpath(str(d / "file2.nc"))

        assert len(files) == 2
        assert expected1 in files
        assert expected2 in files
    except Exception as e:
        pytest.fail(f"LocalConnectionManager test failed: {e}")


def test_argo_auto_builds_master_index_for_catalog(monkeypatch, tmp_path):
    """Missing ARGO master index should trigger an automatic build when time bounds exist."""
    manager = object.__new__(ArgoManager)
    manager._catalog = None
    manager._master_index = None
    manager.start_time = "2024-01-01"
    manager.end_time = "2024-02-01"
    manager.params = SimpleNamespace(local_root=str(tmp_path))

    called = {"build": False}

    def fake_build_time_window_monthly(start, end, temp_dir, n_workers):
        called["build"] = True
        assert pd.Timestamp(start) == pd.Timestamp("2024-01-01")
        assert pd.Timestamp(end) == pd.Timestamp("2024-02-01")
        assert "tmp_argo_refs" in temp_dir
        assert n_workers == 8

    manager.argo_interface = SimpleNamespace(
        build_time_window_monthly=fake_build_time_window_monthly
    )

    def fake_load_master_index():
        if called["build"]:
            manager._master_index = {
                "2024_01": {
                    "start": int(pd.Timestamp("2024-01-01").value),
                    "end": int(pd.Timestamp("2024-01-31").value),
                }
            }

    monkeypatch.setattr(manager, "_load_master_index", fake_load_master_index)

    catalog = manager.list_files_with_metadata()
    assert called["build"] is True
    assert len(catalog) == 1
    assert catalog[0].path == "2024_01"


def test_argo_requires_master_index_for_catalog_without_time_bounds(monkeypatch, tmp_path):
    """Without time bounds, missing ARGO master index should still fail explicitly."""
    manager = object.__new__(ArgoManager)
    manager._catalog = None
    manager._master_index = None
    manager.start_time = None
    manager.end_time = None
    manager.params = SimpleNamespace(local_root=str(tmp_path))
    manager.argo_interface = SimpleNamespace(build_multi_year_monthly=lambda **kwargs: None)

    monkeypatch.setattr(manager, "_load_master_index", lambda: None)

    with pytest.raises(FileNotFoundError):
        manager.list_files_with_metadata()


def test_argo_global_metadata_uses_semantic_mapping(monkeypatch):
    """ARGO global metadata should follow generic semantic mapping behavior."""
    sample_ds = xr.Dataset(
        {
            "TEMP": ("N_POINTS", [10.0, 11.0]),
            "PSAL": ("N_POINTS", [35.1, 35.2]),
        },
        coords={
            "LATITUDE": ("N_POINTS", [42.0, 42.5]),
            "LONGITUDE": ("N_POINTS", [-8.0, -7.5]),
            "TIME": ("N_POINTS", pd.to_datetime(["2024-01-01", "2024-01-02"])),
            "DEPTH": ("N_POINTS", [0.0, 10.0]),
            "N_POINTS": [0, 1],
        },
    )

    manager = object.__new__(ArgoManager)
    manager._global_metadata = None
    manager._catalog = None
    manager._master_index = {
        "2024_01": {
            "start": int(pd.Timestamp("2024-01-01").value),
            "end": int(pd.Timestamp("2024-01-31").value),
        }
    }
    manager.start_time = "2024-01-01"
    manager.end_time = "2024-01-31"
    manager.depth_values = [0, 10, 50, 100]
    manager.params = SimpleNamespace(keep_variables=["TEMP", "PSAL"])
    manager.argo_interface = SimpleNamespace(
        variables=["TEMP", "PSAL"],
        open_time_window=lambda **kwargs: sample_ds,
    )

    monkeypatch.setattr(manager, "estimate_resolution", lambda ds, cs: {})

    metadata = manager.get_global_metadata()

    assert metadata["coord_system"].coord_type == "geographic"
    assert metadata["variables_dict"]["temperature"] == "TEMP"
    assert metadata["variables_dict"]["salinity"] == "PSAL"
    assert metadata["variables_rename_dict"]["TEMP"] == "temperature"
    assert metadata["variables_rename_dict"]["PSAL"] == "salinity"
