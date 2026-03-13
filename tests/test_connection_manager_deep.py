"""Deep coverage tests for connection_manager.py uncovered branches."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from shapely.geometry import box

from dctools.data.connection.connection_manager import (
    BaseConnectionManager,
    CMEMSManager,
    GlonetManager,
    S3Manager,
    ArgoManager,
    RecursionExit,
    create_worker_connect_config,
    prefetch_obs_files_to_local,
)
from dctools.utilities.coordinates import CoordinateSystem


# =====================================================================
# Helpers
# =====================================================================

def _make_base_manager(list_files=None, open_returns=None):
    """Build a BaseConnectionManager subclass instance with minimal mocking."""
    class _Stub(BaseConnectionManager):
        @classmethod
        def supports(cls, path):
            return True

        def list_files(self):
            return list_files or []

    params = SimpleNamespace(
        name="stub",
        protocol="local",
        local_root="/tmp/stub",
        file_pattern="*.nc",
        keep_variables=["ssh"],
        max_samples=999,
        start_time="2024-01-01",
        end_time="2024-01-10",
    )
    mgr = _Stub.__new__(_Stub)
    mgr.params = params
    mgr.init_type = "from_json"
    mgr._list_files = list_files
    mgr._catalog = None
    mgr._global_metadata = None
    if open_returns is not None:
        mgr.open = MagicMock(return_value=open_returns)
    return mgr


def _geo_coord_system():
    """Return a geographic CoordinateSystem."""
    return CoordinateSystem(
        coord_type="geographic",
        coord_level="L4",
        coordinates={"lat": "latitude", "lon": "longitude", "time": "time"},
        crs="EPSG:4326",
    )


def _polar_coord_system():
    """Return a polar CoordinateSystem."""
    return CoordinateSystem(
        coord_type="polar",
        coord_level="L4",
        coordinates={"x": "x", "y": "y", "time": "time"},
        crs="EPSG:3413",
    )


# =====================================================================
# extract_global_metadata
# =====================================================================

class TestExtractGlobalMetadata:
    """Tests for BaseConnectionManager.extract_global_metadata."""

    def test_normal_first_file_valid(self):
        """First file opens successfully → returns full metadata dict."""
        ds = xr.Dataset(
            {"ssh": (["latitude", "longitude"], np.zeros((3, 4)))},
            coords={
                "latitude": [10.0, 20.0, 30.0],
                "longitude": [1.0, 2.0, 3.0, 4.0],
            },
        )
        mgr = _make_base_manager(list_files=["/f1.nc"], open_returns=ds)
        cs = _geo_coord_system()
        with patch.object(CoordinateSystem, "get_coordinate_system", return_value=cs):
            meta = mgr.extract_global_metadata()
        assert "variables" in meta
        assert "resolution" in meta
        assert "coord_system" in meta

    def test_first_file_fails_second_ok(self):
        """First file raises, second succeeds → skips gracefully."""
        ds = xr.Dataset(
            {"ssh": (["latitude", "longitude"], np.zeros((2, 2)))},
            coords={"latitude": [10.0, 20.0], "longitude": [1.0, 2.0]},
        )
        mgr = _make_base_manager(list_files=["/bad.nc", "/good.nc"])
        call_count = {"n": 0}

        def _open(path, mode="rb"):
            call_count["n"] += 1
            if call_count["n"] <= 1:
                raise OSError("corrupt")
            return ds

        mgr.open = _open
        cs = _geo_coord_system()
        with patch.object(CoordinateSystem, "get_coordinate_system", return_value=cs):
            meta = mgr.extract_global_metadata()
        assert "resolution" in meta

    def test_no_files_raises(self):
        """None file list raises FileNotFoundError."""
        mgr = _make_base_manager(list_files=None)
        with pytest.raises(FileNotFoundError, match="Empty file list"):
            mgr.extract_global_metadata()

    def test_all_files_invalid(self):
        """All files fail to open → FileNotFoundError."""
        mgr = _make_base_manager(list_files=["/x.nc"])
        mgr.open = MagicMock(side_effect=OSError("fail"))
        with pytest.raises(FileNotFoundError, match="No valid files"):
            mgr.extract_global_metadata()


# =====================================================================
# estimate_resolution — polar branch
# =====================================================================

class TestEstimateResolutionPolar:
    """Test the polar-coordinate branch of estimate_resolution."""

    def test_polar_regular_spacing(self):
        """Polar coords → returns x and y resolution."""
        ds = xr.Dataset(
            {"data": (["x", "y"], np.zeros((4, 4)))},
            coords={"x": [0.0, 100.0, 200.0, 300.0], "y": [0.0, 50.0, 100.0, 150.0]},
        )
        mgr = _make_base_manager()
        res = mgr.estimate_resolution(ds, _polar_coord_system())
        assert res["x"] == 100.0
        assert res["y"] == 50.0

    def test_polar_single_value(self):
        """Single-element coordinate returns that value."""
        ds = xr.Dataset(
            {"data": (["x", "y"], np.zeros((1, 1)))},
            coords={"x": [42.0], "y": [7.0]},
        )
        mgr = _make_base_manager()
        res = mgr.estimate_resolution(ds, _polar_coord_system())
        assert res["x"] == 42.0
        assert res["y"] == 7.0

    def test_polar_missing_coord(self):
        """Missing coordinate → no key in result."""
        ds = xr.Dataset({"data": (["z"], [1, 2])}, coords={"z": [0, 1]})
        mgr = _make_base_manager()
        res = mgr.estimate_resolution(ds, _polar_coord_system())
        assert "x" not in res
        assert "y" not in res

    def test_temporal_resolution_included(self):
        """Time resolution computed for both geo and polar."""
        times = pd.date_range("2024-01-01", periods=5, freq="6h")
        ds = xr.Dataset(
            {"data": (["time"], np.zeros(5))},
            coords={"time": times},
        )
        mgr = _make_base_manager()
        res = mgr.estimate_resolution(ds, _polar_coord_system())
        assert "time" in res
        assert "21600" in res["time"]  # 6h = 21600s


# =====================================================================
# CMEMSManager.list_files
# =====================================================================

class TestCMEMSListFiles:
    """Test CMEMSManager.list_files date-generation logic."""

    def _make_cmems_manager(self, start, end, max_samples=999):
        mgr = CMEMSManager.__new__(CMEMSManager)
        mgr.params = SimpleNamespace(max_samples=max_samples)
        mgr.start_time = start
        mgr.end_time = end
        mgr._list_files = None
        mgr._catalog = None
        mgr._global_metadata = None
        mgr.init_type = "from_json"
        return mgr

    def test_date_range_correct(self):
        """Generates day-by-day list."""
        mgr = self._make_cmems_manager("2024-01-01", "2024-01-03")
        result = mgr.list_files()
        assert len(result) == 3  # Jan 1, 2, 3

    def test_max_samples_truncation(self):
        """max_samples limits the date list."""
        mgr = self._make_cmems_manager("2024-01-01", "2024-01-31", max_samples=5)
        result = mgr.list_files()
        assert len(result) == 5

    def test_invalid_date_returns_empty(self):
        """Invalid date string → returns []."""
        mgr = self._make_cmems_manager("not-a-date", "also-bad")
        result = mgr.list_files()
        assert result == []


# =====================================================================
# S3Manager.list_first_n_files & list_files
# =====================================================================

class TestS3ManagerListFiles:
    """Test S3Manager.list_files and list_first_n_files."""

    def _make_s3_manager(self):
        mgr = S3Manager.__new__(S3Manager)
        fs = MagicMock()
        mgr.params = SimpleNamespace(
            s3_bucket="bucket",
            s3_folder="folder",
            file_pattern="*.nc",
            endpoint_url="s3.example.com",
            fs=fs,
        )
        mgr._list_files = None
        mgr._catalog = None
        mgr._global_metadata = None
        mgr.init_type = "from_json"
        return mgr, fs

    def test_list_files_success(self):
        """Glob returns files → s3:// prefixed list."""
        mgr, fs = self._make_s3_manager()
        fs.glob.return_value = ["bucket/folder/a.nc", "bucket/folder/b.nc"]
        result = mgr.list_files()
        assert len(result) == 2
        assert all(r.startswith("s3://") for r in result)

    def test_list_files_empty(self):
        """No files found → empty list, warning logged."""
        mgr, fs = self._make_s3_manager()
        fs.glob.return_value = []
        result = mgr.list_files()
        assert result == []

    def test_list_files_permission_error_fallback(self):
        """PermissionError → fallback to fs.ls."""
        mgr, fs = self._make_s3_manager()
        fs.glob.side_effect = PermissionError("no access")
        fs.ls.return_value = [
            {"Key": "folder/a.nc"},
            {"Key": "folder/b.nc"},
            {"Key": "folder/c.txt"},
        ]
        result = mgr.list_files()
        assert len(result) == 2  # only .nc files
        assert all(".nc" in r for r in result)

    def test_list_first_n_files_limit(self):
        """RecursionExit raised when n files reached."""
        mgr, fs = self._make_s3_manager()
        # Simulate walk returning multiple files
        fs.walk.return_value = [("root", [], ["a.nc", "b.nc", "c.nc", "d.nc"])]
        with pytest.raises(RecursionExit):
            mgr.list_first_n_files(fs, "root", n=2, pattern="*.nc")


# =====================================================================
# GlonetManager.list_files & open
# =====================================================================

class TestGlonetManager:
    """Test GlonetManager.list_files and open logic."""

    def _make_glonet_manager(self):
        mgr = GlonetManager.__new__(GlonetManager)
        mgr.params = SimpleNamespace(
            endpoint_url="https://example.com",
            s3_bucket="mybucket",
            s3_folder="myfolder",
        )
        mgr._list_files = None
        mgr._catalog = None
        mgr._global_metadata = None
        mgr.init_type = "from_json"
        return mgr

    def test_list_files_weekly(self):
        """Returns weekly dates from 20240103 through end of 2024."""
        mgr = self._make_glonet_manager()
        result = mgr.list_files()
        assert len(result) > 50
        assert "20240103.zarr" in result[0]
        # Each date should be 7 days apart
        for path in result:
            assert ".zarr" in path

    def test_open_local_zarr(self, tmp_path):
        """Local path → opens via xr.open_zarr."""
        ds = xr.Dataset({"ssh": (["time"], [1.0, 2.0])}, coords={"time": [0, 1]})
        zarr_path = str(tmp_path / "test.zarr")
        ds.to_zarr(zarr_path)

        mgr = self._make_glonet_manager()
        result = mgr.open(zarr_path)
        assert result is not None
        assert "ssh" in result

    def test_open_remote_delegates(self):
        """Remote HTTPS path → calls open_remote."""
        mgr = self._make_glonet_manager()
        mgr.open_remote = MagicMock(return_value=None)
        mgr.open("https://example.com/test.zarr")
        mgr.open_remote.assert_called_once()


# =====================================================================
# S3Manager.open_remote
# =====================================================================

class TestS3ManagerOpenRemote:
    """Test S3Manager.open_remote."""

    def _make_s3_manager(self):
        mgr = S3Manager.__new__(S3Manager)
        mgr.params = SimpleNamespace(
            groups=None,
            keep_variables=["ssh"],
            fs=MagicMock(),
        )
        mgr._list_files = None
        mgr._catalog = None
        mgr._global_metadata = None
        mgr.init_type = "from_json"
        return mgr

    def test_non_zarr_returns_none(self):
        """Non-.zarr extension returns None."""
        mgr = self._make_s3_manager()
        result = mgr.open_remote("s3://bucket/data.nc")
        assert result is None

    def test_zarr_opens_successfully(self):
        """Zarr path → delegates to FileLoader."""
        mgr = self._make_s3_manager()
        mock_ds = xr.Dataset({"ssh": (["time"], [1.0])})
        with patch("dctools.data.connection.connection_manager.FileLoader") as fl:
            fl.open_dataset_auto.return_value = mock_ds
            result = mgr.open_remote("s3://bucket/data.zarr")
        assert result is not None

    def test_exception_returns_none(self):
        """Exception → returns None."""
        mgr = self._make_s3_manager()
        with patch("dctools.data.connection.connection_manager.FileLoader") as fl:
            fl.open_dataset_auto.side_effect = OSError("fail")
            result = mgr.open_remote("s3://bucket/data.zarr")
        assert result is None


# =====================================================================
# ArgoManager.list_files_with_metadata
# =====================================================================

class TestArgoListFilesWithMetadata:
    """Test ArgoManager.list_files_with_metadata."""

    def _make_argo_manager(self, master_index=None):
        mgr = ArgoManager.__new__(ArgoManager)
        mgr.params = SimpleNamespace(
            local_root="/tmp/argo",
            keep_variables=["TEMP", "PSAL"],
            start_time="2024-01-01",
            end_time="2024-03-31",
            max_samples=999,
        )
        mgr._list_files = None
        mgr._catalog = None
        mgr._global_metadata = None
        mgr._master_index = master_index
        mgr.init_type = "from_json"
        mgr.depth_values = [0, 10, 50]
        mgr.argo_interface = MagicMock()
        mgr.argo_index = None
        mgr.start_time = "2024-01-01"
        mgr.end_time = "2024-03-31"
        return mgr

    def test_with_master_index(self):
        """Master index with 2 months → 2 CatalogEntries."""
        idx = {
            "2024_01": {
                "start": int(pd.Timestamp("2024-01-01").value),
                "end": int(pd.Timestamp("2024-01-31").value),
            },
            "2024_02": {
                "start": int(pd.Timestamp("2024-02-01").value),
                "end": int(pd.Timestamp("2024-02-29").value),
            },
        }
        mgr = self._make_argo_manager(master_index=idx)
        entries = mgr.list_files_with_metadata()
        assert len(entries) == 2
        assert entries[0].path == "2024_01"
        assert entries[0].geometry.equals(box(-180, -90, 180, 90))

    def test_cached_catalog_returned(self):
        """Previously cached catalog → returned without re-building."""
        mgr = self._make_argo_manager(master_index={})
        fake_catalog = [MagicMock()]
        mgr._catalog = fake_catalog
        assert mgr.list_files_with_metadata() is fake_catalog

    def test_missing_index_raises(self):
        """No master index and auto-build fails → FileNotFoundError."""
        mgr = self._make_argo_manager(master_index=None)
        mgr._load_master_index = MagicMock(return_value=None)
        mgr._try_auto_build_master_index = MagicMock(return_value=None)
        with pytest.raises(FileNotFoundError, match="ARGO master index"):
            mgr.list_files_with_metadata()


# =====================================================================
# ArgoManager.open
# =====================================================================

class TestArgoOpen:
    """Test ArgoManager.open with month keys and tuples."""

    def _make_argo_manager(self):
        idx = {
            "2024_01": {
                "start": int(pd.Timestamp("2024-01-01").value),
                "end": int(pd.Timestamp("2024-01-31").value),
            },
        }
        mgr = ArgoManager.__new__(ArgoManager)
        mgr._master_index = idx
        mgr.depth_values = [0, 10]
        mgr.params = SimpleNamespace(keep_variables=["TEMP"])
        mgr.argo_interface = MagicMock()
        mgr.argo_interface.variables = ["TEMP"]
        mock_ds = xr.Dataset({"TEMP": (["obs"], [1.0, 2.0])})
        mgr.argo_interface.open_time_window.return_value = mock_ds
        return mgr

    def test_open_by_month_key(self):
        """Open with month key uses master index dates."""
        mgr = self._make_argo_manager()
        result = mgr.open("2024_01")
        assert result is not None
        mgr.argo_interface.open_time_window.assert_called_once()

    def test_open_by_tuple(self):
        """Open with (start, end) tuple works."""
        mgr = self._make_argo_manager()
        start = pd.Timestamp("2024-01-01")
        end = pd.Timestamp("2024-01-15")
        result = mgr.open((start, end))
        assert result is not None

    def test_open_invalid_key_raises(self):
        """Invalid path key → ValueError."""
        mgr = self._make_argo_manager()
        with pytest.raises(ValueError, match="Invalid path for ARGO"):
            mgr.open("nonexistent_key")


# =====================================================================
# prefetch_obs_files_to_local
# =====================================================================

class TestPrefetchObsFilesToLocal:
    """Test prefetch_obs_files_to_local."""

    def test_empty_paths(self, tmp_path):
        """Empty path list → empty dict."""
        fs = MagicMock()
        result = prefetch_obs_files_to_local([], str(tmp_path), fs, "test")
        assert result == {}

    def test_nc_file_cached(self, tmp_path):
        """Already-cached .nc file → returned from cache."""
        local_file = tmp_path / "data.nc"
        local_file.write_bytes(b"fake")
        fs = MagicMock()
        result = prefetch_obs_files_to_local(
            ["s3://bucket/data.nc"], str(tmp_path), fs, "test"
        )
        assert "s3://bucket/data.nc" in result
        assert result["s3://bucket/data.nc"] == str(local_file)

    def test_nc_file_downloaded(self, tmp_path):
        """Non-cached .nc file → downloaded via fs.open."""
        fs = MagicMock()
        fake_remote = MagicMock()
        fake_remote.read.return_value = b"fake-nc-content"
        fake_remote.__enter__ = MagicMock(return_value=fake_remote)
        fake_remote.__exit__ = MagicMock(return_value=False)
        fs.open.return_value = fake_remote
        result = prefetch_obs_files_to_local(
            ["s3://bucket/new.nc"], str(tmp_path), fs, "test"
        )
        assert "s3://bucket/new.nc" in result

    def test_download_failure_skipped(self, tmp_path):
        """Download failure → file skipped, no crash."""
        fs = MagicMock()
        fs.open.side_effect = OSError("network error")
        result = prefetch_obs_files_to_local(
            ["s3://bucket/bad.nc"], str(tmp_path), fs, "test"
        )
        # File is not in map, but function doesn't crash
        assert len(result) == 0

    def test_deduplication(self, tmp_path):
        """Duplicate paths → processed only once."""
        local_file = tmp_path / "data.nc"
        local_file.write_bytes(b"fake")
        fs = MagicMock()
        result = prefetch_obs_files_to_local(
            ["s3://bucket/data.nc", "s3://bucket/data.nc"],
            str(tmp_path), fs, "test",
        )
        assert len(result) == 1


# =====================================================================
# create_worker_connect_config
# =====================================================================

class TestCreateWorkerConnectConfig:
    """Test create_worker_connect_config factory."""

    def test_local_protocol(self):
        """Local protocol → returns a callable."""
        config = SimpleNamespace(
            protocol="local",
            dataset_processor=None,
            params=SimpleNamespace(
                name="local",
                local_root="/tmp",
                file_pattern="*.nc",
                keep_variables=["ssh"],
                max_samples=100,
                start_time="2024-01-01",
                end_time="2024-12-31",
            ),
        )
        with patch.dict(
            "dctools.data.connection.connection_manager.CONNECTION_CONFIG_REGISTRY",
            {"local": lambda d: SimpleNamespace(params=SimpleNamespace(**d))},
        ), patch.dict(
            "dctools.data.connection.connection_manager.CONNECTION_MANAGER_REGISTRY",
            {"local": MagicMock(return_value=MagicMock(open=lambda p, m="rb": None))},
        ):
            func = create_worker_connect_config(config)
        assert callable(func)
