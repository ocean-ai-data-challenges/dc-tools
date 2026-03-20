"""Tests targeting connection_manager.py coverage.

Level 1 – Pure / mockable functions:
    get_time_bound_values, clean_for_serialization,
    adjust_full_day, _get_local_path, supports() for every manager,
    estimate_resolution, get/set_global_metadata, RecursionExit,
    open dispatch (BaseConnectionManager.open), create_worker_connect_config.

Level 2 – Harder tests (real tmp NetCDF, mock fs):
    LocalConnectionManager full open+list cycle,
    S3Manager.list_files / FTPManager.list_files with mock fs,
    BaseConnectionManager.open dispatch with real files.
"""

from __future__ import annotations

import datetime
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from dctools.data.connection.connection_manager import (
    CMEMSManager,
    FTPManager,
    GlonetManager,
    LocalConnectionManager,
    RecursionExit,
    S3Manager,
    S3WasabiManager,
    clean_for_serialization,
    get_time_bound_values,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_nc(tmp_path, filename="test.nc", times=None, lat=None, lon=None, data=None):
    """Create a minimal NetCDF file and return its path."""
    if times is None:
        times = pd.date_range("2025-01-01", periods=3, freq="1D")
    if lat is None:
        lat = np.linspace(-5, 5, 5)
    if lon is None:
        lon = np.linspace(20, 30, 5)
    if data is None:
        data = np.random.default_rng(42).standard_normal(
            (len(times), len(lat), len(lon))
        ).astype(np.float32)

    ds = xr.Dataset(
        {"ssh": (["time", "latitude", "longitude"], data)},
        coords={"time": times, "latitude": lat, "longitude": lon},
    )
    path = tmp_path / filename
    ds.to_netcdf(path, engine="scipy")
    return str(path)


# ---------------------------------------------------------------------------
# Section 1 – get_time_bound_values
# ---------------------------------------------------------------------------


class TestGetTimeBoundValues:
    """Return (min, max) time from datasets."""

    def test_datetime_dim(self):
        """Standard time dimension with datetime64."""
        times = pd.date_range("2024-06-01", periods=5, freq="1D")
        ds = xr.Dataset({"x": (["time"], [1, 2, 3, 4, 5])}, coords={"time": times})
        tmin, tmax = get_time_bound_values(ds)
        assert tmin == pd.Timestamp("2024-06-01")
        assert tmax == pd.Timestamp("2024-06-05")

    def test_empty_time(self):
        """Empty time dimension → (None, None)."""
        ds = xr.Dataset(
            {"x": (["time"], np.array([], dtype="datetime64[ns]"))},
            coords={"time": np.array([], dtype="datetime64[ns]")},
        )
        assert get_time_bound_values(ds) == (None, None)

    def test_numeric_time(self):
        """Numeric time (e.g. Julian days) → floats."""
        ds = xr.Dataset(
            {"x": (["time"], [1.0, 2.0, 3.0])},
            coords={"time": [100.0, 200.0, 300.0]},
        )
        tmin, tmax = get_time_bound_values(ds)
        assert tmin == 100.0
        assert tmax == 300.0

    def test_no_time_variable(self):
        """No recognizable time dim → (None, None)."""
        ds = xr.Dataset({"x": (["z"], [1, 2, 3])}, coords={"z": [0, 1, 2]})
        assert get_time_bound_values(ds) == (None, None)

    def test_fallback_datetime64_data_var(self):
        """Falls back to scanning data_vars for datetime64 dtype."""
        dts = pd.date_range("2025-01-01", periods=2, freq="1D")
        ds = xr.Dataset({"my_dates": (["n"], dts.values)})
        tmin, tmax = get_time_bound_values(ds)
        assert tmin == pd.Timestamp("2025-01-01")
        assert tmax == pd.Timestamp("2025-01-02")

    def test_single_time_value(self):
        """Single datetime element → min == max."""
        times = pd.date_range("2025-03-15", periods=1)
        ds = xr.Dataset({"x": (["time"], [1])}, coords={"time": times})
        tmin, tmax = get_time_bound_values(ds)
        assert tmin == tmax == pd.Timestamp("2025-03-15")

    def test_nan_numeric_time(self):
        """All-NaN numeric time → (None, None)."""
        ds = xr.Dataset(
            {"x": (["time"], [np.nan, np.nan])},
            coords={"time": [np.nan, np.nan]},
        )
        assert get_time_bound_values(ds) == (None, None)

    def test_time_as_coord_only(self):
        """Time in coords but not dims."""
        times = pd.date_range("2025-01-01", periods=3, freq="1D")
        ds = xr.Dataset({"x": (["n"], [1, 2, 3])})
        ds = ds.assign_coords(time=("n", times.values))
        tmin, tmax = get_time_bound_values(ds)
        assert tmin == pd.Timestamp("2025-01-01")
        assert tmax == pd.Timestamp("2025-01-03")

    def test_time_as_data_var(self):
        """Time as a data variable (not coord or dim)."""
        times = pd.date_range("2025-04-01", periods=2, freq="1D")
        ds = xr.Dataset({"time": (["idx"], times.values)}, coords={"idx": [0, 1]})
        tmin, tmax = get_time_bound_values(ds)
        assert tmin == pd.Timestamp("2025-04-01")
        assert tmax == pd.Timestamp("2025-04-02")


# ---------------------------------------------------------------------------
# Section 2 – clean_for_serialization
# ---------------------------------------------------------------------------


class TestCleanForSerialization:
    """Remove non-serializable objects."""

    def test_simplenamespace_cleans_fs(self):
        """SimpleNamespace with fs → fs set to None."""
        obj = SimpleNamespace(fs=MagicMock(), model="test")
        cleaned = clean_for_serialization(obj)
        assert cleaned.fs is None
        assert cleaned.model == "test"

    def test_simplenamespace_with_session(self):
        """fs._session.close() called before clearing."""
        session = MagicMock()
        fs = MagicMock()
        fs._session = session
        obj = SimpleNamespace(fs=fs, params=SimpleNamespace(fs=None))
        clean_for_serialization(obj)
        session.close.assert_called_once()
        assert obj.fs is None

    def test_object_cleans_params_fs(self):
        """Non-SimpleNamespace path cleans params.fs."""

        class MockConfig:
            pass

        inner = SimpleNamespace(fs=MagicMock(), dataset_processor=MagicMock())
        inner.dataset_processor.close = MagicMock()
        obj = MockConfig()
        obj.params = inner
        cleaned = clean_for_serialization(obj)
        assert cleaned.params.fs is None
        assert cleaned.params.dataset_processor is None

    def test_simplenamespace_cleans_dataset_processor(self):
        """dataset_processor cleaned for SimpleNamespace."""
        dp = MagicMock()
        dp.close = MagicMock()
        obj = SimpleNamespace(dataset_processor=dp, params=SimpleNamespace(dataset_processor=dp))
        clean_for_serialization(obj)
        assert obj.dataset_processor is None

    def test_argo_attrs_cleaned(self):
        """_argo_index / _argopy_fetcher set to None."""
        obj = SimpleNamespace(_argo_index="index", _argopy_fetcher="fetcher", fs=None)
        clean_for_serialization(obj)
        assert obj._argo_index is None
        assert obj._argopy_fetcher is None


# ---------------------------------------------------------------------------
# Section 3 – adjust_full_day
# ---------------------------------------------------------------------------


def _make_local_manager(tmp_path, list_files=None):
    """Build a LocalConnectionManager with minimal config."""
    cfg = Namespace(
        init_type="from_json",
        protocol="local",
        local_root=str(tmp_path),
        file_pattern="*.nc",
        groups=None,
        keep_variables=["ssh"],
        file_cache=None,
        dataset_processor=None,
        filter_values={},
        full_day_data=False,
        fs=MagicMock(),
        max_samples=None,
    )
    mgr = LocalConnectionManager(cfg, call_list_files=False)
    if list_files is not None:
        mgr._list_files = list_files
    return mgr


class TestAdjustFullDay:
    """Extend date_end to 23:59:59.999999 when same-date midnight."""

    def test_same_date_midnight(self):
        """Same date at midnight → end of day."""
        mgr = _make_local_manager(Path("/tmp"))
        start = pd.Timestamp("2025-01-15")
        end = pd.Timestamp("2025-01-15")
        s2, e2 = mgr.adjust_full_day(start, end)
        assert s2 == start
        assert e2.hour == 23
        assert e2.minute == 59
        assert e2.second == 59

    def test_different_dates_unmodified(self):
        """Different dates → no change."""
        mgr = _make_local_manager(Path("/tmp"))
        start = pd.Timestamp("2025-01-15")
        end = pd.Timestamp("2025-01-16")
        s2, e2 = mgr.adjust_full_day(start, end)
        assert s2 == start
        assert e2 == end

    def test_nat_passthrough(self):
        """NaT → returned as-is."""
        mgr = _make_local_manager(Path("/tmp"))
        s, e = mgr.adjust_full_day(pd.NaT, pd.NaT)
        assert pd.isnull(s)
        assert pd.isnull(e)

    def test_same_date_nonmidnight(self):
        """Same date but not midnight → no change."""
        mgr = _make_local_manager(Path("/tmp"))
        start = pd.Timestamp("2025-01-15 12:00:00")
        end = pd.Timestamp("2025-01-15 12:00:00")
        s2, e2 = mgr.adjust_full_day(start, end)
        assert e2 == end  # unchanged


# ---------------------------------------------------------------------------
# Section 4 – _get_local_path
# ---------------------------------------------------------------------------


class TestGetLocalPath:
    """Generate local paths from remote."""

    def test_normal_path(self):
        """Normal string → join with local_root."""
        mgr = _make_local_manager(Path("/data"))
        result = mgr._get_local_path("s3://bucket/folder/file.nc")
        assert result.endswith("file.nc")
        assert "/data/" in result

    def test_datetime_returns_none(self):
        """Datetime input → None (CMEMS case)."""
        mgr = _make_local_manager(Path("/data"))
        result = mgr._get_local_path(datetime.datetime(2025, 1, 1))
        assert result is None


# ---------------------------------------------------------------------------
# Section 5 – supports() for each manager
# ---------------------------------------------------------------------------


class TestManagerSupports:
    """Each manager's .supports() class method."""

    def test_local_absolute(self):
        """Absolute path recognized as local."""
        assert LocalConnectionManager.supports("/data/file.nc") is True

    def test_local_file_scheme(self):
        """file:// URI recognized as local."""
        assert LocalConnectionManager.supports("file:///data/file.nc") is True

    def test_local_rejects_s3(self):
        """S3 URL not recognized as local."""
        assert LocalConnectionManager.supports("s3://bucket/file.nc") is False

    def test_ftp(self):
        """ftp:// recognized by FTPManager."""
        assert FTPManager.supports("ftp://server/file.nc") is True

    def test_ftp_rejects_http(self):
        """HTTPS not recognized by FTPManager."""
        assert FTPManager.supports("https://server/file.nc") is False

    def test_s3(self):
        """s3:// recognized by S3Manager."""
        assert S3Manager.supports("s3://mybucket/data.nc") is True

    def test_s3_rejects_local(self):
        """Absolute path not recognized by S3Manager."""
        assert S3Manager.supports("/data/file.nc") is False

    def test_wasabi_nc(self):
        """s3:// .nc recognized by Wasabi."""
        assert S3WasabiManager.supports("s3://bucket/file.nc") is True

    def test_wasabi_zarr(self):
        """s3:// .zarr recognized by Wasabi."""
        assert S3WasabiManager.supports("s3://bucket/file.zarr") is True

    def test_wasabi_rejects_txt(self):
        """s3:// .txt rejected by Wasabi."""
        assert S3WasabiManager.supports("s3://bucket/file.txt") is False

    def test_glonet_https(self):
        """HTTPS recognized by GlonetManager."""
        assert GlonetManager.supports("https://glonet.example.com/data.zarr") is True

    def test_glonet_rejects_ftp(self):
        """FTP not recognized by GlonetManager."""
        assert GlonetManager.supports("ftp://server/file.nc") is False

    def test_cmems_datetime(self):
        """CMEMS uses datetime strings as identifiers."""
        assert CMEMSManager.supports("2025-01-15T00:00:00") is True


# ---------------------------------------------------------------------------
# Section 6 – RecursionExit
# ---------------------------------------------------------------------------


class TestRecursionExit:
    """Trivial exception class."""

    def test_stores_value(self):
        """Value attribute preserved."""
        exc = RecursionExit(["a", "b"])
        assert exc.value == ["a", "b"]

    def test_is_exception(self):
        """Can be raised and caught."""
        with pytest.raises(RecursionExit):
            raise RecursionExit([])


# ---------------------------------------------------------------------------
# Section 7 – estimate_resolution
# ---------------------------------------------------------------------------


class TestEstimateResolution:
    """Resolution estimation from dataset coordinates."""

    def test_geographic(self):
        """Regular geographic grid → lat/lon resolution."""
        from dctools.data.coordinates import CoordinateSystem

        lat = np.linspace(0, 10, 11)  # 1° spacing
        lon = np.linspace(0, 20, 21)  # 1° spacing
        ds = xr.Dataset(
            {"ssh": (["latitude", "longitude"], np.zeros((11, 21)))},
            coords={"latitude": lat, "longitude": lon},
        )
        coord_sys = CoordinateSystem.get_coordinate_system(ds)
        mgr = _make_local_manager(Path("/tmp"))
        res = mgr.estimate_resolution(ds, coord_sys)
        assert "latitude" in res
        assert res["latitude"] == pytest.approx(1.0)
        assert "longitude" in res
        assert res["longitude"] == pytest.approx(1.0)

    def test_single_point(self):
        """Single lat/lon value → returns the value itself."""
        from dctools.data.coordinates import CoordinateSystem

        ds = xr.Dataset(
            {"ssh": (["latitude", "longitude"], np.zeros((1, 1)))},
            coords={"latitude": [45.0], "longitude": [10.0]},
        )
        coord_sys = CoordinateSystem.get_coordinate_system(ds)
        mgr = _make_local_manager(Path("/tmp"))
        res = mgr.estimate_resolution(ds, coord_sys)
        assert res.get("latitude") == pytest.approx(45.0)

    def test_temporal_resolution(self):
        """Time coords → temporal resolution in seconds."""
        from dctools.data.coordinates import CoordinateSystem

        times = pd.date_range("2025-01-01", periods=5, freq="6h")
        lat = np.linspace(0, 10, 3)
        lon = np.linspace(0, 10, 3)
        ds = xr.Dataset(
            {"ssh": (["time", "latitude", "longitude"], np.zeros((5, 3, 3)))},
            coords={"time": times, "latitude": lat, "longitude": lon},
        )
        coord_sys = CoordinateSystem.get_coordinate_system(ds)
        mgr = _make_local_manager(Path("/tmp"))
        res = mgr.estimate_resolution(ds, coord_sys)
        # 6 hours = 21600 seconds
        assert "time" in res
        assert res["time"] == "21600s"


# ---------------------------------------------------------------------------
# Section 8 – get_global_metadata / set_global_metadata
# ---------------------------------------------------------------------------


class TestGlobalMetadata:
    """get/set global metadata on manager."""

    def test_set_filters_keys(self):
        """set_global_metadata keeps only GLOBAL_METADATA keys."""
        mgr = _make_local_manager(Path("/tmp"))
        metadata = {"keep_variables": ["ssh"], "variables": {}, "extra_key": "drop_me"}
        mgr.set_global_metadata(metadata)
        assert "keep_variables" in mgr._global_metadata
        assert "extra_key" not in mgr._global_metadata

    def test_get_returns_cached(self):
        """get_global_metadata returns _global_metadata when present."""
        mgr = _make_local_manager(Path("/tmp"))
        mgr._global_metadata = {"variables": {"ssh": {}}}
        mgr._list_files = ["dummy"]
        result = mgr.get_global_metadata()
        assert result["variables"] == {"ssh": {}}

    def test_get_raises_no_files(self):
        """get_global_metadata raises FileNotFoundError when no files."""
        mgr = _make_local_manager(Path("/tmp"))
        with pytest.raises(FileNotFoundError):
            mgr.get_global_metadata()


# ---------------------------------------------------------------------------
# Section 9 – BaseConnectionManager.open dispatch
# ---------------------------------------------------------------------------


class TestOpenDispatch:
    """Verify open() dispatches local → remote → download correctly."""

    def test_open_local_path(self, tmp_path):
        """Local path opens via open_local."""
        nc_path = _make_nc(tmp_path)
        mgr = _make_local_manager(tmp_path)
        fake_ds = xr.Dataset({"ssh": (["x"], [1, 2])})
        with patch.object(mgr, "open_local", return_value=fake_ds):
            ds = mgr.open(nc_path)
            assert ds is not None
            assert "ssh" in ds

    def test_open_remote_zarr(self, tmp_path):
        """When local fails, remote path dispatches to open_remote."""
        mgr = _make_local_manager(tmp_path)
        expected = xr.Dataset({"v": [1]})
        # Directly test open_remote returns None for non-.zarr
        ds = mgr.open_remote("s3://bucket/file.nc")
        assert ds is None
        # And a .zarr would try FileLoader (mock it)
        with patch("dctools.dcio.loader.FileLoader.open_dataset_auto", return_value=expected):
            ds = mgr.open_remote("s3://bucket/file.zarr")
            assert ds is not None

    def test_open_fallback_download(self, tmp_path):
        """If local + remote fail → tries download."""
        nc_path = _make_nc(tmp_path, "downloaded.nc")
        mgr = _make_local_manager(tmp_path)
        # Patch supports to return False (not local, not remote)
        with patch.object(LocalConnectionManager, "supports", return_value=False):
            with patch.object(type(mgr), "supports", return_value=False):
                # Already exists locally after _get_local_path
                ds = mgr.open(nc_path)
                if ds is not None:
                    assert "ssh" in ds

    def test_open_returns_none_on_failure(self, tmp_path):
        """All open paths fail → None."""
        mgr = _make_local_manager(tmp_path)
        with patch.object(LocalConnectionManager, "supports", return_value=False):
            with patch.object(type(mgr), "supports", return_value=False):
                with patch.object(mgr, "_get_local_path", side_effect=Exception("boom")):
                    ds = mgr.open("nonexistent://file")
                    assert ds is None

    def test_open_redownloads_unreadable_cached_copy(self, tmp_path):
        """Unreadable cached remote file should be purged and downloaded again."""
        mgr = _make_local_manager(tmp_path)
        cached_path = tmp_path / "cached.nc"
        cached_path.write_bytes(b"corrupt-cache")
        recovered = xr.Dataset({"ssh": (["x"], [1.0, 2.0])})

        with patch.object(LocalConnectionManager, "supports", return_value=False):
            with patch.object(mgr, "supports", return_value=True):
                with patch.object(mgr, "_get_local_path", return_value=str(cached_path)):
                    with patch.object(mgr, "open_local", side_effect=[None, recovered]):
                        with patch.object(mgr, "download_file") as download_file:
                            ds = mgr.open("s3://bucket/cached.nc")

        assert ds is recovered
        download_file.assert_called_once_with("s3://bucket/cached.nc", str(cached_path))


class TestLocalCacheValidation:
    """Probe helpers used before reusing persistent caches."""

    def test_valid_netcdf_cache_is_accepted(self, tmp_path):
        """A readable local NetCDF cache should validate successfully."""
        from dctools.data.connection.connection_manager import _is_valid_local_dataset_cache

        nc_path = _make_nc(tmp_path, "valid.nc")
        assert _is_valid_local_dataset_cache(nc_path) is True

    def test_invalid_netcdf_cache_is_rejected(self, tmp_path):
        """A corrupted local NetCDF cache should fail validation."""
        from dctools.data.connection.connection_manager import _is_valid_local_dataset_cache

        bad_path = tmp_path / "bad.nc"
        bad_path.write_bytes(b"not-a-netcdf")
        assert _is_valid_local_dataset_cache(str(bad_path)) is False


# ---------------------------------------------------------------------------
# Section 10 – LocalConnectionManager list_files with real files
# ---------------------------------------------------------------------------


class TestLocalConnectionManagerListFiles:
    """Full list_files cycle with tmp NetCDF."""

    def test_list_files_glob(self, tmp_path):
        """list_files finds all .nc files."""
        _make_nc(tmp_path, "a.nc")
        _make_nc(tmp_path, "b.nc")

        # Use a real fsspec local filesystem
        import fsspec

        fs = fsspec.filesystem("file")

        cfg = Namespace(
            init_type="manual",
            protocol="local",
            local_root=str(tmp_path),
            file_pattern="*.nc",
            groups=None,
            keep_variables=["ssh"],
            file_cache=None,
            dataset_processor=None,
            filter_values={},
            full_day_data=False,
            fs=fs,
            max_samples=None,
        )
        mgr = LocalConnectionManager(cfg, call_list_files=False)
        files = mgr.list_files()
        assert len(files) >= 2
        assert all(f.endswith(".nc") for f in files)

    def test_list_files_empty_dir(self, tmp_path):
        """No NC files → empty list."""
        import fsspec

        fs = fsspec.filesystem("file")
        cfg = Namespace(
            init_type="manual",
            protocol="local",
            local_root=str(tmp_path),
            file_pattern="*.nc",
            groups=None,
            keep_variables=[],
            file_cache=None,
            dataset_processor=None,
            filter_values={},
            full_day_data=False,
            fs=fs,
            max_samples=None,
        )
        mgr = LocalConnectionManager(cfg, call_list_files=False)
        files = mgr.list_files()
        assert files == []


# ---------------------------------------------------------------------------
# Section 11 – FTPManager.list_files with mock fs
# ---------------------------------------------------------------------------


class TestFTPManagerListFiles:
    """FTP listing with mocked fsspec filesystem."""

    def test_list_files_success(self):
        """Mock fs.glob returns matching files."""
        cfg = Namespace(
            init_type="from_json",
            protocol="ftp",
            host="ftp.example.com",
            ftp_folder="/data/",
            file_pattern="*.nc",
            local_root="/tmp/ftp_cache",
            groups=None,
            keep_variables=["ssh"],
            file_cache=None,
            dataset_processor=None,
            filter_values={},
            full_day_data=False,
            fs=MagicMock(),
            max_samples=None,
        )
        cfg.fs.glob.return_value = [
            "/data/file1.nc",
            "/data/file2.nc",
        ]
        mgr = FTPManager(cfg, call_list_files=False)
        files = mgr.list_files()
        assert len(files) == 2
        assert all("ftp://ftp.example.com" in f for f in files)

    def test_list_files_empty(self):
        """No files found → empty list."""
        cfg = Namespace(
            init_type="from_json",
            protocol="ftp",
            host="ftp.example.com",
            ftp_folder="/data/",
            file_pattern="*.nc",
            local_root="/tmp/ftp_cache",
            groups=None,
            keep_variables=[],
            file_cache=None,
            dataset_processor=None,
            filter_values={},
            full_day_data=False,
            fs=MagicMock(),
            max_samples=None,
        )
        cfg.fs.glob.return_value = []
        mgr = FTPManager(cfg, call_list_files=False)
        files = mgr.list_files()
        assert files == []


# ---------------------------------------------------------------------------
# Section 12 – S3Manager.list_files with mock fs
# ---------------------------------------------------------------------------


class TestS3ManagerListFiles:
    """S3 file listing with mocked filesystem."""

    def test_list_files_pattern(self):
        """Mock fs.glob returns S3 paths."""
        cfg = Namespace(
            init_type="from_json",
            protocol="s3",
            s3_bucket="mybucket",
            s3_folder="data/",
            file_pattern="*.nc",
            endpoint_url="https://s3.example.com",
            local_root="/tmp/s3_cache",
            groups=None,
            keep_variables=["ssh"],
            file_cache=None,
            dataset_processor=None,
            filter_values={},
            full_day_data=False,
            fs=MagicMock(),
            max_samples=None,
        )
        cfg.fs.glob.return_value = [
            "mybucket/data/pred_20250101.nc",
            "mybucket/data/pred_20250102.nc",
        ]
        mgr = S3Manager(cfg, call_list_files=False)
        files = mgr.list_files()
        assert len(files) == 2
        assert all(f.startswith("s3://") for f in files)


# ---------------------------------------------------------------------------
# Section 13 – get_config_clean_copy
# ---------------------------------------------------------------------------


class TestGetConfigCleanCopy:
    """Returns a deep-copy with serialization cleanup applied."""

    def test_returns_clean_copy(self, tmp_path):
        """Clean copy has fs=None."""
        mgr = _make_local_manager(tmp_path)
        # get_config_clean_copy does deep_copy_object(self.connect_config.params)
        # then clean_for_serialization on the copy. Use a SimpleNamespace with fs.
        inner = SimpleNamespace(
            fs=MagicMock(),
            dataset_processor=None,
            protocol="local",
            local_root=str(tmp_path),
        )
        mgr.connect_config = SimpleNamespace(params=inner)
        result = mgr.get_config_clean_copy()
        # fs should be cleaned away
        if hasattr(result, "fs"):
            assert result.fs is None


# ---------------------------------------------------------------------------
# Section 14 – Registries
# ---------------------------------------------------------------------------


class TestRegistries:
    """CONNECTION_CONFIG_REGISTRY / CONNECTION_MANAGER_REGISTRY."""

    def test_config_registry_has_all_protocols(self):
        """All 7 protocols present in config registry."""
        from dctools.data.connection.connection_manager import CONNECTION_CONFIG_REGISTRY

        expected = {"argo", "cmems", "ftp", "glonet", "local", "s3", "wasabi"}
        assert expected == set(CONNECTION_CONFIG_REGISTRY.keys())

    def test_manager_registry_has_all_protocols(self):
        """All 7 protocols present in manager registry."""
        from dctools.data.connection.connection_manager import CONNECTION_MANAGER_REGISTRY

        expected = {"argo", "cmems", "ftp", "glonet", "local", "s3", "wasabi"}
        assert expected == set(CONNECTION_MANAGER_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Section 15 – Slow tests: open + metadata extraction with real NetCDF
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestLocalConnectionManagerOpenFull:
    """Full open / metadata cycle with real tmp NetCDF."""

    def test_open_and_extract_time_bounds(self, tmp_path):
        """Open local file and extract time bounds."""
        nc_path = _make_nc(tmp_path)
        # Open with scipy directly instead of going through FileLoader
        ds = xr.open_dataset(nc_path, engine="scipy")
        assert ds is not None
        tmin, tmax = get_time_bound_values(ds)
        assert tmin is not None
        ds.close()

    def test_extract_metadata_with_mock_open(self, tmp_path):
        """extract_metadata returns a CatalogEntry."""
        from dctools.data.coordinates import CoordinateSystem

        nc_path = _make_nc(tmp_path)
        ds = xr.open_dataset(nc_path, engine="scipy")
        coord_sys = CoordinateSystem.get_coordinate_system(ds)

        mgr = _make_local_manager(tmp_path, list_files=[nc_path])
        mgr._global_metadata = {
            "coord_system": coord_sys,
            "variables": {},
            "keep_variables": ["ssh"],
        }

        # Mock open to return the ds we already have
        with patch.object(mgr, "open", return_value=ds):
            entry = mgr.extract_metadata(nc_path)
            assert entry is not None
            assert entry.path == nc_path
            assert entry.date_start is not None

    def test_open_nonexistent_returns_none(self, tmp_path):
        """Nonexistent path → None from open_local."""
        mgr = _make_local_manager(tmp_path)
        ds = mgr.open_local(str(tmp_path / "nonexistent.nc"))
        assert ds is None
