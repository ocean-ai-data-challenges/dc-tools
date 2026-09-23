"""Offline tests for dctools-core (no network, no dctools)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from dctools_core import aliases, altimetry, storage, timebounds
from dctools_core.loader import FileLoader, _fix_nanosecond_time


def test_aliases_map_altimetry_and_model_names():
    assert aliases.get_standardized_var_name("ssha") == "ssh"
    assert aliases.get_standardized_var_name("zos") == "ssh"
    assert aliases.get_standardized_var_name("mean_topography") == "mean_dynamic_topography"
    assert aliases.get_standardized_var_name("thetao") == "temperature"
    assert aliases.get_standardized_var_name("PSAL_ADJUSTED") == "salinity"
    assert aliases.get_standardized_var_name("i_num_pixel") is None
    assert aliases.get_standardized_var_name("x", "sea_surface_height") == "ssh"
    assert "latitude" in aliases.COORD_ALIASES["lat"]


def test_parse_granule_period_gdr_and_swot():
    p = altimetry.parse_granule_period("SRL_GPN_2PfP188_0004_20241209_232245_20241210_001302.CNES.zarr")
    assert p == (pd.Timestamp("2024-12-09T23:22:45"), pd.Timestamp("2024-12-10T00:13:02"))
    p = altimetry.parse_granule_period("SWOT_L3_LR_SSH_Expert_009_020_20240111T232120_20240112T001246_v1.0.zarr")
    assert p == (pd.Timestamp("2024-01-11T23:21:20"), pd.Timestamp("2024-01-12T00:12:46"))
    assert altimetry.parse_granule_period("20240115.zarr") is None


def test_list_granules_filters_by_name(tmp_path):
    names = ["SRL_GPN_2PfP188_0004_20241209_232245_20241210_001302.CNES.zarr",
             "SRL_GPN_2PfP188_0005_20241210_001302_20241210_010319.CNES.zarr",
             "SRL_GPN_2PfP188_0006_20241211_010319_20241211_015336.CNES.zarr", "junk.zarr"]
    for n in names:
        (tmp_path / n).mkdir()
    fs = storage.make_filesystem("file")
    got = altimetry.list_granules(str(tmp_path), fs, first="2024-12-10", last="2024-12-10")
    assert [g.path.rsplit("/", 1)[-1] for g in got] == names[:2]   # first overlaps the day, third does not


def test_granules_from_catalog_dict():
    cat = {"features": [
        {"properties": {"path": "a.zarr", "date_start": "2024-01-01T00:00:00", "date_end": "2024-01-01T01:00:00"}},
        {"properties": {"path": "b.zarr", "date_start": "2024-01-03T00:00:00", "date_end": "2024-01-03T01:00:00"}},
        {"properties": {"path": "c.zarr", "date_start": None, "date_end": None}},
    ]}
    got = altimetry.granules_from_catalog(cat, first="2024-01-02", last="2024-01-05")
    assert [g.path for g in got] == ["b.zarr"]


def test_storage_transient_error_and_split_url():
    assert storage.is_transient_remote_error(Exception("An error occurred (499) ListObjectsV2"))
    assert not storage.is_transient_remote_error(KeyError("nope"))
    assert storage.split_url("s3://bucket/prefix/x.zarr") == ("s3", "bucket/prefix/x.zarr")
    assert storage.split_url("/data/x.zarr") == ("file", "/data/x.zarr")


def test_retry_remote_retries_only_transient():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise OSError("Connection reset by peer")
        return "ok"

    assert storage.retry_remote(flaky, attempts=3, base_delay=0.0) == "ok"
    with pytest.raises(KeyError):
        storage.retry_remote(lambda: (_ for _ in ()).throw(KeyError("x")), attempts=3, base_delay=0.0)


def test_time_bounds_datetime_and_sanity_fallback():
    t = pd.date_range("2024-01-01", periods=5, freq="6h")
    ds = xr.Dataset({"ssha": ("time", np.zeros(5))}, coords={"time": t})
    assert timebounds.get_time_bound_values(ds) == (t[0], t[-1])
    bad = xr.Dataset({"ssha": ("time", np.zeros(2))},
                     coords={"time": np.array(["1970-01-01", "1970-01-01"], dtype="datetime64[ns]")},
                     attrs={"time_coverage_start": "2024-01-11T23:21:20Z", "time_coverage_end": "2024-01-12T00:12:46Z"})
    assert timebounds.get_time_bound_values(bad) == (pd.Timestamp("2024-01-11T23:21:20"), pd.Timestamp("2024-01-12T00:12:46"))
    bad.attrs.clear()
    assert timebounds.get_time_bound_values(bad) == (None, None)


def test_fix_nanosecond_time_uses_units_epoch():
    raw = np.array([0.0, 1e9, 2e9], dtype="float64")
    ds = xr.Dataset({"time": ("n", raw, {"units": "nanoseconds since 2024-12-09 23:22:44"})})
    out = _fix_nanosecond_time(ds)
    assert out.time.dtype == "datetime64[ns]"
    assert pd.Timestamp(out.time.values[1]) == pd.Timestamp("2024-12-09T23:22:45")


def test_loader_opens_local_zarr_and_netcdf(tmp_path):
    t = pd.date_range("2024-01-01", periods=3)
    ds = xr.Dataset({"ssha": ("time", np.arange(3.0))}, coords={"time": t})
    ds.to_zarr(tmp_path / "a.zarr", consolidated=True)
    ds.to_netcdf(tmp_path / "a.nc")
    for name in ("a.zarr", "a.nc"):
        got = FileLoader.open_dataset_auto(str(tmp_path / name))
        assert got is not None and list(got.data_vars) == ["ssha"]
