"""Unit tests for lightweight helper functions in dataloader."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from dctools.data.connection.connection_manager import ArgoManager
from dctools.data.datasets.dataloader import (
    _build_nan_mask,
    _drop_nan_points,
    _nan_mask_numpy,
    add_coords_as_dims,
    add_time_dim,
    filter_by_time,
    preprocess_argo_profiles,
    swath_to_points,
)


def test_add_coords_as_dims_promotes_constant_n_points_coordinate():
    """Promote constant N_POINTS coordinate into singleton dimension and broadcast vars."""
    ds = xr.Dataset(
        {"TEMP": ("N_POINTS", np.array([1.0, 2.0], dtype=np.float32))},
        coords={"LATITUDE": ("N_POINTS", np.array([42.0, 42.0], dtype=np.float32))},
    )

    out = add_coords_as_dims(ds, coords=("LATITUDE",))

    assert "LATITUDE" in out.dims
    assert out.sizes["LATITUDE"] == 1
    assert "LATITUDE" in out["TEMP"].dims


def test_swath_to_points_stacks_swath_dims_into_n_points():
    """Swath dataset should be flattened into n_points dimension."""
    ds = xr.Dataset(
        {
            "ssh": (
                ("num_lines", "num_pixels"),
                np.arange(6, dtype=np.float32).reshape(2, 3),
            )
        },
        coords={
            "LATITUDE": (("num_lines", "num_pixels"), np.ones((2, 3), dtype=np.float32)),
            "LONGITUDE": (("num_lines", "num_pixels"), np.ones((2, 3), dtype=np.float32)),
        },
    )

    out = swath_to_points(ds, coords_to_keep=["LATITUDE", "LONGITUDE"], n_points_dim="n_points")

    assert "n_points" in out.dims
    assert out.sizes["n_points"] == 6


def test_add_time_dim_uses_metadata_mid_time_when_missing_time_coord():
    """When time is missing, current fallback path raises a coordinate/dimension conflict."""
    ds = xr.Dataset(
        {"ssh": ("N_POINTS", np.array([1.0, 2.0], dtype=np.float32))},
        coords={"N_POINTS": np.array([0, 1])},
    )
    meta = pd.DataFrame(
        {
            "date_start": [pd.Timestamp("2024-01-01")],
            "date_end": [pd.Timestamp("2024-01-03")],
        }
    )

    try:
        add_time_dim(ds, meta, n_points_dim="N_POINTS", time_coord=None, idx=0)
    except ValueError as exc:
        assert "time already exists" in str(exc)
    else:
        raise AssertionError("Expected ValueError for missing-time fallback path")


def test_filter_by_time_keeps_overlapping_intervals():
    """Filter should keep rows whose [date_start, date_end] overlaps target interval."""
    df = pd.DataFrame(
        {
            "date_start": ["2024-01-01", "2024-01-05", "2024-01-10"],
            "date_end": ["2024-01-03", "2024-01-08", "2024-01-11"],
        }
    )

    out = filter_by_time(df, pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-06"))

    assert len(out) == 2


def test_preprocess_argo_profiles_requires_argo_manager_binding():
    """Fallback preprocess should return None when open_func is not ArgoManager-bound."""
    out = preprocess_argo_profiles(
        profile_sources=["2024_01"],
        open_func=lambda *_args, **_kwargs: xr.Dataset(),
        alias="argo_profiles",
        time_bounds=(pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")),
        depth_levels=[0.0, 10.0],
    )
    assert out is None


def test_preprocess_argo_profiles_renames_obs_dimension_to_n_points():
    """Kerchunk fallback should rename obs dimension to N_POINTS when needed."""

    class _FakeArgo(ArgoManager):
        def open(self, path, *args, **kwargs):
            del path, args, kwargs
            return xr.Dataset(
                {"TEMP": ("obs", np.array([10.0, 11.0], dtype=np.float32))},
                coords={"obs": np.array([0, 1])},
            )

    mgr = object.__new__(_FakeArgo)
    out = preprocess_argo_profiles(
        profile_sources=["2024_01"],
        open_func=mgr.open,
        alias="argo_profiles",
        time_bounds=(pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")),
        depth_levels=[0.0, 10.0],
        n_points_dim="N_POINTS",
    )

    assert out is not None
    assert "N_POINTS" in out.dims
    assert "obs" not in out.dims


def test_drop_nan_points_removes_fully_nan_points():
    """Drop helper should remove points that are NaN across all data variables."""
    ds = xr.Dataset(
        {
            "A": ("N_POINTS", np.array([1.0, np.nan, 3.0], dtype=np.float32)),
            "B": ("N_POINTS", np.array([np.nan, np.nan, 2.0], dtype=np.float32)),
        }
    )

    out = _drop_nan_points(ds, "N_POINTS")
    assert out.sizes["N_POINTS"] == 2


def test_nan_mask_numpy_and_lazy_builder_return_expected_mask():
    """NaN-mask helpers should mark points with at least one finite data value."""
    ds = xr.Dataset(
        {
            "A": ("N_POINTS", np.array([1.0, np.nan, 3.0], dtype=np.float32)),
            "B": ("N_POINTS", np.array([np.nan, np.nan, 2.0], dtype=np.float32)),
        }
    )

    mask_np = _nan_mask_numpy(ds, "N_POINTS")
    mask_lazy = _build_nan_mask(ds, "N_POINTS")

    assert mask_np is not None
    assert mask_lazy is not None
    np.testing.assert_array_equal(mask_np, np.array([True, False, True]))
    np.testing.assert_array_equal(mask_lazy, np.array([True, False, True]))
