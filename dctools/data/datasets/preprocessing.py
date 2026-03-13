#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Preprocessing helpers for observation and swath datasets.

Contains the per-file preprocessing pipeline:
- ``swath_to_points`` — flatten swath grids to 1-D point collections
- ``add_coords_as_dims`` — promote coordinates to dimensions (Argo)
- ``add_time_dim`` — ensure every dataset has a proper time coordinate
- ``filter_by_time`` — filter DataFrames on an overlapping time window
- ``preprocess_one_npoints`` — full per-file open --> preprocess pipeline
- ``preprocess_argo_profiles`` — Argo-specific fallback path
"""

import gc
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import dask
import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger

from dctools.data.connection.connection_manager import ArgoManager
from dctools.utilities.xarray_utils import filter_variables


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def add_coords_as_dims(ds: xr.Dataset, coords=("LATITUDE", "LONGITUDE")) -> xr.Dataset:
    """
    Add given coordinates as dimensions to all data variables in the dataset.

    Broadcasting them if necessary. Handles the case where coordinates exist
    only as per-point arrays (e.g., Argo profiles with N_POINTS).

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset.
    coords : tuple of str
        Names of the coordinates to promote to dimensions (if present in ds).

    Returns
    -------
    xr.Dataset
        Dataset with new dimensions.
    """
    out = ds.copy()

    for coord in coords:
        if coord not in ds:
            continue

        coord_da = ds[coord]

        # Case Argo: 1D coordinate constant over N_POINTS
        if coord_da.ndim == 1 and coord_da.dims == ("N_POINTS",):
            # Optimize: Avoid full load .to_series().unique() which is eager and RAM heavy
            # Use min/max check instead (lazy on dask)
            cmin = coord_da.min(skipna=True)
            cmax = coord_da.max(skipna=True)

            # If dask, compute small scalars only
            if hasattr(cmin.data, "compute"):
                cmin, cmax = dask.compute(cmin, cmax)
            else:
                cmin = cmin.values
                cmax = cmax.values

            # If min matches max, it's a constant value
            if cmin == cmax:
                value = cmin.item() if hasattr(cmin, "item") else cmin

                # Remove old coordinate to avoid conflict
                out = out.drop_vars(coord)

                # Add as new dimension of size 1
                out = out.expand_dims({coord: [value]})

                # Broadcast over all variables
                for v in out.data_vars:
                    out[v] = out[v].broadcast_like(out[coord])

                continue

        # General case (coords already well defined)
        out = out.assign_coords({coord: coord_da})
        for v in out.data_vars:
            if coord not in out[v].dims:
                out[v] = out[v].broadcast_like(out[coord])

    return out


# ---------------------------------------------------------------------------
# Swath --> points
# ---------------------------------------------------------------------------

def swath_to_points(
        ds: xr.Dataset,
        drop_coords: Optional[List[str]] = None,
        coords_to_keep: Optional[List[str]] = None,
        n_points_dim: str = "n_points",
    ):
    """
    Convert a swath-style Dataset into a 1D point collection along 'n_points'.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset, possibly with swath-like dimensions (e.g. num_lines, num_pixels).
    drop_missing : bool, default True
        If True, drop points where *all* variables are NaN or _FillValue.

    Returns
    -------
    xr.Dataset
        Flattened dataset with dimension 'n_points'.
        Includes time, lat, lon if present.
    """
    if drop_coords is None:
        drop_coords = ["num_lines", "num_pixels", "num_nadir"]

    if coords_to_keep is None:
        coords_to_keep = []

    if "n_points" in ds.dims:
        # Already flat
        return ds

    swath_dims = [d for d in ds.dims if d not in ("time", n_points_dim)]
    if not swath_dims:
        raise ValueError("No swath dims found (already 1D or unexpected format).")

    # Stack swath dims into 'n_points'
    ds_flat = ds.stack(n_points=swath_dims)

    # Save important Coordinates before removal
    coords_to_reassign: Dict[Any, Any] = {}
    for coord in coords_to_keep:
        if coord in ds.coords:
            arr = ds.coords[coord]
            # If the coordinate depends on swath dims, we reindex it on n_points
            if set(arr.dims) <= set(swath_dims):
                # Use .data to preserve dask arrays instead of .values (which forces compute)
                coords_to_reassign[coord] = arr.stack(n_points=swath_dims).data
            elif n_points_dim in arr.dims:
                coords_to_reassign[coord] = arr.data

    # -- Flatten the MultiIndex --> plain integer range ----------------
    # Dropping individual MultiIndex levels triggers a FutureWarning in
    # xarray >= 2024 and will become an error in later versions.
    # Instead, replace the whole MultiIndex at once: drop all level
    # sub-coordinates *and* the composite n_points, then assign a
    # fresh integer range.  This also avoids materialising large
    # per-level arrays (memory-safe for SWOT-scale data).
    if n_points_dim in ds_flat.indexes:
        _idx = ds_flat.indexes[n_points_dim]
        if isinstance(_idx, pd.MultiIndex):
            _n = ds_flat.sizes[n_points_dim]
            _to_drop = list(_idx.names) + [n_points_dim]
            ds_flat = (
                ds_flat
                .drop_vars([c for c in _to_drop if c in ds_flat.coords],
                           errors='ignore')
                .assign_coords({n_points_dim: np.arange(_n, dtype=np.int64)})
            )

    # Remove remaining orphan coordinates (e.g. num_nadir)
    for coord in drop_coords:
        if coord in ds_flat.coords:
            ds_flat = ds_flat.drop_vars(coord)

    # Re-attach important coordinates
    for coord, vals in coords_to_reassign.items():
        ds_flat = ds_flat.assign_coords({coord: (n_points_dim, vals)})

    # Reset attributes to avoid concat conflicts
    ds_flat.attrs = {}

    # -- R4: float32 reduction — 50% RAM saving on swath data ----------
    # SSH precision is ~1 cm, float32 (7 significant digits) is ample.
    for _vn in list(ds_flat.data_vars):
        if ds_flat[_vn].dtype == np.float64:
            ds_flat[_vn] = ds_flat[_vn].astype(np.float32)

    # Ensure time is broadcast to n_points
    if "time" in ds_flat.coords and ds_flat["time"].ndim < ds_flat[n_points_dim].ndim:
        pass
        # Case: time per line only -> broadcast to pixels
        # ds_flat = ds_flat.assign_coords(
        #     time=(n_points_dim, np.repeat(ds_flat["time"].values,
        #           np.prod([ds_flat.sizes[d] for d in swath_dims[1:]])))
        # )

    return ds_flat


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------

def add_time_dim(
    ds: xr.Dataset,
    input_df: pd.DataFrame,
    n_points_dim: str,
    time_coord: Optional[str],
    idx: int,
):
    """
    Ensure that dataset has a 'time' dimension compatible with swath/n-points structure.

    Covers cases:
      - No time info available (fallback: mid_time from metadata).
      - One unique time value for all points.
      - Multiple time values (per-point time).
      - Existing time coordinate.

    Args:
        ds: The xarray Dataset to process.
        input_df: The metadata pandas DataFrame containing time information (date_start, date_end).
        n_points_dim: The name of the dimension representing data points.
        time_coord: The name of the existing time coordinate in ds, if any.
        idx: Index of the current item in input_df.

    Returns:
        xr.Dataset: The dataset with a properly formatted 'time' dimension/coordinate.
    """
    if time_coord is None:
        # Fallback: use metadata mid_time
        file_info = input_df.iloc[idx]
        mid_time = file_info["date_start"] + (file_info["date_end"] - file_info["date_start"]) / 2
        ds = ds.assign_coords(time=(n_points_dim, np.full(ds.sizes[n_points_dim], mid_time)))
        if "time" not in ds.dims:
            ds = ds.expand_dims(time=[mid_time])
        return ds

    # Check if time_coord is a dask array (lazy)
    is_lazy = hasattr(time_coord, "chunks") or (
        hasattr(time_coord, "data") and hasattr(time_coord.data, "chunks")
    )

    if is_lazy:
        # Avoid loading .values and computing unique()
        # Assume per-point coordinates (safest default for n_points dims)

        try:
             # If already datetime, keep as is
             data_to_assign = time_coord

             # Warning: if time_coord has dimensions (e.g. n_points), assign with the dim
             # dims = getattr(time_coord, "dims", (n_points_dim,)) # Tuple of dimensions

             # If time_coord comes from .coords, it already has its dimensions
             # Otherwise we assume n_points_dim

             ds = ds.assign_coords(time=data_to_assign)
             return ds
        except Exception as e:
             logger.warning(f"Lazy time assignment failed, falling back to eager: {e}")
             # Fallback to eager execution below
             pass

    # Standardize time_coord to pandas datetime
    # This forces loading into memory (.values)
    # Check if already datetime64 to avoid costly pd.to_datetime conversion on huge arrays
    raw_values = getattr(time_coord, "values", time_coord)

    if hasattr(raw_values, "dtype") and np.issubdtype(raw_values.dtype, np.datetime64):
         time_values = raw_values
    else:
         try:
             # Fast path for large arrays: pd.to_datetime can be slow on large object arrays
             time_values = pd.to_datetime(raw_values)
         except Exception:
             # Fallback if errors
             time_values = pd.to_datetime(raw_values, errors='coerce')

    # Case: time per point
    # Avoid pd.unique on massive arrays if not strictly necessary
    # Check shape first
    is_scalar = (np.ndim(time_values) == 0) or (
        hasattr(time_values, "size") and time_values.size == 1)

    if n_points_dim in getattr(time_coord, "dims", []) and not is_scalar:
        # If huge array, assume unique times are many -> treat as coordinate
        # Only check uniqueness if relatively small (<100k) to save CPU/RAM
        if hasattr(time_values, "size") and time_values.size < 100_000:
            unique_times = pd.unique(np.asarray(time_values))
            is_single_time = (len(unique_times) == 1)
        else:
            is_single_time = False  # Assume variation to stay safe and fast

        if is_single_time:
            # Only one unique time -> add as scalar dimension if not already there
            unique_val = unique_times[0]
            if "time" in ds.dims or "time" in ds.coords:
                return ds  # already present
            else:
                return ds.expand_dims(time=[unique_val])
        else:
            try:
                # Per-point times: assign as coordinate
                ds = ds.assign_coords(time=(n_points_dim, time_values))
                return ds
            except Exception as e:
                logger.error(f"Error assigning time coordinates: {e}")
                return ds

    else:
        # Case: time scalar or broadcastable
        if np.ndim(time_values) == 0:
            time_val = pd.to_datetime(time_values)
        else:
            time_val = pd.to_datetime(time_values[0])

        if "time" in ds.dims:
            # Already has a time dimension -> just overwrite if needed
            ds = ds.assign_coords(time=[time_val])
            return ds
        else:
            return ds.expand_dims(time=[time_val])


def filter_by_time(df: pd.DataFrame, t0: pd.Timestamp, t1: pd.Timestamp) -> pd.DataFrame:
    """
    Filter the DataFrame to keep only entries where the interval overlaps with [t0, t1].

    The interval is defined by [date_start, date_end].


    Args:
        df (pd.DataFrame): Input DataFrame containing 'date_start' and 'date_end' columns.
        t0 (pd.Timestamp): Start of the time interval.
        t1 (pd.Timestamp): End of the time interval.

    Returns:
        pd.DataFrame: A filtered DataFrame.
    """
    df_copy = df.copy()
    # Convert to datetime if needed
    date_start = pd.to_datetime(df_copy["date_start"])
    date_end = pd.to_datetime(df_copy["date_end"])
    mask = (date_start <= t1) & (date_end >= t0)
    return df_copy[mask]


# ---------------------------------------------------------------------------
# Per-file preprocessing
# ---------------------------------------------------------------------------

def preprocess_one_npoints(
    source,
    is_swath,
    n_points_dim,
    filtered_df,
    idx,
    alias,
    open_func,
    keep_variables_list,
    target_dimensions,
    coordinates,
    time_bounds=None,
    load_to_memory=False,
):
    """Preprocess a single N-point dataset (e.g., swath or track)."""
    try:
        if alias is not None:
            ds = open_func(source, alias)
        else:
            ds = open_func(source)
        if ds is None:
            return None

        if keep_variables_list:
            ds = filter_variables(ds, keep_variables_list)

        if load_to_memory:
            try:
                ds = ds.compute()
            except Exception:
                pass

        if is_swath:
            coords_to_keep = [
                coordinates.get("time", None),
                coordinates.get("depth", None),
                coordinates.get("lat", None),
                coordinates.get("lon", None),
            ]
            coords_to_keep = list(filter(lambda x: x is not None, coords_to_keep))
            ds = swath_to_points(
                ds,
                coords_to_keep=list(coordinates.keys()),
                n_points_dim=n_points_dim,
            )

        time_name = coordinates["time"]
        if time_name in ds.variables and time_name not in ds.coords:
            ds = ds.set_coords(time_name)

        time_coord = ds.coords[time_name]

        if n_points_dim not in ds.dims and "time" in ds.dims and len(ds.dims) == 1:
            ds = ds.assign_coords({n_points_dim: ("time", np.arange(ds.sizes["time"]))})
            ds = ds.swap_dims({"time": n_points_dim})
            if time_name in ds.coords:
                time_coord = ds.coords[time_name]

        if n_points_dim not in ds.dims:
            logger.warning(f"Dataset {idx}: No points dimension found (expected '{n_points_dim}')")
            return None

        ds_with_time = add_time_dim(
            ds,
            filtered_df,
            n_points_dim=n_points_dim,
            time_coord=time_coord,
            idx=idx,
        )

        ds_interp = ds_with_time.chunk({n_points_dim: 500000})

        # Reset MultiIndex on n_points to a simple integer index.
        # After swath_to_points (xr.stack), n_points has a MultiIndex of
        # (num_lines, num_pixels) tuples.  xr.concat of N such datasets must
        # concatenate N such MultiIndexes (each ~500 K entries) even with
        # compat/join="override" — this is a pure-Python pandas operation
        # that can saturate one CPU for seconds per batch.
        # Replacing with a plain integer range lets concat proceed in
        # microseconds while all DATA variables (lat, lon, time, ssh ...)
        # remain as lazy dask arrays untouched.
        if n_points_dim in ds_interp.indexes:
            idx_obj = ds_interp.indexes[n_points_dim]
            if isinstance(idx_obj, pd.MultiIndex):
                n_pts = ds_interp.sizes[n_points_dim]
                # Drop only the MultiIndex level sub-coordinates
                # (e.g. num_lines, num_pixels) plus the composite
                # n_points coordinate itself.  lat/lon/time etc. are
                # ordinary data variables — they are NOT dropped here
                # and stay backed by their lazy dask arrays.
                sub_coords = list(idx_obj.names)  # ['num_lines', 'num_pixels']
                ds_interp = (
                    ds_interp
                    .drop_vars(sub_coords + [n_points_dim], errors='ignore')
                    .assign_coords(
                        {n_points_dim: np.arange(n_pts, dtype=np.int64)}
                    )
                )

        del ds
        del ds_with_time
        gc.collect()

        return ds_interp

    except Exception as e:
        logger.warning(f"Failed to process n_points dataset {idx}: {e}")
        traceback.print_exc()
        return None


# ---------------------------------------------------------------------------
# Argo profiles (fallback path)
# ---------------------------------------------------------------------------

def preprocess_argo_profiles(
    profile_sources: List[str],
    open_func: Callable[..., xr.Dataset],
    alias: str,
    time_bounds: Tuple[pd.Timestamp, pd.Timestamp],
    depth_levels: Union[List[float], np.ndarray],
    n_points_dim: str = "N_POINTS",
):
    """Load ARGO data through ArgoManager for a single time window.

    This is the **fallback** path used when the evaluator's shared-Zarr
    prefetch (``ArgoManager.prefetch_batch_shared_zarr``) did not run or
    failed.  The preferred pipeline is:

    1. Driver merges all batch time-windows and downloads all profiles once
       (``prefetch_batch_shared_zarr``).
    2. Workers open the shared Zarr and filter by ``time_bounds`` via
       ``searchsorted`` (fast, contiguous chunk reads).

    When this fallback IS used, it opens the ArgoManager for the requested
    window, which downloads and interpolates profiles on-demand.

    Parameters
    ----------
    profile_sources : list[str]
        Monthly catalog keys (unused in Kerchunk path — kept for API compat).
    open_func : callable
        ``ArgoManager.open`` bound method (or the ArgoManager itself).
    alias : str
        Dataset alias (``"argo_profiles"``).
    time_bounds : tuple of pd.Timestamp
        ``(start, end)`` time window.
    depth_levels : array-like
        Target depth levels for interpolation.
    n_points_dim : str
        Name of the points dimension (default ``"N_POINTS"``).

    Returns
    -------
    xr.Dataset or None
    """
    argo_manager: Optional[ArgoManager] = None
    if hasattr(open_func, "__self__") and isinstance(open_func.__self__, ArgoManager):
        argo_manager = open_func.__self__
    elif isinstance(open_func, ArgoManager):
        argo_manager = open_func

    if argo_manager is None:
        logger.error(
            "ARGO preprocessing requires ArgoManager.open bound method "
            "(Kerchunk interface)."
        )
        return None

    try:
        ds_window = argo_manager.open((time_bounds[0], time_bounds[1]))  # type: ignore[arg-type]
    except Exception as exc:
        logger.error(f"Kerchunk open_time_window failed: {exc}")
        return None

    if ds_window is None:
        logger.warning("Kerchunk open_time_window returned None")
        return None

    if "obs" in ds_window.dims and n_points_dim not in ds_window.dims:
        ds_window = ds_window.rename({"obs": n_points_dim})

    if n_points_dim not in ds_window.dims:
        logger.error(
            f"Kerchunk dataset missing '{n_points_dim}' dimension "
            f"(available: {list(ds_window.dims)})"
        )
        return None

    if ds_window.sizes.get(n_points_dim, 0) == 0:
        logger.warning("Kerchunk open_time_window returned empty dataset")
        return None

    return ds_window
