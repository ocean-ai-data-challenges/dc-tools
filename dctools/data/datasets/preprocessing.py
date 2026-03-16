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

    # Identify swath dimensions: use drop_coords (known swath dim names)
    # intersected with actual dataset dims.  Fall back to "all non-time,
    # non-n_points dims" only when no drop_coords dim is present.
    _known_swath = [d for d in drop_coords if d in ds.dims]
    if _known_swath:
        swath_dims = _known_swath
    else:
        swath_dims = [d for d in ds.dims if d not in ("time", n_points_dim)]
    if not swath_dims:
        raise ValueError("No swath dims found (already 1D or unexpected format).")

    # -- R4: Reshape WITHOUT MultiIndex --------------------------------
    # ds.stack(n_points=swath_dims) creates a pd.MultiIndex of tuples
    # (~50 MB per file for SWOT) only to be immediately replaced with an
    # integer range.  Instead, compute total n_points and reshape each
    # variable directly — no MultiIndex ever created.
    import dask.array as da

    n_pts = 1
    for d in swath_dims:
        n_pts *= ds.sizes[d]

    # Save important Coordinates before removal
    coords_to_reassign: Dict[Any, Any] = {}
    for coord in coords_to_keep:
        if coord in ds.coords:
            arr = ds.coords[coord]
            if set(arr.dims) <= set(swath_dims):
                # Coordinate spans some/all swath dims.
                # If it covers ALL swath dims, just flatten.
                # If it covers a SUBSET, broadcast first then flatten.
                if set(arr.dims) == set(swath_dims):
                    raw = arr.data
                    if hasattr(raw, 'reshape'):
                        coords_to_reassign[coord] = raw.reshape(-1)
                    else:
                        coords_to_reassign[coord] = np.asarray(arr.values).reshape(-1)
                else:
                    # Partial coverage: e.g. time(num_lines,) needs
                    # repeating across num_pixels to match n_pts.
                    raw = arr.data
                    if isinstance(raw, da.Array):
                        bcast = da.broadcast_to(
                            raw.reshape(
                                tuple(ds.sizes[d] if d in arr.dims else 1 for d in swath_dims)
                            ),
                            tuple(ds.sizes[d] for d in swath_dims),
                        )
                        coords_to_reassign[coord] = bcast.reshape(-1)
                    else:
                        raw_np = np.asarray(arr.values)
                        bcast = np.broadcast_to(
                            raw_np.reshape(
                                tuple(ds.sizes[d] if d in arr.dims else 1 for d in swath_dims)
                            ),
                            tuple(ds.sizes[d] for d in swath_dims),
                        )
                        coords_to_reassign[coord] = bcast.reshape(-1).copy()
            elif n_points_dim in arr.dims:
                coords_to_reassign[coord] = arr.data

    # Helper: broadcast a variable whose dims are a strict subset of
    # swath_dims to the full (num_lines, num_pixels, ...) shape, then
    # flatten.  E.g. time(num_lines,) → repeat across num_pixels.
    def _broadcast_and_flatten(var_data, var_dims):
        """Broadcast partial-swath var to full swath shape and flatten to 1D."""
        target_shape = tuple(ds.sizes[d] for d in swath_dims)
        inter_shape = tuple(
            ds.sizes[d] if d in var_dims else 1 for d in swath_dims
        )
        if isinstance(var_data, da.Array):
            return da.broadcast_to(var_data.reshape(inter_shape), target_shape).reshape(-1)
        else:
            arr_np = np.asarray(var_data)
            return np.broadcast_to(arr_np.reshape(inter_shape), target_shape).reshape(-1).copy()

    # Build new dataset with all variables reshaped to (n_points,)
    # Must handle BOTH data_vars AND non-index coordinates that span
    # swath dims (e.g. latitude, longitude are coords in SWOT zarr).
    new_vars: Dict[str, Any] = {}
    new_coords: Dict[str, Any] = {}

    # Collect all variables + non-index coordinates with swath dims
    _all_items: Dict[str, xr.DataArray] = {}
    _is_coord: set = set()
    for vname in ds.data_vars:
        _all_items[vname] = ds[vname]
    for cname in ds.coords:
        if cname not in ds.dims and cname not in _all_items:
            # Non-dimension coordinate (e.g. latitude, longitude, time)
            _all_items[cname] = ds.coords[cname]
            _is_coord.add(cname)

    for vname, var in _all_items.items():
        # Skip coords that will be re-attached via coords_to_reassign
        if vname in coords_to_reassign:
            continue
        _target = new_coords if vname in _is_coord else new_vars
        var_swath_dims = set(var.dims) & set(swath_dims)
        if var_swath_dims:
            other_dims = [d for d in var.dims if d not in swath_dims]
            if not other_dims:
                # Pure swath variable → flatten (with broadcast if partial)
                if var_swath_dims == set(swath_dims):
                    raw = var.data
                    if isinstance(raw, da.Array):
                        _target[vname] = (n_points_dim, raw.reshape(-1))
                    else:
                        _target[vname] = (n_points_dim, np.asarray(raw).reshape(-1))
                else:
                    _target[vname] = (n_points_dim, _broadcast_and_flatten(var.data, set(var.dims)))
            else:
                # Variable has swath dims + extra dims → reshape swath part.
                # Transpose so that other_dims come first, then swath_dims,
                # before flattening the swath portion into n_points.
                desired_order = other_dims + swath_dims
                if list(var.dims) != desired_order:
                    var = var.transpose(*desired_order)
                raw = var.data
                new_shape = tuple(ds.sizes[d] for d in other_dims) + (n_pts,)
                if isinstance(raw, da.Array):
                    _target[vname] = (tuple(other_dims) + (n_points_dim,), raw.reshape(new_shape))
                else:
                    _target[vname] = (tuple(other_dims) + (n_points_dim,), np.asarray(raw).reshape(new_shape))
        else:
            # Variable doesn't involve swath dims → keep as-is
            _target[vname] = var

    # Merge dimension coord + flattened non-index coords
    _all_coords = {n_points_dim: np.arange(n_pts, dtype=np.int64)}
    _all_coords.update(new_coords)

    ds_flat = xr.Dataset(
        new_vars,
        coords=_all_coords,
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

    # Ensure time is broadcast to n_points when it has fewer elements
    # (e.g. time defined per-line only, not per-pixel).
    if "time" in ds_flat.coords:
        _time_arr = ds_flat.coords["time"]
        if _time_arr.ndim == 1 and _time_arr.sizes.get(n_points_dim, 0) != n_pts:
            # time was flattened from a partial swath dim (e.g. num_lines)
            # and needs to be broadcast to the full n_points length.
            _t_vals = _time_arr.values
            if len(_t_vals) > 0 and n_pts % len(_t_vals) == 0:
                _repeat_factor = n_pts // len(_t_vals)
                ds_flat = ds_flat.assign_coords(
                    time=(n_points_dim, np.repeat(_t_vals, _repeat_factor))
                )

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
            coords_to_keep = [c for c in coords_to_keep if c is not None]
            ds = swath_to_points(
                ds,
                coords_to_keep=coords_to_keep,
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

        # Skip useless re-chunking: callers (.compute() or to_zarr)
        # will materialise the data immediately, making the dask rechunk
        # graph pure overhead.  Return the dataset as-is.

        del ds
        gc.collect()

        return ds_with_time

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
