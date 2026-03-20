#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Dataloder."""

import atexit
import gc
import math
import os
import shutil
import tempfile
import traceback
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Type, Union

import dask
import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger
from oceanbench.core.distributed import DatasetProcessor

from dctools.data.connection.connection_manager import (
    ArgoManager,
    BaseConnectionManager,
    CMEMSManager,
    FTPManager,
    GlonetManager,
    LocalConnectionManager,
    S3Manager,
    S3WasabiManager,
)
from dctools.utilities.xarray_utils import filter_variables


# Dictionary mapping names to classes
CLASS_REGISTRY: Dict[str, Type[BaseConnectionManager]] = {
    "S3WasabiManager": S3WasabiManager,
    "FTPManager": FTPManager,
    "GlonetManager": GlonetManager,
    "ArgoManager": ArgoManager,
    "CMEMSManager": CMEMSManager,
    "S3Manager": S3Manager,
    "LocalConnectionManager": LocalConnectionManager,
}


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
    # Reset the MultiIndex immediately so that the stacked dimension levels
    # (e.g. num_lines, num_pixels) become plain coordinates.  Without this,
    # dropping a single MultiIndex level triggers a FutureWarning in recent
    # xarray versions and will become an error in the future.
    ds_flat = ds_flat.reset_index("n_points")

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

    # Remove orphan coordinates (unused)
    for coord in drop_coords:
        if coord in ds_flat.coords:
            ds_flat = ds_flat.drop_vars(coord)

    # Re-attach important coordinates
    for coord, vals in coords_to_reassign.items():
        ds_flat = ds_flat.assign_coords({coord: (n_points_dim, vals)})

    # Reset attributes to avoid concat conflicts
    ds_flat.attrs = {}

    # Ensure time is broadcast to n_points
    if "time" in ds_flat.coords and ds_flat["time"].ndim < ds_flat[n_points_dim].ndim:
        pass
        # Case: time per line only -> broadcast to pixels
        # ds_flat = ds_flat.assign_coords(
        #     time=(n_points_dim, np.repeat(ds_flat["time"].values,
        #           np.prod([ds_flat.sizes[d] for d in swath_dims[1:]])))
        # )

    return ds_flat


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

    # Only attempt CF numeric decoding when the data is NOT lazy.
    # Calling .values on a lazy (dask-backed) coordinate forces an eager
    # compute of potentially millions of points — bypassing the lazy path
    # below.  When is_lazy is True, xarray has already decoded the time
    # to datetime64; any further decoding is redundant and costly.
    decoded_cf_time = None
    if not is_lazy and hasattr(time_coord, "attrs"):
        _units = str(time_coord.attrs.get("units", "")).lower()
        if "since" in _units:
            decoded_cf_time = _decode_numeric_cf_time(
                getattr(time_coord, "values", time_coord),
                time_coord.attrs,
            )

    if decoded_cf_time is not None:
        time_values = decoded_cf_time
    elif is_lazy:
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
    if decoded_cf_time is not None:
         time_values = decoded_cf_time
    else:
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
            is_single_time = False # Assume variation to stay safe and fast

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

        ds_interp = ds_with_time.chunk({n_points_dim: 100000})

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


class EvaluationDataloader:
    """Class to manage loading and batching of evaluation data."""

    pred_catalog: Any
    ref_catalogs: Dict[str, Any]
    ref_aliases: List[str]
    forecast_mode: bool
    forecast_index: Optional[pd.DataFrame]
    n_days_forecast: int
    time_tolerance: Any
    keep_variables: Dict[str, List[str]]
    metadata: Dict[str, Any]
    optimize_for_parallel: bool
    min_batch_size_for_parallel: int
    pred_coords: Any
    ref_coords: Any
    pred_manager: Any
    ref_managers: Dict[str, Any]
    target_dimensions: Dict[str, Any]
    lead_time_unit: str
    file_cache: Any
    batch_size: int
    pred_alias: str
    pred_connection_params: Any
    ref_connection_params: Dict[str, Any]
    pred_transform: Any
    ref_transforms: Optional[Dict[str, Any]]

    def __init__(
        self,
        params: dict,
    ):
        """
        Initializes the dataloader for data collections.

        Args:
            params: parameter dictionary
        """
        for key, value in params.items():
            setattr(self, key, value)
        self.pred_coords = self.pred_catalog.get_global_metadata().get("coord_system", None)
        self.ref_coords = {
            ref_alias: ref_catalog.get_global_metadata().get("coord_system", None)
            for ref_alias, ref_catalog in self.ref_catalogs.items()
        }

        self.optimize_for_parallel = True
        self.min_batch_size_for_parallel = 4

    def __len__(self):
        """Return the number of batches."""
        if self.forecast_mode and self.forecast_index is not None:
            return len(self.forecast_index)
        return len(self.pred_catalog.get_dataframe())

    def __iter__(self):
        """Iterate over batches of data."""
        return self._generate_batches()

    def _find_matching_ref(self, valid_time, ref_alias):
        """Find matching reference file covering valid_time for ref_alias."""
        ref_df = self.ref_catalogs[ref_alias].get_dataframe()
        match = ref_df[(ref_df["date_start"] <= valid_time) & (ref_df["date_end"] >= valid_time)]
        if not match.empty:
            return match.iloc[0]["path"]
        return None

    def _generate_batches(self) -> Generator[List[Dict[str, Any]], None, None]:
        batch: List[Any] = []
        try:
            # Check maximum available date in reference data
            if self.forecast_index is None:
                logger.error("forecast_index is None, cannot generate batches")
                return

            for ref_alias in self.ref_aliases:
                for _, row in self.forecast_index.iterrows():
                    # Check if enough data for this forecast
                    forecast_reference_time = row["forecast_reference_time"]
                    lead_time = row["lead_time"]
                    valid_time = row["valid_time"]

                    # Calculate full forecast end (last lead time)
                    max_lead_time = self.n_days_forecast - 1  # 0-indexed
                    if (
                        hasattr(self, 'lead_time_unit')
                        and self.lead_time_unit == "hours"
                    ):
                        forecast_end_time = (
                            forecast_reference_time + pd.Timedelta(hours=max_lead_time)
                        )
                    else:
                        forecast_end_time = (
                            forecast_reference_time + pd.Timedelta(days=max_lead_time)
                        )

                    entry = {
                        "forecast_reference_time": forecast_reference_time,
                        "lead_time": lead_time,
                        "valid_time": valid_time,
                        "pred_data": row["file"],
                        "ref_data": None,
                        "ref_alias": ref_alias,
                        "pred_coords": self.pred_coords,
                        "ref_coords": self.ref_coords[ref_alias] if ref_alias else None,
                    }
                    if ref_alias:
                        ref_catalog = self.ref_catalogs[ref_alias]
                        ref_df = ref_catalog.get_dataframe()

                        # Find max available date in reference data
                        max_available_date = ref_df["date_end"].max()

                        # If forecast end > available data, skip this entry
                        if forecast_end_time > max_available_date:
                            logger.debug(
                                f"Skipping forecast starting at {forecast_reference_time}: "
                                f"forecast ends at {forecast_end_time} "
                                f"but data only available until {max_available_date}"
                            )
                            if batch:
                                yield batch
                                batch = []
                            break

                        # ref_catalog = self.ref_catalogs[ref_alias]
                        coord_system = ref_catalog.get_global_metadata().get("coord_system")
                        if coord_system:
                            is_observation = coord_system.is_observation_dataset()
                        else:
                            is_observation = False

                        if is_observation:
                            # Observation logic: filter observation catalog
                            # on forecast_index time interval
                            obs_time_interval = (valid_time, valid_time)
                            keep_vars = self.keep_variables[ref_alias]
                            rename_vars_dict = self.metadata[ref_alias]['variables_dict']
                            keep_vars = [
                                rename_vars_dict[var]
                                for var in keep_vars
                                if var in rename_vars_dict
                            ]

                            t0, t1 = obs_time_interval
                            t0 = t0 - self.time_tolerance
                            t1 = t1 + self.time_tolerance
                            time_bounds = (t0, t1)

                            entry["ref_data"] = {
                                "source": ref_catalog,
                                "keep_vars": keep_vars,
                                "target_dimensions": self.target_dimensions,
                                "metadata": self.metadata[ref_alias],
                                "time_bounds": time_bounds,
                            }
                            entry["ref_is_observation"] = True
                            entry["obs_time_interval"] = obs_time_interval
                        else:
                            # Gridded logic: associate reference file covering valid_time
                            ref_path = self._find_matching_ref(valid_time, ref_alias)
                            if ref_path is None:
                                logger.debug(f"No reference data found for valid_time {valid_time}")
                                continue
                            entry["ref_data"] = ref_path
                            entry["ref_is_observation"] = False

                    batch.append(entry)
                    # Adapt batch size according to observation/gridded type:
                    # - observation datasets (SWOT, saral ...): use obs_batch_size
                    #   to limit per-batch S3 download volume.
                    # - gridded datasets (GLORYS ...): use gridded_batch_size
                    #   to limit per-batch 3-D I/O and worker RAM pressure.
                    # - fallback: global batch_size.
                    _effective_bs = self.batch_size
                    if entry.get("ref_is_observation") and getattr(
                        self, "obs_batch_size", None
                    ):
                        _effective_bs = self.obs_batch_size  # type: ignore[attr-defined]
                    elif not entry.get("ref_is_observation") and getattr(
                        self, "gridded_batch_size", None
                    ):
                        _effective_bs = self.gridded_batch_size  # type: ignore[attr-defined]
                    if len(batch) >= _effective_bs:
                        yield batch
                        batch = []
                if batch:  # last batch of ref_alias
                    yield batch
                    batch = []
        except Exception as e:
            logger.error(f"Error generating batches: {e}")
            traceback.print_exc()


    def open_pred(self, pred_entry: str) -> xr.Dataset:
        """Open a prediction dataset."""
        pred_data: xr.Dataset = self.pred_manager.open(pred_entry, self.file_cache)
        return pred_data

    def open_ref(self, ref_entry: str, ref_alias: str) -> xr.Dataset:
        """Open a reference dataset."""
        ref_data: xr.Dataset = self.ref_managers[ref_alias].open(ref_entry, self.file_cache)
        return ref_data


def _drop_nan_points(ds: xr.Dataset, n_points_dim: str) -> xr.Dataset:
    """Drop points where ALL data variables are NaN along n_points_dim.

    SWOT swath grids stack every (num_lines, num_pixels) cell, including
    land, ice, and orbital-gap areas that are entirely fill-value.  These
    NaN-only points can represent 60-90 % of each file's size and have
    zero scientific value.  Removing them immediately after per-file
    compute() keeps the accumulator list lean before the final concat.
    """
    if n_points_dim not in ds.dims:
        return ds

    n_pts = ds.sizes[n_points_dim]
    if n_pts == 0:
        return ds

    valid_mask: Optional[np.ndarray] = None
    for vname in ds.data_vars:
        v = ds[vname]
        if n_points_dim not in v.dims:
            continue
        arr = v.values
        if arr.ndim == 1:
            finite = np.isfinite(arr)
        else:
            # Multi-dim variable: a point is valid if any element along
            # non-n_points axes is finite.
            ax0_size = arr.shape[v.dims.index(n_points_dim)]
            try:
                flat = arr.reshape(ax0_size, -1)
                finite = np.any(np.isfinite(flat), axis=1)
            except Exception:
                continue
        valid_mask = finite if valid_mask is None else (valid_mask | finite)

    if valid_mask is None or valid_mask.all():
        return ds

    n_valid = int(valid_mask.sum())
    if n_valid == 0:
        return ds  # keep caller's None-check handling

    return ds.isel({n_points_dim: valid_mask})


# ---------------------------------------------------------------------------
# Module-level temp-dir registry: cleaned on worker/process exit.
# Zarr stores written during SWOT preprocessing are registered here so that
# they are reliably removed even if an exception propagates.
# ---------------------------------------------------------------------------
_SWOT_TEMP_DIRS: List[str] = []


def _atexit_cleanup_swot_dirs() -> None:  # noqa: D401
    """Remove all temporary zarr directories created during preprocessing."""
    for _d in _SWOT_TEMP_DIRS:
        shutil.rmtree(_d, ignore_errors=True)


atexit.register(_atexit_cleanup_swot_dirs)


def _build_nan_mask(ds: xr.Dataset, n_points_dim: str) -> Optional[np.ndarray]:
    """Return a pre-computed 1-D boolean mask of *valid* (non-all-NaN) points.

    Unlike :func:`_drop_nan_points` (which works on already-computed numpy
    arrays), this function builds the mask from the **lazy** dask graph so
    that xarray never needs to materialise the full dataset in RAM.  Dask
    evaluates the mask chunk-by-chunk (peak ≈ one chunk); the returned numpy
    array is only *n_pts* booleans (≪ full data).

    Returns
    -------
    np.ndarray or None
        1-D boolean mask (True = valid point).  ``None`` when all points are
        valid (caller should skip filtering).
    """
    if n_points_dim not in ds.dims or ds.sizes.get(n_points_dim, 0) == 0:
        return None

    import dask.array as da  # local import to keep module-level deps clean

    combined_mask = None
    for vname in ds.data_vars:
        v = ds[vname]
        if n_points_dim not in v.dims:
            continue
        raw = v.data  # dask array or numpy
        n_pts_axis = v.dims.index(n_points_dim)
        other_axes = tuple(i for i in range(raw.ndim) if i != n_pts_axis)
        if isinstance(raw, da.Array):
            finite: Any = da.isfinite(raw)
            if other_axes:
                finite = da.any(finite, axis=other_axes)
        else:
            finite = np.isfinite(raw)
            if other_axes:
                finite = np.any(finite, axis=other_axes)
        combined_mask = (
            finite if combined_mask is None else (combined_mask | finite)
        )

    if combined_mask is None:
        return None

    # Compute only the small 1-D mask (cheap: n_pts × 1 byte).
    if hasattr(combined_mask, "compute"):
        mask_np: np.ndarray = combined_mask.compute()
    else:
        mask_np = np.asarray(combined_mask)

    if mask_np.all():
        return None  # all valid — caller can skip isel
    return mask_np


# ---------------------------------------------------------------------------
# Module-level helpers for process-pool based batch preprocessing.
# These must be at module scope so they are picklable by ProcessPoolExecutor.
# ---------------------------------------------------------------------------

_CF_TIME_UNIT_TO_NS: Dict[str, int] = {
    "nanosecond": 1,
    "nanoseconds": 1,
    "ns": 1,
    "microsecond": 1_000,
    "microseconds": 1_000,
    "us": 1_000,
    "millisecond": 1_000_000,
    "milliseconds": 1_000_000,
    "ms": 1_000_000,
    "second": 1_000_000_000,
    "seconds": 1_000_000_000,
    "sec": 1_000_000_000,
    "secs": 1_000_000_000,
    "s": 1_000_000_000,
    "minute": 60 * 1_000_000_000,
    "minutes": 60 * 1_000_000_000,
    "min": 60 * 1_000_000_000,
    "mins": 60 * 1_000_000_000,
    "hour": 3_600 * 1_000_000_000,
    "hours": 3_600 * 1_000_000_000,
    "hr": 3_600 * 1_000_000_000,
    "hrs": 3_600 * 1_000_000_000,
    "h": 3_600 * 1_000_000_000,
    "day": 86_400 * 1_000_000_000,
    "days": 86_400 * 1_000_000_000,
    "d": 86_400 * 1_000_000_000,
}

_STANDARD_CF_CALENDARS = {
    "",
    "standard",
    "gregorian",
    "proleptic_gregorian",
    "julian",
}


def _decode_numeric_cf_time(
    raw_values: Any,
    attrs: Optional[Dict[str, Any]],
) -> Optional[np.ndarray]:
    """Decode numeric CF-style time offsets into datetime64[ns].

    This is used as a fallback when xarray cannot decode a coordinate like
    ``nanoseconds since ...`` during ``open_zarr``. Out-of-range values are
    coerced to ``NaT`` instead of aborting the whole file open.
    """
    if not attrs:
        return None

    units_text = str(attrs.get("units", "")).strip()
    units_lower = units_text.lower()
    if "since" not in units_lower:
        return None

    since_idx = units_lower.index("since")
    unit_name = units_lower[:since_idx].strip()
    ref_text = units_text[since_idx + len("since"):].strip()
    if not unit_name or not ref_text:
        return None

    calendar = str(attrs.get("calendar", "standard")).strip().lower()
    if calendar not in _STANDARD_CF_CALENDARS:
        return None

    scale_ns = _CF_TIME_UNIT_TO_NS.get(unit_name)
    if scale_ns is None:
        return None

    arr = np.asarray(raw_values)
    if not np.issubdtype(arr.dtype, np.number):
        return None

    try:
        ref_ns = np.datetime64(pd.Timestamp(ref_text).to_datetime64(), "ns").astype("int64")
    except Exception:
        return None

    min_ns = np.iinfo(np.int64).min + 1
    max_ns = np.iinfo(np.int64).max
    min_raw = math.ceil((min_ns - int(ref_ns)) / scale_ns)
    max_raw = math.floor((max_ns - int(ref_ns)) / scale_ns)
    nat_ns = np.datetime64("NaT", "ns").astype("int64")

    flat = arr.reshape(-1)
    decoded_ns = np.full(flat.shape, nat_ns, dtype=np.int64)

    if np.issubdtype(flat.dtype, np.floating):
        valid_mask = np.isfinite(flat) & (flat >= min_raw) & (flat <= max_raw)
        if np.any(valid_mask):
            delta_ns = np.rint(flat[valid_mask] * scale_ns).astype(np.int64, copy=False)
            decoded_ns[valid_mask] = ref_ns + delta_ns
    elif np.issubdtype(flat.dtype, np.unsignedinteger):
        if max_raw >= 0:
            valid_mask = flat <= np.array(max_raw, dtype=flat.dtype)
            if np.any(valid_mask):
                delta_ns = flat[valid_mask].astype(np.int64, copy=False) * scale_ns
                decoded_ns[valid_mask] = ref_ns + delta_ns
    else:
        valid_mask = (flat >= min_raw) & (flat <= max_raw)
        if np.any(valid_mask):
            delta_ns = flat[valid_mask].astype(np.int64, copy=False) * scale_ns
            decoded_ns[valid_mask] = ref_ns + delta_ns

    return decoded_ns.view("datetime64[ns]").reshape(arr.shape)


def _is_time_like_variable(name: str, var: xr.DataArray) -> bool:
    """Return True for variables that plausibly represent time."""
    lname = str(name).lower()
    if "time" in lname or "date" in lname or "juld" in lname:
        return True

    axis = str(var.attrs.get("axis", "")).upper()
    if axis == "T":
        return True

    standard_name = str(var.attrs.get("standard_name", "")).lower()
    long_name = str(var.attrs.get("long_name", "")).lower()
    return (
        "time" in standard_name
        or "time" in long_name
        or "date" in standard_name
        or "date" in long_name
    )


def _repair_undecoded_time_variables(ds: xr.Dataset) -> Tuple[xr.Dataset, bool]:
    """Decode numeric CF time variables after ``decode_times=False`` fallback."""
    repaired = False
    coord_names = set(ds.coords)

    for name in list(ds.variables):
        var = ds[name]
        if not _is_time_like_variable(str(name), var):
            continue

        decoded = _decode_numeric_cf_time(var.values, dict(var.attrs))
        if decoded is None:
            continue

        cleaned_attrs = {
            key: value
            for key, value in dict(var.attrs).items()
            if key not in {"units", "calendar"}
        }
        decoded_da = xr.DataArray(decoded, dims=var.dims, attrs=cleaned_attrs, name=name)

        if name in coord_names:
            ds = ds.assign_coords({name: decoded_da})
        else:
            ds[name] = decoded_da
        repaired = True

    return ds, repaired


def _is_consolidated_metadata_error(exc: Exception) -> bool:
    """Return True when a Zarr open should retry without consolidated metadata."""
    message = str(exc).lower()
    return "zmetadata" in message or (
        "consolidated" in message and "metadata" in message
    )


def _is_cf_time_decode_error(exc: Exception) -> bool:
    """Return True for xarray CF time decode failures that should use a raw retry."""
    message = f"{type(exc).__name__}: {exc}".lower()
    return (
        "out of bounds nanosecond timestamp" in message
        or ("units must be one of" in message and "nanoseconds" in message)
        or "unable to decode time units" in message
    )


def _is_invalid_local_zarr_error(exc: Exception) -> bool:
    """Return True when the path clearly does not point to a readable Zarr group."""
    message = f"{type(exc).__name__}: {exc}".lower()
    return (
        "group not found at path" in message
        or "nothing found at path" in message
        or "_array_dimensions" in message
        or "_nczarr_array" in message
    )


def _is_probable_local_zarr_group(path: str) -> bool:
    """Cheap validation for local Zarr directories before xarray touches them."""
    if not path or not os.path.isdir(path):
        return False

    if any(
        os.path.exists(os.path.join(path, marker))
        for marker in (".zgroup", ".zmetadata", "zarr.json")
    ):
        return True

    return not os.path.exists(os.path.join(path, ".zarray"))


def _open_local_zarr_simple(path: str, _alias: Any = None) -> Optional[xr.Dataset]:
    """Open a local zarr file — prefer consolidated metadata.

    Module-level function (picklable for multiprocessing).
    """
    if not _is_probable_local_zarr_group(path):
        return None

    try:
        return xr.open_zarr(path, consolidated=True, chunks={})  # type: ignore[no-any-return]
    except Exception as exc:
        if _is_cf_time_decode_error(exc):
            ds = xr.open_zarr(
                path,
                consolidated=True,
                chunks={},
                decode_times=False,
            )
            ds, _ = _repair_undecoded_time_variables(ds)
            return ds

        if _is_invalid_local_zarr_error(exc):
            return None

        if _is_consolidated_metadata_error(exc):
            try:
                return xr.open_zarr(path, consolidated=False, chunks={})  # type: ignore[no-any-return]
            except Exception as fallback_exc:
                if _is_invalid_local_zarr_error(fallback_exc):
                    return None
                raise

        raise


def _nan_mask_numpy(ds: xr.Dataset, n_points_dim: str) -> Optional[np.ndarray]:
    """Build NaN mask from an already-computed (in-memory) dataset.

    Unlike :func:`_build_nan_mask` (which creates a dask graph and reads data
    lazily), this operates on numpy arrays directly.  Use it **after**
    ``.compute()`` to avoid the double-read penalty.
    """
    if n_points_dim not in ds.dims or ds.sizes.get(n_points_dim, 0) == 0:
        return None
    combined: Optional[np.ndarray] = None
    for vname in ds.data_vars:
        v = ds[vname]
        if n_points_dim not in v.dims:
            continue
        vals = v.values  # already numpy
        n_pts_axis = v.dims.index(n_points_dim)
        other_axes = tuple(i for i in range(vals.ndim) if i != n_pts_axis)
        finite = np.isfinite(vals)
        if other_axes:
            finite = np.any(finite, axis=other_axes)
        combined = finite if combined is None else (combined | finite)
    if combined is None or combined.all():
        return None
    return combined


# Shared dummy dataframe for add_time_dim fallback (SWOT files contain
# their own per-point time coordinate — this is the fallback sentinel).
_DUMMY_TIME_DF = pd.DataFrame({
    "path": [""],
    "date_start": [pd.Timestamp("2000-01-01")],
    "date_end": [pd.Timestamp("2100-01-01")],
})


def _process_file_to_zarr(args: Tuple) -> Tuple[Optional[str], int]:
    """Process one observation file → compute → NaN-filter → write mini zarr.

    Designed for ``ProcessPoolExecutor``: all arguments are picklable, each
    invocation is self-contained, and the result is just a ``(path, n_pts)``
    tuple.

    **Single-pass I/O** — the file is read/decompressed exactly once
    (the old code did it twice: once for the dask NaN mask, once for the
    final ``.compute()``).
    """
    (path, file_idx, is_swath, n_points_dim, alias,
     keep_vars, coordinates, output_dir) = args
    try:
        # Force synchronous dask scheduler — avoid any interaction with
        # a distributed Client that may have been inherited via fork().
        import dask as _dask_mod
        _dask_mod.config.set(scheduler="synchronous")

        result = preprocess_one_npoints(
            path, is_swath, n_points_dim,
            _DUMMY_TIME_DF, 0,
            alias, _open_local_zarr_simple,
            keep_vars, None, coordinates,
            None, False,
        )
        if result is None:
            return (None, 0)

        # ── Single-pass compute ─────────────────────────────────────
        # Read + decompress data ONCE into memory.
        result = result.compute(scheduler="synchronous")

        # NaN filter on in-memory numpy arrays (negligible cost).
        _nmask = _nan_mask_numpy(result, n_points_dim)
        if _nmask is not None:
            if int(_nmask.sum()) == 0:
                del result
                return (None, 0)
            result = result.isel({n_points_dim: _nmask})

        n_pts = result.sizes.get(n_points_dim, 0)
        if n_pts == 0:
            del result
            return (None, 0)

        # Write mini zarr directly from this worker process.
        zarr_path = os.path.join(output_dir, f"file_{file_idx}.zarr")
        result.to_zarr(zarr_path, mode="w")
        del result
        gc.collect()
        return (zarr_path, n_pts)
    except Exception as exc:
        logger.debug(f"Batch preproc ({alias}) file {file_idx}: {exc}")
        return (None, 0)


def concat_with_dim_delayed(
    datasets: List[xr.Dataset],
    concat_dim: str,
    sort: bool = True,
):
    """Concatenate datasets along a dimension using dask.delayed."""
    datasets_with_dim: List[Any] = []
    for i, ds in enumerate(datasets):
        if concat_dim not in ds.dims:
                ds = ds.expand_dims({concat_dim: [i]})
        datasets_with_dim.append(ds)

    result = dask.delayed(xr.concat)(
        datasets_with_dim,
        dim=concat_dim,
        coords="minimal",
        compat="override",
        join="outer"
    )
    if sort:
        result = dask.delayed(result.sortby)(concat_dim)
    return result


def concat_with_dim(
    datasets: List[xr.Dataset],
    concat_dim: str,
    sort: bool = True,
):
    """Concatenate datasets along a dimension eagerly."""
    datasets_refs: List[Any] = list(datasets)
    datasets_with_dim: List[Any] = []
    for i in range(len(datasets_refs)):
        ds = datasets_refs[i]
        # Release reference in source list immediately (memory optimization)
        datasets_refs[i] = None

        if "time" in ds.coords:
            # Check dtype without loading data
            dtype = ds.coords["time"].dtype
            if np.issubdtype(dtype, np.integer):
                pass
            elif dtype == "O":
                pass
            else:
                 pass

        if concat_dim not in ds.dims:
                ds = ds.expand_dims({concat_dim: [i]})
        datasets_with_dim.append(ds)

    # Always use compat="override" to stay fully lazy.
    # compat="no_conflicts" forces xarray to compare coordinates across all
    # datasets — for dask-backed coords (lat, lon, etc.) this triggers an
    # eager compute of every coordinate array, saturating RAM on large SWATH
    # batches.  For n_points concatenation the coordinates never conflict
    # (each dataset contributes its own disjoint slice), so "override" is safe.
    result: xr.Dataset = xr.concat(  # type: ignore[call-overload]
        datasets_with_dim, dim=concat_dim,
        coords="minimal",
        compat="override", join="outer",
    )
    if sort:
        result = result.sortby(concat_dim)
    return result

def preprocess_batch_obs_files(
    local_paths: List[str],
    alias: str,
    keep_vars: Optional[List[str]],
    coordinates: Dict[str, str],
    n_points_dim: str = "n_points",
    output_zarr_dir: Optional[str] = None,
) -> Optional[str]:
    """Preprocess all unique observation files on the driver into a single zarr.

    Eliminates redundant per-worker preprocessing when multiple tasks share
    the same observation files (typical for SWOT/swath data with wide
    time_tolerance).  Each unique file is processed exactly once:
    ``open → swath_to_points → NaN-mask → compute``.

    The resulting zarr contains all valid ocean points with a ``time``
    coordinate along *n_points_dim*, enabling per-worker time filtering.

    Parameters
    ----------
    local_paths : list of str
        Unique local file paths (post-prefetch).
    alias : str
        Dataset alias (e.g. ``"swot"``).
    keep_vars : list of str or None
        Variables to retain in the output.
    coordinates : dict
        Coordinate name mapping, must contain ``"time"`` key.
    n_points_dim : str
        Name of the points dimension (default ``"n_points"``).
    output_zarr_dir : str or None
        Directory for temp zarr files.  Created automatically when *None*.

    Returns
    -------
    str or None
        Absolute path to the shared batch zarr, or *None* on failure.
    """
    import time as _time

    if not local_paths:
        return None

    # ── File-count guard ──────────────────────────────────────────────
    # For very large observation sets (e.g. SWOT with 9 000+ files over
    # a full year), building the shared Zarr on the driver can take over
    # an hour (one mini-flush every ~12 s × 300 flushes).  Skip the
    # shared build when the number of unique files exceeds the limit;
    # workers will fall back to processing only their ~25 files each.
    _MAX_SHARED_OBS_FILES = int(
        os.environ.get("DCTOOLS_SHARED_OBS_MAX_FILES", "5000")
    )
    if len(local_paths) > _MAX_SHARED_OBS_FILES:
        logger.info(
            f"Shared batch preprocessing ({alias}): {len(local_paths)} "
            f"files exceeds limit {_MAX_SHARED_OBS_FILES} — skipping. "
            f"Workers will process files individually."
        )
        return None

    t_start = _time.perf_counter()

    if output_zarr_dir is None:
        output_zarr_dir = tempfile.mkdtemp(prefix=f"dctools_batch_{alias}_")
    os.makedirs(output_zarr_dir, exist_ok=True)
    output_zarr_path = os.path.join(output_zarr_dir, "batch_shared.zarr")

    # If a previous run already produced this zarr, reuse it.
    if os.path.isdir(output_zarr_path):
        try:
            _probe = xr.open_zarr(output_zarr_path, consolidated=False)
            if _probe.sizes.get(n_points_dim, 0) > 0:
                logger.debug(
                    f"Shared batch zarr ({alias}): reusing existing "
                    f"({_probe.sizes[n_points_dim]:,} pts)"
                )
                _probe.close()
                return output_zarr_path
            _probe.close()
        except Exception:
            pass  # stale zarr — rebuild

    # ── Probe first file to detect swath structure ──────────────────────
    # These zarr stores often lack per-variable _ARRAY_DIMENSIONS
    # attributes; dimension info lives only in the consolidated
    # .zmetadata file.  Use consolidated=True first (with fallback).
    try:
        _first = _open_local_zarr_simple(local_paths[0], alias)
        if _first is None:
            logger.error(
                f"Cannot probe first obs file for batch preproc: invalid zarr store {local_paths[0]}"
            )
            return None
        is_swath = {"num_lines", "num_pixels"}.issubset(_first.dims)
        _first.close()
        del _first
    except Exception as exc:
        logger.error(f"Cannot probe first obs file for batch preproc: {exc}")
        return None

    # ── Per-file parallel preprocessing ──────────────────────────────
    # Architecture overview (v2 — ProcessPoolExecutor):
    #   • Each unique obs file is processed by a *separate OS process*
    #     → open_zarr → swath_to_points → single-pass compute → NaN mask
    #     → write mini zarr.  True CPU parallelism (no GIL).
    #   • Single-pass I/O: data is read/decompressed ONCE per file
    #     (v1 read twice: once for the dask NaN mask, once for compute).
    #   • No mini-batch accumulation or serial flush — each process
    #     writes its own zarr directly.
    #   • Falls back to ThreadPoolExecutor if process creation fails
    #     (e.g. container restrictions, fork issues).
    _mini_zarr_paths: List[str] = []
    n_ok = 0
    n_pts_total = 0

    _cpu_count = os.cpu_count() or 4
    _env_max = int(os.environ.get("DCTOOLS_PREP_WORKERS", "0"))

    # Default: use threads, not processes.
    # ProcessPoolExecutor spawns/forks complete Python processes; each forked
    # child accumulates ~400 MB of copied pages (interpreter + xarray/zarr
    # imports) on Linux copy-on-write, so 16 processes = ~6 GB overhead
    # before processing a single SWOT file.  Threads share the parent's
    # address space and imported modules, so only the numpy arrays (a few
    # tens of MB per file) are duplicated.  The GIL is released during zarr
    # I/O, Blosc decompression, and numpy operations, giving near-equivalent
    # throughput at a fraction of the memory cost.
    # Set DCTOOLS_PREP_USE_PROCESSES=1 to restore the old process-based path
    # (useful on pure-compute workloads with no Blosc-compressed I/O).
    _use_processes = os.environ.get(
        "DCTOOLS_PREP_USE_PROCESSES", ""
    ).lower() in ("1", "true", "yes")

    # Default worker cap: 8 threads (plenty for I/O-bound zarr reads).
    # When using processes, cap at 4 to limit forked-process RAM overhead.
    _default_max = 4 if _use_processes else 8
    _MAX_PREP_WORKERS = min(
        _env_max if _env_max > 0 else min(_cpu_count, _default_max),
        len(local_paths),
    )

    _args_list = [
        (p, idx, is_swath, n_points_dim, alias,
         keep_vars, dict(coordinates) if not isinstance(coordinates, dict) else coordinates,
         output_zarr_dir)
        for idx, p in enumerate(local_paths)
    ]

    logger.info(
        f"Shared batch preprocessing ({alias}): "
        f"{len(local_paths)} unique files, "
        f"{_MAX_PREP_WORKERS} {'processes' if _use_processes else 'threads'}"
    )

    try:
        if _use_processes:
            from concurrent.futures import ProcessPoolExecutor as _FilePool
            with _FilePool(max_workers=_MAX_PREP_WORKERS) as pool:
                from concurrent.futures import as_completed as _proc_ac
                _futs = [pool.submit(_process_file_to_zarr, a) for a in _args_list]
                _fut_to_path = {f: a[0] for f, a in zip(_futs, _args_list, strict=False)}
                for fut in _proc_ac(_futs):
                    try:
                        zarr_path, n_pts = fut.result()
                    except Exception as _fe:
                        _fpath = os.path.basename(str(_fut_to_path.get(fut, "unknown")))
                        logger.warning(
                            f"Batch preproc ({alias}): skipping file {_fpath}: {_fe!r}"
                        )
                        continue
                    if zarr_path is not None:
                        _mini_zarr_paths.append(zarr_path)
                        n_ok += 1
                        n_pts_total += n_pts
        else:
            from concurrent.futures import ThreadPoolExecutor as _ThrPool
            from concurrent.futures import as_completed as _thr_ac
            with _ThrPool(max_workers=_MAX_PREP_WORKERS) as tpool:
                _futs = [tpool.submit(_process_file_to_zarr, a) for a in _args_list]
                _fut_to_path = {f: a[0] for f, a in zip(_futs, _args_list, strict=False)}
                for fut in _thr_ac(_futs):
                    try:
                        zarr_path, n_pts = fut.result()
                    except Exception as _fe:
                        _fpath = os.path.basename(str(_fut_to_path.get(fut, "unknown")))
                        logger.warning(
                            f"Batch preproc ({alias}): skipping file {_fpath}: {_fe!r}"
                        )
                        continue
                    if zarr_path is not None:
                        _mini_zarr_paths.append(zarr_path)
                        n_ok += 1
                        n_pts_total += n_pts

    except (OSError, BrokenPipeError) as _pool_exc:
        logger.warning(
            f"Batch preproc ({alias}): executor failed "
            f"({_pool_exc!r}), retrying with ThreadPoolExecutor"
        )
        from concurrent.futures import ThreadPoolExecutor as _ThrPool
        from concurrent.futures import as_completed as _thr_ac

        with _ThrPool(max_workers=_MAX_PREP_WORKERS) as tpool:
            _futs = [tpool.submit(_process_file_to_zarr, a) for a in _args_list]
            _fut_to_path = {f: a[0] for f, a in zip(_futs, _args_list, strict=False)}
            for fut in _thr_ac(_futs):
                try:
                    zarr_path, n_pts = fut.result()
                except Exception as _fe:
                    _fpath = os.path.basename(str(_fut_to_path.get(fut, "unknown")))
                    logger.warning(
                        f"Batch preproc ({alias}): skipping file {_fpath}: {_fe!r}"
                    )
                    continue
                if zarr_path is not None:
                    _mini_zarr_paths.append(zarr_path)
                    n_ok += 1
                    n_pts_total += n_pts

    if not _mini_zarr_paths:
        logger.warning(f"Shared batch preprocessing ({alias}): no valid data")
        shutil.rmtree(output_zarr_dir, ignore_errors=True)
        return None

    # ── Pre-sort mini-zarrs by first time element (1 chunk read per file) ──
    # Reading one element per file costs O(n_files × chunk_size) ≈ a few MB,
    # versus the O(n_total_pts) = 191 M element global argsort that previously
    # consumed ~3 GB (time array + index array) and serialised on a single CPU.
    # For non-overlapping SWOT/SARAL passes this guarantees that the final
    # xr.concat produces a globally time-sorted dataset without any permutation.
    _time_name = coordinates.get("time", "time")
    if len(_mini_zarr_paths) > 1 and _time_name:
        _t_starts: List[Any] = []
        for _mp in _mini_zarr_paths:
            _t0: Any = None
            try:
                _zt = xr.open_zarr(_mp, consolidated=False)
                if _time_name in _zt.variables:
                    _tv0 = np.asarray(_zt[_time_name].values[0:1])
                    if len(_tv0) > 0:
                        _t0 = _tv0[0]
                _zt.close()
            except Exception:
                pass
            _t_starts.append(_t0)

        _valid_pairs = [
            (t, p) for t, p in zip(_t_starts, _mini_zarr_paths) if t is not None
        ]
        _none_paths = [
            p for t, p in zip(_t_starts, _mini_zarr_paths) if t is None
        ]
        try:
            _valid_pairs.sort(key=lambda x: x[0])
        except TypeError:
            pass  # mixed dtypes – keep existing order
        _mini_zarr_paths = [p for _, p in _valid_pairs] + _none_paths
        logger.debug(
            f"Shared batch ({alias}): sorted {len(_mini_zarr_paths)} "
            "mini-zarrs by time start"
        )

    # ── Final concat → shared zarr ─────────────────────────────────────
    lazy_parts = [
        xr.open_zarr(p, consolidated=False) for p in _mini_zarr_paths
    ]
    if len(lazy_parts) == 1:
        combined = lazy_parts[0]
    else:
        combined = xr.concat(
            lazy_parts, dim=n_points_dim,
            coords="minimal", compat="override", join="override",
        )

    if n_points_dim in combined.coords:
        combined = combined.drop_vars(n_points_dim)

    # Drop inherited zarr encoding from mini-batch stores — the lazy
    # concat produces dask chunks that don't align with the original
    # per-file zarr chunk sizes, causing a validation error on write.
    # IMPORTANT: preserve datetime encoding for the time coordinate
    # so that workers can decode it correctly after open_zarr().
    _time_encoding = None
    if "time" in combined.variables:
        _time_encoding = dict(combined["time"].encoding)
    for var in combined.variables:
        combined[var].encoding.clear()
    if _time_encoding and "time" in combined.variables:
        # Restore only the CF datetime keys needed for correct roundtrip
        for _ek in ("units", "calendar", "dtype"):
            if _ek in _time_encoding:
                combined["time"].encoding[_ek] = _time_encoding[_ek]

    # ── Sort by time so workers can use fast contiguous slicing ────────
    # The mini-zarrs were pre-sorted by their first time element above,
    # so xr.concat produces an already-monotone time axis for
    # non-overlapping SWOT/SARAL passes.  The expensive global argsort
    # (191 M timestamps → 3 GB RAM + single-threaded CPU) is no longer
    # needed and has been eliminated.

    # Rechunk to uniform sizes — xr.concat of files with varying n_points
    # produces non-uniform dask chunks which zarr cannot store.
    # Use a fixed chunk size; the total is n_pts_total accumulated above.
    _target_chunk = max(1, n_pts_total // max(1, len(lazy_parts)))
    combined = combined.chunk({n_points_dim: _target_chunk})

    # Write the shared zarr (streams chunk-by-chunk from mini zarrs)
    combined.to_zarr(output_zarr_path, mode="w")

    # ── Save time index as sidecar .npy for zero-copy worker access ───
    # Workers need the full time array for searchsorted filtering.
    # We stream through the pre-sorted mini-zarrs reading only the time
    # variable from each, writing directly into a memory-mapped .npy file.
    # Peak RAM = one mini-zarr's time array (a few MB) versus the previous
    # approach of loading all 191 M timestamps at once (~1.5 GB).
    #
    # Size guard: for very large swath datasets (SWOT, SARAL) a full-year
    # batch can total hundreds of millions of points.  Each point requires
    # 8 bytes in the time index.  Skip the sidecar when the estimated file
    # size would exceed DCTOOLS_TIME_NPY_MAX_GB (default 1 GB) to avoid
    # exhausting disk space and OS dirty-page RAM during the mmap write.
    # Workers fall back to loading the time coordinate from the zarr store.
    _TIME_NPY_MAX_GB = float(os.environ.get("DCTOOLS_TIME_NPY_MAX_GB", "1.0"))
    _time_npy_size_gb = n_pts_total * 8 / 1e9
    _time_npy_path = os.path.join(output_zarr_dir, "time_index.npy")
    try:
        _time_name_coord = coordinates.get("time", "time")
        if _time_npy_size_gb > _TIME_NPY_MAX_GB:
            logger.debug(
                f"Shared batch ({alias}): skipping time index .npy "
                f"(estimated {_time_npy_size_gb:.2f} GB > "
                f"limit {_TIME_NPY_MAX_GB} GB). "
                "Workers will read time from the shared zarr."
            )
        elif n_pts_total > 0 and any(
            _time_name_coord in _lp.variables for _lp in lazy_parts
        ):
            _tv_mmap = np.lib.format.open_memmap(
                _time_npy_path, mode="w+",
                dtype="datetime64[ns]", shape=(n_pts_total,),
            )
            _offset = 0
            for _lp in lazy_parts:
                if _time_name_coord not in _lp.variables:
                    continue
                _tv = np.asarray(_lp[_time_name_coord].values)
                if np.issubdtype(_tv.dtype, np.integer):
                    _tv = _tv.astype("datetime64[ns]")
                elif not np.issubdtype(_tv.dtype, np.datetime64):
                    _tv = pd.to_datetime(_tv).values.astype("datetime64[ns]")
                elif _tv.dtype != np.dtype("datetime64[ns]"):
                    _tv = _tv.astype("datetime64[ns]")
                _n = len(_tv)
                _tv_mmap[_offset:_offset + _n] = _tv
                _offset += _n
                del _tv
            del _tv_mmap  # flush to disk
            logger.debug(
                f"Shared batch ({alias}): saved time index "
                f"({_offset:,} pts) to {_time_npy_path}"
            )
    except Exception as _exc_npy:
        logger.debug(f"Could not save time index .npy: {_exc_npy}")

    # Consolidate metadata so worker threads don't each stat hundreds
    # of small .zarray/.zattrs files simultaneously.
    try:
        import zarr as _zarr_mod
        _zarr_mod.consolidate_metadata(output_zarr_path)
    except Exception:
        pass  # non-critical — workers fall back to consolidated=False

    # Cleanup mini zarrs
    for lp in lazy_parts:
        lp.close()
    for p in _mini_zarr_paths:
        shutil.rmtree(p, ignore_errors=True)

    # Register for cleanup at exit
    _SWOT_TEMP_DIRS.append(output_zarr_dir)

    elapsed = _time.perf_counter() - t_start
    logger.info(
        f"Shared batch preprocessing ({alias}): "
        f"{n_ok}/{len(local_paths)} files → "
        f"{n_pts_total:,} points in {elapsed:.1f}s"
    )

    return output_zarr_path


class ObservationDataViewer:
    """Class to view and preprocess observation data."""

    def __init__(
        self,
        source: Union[xr.Dataset, List[xr.Dataset], pd.DataFrame, gpd.GeoDataFrame],
        load_fn: Callable[..., xr.Dataset],
        alias: str,
        keep_vars: List[str],
        target_dimensions: Dict[str, Any],
        dataset_metadata: Any,
        time_bounds: Tuple[pd.Timestamp, pd.Timestamp],
        # time_tolerance: pd.Timedelta = pd.Timedelta("12h"),
        n_points_dim: str,
        dataset_processor: Optional[Optional[DatasetProcessor]] = None,
        results_dir: Optional[str] = None,
        include_geometry: bool = False,
        save_preprocessed: bool = False,
    ):
        """
        Initialize the ObservationDataViewer.

        Parameters:
            source: either
                - one or more xarray Datasets (data already loaded)
                - a DataFrame containing metadata, including file links
            load_fn: a callable that loads a dataset given a link
            alias: optional alias to pass to load_fn if needed
            keep_vars: extracted variables to keep
            target_dimensions: target dimensions dict
            dataset_metadata: metadata dict
            time_bounds: time bounds tuple
            n_points_dim: name of points dimension
            dataset_processor: optional processor
            include_geometry: whether to include geometry column
            save_preprocessed: whether to persist preprocessed data to Zarr
        """
        self.is_metadata = isinstance(source, (pd.DataFrame, gpd.GeoDataFrame))
        self.load_fn = load_fn
        # self.time_tolerance = time_tolerance
        self.alias = alias
        self.keep_vars = keep_vars
        self.target_dimensions = target_dimensions
        self.n_points_dim = n_points_dim
        self.dataset_processor = dataset_processor
        self.time_bounds = time_bounds
        self.results_dir = results_dir
        self.save_preprocessed = save_preprocessed

        if self.is_metadata:
            if self.load_fn is None:
                raise ValueError("A 'load_fn(link: str)' must be provided when using metadata.")
            self.meta_df = source
        else:
            self.datasets = source if isinstance(source, list) else [source]
        self.coordinates = dataset_metadata['coord_system'].coordinates
        self.include_geometry = include_geometry

    def save_to_zarr(self, dataset: xr.Dataset, root_path: str):
        """
        Save preprocessed dataset to a Zarr file in the specified root path.

        Parameters:
            dataset: The xarray Dataset to save.
            root_path: The root directory path where the Zarr file will be saved.
        """
        # Save preprocessed dataset in a Zarr file
        time_val = dataset.coords["time"].values

        # If it's an array with a single value
        if isinstance(time_val, (np.ndarray, list)) and len(time_val) == 1:
            time_str = str(pd.to_datetime(time_val[0]))
        else:
            time_str = str(pd.to_datetime(time_val))
        argo_name = f"argo_profiles_{time_str}.zarr"
        path = os.path.join(root_path, argo_name)
        dataset.to_zarr(path, mode="w", consolidated=True)

    def preprocess_datasets(
        self,
        dataframe: pd.DataFrame,
        load_to_memory: bool = False,
    ) -> Optional[xr.Dataset]:
        """
        Preprocess the input DataFrame and single observations files.

        Returns:
            xr.Dataset: The preprocessed dataset.
        """
        # remove "geometry" fields if needed:
        if not self.include_geometry and "geometry" in dataframe.columns:
            dataframe = dataframe.drop(columns=["geometry"])

        # File loading
        dataset_paths = [row["path"] for _, row in dataframe.iterrows()]
        if not dataset_paths:
            logger.warning(f"No dataset paths found for alias '{self.alias}'")
            return None

        # log diagnostic
        total_files = len(dataset_paths)
        if total_files > 100:
            logger.info(
                f"Preprocessing large batch of files for {self.alias}: {total_files} files. "
                "This may generate a large Dask graph."
            )

        # swath_dims = {"num_lines", "num_pixels", "num_nadir"}
        reduced_swath_dims = {"num_lines", "num_pixels"}

        # if argo profiles, special preprocessing:
        # NOTE: This is the FALLBACK path.  The preferred pipeline is the
        # shared-Zarr approach in evaluator._evaluate_batch() which merges
        # all batch time-windows, downloads once, and lets workers filter
        # by searchsorted.  This branch only runs when the evaluator's
        # prefetch did not happen (e.g. standalone ObservationDataViewer
        # usage outside the Evaluator class).

        if self.alias == "argo_profiles":
            try:
                result = preprocess_argo_profiles(
                    profile_sources=dataset_paths,
                    open_func=self.load_fn,
                    alias=self.alias,
                    time_bounds=self.time_bounds,
                    depth_levels=self.target_dimensions.get('depth', np.array([])),
                )
                if result is None:
                    logger.error("No Argo profiles could be processed")
                    return None

                if load_to_memory:
                    result = result.compute()

                if self.save_preprocessed and self.results_dir:
                    save_path = os.path.join(self.results_dir, self.alias + "_preprocessed")
                    os.makedirs(save_path, exist_ok=True)
                    self.save_to_zarr(result, save_path)
                result_ds: xr.Dataset = xr.Dataset(result)
                return result_ds
            except Exception as e:
                logger.error(f"Argo preprocessing failed: {e}")
                traceback.print_exc()
                return None

        first_ds = None
        try:
            if self.alias is not None:
                first_ds = self.load_fn(dataset_paths[0], self.alias)
            else:
                first_ds = self.load_fn(dataset_paths[0])
        except Exception as exc:
            logger.error(
                f"Failed to open first dataset for alias '{self.alias}' "
                f"during preprocessing probe: {exc}"
            )
            traceback.print_exc()
            return None

        if first_ds is None:
            logger.error(
                f"Failed to open first dataset for alias '{self.alias}' "
                "during preprocessing probe (received None)."
            )
            return None

        # --- Capture dimension flags, then immediately free the probe dataset ---
        # first_ds is only needed to decide which preprocessing branch to take.
        # Keeping it alive through the N-file loop wastes RAM equal to one full
        # SWOT file (often 50-300 MB) and can double-open the first file.
        _probe_has_npoints = (
            self.n_points_dim in first_ds.dims
            or ("time" in first_ds.dims and len(first_ds.dims) == 1)
        )
        _probe_is_swath = reduced_swath_dims.issubset(first_ds.dims)
        _probe_dims = dict(first_ds.sizes)  # lightweight copy
        del first_ds
        gc.collect()

        # Data with n_points/N_POINTS dimension only
        # OR special case: unique "time" dimension (saral tracks, etc)
        if _probe_has_npoints and not _probe_is_swath:
            try:
                # Clean and process datasets
                # Use the processor only if loading into memory (eager loading distributed)
                # Othewise, local graph construction (lazy) to avoid serialization issues
                # of lazy datasets between workers and client.
                if self.dataset_processor is not None and load_to_memory:
                    delayed_tasks: List[Any] = []
                    for idx, dataset_path in enumerate(dataset_paths):
                        delayed_tasks.append(dask.delayed(preprocess_one_npoints)(
                            dataset_path, False, self.n_points_dim, dataframe, idx,
                            self.alias, self.load_fn,
                            self.keep_vars, self.target_dimensions,
                            self.coordinates,
                            self.time_bounds,
                            load_to_memory,
                        ))
                    batch_results = self.dataset_processor.compute_delayed_tasks(
                        delayed_tasks, sync=False
                    )
                else:
                    batch_results_sync: List[Any] = []
                    for idx, dataset_path in enumerate(dataset_paths):
                        result = preprocess_one_npoints(
                            dataset_path, False, self.n_points_dim, dataframe, idx,
                            self.alias, self.load_fn,
                            self.keep_vars, self.target_dimensions,
                            self.coordinates,
                            self.time_bounds,
                            load_to_memory,
                        )
                        if result is not None:
                            batch_results_sync.append(result)

                    batch_results = batch_results_sync

                if not batch_results:
                    return None

                # Combine results
                combined = concat_with_dim(batch_results, self.n_points_dim)

                # if load_to_memory:
                #    combined = combined.compute()

                return xr.Dataset(combined) if combined is not None else None

            except Exception as e:
                logger.error(f"Preprocessing failed for {self.alias}: {e}")
                traceback.print_exc()
                return None

        # Swath data (num_lines, num_pixels)
        elif _probe_is_swath:
            try:
                # Clean and process datasets
                if self.dataset_processor is not None and load_to_memory:
                    delayed_tasks_swath: List[Any] = []
                    for idx, dataset_path in enumerate(dataset_paths):
                        delayed_tasks_swath.append(dask.delayed(preprocess_one_npoints)(
                            dataset_path, True, self.n_points_dim, dataframe, idx,
                            self.alias, self.load_fn,
                            self.keep_vars, self.target_dimensions,
                            self.coordinates,
                            self.time_bounds,
                            load_to_memory,
                        ))
                    batch_results = self.dataset_processor.compute_delayed_tasks(
                        delayed_tasks_swath, sync=False
                    )
                else:
                    batch_results_sync2: List[Any] = []
                    for idx, dataset_path in enumerate(dataset_paths):
                        result = preprocess_one_npoints(
                            dataset_path, True, self.n_points_dim, dataframe, idx,
                            self.alias, self.load_fn,
                            self.keep_vars, self.target_dimensions,
                            self.coordinates,
                            self.time_bounds,
                            load_to_memory,
                        )
                        if result is not None:
                            batch_results_sync2.append(result)

                    batch_results = batch_results_sync2

                if not batch_results:
                    return None

                # Combine results
                combined = concat_with_dim(batch_results, self.n_points_dim)

                return xr.Dataset(combined) if combined is not None else None

            except Exception as e:
                logger.error(f"Preprocessing failed for {self.alias}: {e}")
                traceback.print_exc()
                return None
        else:
            logger.error(
                f"Dataset for {self.alias} has unsupported dimensions: {_probe_dims}"
            )
            return None
