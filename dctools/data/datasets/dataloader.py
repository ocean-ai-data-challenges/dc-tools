#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Dataloder."""

import gc
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Type, Union

import dask
import geopandas as gpd
import numpy as np
import pandas as pd
import torch
import xarray as xr
from loguru import logger
from oceanbench.core.distributed import DatasetProcessor
from xrpatcher import XRDAPatcher

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


def extrapolate_to_surface(var_name, valid_depths, valid_vals):
    """
    Extrapolate profile data to the surface (depth=0) using linear gradient or constant.

    Args:
        var_name (str): Name of the variable (e.g., 'TEMP', 'PSAL').
        valid_depths (array-like): Array of depth values where data exists.
        valid_vals (array-like): Array of valid data values corresponding to depths.

    Returns:
        float: The extrapolated value at the surface.
    """
    if len(valid_vals) == 0:
        return np.nan

    if var_name in ["TEMP", "temperature"]:
        # Temperature: reduced gradient towards surface
        if len(valid_depths) >= 2:
            # Search for depth different from valid_depths[0]
            depth_diff = 0
            i = 1
            while i < len(valid_depths) and abs(depth_diff) < 1e-6:
                depth_diff = valid_depths[i] - valid_depths[0]
                i += 1

            if abs(depth_diff) < 1e-6:  # All depths are identical
                surface_val = valid_vals[0]
            else:
                gradient = (valid_vals[i-1] - valid_vals[0]) / depth_diff
                surface_val = valid_vals[0] - gradient * 0.5 * valid_depths[0]
        else:
            surface_val = valid_vals[0]

    elif var_name in ["PSAL", "salinity"]:
        # Same logic for salinity
        if len(valid_depths) >= 2:
            depth_diff = 0
            i = 1
            while i < len(valid_depths) and abs(depth_diff) < 1e-6:
                depth_diff = valid_depths[i] - valid_depths[0]
                i += 1

            if abs(depth_diff) < 1e-6:
                surface_val = valid_vals[0]
            else:
                gradient = (valid_vals[i-1] - valid_vals[0]) / depth_diff
                surface_val = valid_vals[0] - gradient * 0.3 * valid_depths[0]
        else:
            surface_val = valid_vals[0]

    else:
        # Other variables: standard linear extrapolation
        if len(valid_depths) >= 2:
            depth_diff = 0
            i = 1
            while i < len(valid_depths) and abs(depth_diff) < 1e-6:
                depth_diff = valid_depths[i] - valid_depths[0]
                i += 1

            if abs(depth_diff) < 1e-6:
                surface_val = valid_vals[0]
            else:
                gradient = (valid_vals[i-1] - valid_vals[0]) / depth_diff
                surface_val = valid_vals[0] - gradient * valid_depths[0]
        else:
            surface_val = valid_vals[0]

    return surface_val

def preprocess_argo_profiles(
    profile_sources: List[str],
    open_func: Callable[..., xr.Dataset],
    alias: str,
    time_bounds: Tuple[pd.Timestamp, pd.Timestamp],
    depth_levels: Union[List[float], np.ndarray],
    n_points_dim: str = "N_POINTS"
):
    """
    Load, filter, interpolate, and combine multiple Argo profile files into a single dataset.

    Args:
        profile_sources (List[str]): List of file paths or identifiers for Argo profiles.
        open_func (Callable): Function to open a single profile.
        alias (str): Dataset alias.
        time_bounds (Tuple): Start and end time for filtering.
        depth_levels (List[float]): Target depth levels for interpolation.
        n_points_dim (str): Name of the points dimension (default "N_POINTS").

    Returns:
        xr.Dataset: Combined and interpolated dataset.
    """
    interp_profiles: List[Any] = []
    time_vals: List[Any] = []
    # TODO : remove this after storing preprocessed profiles to
    # avoid reprocessing them at each timestep
    threshold_list_profiles = 20

    def process_one_profile(profile_source):
        try:
            # open dataset
            if alias is not None:
                ds = open_func(profile_source, alias)
            else:
                ds = open_func(profile_source)

            if ds is None:
                return None, None

            # ds = ds.argo.interp_std_levels(target_dimensions['depth'])
            # unusable on a single profile
            # ds = ds.argo.filter_qc()   # useless in "research" mode of argopy
            # ds_filtered = filter_variables(ds, keep_variables_list)

            ds = ds.rename({"PRES_ADJUSTED": "depth"})
            ds = ArgoManager.filter_argo_profile_by_time(
                ds,
                tmin=time_bounds[0],
                tmax=time_bounds[1],
            )
            if n_points_dim not in ds.dims or ds.sizes.get(n_points_dim, 0) == 0:
                logger.warning(
                    f"Argo profile {profile_source} is empty after time filtering, skipping."
                )
                return None, None

            lat = ds["LATITUDE"].isel(N_POINTS=0).values.item()
            lon = ds["LONGITUDE"].isel(N_POINTS=0).values.item()
            time = pd.to_datetime(ds["TIME"].values)
            if isinstance(time, (np.ndarray, list)) and len(time) > 1:
                mean_time = pd.to_datetime(time).mean()
            else:
                mean_time = pd.to_datetime(
                    time[0] if isinstance(time, (np.ndarray, list)) else time
                )

            depths = ds["depth"].values

            data_dict: Dict[Any, Any] = {}
            for v in ds.data_vars:
                if v == "depth":
                    continue
                vals = ds[v].values
                # Filter NaNs
                valid_mask = ~np.isnan(vals)
                if not np.any(valid_mask):
                    # If all are NaNs, create array of NaNs
                    interp_vals = np.full_like(depth_levels, np.nan, dtype=float)
                else:
                    # Filter corresponding values and depths
                    valid_vals = vals[valid_mask]
                    valid_depths = depths[valid_mask]

                    # Extrapolation to surface
                    surface_val = extrapolate_to_surface(v, valid_depths, valid_vals)

                    interp_vals = np.interp(
                        depth_levels,
                        valid_depths,
                        valid_vals,
                        left=surface_val,
                        right=np.nan
                    )
                data_dict[v] = ("depth", interp_vals)

            # Clean up to avoid memory retention
            del vals, depths
            if 'valid_mask' in locals():
                del valid_mask, valid_vals, valid_depths
            ds.close()
            del ds

            interp_ds = xr.Dataset(
                data_dict,
                coords={
                    "depth": depth_levels,
                    "lat": lat,
                    "lon": lon,
                    "time": time
                }
            )
            return interp_ds, mean_time
        except Exception as e:
            logger.warning(f"Failed to process Argo profile {profile_source}: {e}")
            traceback.print_exc()
            return None, None

    # Parallelization with ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=2) as executor:  # adapt max_workers to your CPU
        # TODO remove this shortcut after optimizing the processing:
        # thousands of files to preprocess at each timestep !
        subset_sources = profile_sources[0:threshold_list_profiles]
        futures = [executor.submit(process_one_profile, src) for src in subset_sources]
        for future in as_completed(futures):
            interp_ds, mean_time = future.result()
            if interp_ds is not None:
                interp_profiles.append(interp_ds)
                time_vals.append(mean_time)

    if len(interp_profiles) == 0:
        return None
    # Convert each element to scalar pd.Timestamp
    clean_time_vals: List[Any] = []
    for t in time_vals:
        if isinstance(t, (pd.DatetimeIndex, np.ndarray, list)):
            # Takes first element if index or list
            clean_time_vals.append(pd.to_datetime(t[0]))
        else:
            clean_time_vals.append(pd.to_datetime(t))
    mean_time = pd.Series(clean_time_vals).mean()
    interp_profiles = [
        ds.drop_vars("time") if "time" in ds.coords else ds
        for ds in interp_profiles
    ]

    if len(interp_profiles) == 1:
        combined = interp_profiles[0]
    else:
        combined = xr.concat(interp_profiles, dim=n_points_dim)

    combined = combined.assign_coords(time=mean_time)

    # Clean up temporary structures
    del interp_profiles, time_vals, clean_time_vals
    gc.collect()

    return combined


def preprocess_one_npoints(
    source, is_swath,
    n_points_dim,
    filtered_df, idx,
    alias, open_func,
    keep_variables_list,
    target_dimensions,
    coordinates,
    time_bounds=None,
    load_to_memory=False,
):
    """
    Preprocess a single N-point dataset (e.g., swath or track).

    Args:
        source: Source dataset or identifier.
        is_swath (bool): Whether the data is a swath (2D) or track (1D).
        n_points_dim (str): Name of the points dimension.
        filtered_df: Filtered metadata DataFrame.
        idx: Index of the current file in filtered_df.
        alias (str): Dataset alias.
        open_func (Callable): Function to open the dataset.
        keep_variables_list (List[str]): Variables to keep.
        target_dimensions (Dict): Format definitions.
        coordinates (Dict): Coordinate system info.
        time_bounds (Tuple, optional): Time bounds for filtering.
        load_to_memory (bool): Whether to load data into memory.

    Returns:
        xr.Dataset: Preprocessed dataset.
    """
    try:
        # open dataset
        if alias is not None:
            ds = open_func(source, alias)
        else:
            ds = open_func(source)
        if ds is None:
            return None

        # Filter variables early to reduce graph size
        # Only keep what is needed + coordinates
        if keep_variables_list:
            ds = filter_variables(ds, keep_variables_list)

        # Load individual files immediately into memory.
        # Single observation files (traces) are small. Managing a Dask graph
        # for each operation (where, stack, assign) on these small files costs
        # more in "scheduling" than the computation itself.
        # Switching to pure NumPy here drastically speeds up preprocessing.
        if load_to_memory:
            try:
                ds = ds.compute()
            except Exception:
                pass

        if is_swath:
            coords_to_keep = [
                coordinates.get('time', None),
                coordinates.get('depth', None),
                coordinates.get('lat', None),
                coordinates.get('lon', None),
            ]
            coords_to_keep = list(filter(lambda x: x is not None, coords_to_keep))
            ds = swath_to_points(
                ds,
                coords_to_keep=list(coordinates.keys()),
                n_points_dim=n_points_dim,
            )

        # Search for coordinate/time variable
        time_name = coordinates['time']
        if time_name in ds.variables and time_name not in ds.coords:
            ds = ds.set_coords(time_name)

        time_coord = ds.coords[time_name]

        # Check for alternative 'time' dimension for 1D tracks and rename it
        if n_points_dim not in ds.dims and "time" in ds.dims and len(ds.dims) == 1:
            # Create integer index for n_points and swap, preserving original time variable
            ds = ds.assign_coords({n_points_dim: ("time", np.arange(ds.sizes["time"]))})
            ds = ds.swap_dims({"time": n_points_dim})
            # Refresh time_coord to ensure it has the correct dimension name (n_points)
            if time_name in ds.coords:
                time_coord = ds.coords[time_name]

        if n_points_dim not in ds.dims:
            logger.warning(f"Dataset {idx}: No points dimension found (expected '{n_points_dim}')")
            return None

        ds_with_time = add_time_dim(
            ds, filtered_df, n_points_dim=n_points_dim, time_coord=time_coord, idx=idx
        )

        import gc
        ds_interp = ds_with_time
        # Use balanced chunks for n_points to avoid memory spiking in concatenation
        # 100k points is a moderate compromise (~8-10MB per var)
        ds_interp = ds_interp.chunk({n_points_dim: 100000})

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
                logger.info(
                    f"=============  PREPROCESSING REFERENCE DATASET: {ref_alias} =============="
                )
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
                    # Adapt batch size according to parallelization mode
                    # target_batch_size = self._get_optimal_batch_size()
                    if len(batch) >= self.batch_size:
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


class TorchCompatibleDataloader:
    """Adapter to make EvaluationDataloader compatible with PyTorch."""

    def __init__(
        self,
        dataloader: EvaluationDataloader,
        patch_size: Tuple[int, int],
        stride: Tuple[int, int],
    ):
        """
        Initializes a PyTorch compatible dataloader.

        Args:
            dataloader (EvaluationDataloader): The existing dataloader.
            patch_size (Tuple[int, int]): Size of the patches (height, width).
            stride (Tuple[int, int]): Stride step for patches.
        """
        self.dataloader = dataloader
        self.patch_size = patch_size
        self.stride = stride

    def __len__(self):
        """Returns the total number of batches in the dataloader."""
        return len(self.dataloader)

    def __iter__(self) -> Generator[Dict[str, torch.Tensor], None, None]:
        """
        Generates PyTorch compatible data batches.

        Yields:
            Dict[str, torch.Tensor]: A dictionary containing data patches.
        """
        for batch in self.dataloader:
            for entry in batch:
                pred_data = entry["pred_data"]
                ref_data = entry["ref_data"]

                # Generate patches for prediction data
                pred_patches = self._generate_patches(pred_data)

                # Generate patches for reference data (if available)
                ref_patches = self._generate_patches(ref_data) if ref_data is not None else None

                # Return patches as PyTorch tensors
                yield {
                    "date": entry["date"],
                    "pred_patches": pred_patches,
                    "ref_patches": ref_patches,  # type: ignore[dict-item]
                }

    def _generate_patches(self, dataset: xr.Dataset) -> torch.Tensor:
        """
        Generates patches from an xarray dataset.

        Args:
            dataset (xr.Dataset): The xarray dataset.

        Returns:
            torch.Tensor: Patches as PyTorch tensor.
        """
        patcher = XRDAPatcher(
            data=dataset,
            patch_size=self.patch_size,
            stride=self.stride,
        )
        patches = patcher.extract_patches()
        return torch.tensor(patches)


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

    # Optimization: override to avoid comparing all coordinates (slow on S3/Dask)
    # join='override' assumes that non-concatenated coordinates are identical.
    join_mode = "outer"
    compat_mode = "no_conflicts"

    # If we concatenate on time or n_points for massive data, override is much faster
    if len(datasets) > 10:
        join_mode = "override"
        compat_mode = "override"

    result: xr.Dataset = xr.concat(  # type: ignore[call-overload]
        datasets_with_dim, dim=concat_dim,
        coords="minimal",
        compat=compat_mode, join=join_mode,
    )
    if sort:
        result = result.sortby(concat_dim)
    return result

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
        include_geometry: bool = False,
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

        if self.is_metadata:
            if self.load_fn is None:
                raise ValueError("A 'load_fn(link: str)' must be provided when using metadata.")
            self.meta_df = source
        else:
            self.datasets = source if isinstance(source, list) else [source]
        self.coordinates = dataset_metadata['coord_system'].coordinates
        self.include_geometry = include_geometry


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

        # log diagnostic
        total_files = len(dataset_paths)
        if total_files > 100:
            logger.info(
                f"Preprocessing large batch of files for {self.alias}: {total_files} files. "
                "This may generate a large Dask graph."
            )

        first_ds = None
        while first_ds is None:
            if self.alias is not None:
                first_ds = self.load_fn(dataset_paths[0], self.alias)
            else:
                first_ds = self.load_fn.open(dataset_paths[0])

        # swath_dims = {"num_lines", "num_pixels", "num_nadir"}
        reduced_swath_dims = {"num_lines", "num_pixels"}

        # if argo profiles, special preprocessing:

        if self.alias == "argo_profiles":
            # logger.info("Argo profiles detected - special preprocessing")
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
                # logger.info(f"Final Argo result: {result.sizes.get('profile', 1)} profiles, "
                #        f"{len(result.data_vars)} variables")
                if load_to_memory:
                    result = result.compute()

                '''# Save preprocessed dataset in a Zarr file
                argo_dir = "..."
                time_val = result.coords["time"].values

                # If it's an array with a single value
                if isinstance(time_val, (np.ndarray, list)) and len(time_val) == 1:
                    time_str = str(pd.to_datetime(time_val[0]))
                else:
                    time_str = str(pd.to_datetime(time_val))
                argo_name = f"argo_profiles_{time_str}.zarr"
                import os
                argo_path = os.path.join(argo_dir, argo_name)
                result.to_zarr(argo_path, mode="w", consolidated=True)'''

                return xr.Dataset(result) if result is not None else None
            except Exception as e:
                logger.error(f"Argo preprocessing failed: {e}")
                traceback.print_exc()
                return None

        # Data with n_points/N_POINTS dimension only
        # OR special case: unique "time" dimension (saral tracks, etc)
        elif (
            (
                self.n_points_dim in first_ds.dims
                or ("time" in first_ds.dims and len(first_ds.dims) == 1)
            )
            and not reduced_swath_dims.issubset(first_ds.dims)
        ):
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
        elif reduced_swath_dims.issubset(first_ds.dims):
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
            logger.error(f"Dataset for {self.alias} has unsupported dimensions: {first_ds.dims}")
            return None
