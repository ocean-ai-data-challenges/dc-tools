# -*- coding: UTF-8 -*-

"""Functions for processing `xr.Datasets` and `DataArrays`."""

import ast
import traceback
from typing import Any, Dict, List, Optional

# New (compatible with recent versions)
# from pangeo_forge_recipes.recipes.xarray.zarr import XarrayZarrRecipe
# from pangeo_forge_recipes.patterns import FilePattern
# from pangeo_forge_recipes.executors.python import PythonPipelineExecutor
import cftime
from loguru import logger
import numpy as np
import pandas as pd
from pathlib import Path
import shutil
import xarray as xr


def create_empty_dataset(dimensions: dict) -> xr.Dataset:
    """
    Create an empty xarray Dataset from a dictionary of dimensions.

    Args:
        dimensions (dict): Dictionary where keys are dimension names
                           and values are dimension sizes.

    Returns:
        xr.Dataset: Empty xarray Dataset with specified dimensions.
    """
    coords = {dim: range for dim, range in dimensions.items()}
    return xr.Dataset(coords=coords)


def rename_coordinates(ds: xr.Dataset, rename_dict: dict) -> xr.Dataset:
    """
    Rename coordinates/dimensions of an xarray Dataset using swap_dims.

    Ensures new coordinates become indexed dimensions (avoids xarray warnings).

    Args:
        ds (xr.Dataset): Dataset to modify.
        rename_dict (dict): Dictionary {old_name: new_name}.

    Returns:
        xr.Dataset: Renamed dataset.
    """
    # Rename coordinates (without touching dimensions)
    coords_to_rename = {
        k: v for k, v in rename_dict.items() if k in ds.coords and k != v and v not in ds.coords
    }
    if coords_to_rename:
        ds = ds.rename(coords_to_rename)

    # Ensure new coordinates are present
    for _old, new in coords_to_rename.items():
        if new in ds.variables and new not in ds.coords:
            ds = ds.set_coords(new)

    # Use swap_dims to transform coordinates into main dimensions
    swap_dict: Dict[Any, Any] = {}
    for old, new in coords_to_rename.items():
        if old in ds.dims and new in ds.coords and old != new:
            swap_dict[old] = new
    if swap_dict:
        ds = ds.swap_dims(swap_dict)

    # Remove old coordinates if they still exist
    for old, new in coords_to_rename.items():
        if old in ds.coords and old != new:
            ds = ds.drop_vars(old)
    return ds

def rename_variables(ds: xr.Dataset, rename_dict: Optional[Optional[dict]] = None):
    """
    Rename variables according to a given dictionary.

    Args:
        ds (xr.Dataset): Dataset to modify.
        rename_dict (dict or str): Dictionary mapping old names to new names.
            Can be a string representation of a dictionary.

    Returns:
        xr.Dataset: Renamed dataset.
    """
    try:
        if not rename_dict:
            return ds
        if isinstance(rename_dict, str):
            rename_dict = ast.literal_eval(rename_dict)

        # keep only keys that are in the dataset variables
        rename_dict = {k: v for k, v in rename_dict.items() if k in list(ds.data_vars)}
        # Remove invalid values from the dictionary
        rename_dict = {k: v for k, v in rename_dict.items() if k != v and v is not None}
        if not rename_dict or len(rename_dict) == 0:
            return ds

        ds = ds.rename_vars(rename_dict)

        # Delete old variables if they still exist
        for old, new in rename_dict.items():
            if old in ds.variables and new in ds.variables and old != new:
                ds = ds.drop_vars(old)
        return ds
    except Exception as e:
        logger.error(f"Error renaming variables: {e}")
        raise ValueError(f"Failed to rename variables in dataset: {e}") from e


def rename_coords_and_vars(
    ds: xr.Dataset,
    rename_coords_dict: Optional[Optional[dict]] = None,
    rename_vars_dict: Optional[Optional[dict]] = None,
):
    """
    Rename variables and coordinates according to given dictionaries.

    Wrapper around rename_variables and rename_coordinates.
    """
    try:
        ds = rename_variables(ds, rename_vars_dict)
        ds = rename_coordinates(ds, rename_coords_dict or {})

        return ds
    except Exception as e:
        logger.error(f"Error renaming coordinates or variables: {e}")
        raise ValueError(f"Failed to rename coordinates or variables in dataset: {e}") from e


def subset_variables(ds: xr.Dataset, list_vars: List[str]):
    """
    Extract a sub-dataset containing only listed variables, preserving attributes.

    Also handles coordinate preservation and standard_name attribute updates.
    """
    for variable_name in ds.variables:
        var_std_name = str(ds[variable_name].attrs.get("standard_name", "")).lower()
        if not var_std_name:
            var_std_name = str(ds[variable_name].attrs.get("std_name", "")).lower()

    real_vars = [var for var in list_vars if var in ds.data_vars]
    # Creates a sub-dataset with only listed variables
    subset = ds[real_vars]

    # Detect coordinates present in sub-dataset
    coords_to_set = [c for c in subset.data_vars if c in ds.coords]
    if coords_to_set:
        subset = subset.set_coords(coords_to_set)

    for variable_name in subset.variables:
        var_std_name = str(subset[variable_name].attrs.get("standard_name", "")).lower()
        if not var_std_name:
            var_std_name = str(subset[variable_name].attrs.get("std_name", "")).lower()

    return subset

def assign_coordinate(
    ds: xr.Dataset, coord_name: str, coord_vals: List[Any], coord_attrs: Dict[str, str]
):
    """Assign a coordinate with values and attributes to a dataset."""
    ds = ds.assign_coords({coord_name: (coord_name, coord_vals, coord_attrs)})
    return ds

def get_time_info(ds: xr.Dataset):
    """
    Analyze the main time axis of an xarray Dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset to analyze.

    Returns
    -------
    dict
        A dictionary containing:
        - start: the earliest timestamp (pandas.Timestamp or string from attributes)
        - end: the latest timestamp
        - duration: the time range duration (if computable)
        - calendar: the calendar used (if specified)
    """
    # Step 1: Try to find a valid time coordinate
    for name in ds.coords:
        if str(name).lower() in {"time", "t", "date", "datetime"}:
            time_coord = ds.coords[name]
            calendar = time_coord.attrs.get("calendar", "standard")

            # Case 1: Already decoded to datetime-like values
            if np.issubdtype(time_coord.dtype, np.datetime64) or isinstance(
                time_coord.values[0], cftime.DatetimeBase
            ):
                times = pd.to_datetime(time_coord.values)
                break

            # Case 2: Try decoding from CF metadata
            units = time_coord.attrs.get("units")
            if units:
                try:
                    decoded_time = xr.decode_cf(xr.Dataset({name: time_coord}))[name]
                    times = pd.to_datetime(decoded_time.values)
                    break
                except Exception as exc:
                    logger.error(f"Error decoding time axis: {repr(exc)}")
                    continue
    else:
        # Step 2 fallback: Use global attributes
        start = ds.attrs.get("time_coverage_start")
        end = ds.attrs.get("time_coverage_end")
        duration = None
        if start and end:
            try:
                parsed_start = pd.to_datetime(start)
                parsed_end = pd.to_datetime(end)
                duration = parsed_end - parsed_start
            except Exception:
                pass
        return {"start": start, "end": end, "duration": duration, "calendar": None}

    # Step 3: Extract times and compute
    if len(times) == 0:
        return {"start": None, "end": None, "duration": None, "calendar": calendar}

    start = times.min()
    end = times.max()
    duration = end - start

    return {"start": start, "end": end, "duration": duration, "calendar": calendar}


def filter_time_interval(ds: xr.Dataset, start_time: str, end_time: str) -> Optional[xr.Dataset]:
    """
    Filters an xarray Dataset to only include data within a specified time range.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset containing time coordinates.
    start_time : str
        The start time in ISO format (e.g., "2024-05-02").
    end_time : str
        The end time in ISO format (e.g., "2024-05-11").

    Returns
    -------
    xr.Dataset
        A filtered dataset containing only data within the time range.
        Returns None if no data is within the time range.
    """
    # Analyze the time axis
    time_info = get_time_info(ds)

    # Convert start and end times to pandas Timestamp
    start_time = pd.to_datetime(start_time)
    end_time = pd.to_datetime(end_time)

    # Filter the time coordinate within the given interval
    try:
        # Check if the time axis is present and valid
        if time_info["start"] is None or time_info["end"] is None:
            return None

        time_coord = ds.coords["time"]
        mask = (time_coord >= start_time) & (time_coord <= end_time)  # type: ignore

        # If no data falls within the time range, return None
        if mask.sum() == 0:
            logger.warning("No data found in the specified time interval.")
            return None

        return ds.sel(time=mask)
    except TypeError:
        # Fallback for incompatible types
        logger.warning(f"Failed to filter time interval {start_time} - {end_time}")
        return ds

def filter_spatial_area(
    ds: xr.Dataset, lat_min: float, lat_max: float, lon_min: float, lon_max: float
) -> Optional[xr.Dataset]:
    """
    Filters an xarray Dataset to only include data within a specified spatial area.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset containing latitude and longitude coordinates.
    lat_min : float
        Minimum latitude.
    lat_max : float
        Maximum latitude.
    lon_min : float
        Minimum longitude.
    lon_max : float
        Maximum longitude.

    Returns
    -------
    xr.Dataset
        A filtered dataset containing only data within the specified spatial area.
        Returns None if no data is within the area.
    """
    # Step 1: Check if latitude and longitude coordinates exist
    if "lat" not in ds.coords or "lon" not in ds.coords:
        logger.warning("Latitude or longitude coordinates not found in the dataset.")
        return None

    # Step 2: Apply the spatial filter
    lat_coord = ds.coords["lat"]
    lon_coord = ds.coords["lon"]

    lat_mask = (lat_coord >= lat_min) & (lat_coord <= lat_max)
    lon_mask = (lon_coord >= lon_min) & (lon_coord <= lon_max)

    # Step 3: Filter the dataset using the masks
    if lat_mask.sum() == 0 or lon_mask.sum() == 0:
        logger.warning("No data found in the specified spatial area.")
        return None

    # Step 4: Return the filtered dataset
    return ds.sel(lat=lat_mask, lon=lon_mask)

def reset_time_coordinates(dataset: xr.Dataset) -> xr.Dataset:
    """
    Reset time coordinates values to a sequence starting at 0.

    Args:
        dataset (xr.Dataset): Input xarray Dataset.

    Returns:
        xr.Dataset: Dataset with modified time coordinates.
    """
    if "time" not in dataset.coords:
        raise ValueError("The dataset does not contain 'time' coordinates.")

    # Generate a sequence of values starting at 0
    new_time_values = range(len(dataset.coords["time"]))

    # Replace time coordinate values
    dataset = dataset.assign_coords(time=new_time_values)
    return dataset


def filter_variables(ds: xr.Dataset, keep_vars: List[str]) -> xr.Dataset:
    """
    Filter an xarray Dataset by keeping only some variables/coordinates.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset.
    keep_vars : list[str]
        Names of variables/coords to keep.

    Returns
    -------
    xr.Dataset
        The filtered Dataset containing the selected data vars, any explicitly
        requested coords, and any coords required by the data vars, as well as
        global attributes.
    """
    # Normalize and remove duplicates (keeping appearance order)
    keep_vars = [str(v) for v in keep_vars]
    seen: set = set()
    unique_keep_vars = []
    for v in keep_vars:
        if v not in seen:
            unique_keep_vars.append(v)
            seen.add(v)
    keep_vars = unique_keep_vars

    # Set of present variables (includes data_vars + coords)
    present = set(ds.variables.keys())

    # Separate what is present / absent
    keep_present = [v for v in keep_vars if v in present]
    not_found = [v for v in keep_vars if v not in present]
    if not_found:
        print(f"Warning: these names were not found in dataset and will be ignored: {not_found}")

    # Data variables to keep (intersection with ds.data_vars)
    data_vars_to_keep = [v for v in keep_present if v in ds.data_vars]

    # Build the new dataset from selected data_vars
    if data_vars_to_keep:
        new_ds = ds[data_vars_to_keep].copy()  # xarray keeps necessary coords
    else:
        # No data_var requested -> empty dataset
        new_ds = xr.Dataset()

    # Explicitly re-attach requested coords (even if not referenced by data_vars)
    for name in keep_present:
        if name in ds.coords and name not in new_ds.coords:
            new_ds = new_ds.assign_coords({name: ds.coords[name]})

    # Preserve global attributes
    new_ds.attrs = ds.attrs.copy()

    return new_ds


def filter_dataset_by_depth(ds: xr.Dataset, depth_vals, depth_tol=1) -> xr.Dataset:
    """
    Filter a dataset (Argo-like) by keeping only values close to depth_vals within depth_tol.

    Works whether 'depth' is a dimension or a variable aligned with 'n_points'.
    """
    if "depth" in ds.dims:
        # Case 1: depth is a dimension
        depth_array = ds["depth"].values
    elif "depth" in ds.variables:
        # Case 2: depth is a coordinate-type variable
        depth_array = ds["depth"].values
    else:
        raise ValueError("No 'depth' dimension or variable found in dataset")

    # Strict conversion to float (replaces invalid strings/objects with NaN)
    depth_array = pd.to_numeric(depth_array.ravel(), errors="coerce").astype(float)

    # Mask construction
    mask = np.zeros_like(depth_array, dtype=bool)
    for d in depth_vals:
        try:
            d_float = float(d)
            mask |= np.isclose(depth_array, d_float, atol=depth_tol, rtol=0.0)
        except Exception:
            continue

    # Apply the mask
    if "depth" in ds.dims:
        return ds.sel(depth=depth_array[mask])
    else:  # aligned on n_points
        return ds.isel(N_POINTS=mask)

def sanitize_for_zarr(ds: xr.Dataset) -> xr.Dataset:
    """Prepare dataset for Zarr writing by cleaning encoding and converting data types."""
    # ds = ds_init.copy()
    try:
        for v in ds.variables:
            var = ds[v]
            # If variable contains NaNs, cast to float
            if np.any(pd.isnull(var.values)):
                if not np.issubdtype(var.dtype, np.floating):
                    ds[v] = var.astype("float32")
            # Clean existing encoding
            enc = dict(ds[v].encoding)
            # Remove all inherited NetCDF encodings that cause issues
            keys_to_remove = [
                "_FillValue",
                "dtype",
                "scale_factor",
                "add_offset",
                "zlib",
                "complevel",
                "shuffle",
                "fletcher32",
                "preferred_chunks",
                "source",
                "original_shape",
            ]
            for key in keys_to_remove:
                if key in enc:
                    del enc[key]
            # Specific correction for time variables
            if str(v).lower() == "time":
                if "units" in enc:
                    del enc["units"]
                if "calendar" in enc:
                    del enc["calendar"]
            # If float -> add explicit FillValue
            if np.issubdtype(ds[v].dtype, np.floating):
                enc["_FillValue"] = np.nan
            # If int -> add integer FillValue if needed
            elif np.issubdtype(ds[v].dtype, np.integer):
                enc["_FillValue"] = -9999
            ds[v].encoding = enc
        return ds
    except Exception as e:
        logger.error(f"Error sanitizing dataset for Zarr: {e}")
        traceback.print_exc()
        return ds


def netcdf_to_zarr(
    ds: xr.Dataset,
    zarr_path: str,
    overwrite: bool = True,
    chunk_size: Optional[dict] = None,
    compression: str = "zlib",
    compression_level: int = 3,
):
    """
    Convert NetCDF file to Zarr (fully written, no lazy graph left).

    Saves in the same folder with suffix `.zarr`.
    """
    try:
        if overwrite and Path(zarr_path).exists():
            shutil.rmtree(zarr_path)
        # ds = xr.open_dataset(nc_path, decode_cf=True)
        ds_clean = sanitize_for_zarr(ds)
        ds_clean = ds_clean.chunk()  # automatic chunk, Zarr compatible
        ds_clean.to_zarr(str(zarr_path), mode="w", consolidated=True)
        return str(zarr_path)

    except Exception as e:
        logger.error(f"Error converting NetCDF file to Zarr {zarr_path}: {e}")
        traceback.print_exc()
        return None
