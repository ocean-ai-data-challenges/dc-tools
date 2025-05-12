#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Misc. functions to aid in the processing xr.Datasets and DataArrays."""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import cftime
from loguru import logger
import numpy as np
import pandas as pd
from pathlib import Path
import xarray as xr
import xesmf as xe

from dctools.data.coordinates import CoordinateSystem

# Possible names of coordinates that we want to check for
LATITUDE_NAMES = ["lat", "latitude", "LAT", "LATITUDE"]
LONGITUDE_NAMES = ["lon", "longitude", "LON", "LONGITUDE"]
DEPTH_NAMES = ["depth", "DEPTH", "height", "HEIGHT"]
TIME_NAMES = ["time", "TIME"]
DICT_RENAME_CMEMS = dict(longitude="lon", latitude="lat")
LIST_VARS_GLONET = ["thetao", "zos", "uo", "vo", "so", "depth", "lat", "lon", "time"]
LIST_VARS_GLONET_NO_DIMS = ["thetao", "zos", "uo", "vo", "so"]
LIST_VARS_GLONET_UNITTEST = ["thetao", "zos", "uo"]
GLONET_DEPTH_VALS = [0.494025, 47.37369, 92.32607, 155.8507, 222.4752, 318.1274, 380.213, 
        453.9377, 541.0889, 643.5668, 763.3331, 902.3393, 1245.291, 1684.284, 
        2225.078, 3220.82, 3597.032, 3992.484, 4405.224, 4833.291, 5274.784]

GLONET_TIME_VALS = range(0, 10)

RANGES_GLONET = {
    "lat": np.arange(-78, 90, 0.25),
    "lon": np.arange(-180, 180, 0.25),
    "depth": GLONET_DEPTH_VALS,
    #"time": GLONET_TIME_VALS,
}

GLONET_ENCODING = {"depth": {"dtype": "float32"},
                   "lat": {"dtype": "float64"},
                   "lon": {"dtype": "float64"},
                   "time": {"dtype": "str"},
                   "so": {"dtype": "float32"},
                   "thetao": {"dtype": "float32"},
                   "uo": {"dtype": "float32"},
                   "vo": {"dtype": "float32"},
                   "zos": {"dtype": "float32"},
}

STD_COORDS_NAMES = {"lon": "lon", "lat": "lat", "depth": "depth", "time": "time"}

def create_empty_dataset(dimensions: dict) -> xr.Dataset:
    """
    Crée un Dataset Xarray vide à partir d'un dictionnaire de dimensions.

    Args:
        dimensions (dict): Dictionnaire où les clés sont les noms des dimensions
                           et les valeurs sont les tailles des dimensions.

    Returns:
        xr.Dataset: Dataset Xarray vide avec les dimensions spécifiées.
    """
    coords = {dim: range for dim, range in dimensions.items()}
    return xr.Dataset(coords=coords)

def get_grid_coord_names(
    data: xr.Dataset | xr.DataArray,
) -> Dict[str, str | None]:
    """
    Get the names of the coordinates in `data`.

    Return a dictionary with "lat", "lon", "depth" and "time" as keys
    and the names of the corresponding coordinates (if they exist) as
    values.
    The names are determined by checking against the `XXX_NAMES` constants defined
    in this file for any existing coordinates in `data`
    """
    coord_name_dict = {}
    if hasattr(data, "coords")and len(data.coords) > 0:
        # If data is a Dataset, get the coordinates
        list_coords = list(data.coords) if hasattr(data, "coords") else list(data)
    elif hasattr(data, "dims") and len(data.dims) > 0:
        # If data is a DataArray, get the dimensions
        list_coords = list(data.dims)
    else:
        # If data is neither a Dataset nor a DataArray, return an empty dictionary
        list_coords = list(data)
    # There's probably a less disgusting way of doing this...
    lon_set = set(LONGITUDE_NAMES).intersection(list_coords)
    coord_name_dict["lon"] = None if len(lon_set) == 0 else next(iter(lon_set))
    lat_set = set(LATITUDE_NAMES).intersection(list_coords)
    coord_name_dict["lat"] = None if len(lat_set) == 0 else next(iter(lat_set))
    depth_set = set(DEPTH_NAMES).intersection(list_coords)
    coord_name_dict["depth"] = None if len(depth_set) == 0 else next(iter(depth_set))
    time_set = set(TIME_NAMES).intersection(list_coords)
    coord_name_dict["time"] = None if len(time_set) == 0 else next(iter(time_set))

    return coord_name_dict

def create_coords_rename_dict(ds: xr.Dataset):
    ds_coords = get_grid_coord_names(ds)
    dict_rename = {
        ds_coords["lon"]: STD_COORDS_NAMES["lon"],
        ds_coords["lat"]: STD_COORDS_NAMES["lat"],
        ds_coords["depth"]: STD_COORDS_NAMES["depth"],
        ds_coords["time"]: STD_COORDS_NAMES["time"],
    }
    return dict_rename

def standard_rename_coords(ds: xr.Dataset):
    """Rename coordinates to standard names."""
    dict_rename = create_coords_rename_dict(ds)
    # Remove None values from the dictionary
    dict_rename = {k: v for k, v in dict_rename.items() if v is not None}
    # Rename coordinates
    ds = rename_coordinates(ds, dict_rename)
    return ds

def rename_coordinates(ds: xr.Dataset, rename_dict):
    """Rename coordinates according to a given dictionary."""
    return ds.rename(rename_dict)

def rename_variables(ds: xr.Dataset, rename_dict):
    """Rename variables according to a given dictionary."""
    return ds.rename_vars(rename_dict)

def subset_variables(ds: xr.Dataset, list_vars: List[str]):
    """Extract a sub-dataset containing only listed variables."""
    return ds[list_vars]

def interpolate_dataset(
        ds: xr.Dataset, ranges: Dict[str, np.ndarray],
        weights_filepath: Optional[str] = None,
    ) -> xr.Dataset:
    for key in ranges.keys():
        assert(key in list(ds.dims))

    out_dict = {}
    for key in ranges.keys():
        out_dict[key] = ranges[key]
    for dim in STD_COORDS_NAMES.keys():
        if dim not in out_dict.keys():
            out_dict[dim] = ds.coords[dim].values
    ds_out = create_empty_dataset(out_dict)

    # TODO : adapt chunking depending on the dataset type
    ds_out = ds_out.chunk(chunks={"lat": -1, "lon": -1, "time": 1})

    if weights_filepath and Path(weights_filepath).is_file():
        # Use precomputed weights
        logger.info(f"Using precomputed weights from {weights_filepath}")
        regridder = xe.Regridder(
            ds, ds_out, "bilinear", reuse_weights=True, filename=weights_filepath
        )
    else:
        # Compute weights
        regridder = xe.Regridder(
            ds, ds_out, "bilinear"
        )
        # Save the weights to a file
        regridder.to_netcdf(weights_filepath)
    # Regrid the dataset
    ds_out = regridder(ds)

    return ds_out

def assign_coordinate(
    ds: xr.Dataset, coord_name: str, coord_vals: List[Any], coord_attrs: Dict[str, str]
):
    ds = ds.assign_coords({coord_name: (coord_name, coord_vals, coord_attrs)})
    return ds

def get_glonet_time_attrs(start_date: str):
    glonet_time_attrs = {
        'units': f"days since {start_date} 00:00:00", 'calendar': "proleptic_gregorian"
    }
    return glonet_time_attrs

def get_vars_dims(ds: xr.Dataset) -> Tuple[List[str]]:
    """
    Get the variables and their dimensions from an xarray dataset.
    """
    vars_2d = []
    vars_3d = []
    for var in ds.data_vars:
        # dims = list(ds[var].dims)
        dict_coords = get_grid_coord_names(ds[var])

        if "lat" in dict_coords and "lon" in dict_coords:
            if "depth" not in dict_coords:
                vars_2d.append(var)
            else:
                vars_3d.append(var)
    return (vars_2d, vars_3d)

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
        if name.lower() in {"time", "t", "date", "datetime"}:
            time_coord = ds.coords[name]
            calendar = time_coord.attrs.get("calendar", "standard")

            # Case 1: Already decoded to datetime-like values
            if np.issubdtype(time_coord.dtype, np.datetime64) or isinstance(time_coord.values[0], cftime.DatetimeBase):
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
        return {
            "start": start,
            "end": end,
            "duration": duration,
            "calendar": None
        }

    # Step 3: Extract times and compute
    if len(times) == 0:
        return {
            "start": None,
            "end": None,
            "duration": None,
            "calendar": calendar
        }

    start = times.min()
    end = times.max()
    duration = end - start

    return {
        "start": start,
        "end": end,
        "duration": duration,
        "calendar": calendar
    }

def filter_time_interval(ds: xr.Dataset, start_time: str, end_time: str) -> xr.Dataset:
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
    # Step 1: Analyze the time axis
    time_info = get_time_info(ds)
    
    # Convert start and end times to pandas Timestamp
    start_time = pd.to_datetime(start_time)
    end_time = pd.to_datetime(end_time)

    # Step 2: Check if the time axis is present and valid
    if time_info["start"] is None or time_info["end"] is None:
        return None

    # Step 3: Filter the time coordinate within the given interval
    time_coord = ds.coords["time"]
    mask = (time_coord >= start_time) & (time_coord <= end_time)

    # If no data falls within the time range, return None
    if mask.sum() == 0:
        logger.warning("No data found in the specified time interval.")
        return None

    # Step 4: Return the filtered dataset
    return ds.sel(time=mask)


def extract_spatial_bounds(ds: xr.Dataset) -> dict:
    """
    Extract spatial bounds from an xarray Dataset, handling various coordinate naming conventions.

    Args:
        ds (xr.Dataset): The xarray dataset.

    Returns:
        dict: Dictionary with lat/lon min/max.
    """
    # Tentatives courantes pour les noms de coordonnées
    lat_names = ["lat", "latitude"]
    lon_names = ["lon", "longitude"]

    lat_var = next((name for name in lat_names if name in ds.coords), None)
    lon_var = next((name for name in lon_names if name in ds.coords), None)

    if lat_var is None or lon_var is None:
        raise ValueError("Could not identify latitude or longitude coordinates in dataset.")

    lat_vals = ds[lat_var].values
    lon_vals = ds[lon_var].values

    return {
        "lat_min": float(lat_vals.min()),
        "lat_max": float(lat_vals.max()),
        "lon_min": float(lon_vals.min()),
        "lon_max": float(lon_vals.max()),
    }

def filter_spatial_area(
    ds: xr.Dataset,
    lat_min: float, lat_max: float,
    lon_min: float, lon_max: float
) -> xr.Dataset:
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
    if "latitude" not in ds.coords or "longitude" not in ds.coords:
        logger.warning("Latitude or longitude coordinates not found in the dataset.")
        return None

    # Step 2: Apply the spatial filter
    lat_coord = ds.coords["latitude"]
    lon_coord = ds.coords["longitude"]
    
    lat_mask = (lat_coord >= lat_min) & (lat_coord <= lat_max)
    lon_mask = (lon_coord >= lon_min) & (lon_coord <= lon_max)

    # Step 3: Filter the dataset using the masks
    if lat_mask.sum() == 0 or lon_mask.sum() == 0:
        logger.warning("No data found in the specified spatial area.")
        return None

    # Step 4: Return the filtered dataset
    return ds.sel(latitude=lat_mask, longitude=lon_mask)


def extract_variables(ds: xr.Dataset) -> List[str]:
    return list(ds.data_vars.keys())


def reset_time_coordinates(dataset: xr.Dataset) -> xr.Dataset:
    """
    Remplace les valeurs des coordonnées de temps par une suite débutant à 0.

    Args:
        dataset (xr.Dataset): Le Dataset Xarray à modifier.

    Returns:
        xr.Dataset: Le Dataset avec les coordonnées de temps modifiées.
    """
    if "time" not in dataset.coords:
        raise ValueError("Le dataset ne contient pas de coordonnées 'time'.")

    # Générer une suite de valeurs commençant à 0
    new_time_values = range(len(dataset.coords["time"]))

    # Remplacer les valeurs des coordonnées de temps
    dataset = dataset.assign_coords(time=new_time_values)
    return dataset

def detect_coordinate_system(ds: xr.Dataset) -> CoordinateSystem:
    if "lat" in ds.dims and "lon" in ds.dims:
        return CoordinateSystem("geographic", ("lat", "lon"), crs="EPSG:4326")
    elif "x" in ds.dims and "y" in ds.dims:
        crs = ds.attrs.get("crs", "EPSG:3413")  # ex: stéréographique arctique
        return CoordinateSystem("polar", ("x", "y"), crs=crs)
    else:
        raise ValueError("Unknown coordinate system in dataset.")
