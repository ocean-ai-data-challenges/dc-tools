#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Misc. functions to aid in the processing xr.Datasets and DataArrays."""

import ast
from datetime import datetime
import traceback
from typing import Any, Dict, List, Optional, Tuple

import cftime
from loguru import logger
import numpy as np
import pandas as pd
from pathlib import Path
import xarray as xr
import xesmf as xe

from dctools.data.coordinates import (
    GEO_STD_COORDS
)


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


def rename_coordinates(ds: xr.Dataset, rename_dict: Optional[dict] = None):
    """Rename coordinates according to a given dictionary."""
    try:
        if not rename_dict:
            return ds

        if isinstance(rename_dict, str):
            rename_dict = ast.literal_eval(rename_dict)

        # Remove None values from the dictionary
        #rename_dict = {k: v for k, v in rename_dict.items() if v is not None}
        # keep only keys that are in the dataset dimensions
        rename_dict = {k: v for k, v in rename_dict.items() if k in list(ds.dims)}
        # remove entries when key is same as value, and None values
        rename_dict = {k: v for k, v in rename_dict.items() if k != v  and v is not None}
        if not rename_dict or len (rename_dict) == 0:
            return ds
        ds = ds.rename_dims(rename_dict)
        return ds.rename_dims(rename_dict)
    except Exception as e:
        logger.error(f"Error renaming coordinates: {e}")
        raise ValueError(f"Failed to rename coordinates in dataset: {e}") from e


def rename_variables(ds: xr.Dataset, rename_dict: Optional[dict] = None):
    """Rename variables according to a given dictionary."""
    try:
        if not rename_dict:
            return ds
 
        if isinstance(rename_dict, str):
            rename_dict = ast.literal_eval(rename_dict)
        # keep only keys that are in the dataset dimensions
        rename_dict = {k: v for k, v in rename_dict.items() if k in list(ds.data_vars)}
        # Remove invalid values from the dictionary
        rename_dict = {k: v for k, v in rename_dict.items() if k != v and v is not None}
        if not rename_dict or len (rename_dict) == 0:
            return ds
        rename_ds = ds.rename_vars(rename_dict)
        new_vars = list(rename_ds.data_vars)
        logger.debug(f"Dataset variables after renaming: {new_vars}")
        return rename_ds
    except Exception as e:
        logger.error(f"Error renaming variables: {e}")
        raise ValueError(f"Failed to rename variables in dataset: {traceback.format_exc()}") from e


def rename_coords_and_vars(
    ds: xr.Dataset,
    rename_coords_dict: Optional[dict] = None,
    rename_vars_dict: Optional[dict] = None,
):
    """Rename variables and coordinates according to given dictionaries."""
    #logger.debug(f"Renaming coordinates in dataset with dictionary: {rename_coords_dict}")
    try:
        rename_ds = rename_coordinates(ds, rename_coords_dict)
        #logger.debug(f"Renaming variables in dataset with dictionary: {rename_vars_dict}")
        rename_ds = rename_variables(rename_ds, rename_vars_dict)

        return rename_ds
    except Exception as e:
        logger.error(f"Error renaming coordinates or variables: {e}")
        raise ValueError(f"Failed to rename coordinates or variables in dataset: {e}") from e

def subset_variables(ds: xr.Dataset, list_vars: List[str]):
    """Extract a sub-dataset containing only listed variables, preserving attributes."""

    for variable_name in ds.variables:
        var_std_name = ds[variable_name].attrs.get("standard_name",'').lower()
        if not var_std_name:
            var_std_name = ds[variable_name].attrs.get("std_name", '').lower()

    # Crée un sous-dataset avec uniquement les variables listées
    subset = xr.Dataset({var: ds[var].copy(deep=True) for var in list_vars if var in ds})

    # Détecter les coordonnées présentes dans le sous-dataset
    coords_to_set = [c for c in subset.data_vars if c in ds.coords]
    if coords_to_set:
        subset = subset.set_coords(coords_to_set)

    # Supprime les coordonnées orphelines (non utilisées)
    subset = subset.reset_coords(drop=True)

    for variable_name in subset.variables:
        var_std_name = subset[variable_name].attrs.get("standard_name",'').lower()
        if not var_std_name:
            var_std_name = subset[variable_name].attrs.get("std_name", '').lower()

    return subset


def interpolate_dataset(
        ds: xr.Dataset, ranges: Dict[str, np.ndarray],
        weights_filepath: Optional[str] = None,
    ) -> xr.Dataset:

    for variable_name in ds.variables:
        var_std_name = ds[variable_name].attrs.get("standard_name",'').lower()
        if not var_std_name:
            var_std_name = ds[variable_name].attrs.get("std_name", '').lower()

    # 1. Sauvegarder les attributs des coordonnées AVANT interpolation
    coords_attrs = {}
    for coord in ds.coords:
        coords_attrs[coord] = ds.coords[coord].attrs.copy()

    # (optionnel) Sauvegarder aussi les attrs des variables si besoin
    vars_attrs = {}
    for var in ds.data_vars:
        vars_attrs[var] = ds[var].attrs.copy()

    for key in ranges.keys():
        assert(key in list(ds.dims))

    out_dict = {}
    for key in ranges.keys():
        out_dict[key] = ranges[key]
    for dim in GEO_STD_COORDS.keys():
        if dim not in out_dict.keys():
            out_dict[dim] = ds.coords[dim].values
    ds_out = create_empty_dataset(out_dict)

    # TODO : adapt chunking depending on the dataset type
    ds_out = ds_out.chunk(chunks={"lat": -1, "lon": -1, "time": 1})

    if weights_filepath and Path(weights_filepath).is_file():
        # Use precomputed weights
        logger.debug(f"Using interpolation precomputed weights from {weights_filepath}")
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

    # 2. Réaffecter les attributs des variables (déjà fait dans ton code)
    for var in ds_out.data_vars:
        if var in vars_attrs:
            ds_out[var].attrs = vars_attrs[var].copy()

    # 3. Réaffecter les attributs des coordonnées
    for coord in ds_out.coords:
        if coord in coords_attrs:
            # Crée un nouveau DataArray avec les attrs sauvegardés
            new_coord = xr.DataArray(
                ds_out.coords[coord].values,
                dims=ds_out.coords[coord].dims,
                attrs=coords_attrs[coord].copy()
            )
            ds_out = ds_out.assign_coords({coord: new_coord})

    for variable_name in ds.variables:
        var_std_name = ds[variable_name].attrs.get("standard_name",'').lower()
        if not var_std_name:
            var_std_name = ds[variable_name].attrs.get("std_name", '').lower()

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
