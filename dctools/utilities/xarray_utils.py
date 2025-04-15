#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Misc. functions to aid in the processing xr.Datasets and DataArrays."""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pathlib import Path
import xarray as xr
import xesmf as xe

# Possible names of coordinates that we want to check for
LATITUDE_NAMES = ["lat", "latitude", "LAT", "LATITUDE"]
LONGITUDE_NAMES = ["lon", "longitude", "LON", "LONGITUDE"]
DEPTH_NAMES = ["depth", "DEPTH", "height", "HEIGHT"]
TIME_NAMES = ["time", "TIME"]
DICT_RENAME_CMEMS = dict(longitude="lon", latitude="lat")
LIST_VARS_GLONET = ["thetao", "zos", "uo", "vo", "so", "depth", "lat", "lon", "time"]
LIST_VARS_GLONET_UNITTEST = ["thetao", "zos", "uo"]
GLONET_DEPTH_VALS = [0.494025, 47.37369, 92.32607, 155.8507, 222.4752, 318.1274, 380.213, 
        453.9377, 541.0889, 643.5668, 763.3331, 902.3393, 1245.291, 1684.284, 
        2225.078, 3220.82, 3597.032, 3992.484, 4405.224, 4833.291, 5274.784]

GLONET_TIME_VALS = range(0, 10)

RANGES_GLONET = {
    "lat": np.arange(-78, 90, 0.25),
    "lon": np.arange(-180, 180, 0.25),
    "depth": GLONET_DEPTH_VALS,
    "time": GLONET_TIME_VALS,
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
    list_coords = list(data.coords) if hasattr(data, "coords") else list(data)
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

def get_grid_dim_names(
    data: xr.Dataset | xr.DataArray,
) -> Dict[str, Optional[str]]:
    """
    Get the names of the dimensions in `data`.

    Return a dictionary with "lat", "lon", "depth" and "time" as keys
    and the names of the corresponding coordinates (if they exist) as
    values.
    The names are determined by checking against the `XXX_NAMES` constants defined
    in this file for any existing coordinates in `data`
    """
    dim_name_dict = {}
    list_dims = list(data.coords) if hasattr(data, "coords") else list(data)

    # There's probably a less disgusting way of doing this...
    lon_set = set(LONGITUDE_NAMES).intersection(list_dims)
    dim_name_dict["lon"] = None if len(lon_set) == 0 else next(iter(lon_set))
    lat_set = set(LATITUDE_NAMES).intersection(list_dims)
    dim_name_dict["lat"] = None if len(lat_set) == 0 else next(iter(lat_set))
    depth_set = set(DEPTH_NAMES).intersection(list_dims)
    dim_name_dict["depth"] = None if len(depth_set) == 0 else next(iter(depth_set))
    time_set = set(TIME_NAMES).intersection(list_dims)
    dim_name_dict["time"] = None if len(time_set) == 0 else next(iter(time_set))

    return dim_name_dict

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

    if "depth" in ds.dims:
        ds_out = xr.Dataset(
            {
                "lat": (["lat"], ranges['lat']),
                "lon": (["lon"], ranges['lon']),
                "depth": (["depth"], ranges['depth']),
                "time": (["time"], ranges['time']),
            }
        )
    else:
        ds_out = xr.Dataset(
            {
                "lat": (["lat"], ranges['lat']),
                "lon": (["lon"], ranges['lon']),
                "time": (["time"], ranges['time']),
            }
        )
    # TODO : adapt chunking depending on the dataset type
    ds_out = ds_out.chunk(chunks={"lat": -1, "lon": -1, "time": 1})

    if weights_filepath and Path(weights_filepath).is_file():
        # Use precomputed weights
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