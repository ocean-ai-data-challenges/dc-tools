"""Misc. functions to aid in the processing xr.Datasets and DataArrays."""

from typing import Dict

import xarray as xr


# Possible names of coordinates that we want to check for
LATITUDE_NAMES = ["lat", "latitude", "LAT", "LATITUDE"]
LONGITUDE_NAMES = ["lon", "longitude", "LON", "LONGITUDE"]
DEPTH_NAMES = ["depth", "DEPTH", "height", "HEIGHT"]
TIME_NAMES = ["time", "TIME"]


def get_grid_coord_names(
    data: xr.Dataset | xr.DataArray,
) -> Dict[str, str]:
    """
    Get the names of the coordinates in `data`.

    Return a dictionary with "lat", "lon", "depth" and "time" as keys
    and the names of the corresponding coordinates (if they exist) as
    values.
    The names are determined by checking against the `XXX_NAMES` constants defined
    in this file for any existing coordinates in `data`
    """
    coord_name_dict = {}

    # There's probably a less disgusting way of doing this...
    lon_set = set(LONGITUDE_NAMES).intersection(data.coords)
    coord_name_dict["lon"] = None if len(lon_set) == 0 else next(iter(lon_set))
    lat_set = set(LATITUDE_NAMES).intersection(data.coords)
    coord_name_dict["lat"] = None if len(lat_set) == 0 else next(iter(lat_set))
    depth_set = set(DEPTH_NAMES).intersection(data.coords)
    coord_name_dict["depth"] = None if len(depth_set) == 0 else next(iter(depth_set))
    time_set = set(TIME_NAMES).intersection(data.coords)
    coord_name_dict["time"] = None if len(time_set) == 0 else next(iter(time_set))

    return coord_name_dict


def get_grid_dim_names(
    data: xr.Dataset | xr.DataArray,
) -> Dict[str, str]:
    """
    Get the names of the dimensions in `data`.

    Return a dictionary with "lat", "lon", "depth" and "time" as keys
    and the names of the corresponding coordinates (if they exist) as
    values.
    The names are determined by checking against the `XXX_NAMES` constants defined
    in this file for any existing coordinates in `data`
    """
    dim_name_dict = {}

    # There's probably a less disgusting way of doing this...
    lon_set = set(LONGITUDE_NAMES).intersection(data.dims)
    dim_name_dict["lon"] = None if len(lon_set) == 0 else next(iter(lon_set))
    lat_set = set(LATITUDE_NAMES).intersection(data.dims)
    dim_name_dict["lat"] = None if len(lat_set) == 0 else next(iter(lat_set))
    depth_set = set(DEPTH_NAMES).intersection(data.dims)
    dim_name_dict["depth"] = None if len(depth_set) == 0 else next(iter(depth_set))
    time_set = set(TIME_NAMES).intersection(data.dims)
    dim_name_dict["time"] = None if len(time_set) == 0 else next(iter(time_set))

    return dim_name_dict
