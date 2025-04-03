#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Tools for handling Copernicus Marine data."""

import datetime
import logging
import os
from typing import List, Optional

import numpy as np
import pandas as pd
import xarray as xr
import xesmf as xe

from dctools.dcio.loader import FileLoader
from dctools.dcio.saver import DataSaver
from dctools.utilities.errors import DCExceptionHandler
from dctools.processing.gridded_data import GriddedDataProcessor
from dctools.utilities.xarray_utils import rename_coordinates,\
    get_glonet_time_attrs, assign_coordinate, RANGES_GLONET, GLONET_ENCODING


def create_glorys_ndays_forecast(
    nc_path: str,
    list_nc_files: List[str],
    # ref_data: xr.Dataset,
    start_date: datetime.datetime,
    zarr_path: str,
    transform_fct: Optional[callable],
    dclogger: logging.Logger,
    exception_handler: DCExceptionHandler
) -> xr.Dataset:
    """Create a forecast dataset from a list of CMEMS files.

    Args:
        `nc_path` (str): path to the CMEMS files
        `list_nc_files` (List[str]): list of CMEMS files
        `ref_data` (xr.Dataset): reference dataset
        `start_time` (str): start date for forecast
        `zarr_path` (str): path to the zarr file
        `dclogger` (logging.Logger): logger instance
        `exception_handler` (DCExceptionHandler): exception handler instance

    Returns:
        `glorys_data` (xr.Dataset): Glorys forecast dataset
    """

    """dim_lat = len(RANGES_GLONET['lat'])
    dim_lon = len(RANGES_GLONET['lon'])
    dim_depth = len(RANGES_GLONET['depth'])
    dim_time = len(RANGES_GLONET['time'])
    times = [dat.strftime('%Y-%m-%d') for dat in pd.date_range(start=start_date,freq='1D',periods=dim_time)]
    glorys_data = xr.Dataset(
        data_vars=dict(
            thetao=(["time", "depth", "lat", "lon"], np.random.randn(dim_time, dim_depth, dim_lat, dim_lon)),
            zos=(["time", "lat", "lon"], np.random.randn(dim_time, dim_lat, dim_lon)),
            uo=(["time", "depth", "lat", "lon"], np.random.randn(dim_time, dim_depth, dim_lat, dim_lon)),
            vo=(["time", "depth", "lat", "lon"], np.random.randn(dim_time, dim_depth, dim_lat, dim_lon)),
            so=(["time", "depth", "lat", "lon"], np.random.randn(dim_time, dim_depth, dim_lat, dim_lon)),

        ),
        coords=dict(
            lon=("lon", RANGES_GLONET['lon']),
            lat=("lat", RANGES_GLONET['lat']),
            depth=("depth", RANGES_GLONET['depth']),
            time=times, #RANGES_GLONET['time'], "attrs", get_glonet_time_attrs(start_date)),
        ),
    )"""
    dclogger.info(f"Concatenate {len(list_nc_files)} Glorys forecast files.")
    time_step = 0
    try:
        # concatenate downloaded files from CMEMS
        for fname in list_nc_files:
            fpath = os.path.join(nc_path, fname)
            tmp_ds = FileLoader.lazy_load_dataset(fpath, exception_handler)
            tmp_ds = transform_fct(tmp_ds)

            tmp_ds = assign_coordinate(
                tmp_ds, "time", coord_vals=[time_step],
                #tmp_ds, "time", coord_vals=[times[time_step]],
                coord_attrs=get_glonet_time_attrs(start_date)
            )
            assert tmp_ds is not None, f"Error while loading dataset: {tmp_ds}."

            if time_step == 0:
                DataSaver.save_dataset(
                    tmp_ds, zarr_path, exception_handler,
                    #file_format="zarr", mode="r+",
                    file_format="zarr", mode="w",
                    compute=True,
                )
            else:
                DataSaver.save_dataset(
                    tmp_ds, zarr_path, exception_handler,
                    file_format="zarr", mode="a", append_dim='time',
                    compute=True,
                )
            tmp_ds.close()
            time_step += 1

        glorys_data = FileLoader.lazy_load_dataset(zarr_path, exception_handler)

    except Exception as e:
        exception_handler.handle_exception(
            e, "Error while creating Glorys forecast dataset."
        )

    return glorys_data
