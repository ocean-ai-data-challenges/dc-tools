#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Tools for handling Copernicus Marine data."""

import logging
import os
from typing import List

import xarray as xr
import xesmf as xe

from dctools.dcio.loader import DataLoader
from dctools.utilities.errors import DCExceptionHandler
from dctools.processing.gridded_data import GriddedDataProcessor
from dctools.utilities.xarray_utils import rename_coordinates, DICT_RENAME_CMEMS


def create_glorys_ndays_forecast(
    nc_path: str, list_nc_files: List[str], ref_data: xr.Dataset,
    start_date: str,
    dclogger: logging.Logger, exception_handler: DCExceptionHandler
) -> xr.Dataset:
    """Create a forecast dataset from a list of CMEMS files.

    Args:
        `nc_path` (str): path to the CMEMS files
        `list_nc_files` (List[str]): list of CMEMS files
        `ref_data` (xr.Dataset): reference dataset
        `start_time` (str): start date for forecast
        `dclogger` (logging.Logger): logger instance
        `exception_handler` (DCExceptionHandler): exception handler instance

    Returns:
        `glorys_data` (xr.Dataset): Glorys forecast dataset
    """
    glorys_data = xr.Dataset()
    dclogger.info(f"Concatenate {len(list_nc_files)} Glorys forecast files.")
    time_step = 0
    try:
        # concatenate downloaded files from CMEMS
        for fname in list_nc_files:
            fpath = os.path.join(nc_path, fname)
            tmp_ds = DataLoader.lazy_load_dataset(fpath, exception_handler)
            assert tmp_ds is not None, f"Error while loading dataset: {tmp_ds}."
            tmp_ds = tmp_ds.drop_vars(
                ["bottomT", "usi", "vsi", "mlotst", "siconc", "sithick"]
            )
            if time_step == 0:
                glorys_data = tmp_ds
            else:
                glorys_data = GriddedDataProcessor.concatenate(
                    glorys_data, tmp_ds, dim='time'
                )
            tmp_ds.close()
            time_step += 1

        # longitude --> lon, latitude --> lat
        glorys_data = rename_coordinates(glorys_data, DICT_RENAME_CMEMS)

        # select only the subset of depth values that matches the values in Glonet forecast
        depth_vals = ref_data.coords['depth'].values
        depth_indices = [
            idx for idx in range(
                0, glorys_data.depth.values.size
            ) if glorys_data.depth.values[idx] in depth_vals
        ]
        glorys_data = glorys_data.isel(
            depth=depth_indices
        )
        # regrid to match the resolution of the reference dataset
        dclogger.info("Regridding to match Glonet resolution (1/4 degree).")
        regridder = xe.Regridder(
            glorys_data, ref_data, method='bilinear', unmapped_to_nan=True
        )
        glorys_data = regridder(glorys_data)

        '''# Get time attributes from reference dataset
        units_ref_time_attrs = ref_data.coords['time'].attrs
        calendar_ref_time_attrs = ref_data.coords['time'].attrs
        print(calendar_ref_time_attrs)
        dclogger.info(f"Time attibutes (units): {units_ref_time_attrs}")
        dclogger.info(f"Time attibutes (calendar): {calendar_ref_time_attrs}")'''

        # modify time coordinate to mathe Glonet forecast's time coordinate
        time_attrs = {
            'units': f"days since {start_date} 00:00:00", 'calendar': "proleptic_gregorian"
        }
        glorys_data = glorys_data.assign_coords(
            # {'time': ('time', [i for i in range(0,len(list_nc_files))], time_attrs)}
            {'time': ('time', [i for i in range(0,len(list_nc_files))], time_attrs)}
        )
    except Exception as e:
        exception_handler.handle_exception(
            e, "Error while creating Glorys forecast dataset."
        )

    return glorys_data
