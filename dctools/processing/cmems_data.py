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
    dclogger: logging.Logger, exception_handler: DCExceptionHandler
) -> xr.Dataset:
    """Create a forecast dataset from a list of CMEMS files.

    Args:
        `nc_path` (str): path to the CMEMS files
        `list_nc_files` (List[str]): list of CMEMS files
        `ref_data` (xr.Dataset): reference dataset
        `dclogger` (logging.Logger): logger instance
        `exception_handler` (DCExceptionHandler): exception handler instance

    Returns:
        `glorys_data` (xr.Dataset): Glorys forecast dataset
    """
    glorys_data = xr.Dataset()
    dclogger.info(f"Concatenate {len(list_nc_files)} Glorys forecast files.")
    time_step = 0
    try:
        for fname in list_nc_files:
            fpath = os.path.join(nc_path, fname)
            tmp_ds = DataLoader.lazy_load_dataset(fpath, exception_handler)
            assert tmp_ds is not None, f"Error while loading dataset: {tmp_ds}."
            tmp_ds = tmp_ds.drop_vars(
                ["bottomT", "usi", "vsi", "mlotst", "siconc", "sithick"]
            )
            tmp_ds = tmp_ds.assign_coords(time=[time_step])
            tmp_ds['time'].attrs = {
                "units": "days since 2024-01-04 00:00:00",
                "calendar": "proleptic_gregorian"
            }
            if time_step == 0:
                glorys_data = tmp_ds
            else:
                glorys_data = GriddedDataProcessor.concatenate(
                    glorys_data, tmp_ds, dim='time'
                )
            tmp_ds.close()
            time_step += 1

        time_attrs = {
            'units': "days since 2024-05-02 00:00:00", 'calendar': "proleptic_gregorian"
        }
        glorys_data = glorys_data.assign_coords(
            {'time': ('time', [i for i in range(0,len(list_nc_files))], time_attrs)}
        )
        glorys_data = rename_coordinates(glorys_data, DICT_RENAME_CMEMS)

        depth_vals = ref_data.coords['depth'].values
        depth_indices = [
            idx for idx in range(
                0, glorys_data.depth.values.size
            ) if glorys_data.depth.values[idx] in depth_vals
        ]
        glorys_data = glorys_data.isel(
            depth=depth_indices
        )

        dclogger.info("Regridding to match Glonet resolution (1/4 degree).")
        regridder = xe.Regridder(
            glorys_data, ref_data, method='bilinear', unmapped_to_nan=True
        )
        glorys_data = regridder(glorys_data)
    except Exception as e:
        exception_handler.handle_exception(
            e, "Error while creating Glorys forecast dataset."
        )

    return glorys_data
