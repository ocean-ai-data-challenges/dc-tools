#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Tools for handling CMEMS data."""

import os
import re
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger
import xarray as xr

from dctools.dcio.loader import FileLoader
from dctools.dcio.saver import DataSaver
from dctools.utilities.xarray_utils import (
    get_glonet_time_attrs, assign_coordinate
)

def create_glorys_ndays_forecast(
    nc_path: str,
    list_nc_files: List[str],
    start_date: str,
    zarr_path: str,
    transform_fct: Optional[callable],
) -> xr.Dataset:
    """Create a forecast dataset from a list of CMEMS files.

    Args:
        `nc_path` (str): path to the CMEMS files
        `list_nc_files` (List[str]): list of CMEMS files
        `ref_data` (xr.Dataset): reference dataset
        `start_time` (str): start date for forecast
        `zarr_path` (str): path to the zarr file

    Returns:
        `glorys_data` (xr.Dataset): Glorys forecast dataset
    """

    logger.info(f"Concatenate {len(list_nc_files)} Glorys forecast files.")
    time_step = 0
    try:
        # concatenate downloaded files from CMEMS
        for fname in list_nc_files:
            fpath = os.path.join(nc_path, fname)
            tmp_ds = FileLoader.lazy_load_dataset(fpath)
            tmp_ds = transform_fct(tmp_ds)
            tmp_ds = assign_coordinate(
                tmp_ds, "time", coord_vals=[time_step],
                coord_attrs=get_glonet_time_attrs(start_date)
            )
            assert tmp_ds is not None, f"Error while loading dataset: {tmp_ds}."

            if time_step == 0:
                DataSaver.save_dataset(
                    tmp_ds, zarr_path,
                    file_format="zarr", mode="w",
                    compute=True,
                )
            else:
                DataSaver.save_dataset(
                    tmp_ds, zarr_path,
                    file_format="zarr", mode="a", append_dim='time',
                    compute=True,
                )
            time_step += 1

        glorys_data = FileLoader.lazy_load_dataset(zarr_path)

    except Exception as err:
        logger.error(
            f"Error while creating Glorys forecast dataset: {repr(err)}"
        )
        raise

    return glorys_data



def extract_dates_from_filename(filename: str) -> Optional[Tuple[str, str]]:
    """
    Extract start and end dates from a CMEMS filename.

    Args:
        filename (str): The name of the file.

    Returns:
        Optional[Tuple[str, str]]: A tuple of (start_date, end_date) in 'YYYY-MM-DD' format,
                                   or None if no dates are found.
    """
    # Regex pour extraire une plage de dates (YYYYMMDD-YYYYMMDD)
    match_range = re.search(r"(\d{8})-(\d{8})", filename)
    if match_range:
        start_date = match_range.group(1)
        end_date = match_range.group(2)
        return start_date[:4] + "-" + start_date[4:6] + "-" + start_date[6:], \
               end_date[:4] + "-" + end_date[4:6] + "-" + end_date[6:]

    # Regex pour extraire une seule date (YYYYMMDD)
    match_single = re.search(r"(\d{8})", filename)
    if match_single:
        date = match_single.group(1)
        return date[:4] + "-" + date[4:6] + "-" + date[6:], date[:4] + "-" + date[4:6] + "-" + date[6:]

    return None

