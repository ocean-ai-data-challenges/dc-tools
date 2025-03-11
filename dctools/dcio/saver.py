#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Classes and functions for saving xarray datasets."""

import xarray as xr

from dctools.utilities.errors import DCExceptionHandler

class DataSaver:
    """Saving datasets."""

    @staticmethod
    def save_dataset(
        ds: xr.Dataset,
        file_path: str,
        exception_handler: DCExceptionHandler,
    ) -> None:
        """Save a dataset in a NetCDF file.

        Args:
            ds (xr.Dataset): Dataset to save.
            file_path (str): path to output file.
        """
        try:
            ds.to_netcdf(file_path, format="NETCDF4", engine="netcdf4")
        except Exception as error:
            message = (f"Error when saving dataset to {file_path}")
            exception_handler.handle_exception(error, message)
