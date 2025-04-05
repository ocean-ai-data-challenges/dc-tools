#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Classes and functions for loading xarray datasets."""

from typing import Optional

from dask.diagnostics import ProgressBar
import xarray as xr
import zarr

from dctools.dcio.dclogger import DCLogger
from dctools.utilities.errors import DCExceptionHandler

class FileLoader:
    """Loading NetCDF or Zarr files."""

    @staticmethod
    def load_dataset(
        file_path: str, exc_handler: DCExceptionHandler,
        dclogger: DCLogger, fail_on_error=True
    ) -> Optional[xr.Dataset]:
        """Load a dataset from NetCDF or Zarr file.

        Args:
            file_path (str): path to the file to load from.

        Returns:
            xr.Dataset: loaded dataset.
        """
        try:
            if file_path.endswith(".nc"):
                dclogger.info(
                    f"Loading dataset from NetCDF file: {file_path}"
                )
                #with ProgressBar():
                ds = xr.open_dataset(file_path)
                return ds
            elif file_path.endswith(".zarr"):
                dclogger.info(
                    f"Loading dataset from Zarr file: {file_path}"
                )
                #with ProgressBar():
                ds = xr.open_zarr(file_path)
                #ds = xr.open_zarr(file_path, chunks='auto')
                return ds
            else:
                raise ValueError("Unsupported file format.")
        except Exception as error:
            exc_handler.handle_exception(
                error, f"Error while loading file: {file_path}",
                fail_on_error=fail_on_error,
            )
            return None

    @staticmethod
    def lazy_load_dataset(
        file_path: str, exc_handler: DCExceptionHandler,
        dclogger: DCLogger,
        fail_on_error=True,
    ) -> Optional[xr.Dataset]:
        """Load a dataset from NetCDF or Zarr file.

        Args:
            file_path (str): path to the file to load from.

        Returns:
            xr.Dataset: loaded dataset.
        """
        try:
            if file_path.endswith(".nc"):
                dclogger.info(
                    f"Loading dataset from NetCDF file: {file_path}"
                )
                #with ProgressBar():
                ds = xr.open_mfdataset(
                    file_path, parallel=True
                )
                #ds = xr.open_dataset(file_path, chunks='auto')
                # ds = xr.open_mfdataset(file_path, parallel=True)
                #ds = ds.chunk(chunks={})
                if "longitude" in ds.variables:
                    # TODO: adapt chunking for each dataset
                    ds = ds.chunk(chunks={"latitude": -1, "longitude": -1, "time": 1})
                else:
                    ds = ds.chunk(chunks={"lat": -1, "lon": -1, "time": 1})
                '''if "longitude" in ds.variables:
                    ds = ds.chunk(chunks={"latitude": 20, "longitude": 20, "time": 1})
                else:
                    ds = ds.chunk(chunks={"lat": 20, "lon": 20, "time": 1})'''
                #ds = xr.open_dataset(file_path, chunks='auto')
                return ds
            elif file_path.endswith(".zarr"):
                #ds = xr.open_zarr(file_path, chunks='auto')
                #ds = xr.open_dataset(file_path, engine="zarr", chunks='auto')
                # use tqdm to show progress bar
                #dclogger.info(
                #    f"Loading dataset from Zarr file: {file_path}"
                #)
                #with ProgressBar():
                #ds = xr.open_zarr(file_path)
                #ds = xr.open_zarr(file_path, chunks='auto')
                #ds = xr.open_dataset(file_path, engine="zarr") #, chunks='auto')
                ds = xr.open_zarr(file_path)
                ds = ds.chunk(chunks='auto')
                return ds
            else:
                raise ValueError(f"Unsupported file format {file_path}.")

        except Exception as error:
            exc_handler.handle_exception(
                error, f"Error when loading file {file_path}",
                fail_on_error=fail_on_error,
            )
            return None
