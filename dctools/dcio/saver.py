#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Classes and functions for saving xarray datasets."""
from typing import Any, Dict, Optional
import xarray as xr

from dctools.utilities.errors import DCExceptionHandler

class DataSaver:
    """Saving datasets."""

    @staticmethod
    def save_dataset(
        ds: xr.Dataset,
        file_path: str,
        exception_handler: DCExceptionHandler,
        file_format: Optional[str] = "netcdf",
        mode: Optional[str] = "w",
        append_dim: Optional[str] = None,
        compute: Optional[bool] = True,
    ) -> None:
        """Save a dataset in a NetCDF file.

        Args:
            ds (xr.Dataset): Dataset to save.
            file_path (str): path to output file.
        """
        try:
            if file_format == "netcdf":
                ds.to_netcdf(file_path, format="NETCDF4", engine="netcdf4")
            elif file_format == "zarr":
                #print(f"Saving dataset in zarr format: {file_path}")
                if append_dim:
                    ds.to_zarr(
                        store=file_path,
                        compute=compute,
                        append_dim=append_dim,
                    )
                else:
                    ds.to_zarr(
                        store=file_path, compute=compute, mode=mode,
                    )
            else:
                raise ValueError(f"Unsupported file format: {file_format}")
        except FileNotFoundError as error:
            message = (f"File not found: {file_path}")
            exception_handler.handle_exception(error, message)
        except PermissionError as error:
            message = (f"Permission denied: {file_path}")
            exception_handler.handle_exception(error, message)
        except Exception as error:
            message = (f"Error when saving dataset to {file_path}")
            exception_handler.handle_exception(error, message)
