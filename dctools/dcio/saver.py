#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Classes and functions for saving xarray datasets."""
from typing import Any, Dict, Optional

import xarray as xr

from dctools.dcio.dclogger import DCLogger
from dctools.utilities.errors import DCExceptionHandler

class DataSaver:
    """Saving datasets."""

    @staticmethod
    def save_dataset(
        ds: xr.Dataset,
        file_path: str,
        exception_handler: DCExceptionHandler,
        dclogger: DCLogger,
        **kwargs: Dict[str, Any]
    ) -> None:
        """Save a dataset in a NetCDF file.

        Args:
            ds (xr.Dataset): Dataset to save.
            file_path (str): path to output file.
        """
        default_attrs = dict(
            file_format="netcdf", mode="w", append_dim=str(), compute=True,
        )
        kwargs.update((k,v) for k,v in default_attrs.items() if k not in kwargs)
        try:
            if kwargs["file_format"] == "netcdf":
                dclogger.info(
                    f"Saving dataset in netcdf format: {file_path}"
                )
                ds.to_netcdf(file_path, format="NETCDF4", engine="netcdf4")
            elif kwargs["file_format"] == "zarr":
                dclogger.info(
                    f"Saving dataset in zarr format: {file_path}"
                )
                if "append_dim" in kwargs:
                    # Append to existing zarr file
                    ds.to_zarr(
                        store=file_path,
                        append_dim=kwargs["append_dim"],
                        compute=kwargs["compute"],
                    )
                else:
                    # Create new zarr file
                    ds.to_zarr(
                        store=file_path, compute=kwargs["compute"], mode=kwargs["mode"],
                    )
            else:
                raise ValueError(f"Unsupported file format: {kwargs["file_format"]}")
        except FileNotFoundError as error:
            message = (f"File not found: {file_path}")
            exception_handler.handle_exception(error, message)
        except PermissionError as error:
            message = (f"Permission denied: {file_path}")
            exception_handler.handle_exception(error, message)
        except Exception as error:
            message = (f"Error when saving dataset to {file_path}")
            exception_handler.handle_exception(error, message)
