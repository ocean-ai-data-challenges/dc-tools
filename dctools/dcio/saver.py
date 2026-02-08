#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Classes and functions for saving xarray datasets."""
from typing import Any

from loguru import logger
import xarray as xr
import zarr


class DataSaver:
    """Saving datasets."""

    @staticmethod
    def save_dataset(
        ds: xr.Dataset,
        file_path: str,
        **kwargs: Any
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
                logger.info(
                    f"Saving dataset in netcdf format: {file_path}"
                )
                ds.to_netcdf(file_path, format="NETCDF4", engine="netcdf4")
            elif kwargs["file_format"] == "zarr":
                logger.info(
                    f"Saving dataset in zarr format: {file_path}"
                )
                if kwargs["append_dim"] is not None and len(kwargs["append_dim"]) > 0:
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
                raise ValueError(f"Unsupported file format: {kwargs['file_format']}")
        except FileNotFoundError as error:
            message = (f"File not found: {file_path}: {repr(error)}")
            logger.error(message)
        except PermissionError as error:
            message = (f"Permission denied: {file_path}: {repr(error)}")
            logger.error(message)
        except Exception as error:
            message = (f"Error when saving dataset to {file_path}: {repr(error)}")
            logger.error(message)


def progressive_zarr_save(ds: xr.Dataset, zarr_path: str):
    """
    Save an xarray Dataset to Zarr format progressively (variable by variable).

    This helps in reducing memory usage when saving large datasets.

    Args:
        ds (xr.Dataset): The dataset to save.
        zarr_path (str): The path to the Zarr store.
    """
    try:
        # Create Zarr store (directory or file)
        store = zarr.DirectoryStore(zarr_path)

        # Write first variable (creates the group)
        first_var = list(ds.data_vars)[0]
        ds[[first_var]].to_zarr(store, mode="w", consolidated=False)

        # Add other variables one by one
        for var in list(ds.data_vars)[1:]:
            ds[[var]].to_zarr(store, mode="a", consolidated=False)
    except Exception as exc:
        logger.error(f"Error during progressive save to Zarr: {exc}")
