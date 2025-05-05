#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Classes and functions for loading xarray datasets."""

from typing import Any, Optional

from loguru import logger
import xarray as xr
import zarr

# from dctools.utilities.xarray_utils import standard_rename_coords

class FileLoader:
    """Loading NetCDF or Zarr files."""

    @staticmethod
    def open_dataset_auto(path: str, manager: Any) -> xr.Dataset:
        """
        Open a dataset automatically, handling both NetCDF and Zarr formats.

        Args:
            path (str): Path to the dataset (local or remote).
            manager (Any): Connection manager providing the filesystem.

        Returns:
            xr.Dataset: Opened dataset.
        """
        try:
            if path.endswith(".zarr"):
                #logger.info(f"Opening Zarr dataset: {path}")
                return xr.open_zarr(manager.params.fs.get_mapper(path))
            else:
                #logger.info(f"Opening NetCDF dataset: {path}")
                return xr.open_dataset(manager.params.fs.open(path), engine="netcdf4")
        except Exception as exc:
            logger.error(f"Failed to open dataset {path}: {repr(exc)}")
            raise

    @staticmethod
    def load_dataset(
        file_path: str,
        adaptive_chunking: bool = False,
    ) -> xr.Dataset | None:
        """Load a dataset from a local NetCDF or Zarr file.

        Parameters
        ----------
        file_path : str
            Path to NetCDF or Zarr file.
        adaptive_chunking : bool, optional
            Whether to adapt chunking to the specific dataset being loaded. This
            feature is not supported for Zarr datasets and is experimental at
            best. By default False.

        Returns
        -------
        xr.Dataset | None
            Loaded Xarray Dataset, or None if error while loading and `fail_on_error = True`

        Raises
        ------
        ValueError
            If `file_path` does not point to a NetCDF of Zarr file.
        """
        try:
            if file_path.endswith(".nc"):
                #logger.info(
                #    f"Loading dataset from NetCDF file: {file_path}"
                #)
                ds = xr.open_dataset(file_path, chunks='auto')

                if adaptive_chunking:
                    if "latitude" in ds.dims and "time" in ds.dims:
                        # TODO: adapt chunking for each dataset
                        ds = ds.chunk(chunks={"latitude": -1, "longitude": -1, "time": 1})
                    elif "lat" in ds.dims and "time" in ds.dims:
                        ds = ds.chunk(chunks={"lat": -1, "lon": -1, "time": 1})
                    else:
                        ds = ds.chunk(chunks='auto')
                else:
                    ds = ds.chunk(chunks='auto')
                return ds
            elif file_path.endswith(".zarr"):
                ds = xr.open_zarr(file_path)
                return ds
            else:
                raise ValueError(f"Unsupported file format {file_path}.")

        except Exception as error:
            logger.error(
                f"Error when loading file {file_path}: {repr(error)}"
            )
            return None
