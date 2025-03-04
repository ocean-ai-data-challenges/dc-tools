#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Classes and functions for loading xarray datasets."""

from typing import Optional

import xarray as xr


class DataLoader:
    """Loading NetCDF or Zarr files."""

    @staticmethod
    def load_dataset(file_path: str) -> Optional[xr.Dataset]:
        """Load a dataset from NetCDF or Zarr file.

        Args:
            file_path (str): path to the file to load from.

        Returns:
            xr.Dataset: loaded dataset.
        """
        try:
            if file_path.endswith(".nc"):
                return xr.open_dataset(file_path)
            if file_path.endswith(".zarr"):
                return xr.open_zarr(file_path)
            raise ValueError("Unsupported file format.")
        except Exception as error:
            print(f"Error when loading file {file_path} : {error}")
            return None

    @staticmethod
    def lazy_load_dataset(file_path: str) -> Optional[xr.Dataset]:
        """Load a dataset from NetCDF or Zarr file.

        Args:
            file_path (str): path to the file to load from.

        Returns:
            xr.Dataset: loaded dataset.
        """
        try:
            ds = xr.open_mfdataset(file_path)
            if "longitude" in ds.variables:
                ds = ds.chunk(chunks={"latitude": -1, "longitude": -1, "time": 1})
            else:
                ds = ds.chunk(chunks={"lat": -1, "lon": -1, "time": 1})
            #ds = xr.open_dataset(file_path, chunks='auto')
            return ds
        except Exception as error:
            print(f"Error when loading file {file_path} : {error}")
            return None