#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Classes and functions for loading xarray datasets."""

import xarray as xr


class DataLoader:
    """Loading NetCDF or Zarr files."""

    @staticmethod
    def load_dataset(file_path: str) -> xr.Dataset | None:
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
            print(f"Error when loading file: {error}")
            return None
