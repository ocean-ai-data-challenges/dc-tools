#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Classes and functions for processing Argo float data."""

from typing import Optional, Tuple

import numpy as np
import xarray as xr

from dctools.data.coordinates import CoordinateSystem

class ArgoDataProcessor:
    """Processor for argo data."""

    @staticmethod
    def subset_argo(
        data: xr.Dataset | xr.DataArray,
        lat_range: Optional[Tuple[float, float]] = None,
        lon_range: Optional[Tuple[float, float]] = None,
        time_range: Optional[Tuple[np.datetime64, np.datetime64]] = None,
        coord_name_dict: Optional[dict] = None,
    ) -> xr.Dataset | xr.DataArray:
        """
        Subset the area defined by `lat_range`, `lon_range` and `time_range`.

        Args:
            `data` (xr.Dataset or xr.DataArray): Xarray object from argopy
            `lat_range` (Tuple of two floats): lower and upper latitude
                bounds
            `lon_range` (Tuple of two floats): lower and upper longitude
                bounds
            `time_range` (Tuple of two datetimes): start and end of
                selected period

        Returns:
            `subset`: An object of the same type as `data` containing the
                specified range
        """
        result = data
        # We can't use .sel since argopy data is only indexed by N_POINTS
        if not coord_name_dict:
            coord_sys = CoordinateSystem.get_coordinate_system(data)
            coord_name_dict = coord_sys.coordinates

        # Create mask for .where
        mask = xr.ones_like(data[coord_name_dict["lat"]])
        if lat_range is not None:
            mask = xr.DataArray(np.logical_and(
                mask,
                np.logical_and(
                    data[coord_name_dict["lat"]] >= lat_range[0],
                    data[coord_name_dict["lat"]] <= lat_range[1],
                ),
            ))
        if lon_range is not None:
            mask = xr.DataArray(np.logical_and(
                mask,
                np.logical_and(
                    data[coord_name_dict["lon"]] >= lon_range[0],
                    data[coord_name_dict["lon"]] <= lon_range[1],
                ),
            ))
        if time_range is not None:
            mask = xr.DataArray(np.logical_and(
                mask,
                np.logical_and(
                    data[coord_name_dict["time"]] >= time_range[0],
                    data[coord_name_dict["time"]] <= time_range[1],
                ),
            ))

        # .compute() needed if data is a dask array
        # https://github.com/hainegroup/oceanspy/issues/332
        result = result.where(mask.compute(), drop=True)

        # TODO: Figure out how to subset depth (we only have a pressure
        # variable that's a proxy)

        return result
