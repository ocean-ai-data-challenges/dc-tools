"""Classes and functions for processing Argo float data."""

from typing import Tuple

import numpy as np
import xarray as xr

from dctools.utilities.xarray_utils import get_grid_coord_names

class ArgoDataProcessor:
    """Processor for argo data."""

    @staticmethod
    def subset_argo(
        data: xr.Dataset | xr.DataArray,
        lat_range: Tuple[float, float] | None = None,
        lon_range: Tuple[float, float] | None = None,
        time_range: Tuple[np.datetime64, np.datetime64] | None = None,
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
        coord_name_dict = get_grid_coord_names(data)

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

        
