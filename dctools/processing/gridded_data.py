"""Classes and functions for processing gridded Xarray objects."""

from typing import Tuple, Optional

import numpy as np
import xarray as xr

from dctools.utilities.xarray_utils import get_grid_coord_names, get_grid_dim_names


class GriddedDataProcessor:
    """Processor for gridded data (in 2D or 3D, with or without t)."""

    @staticmethod
    def subset_grid(
        data: xr.Dataset | xr.DataArray,
        lat_range: Optional[Tuple[float, float]] = None,
        lon_range: Optional[Tuple[float, float]] = None,
        vert_range: Optional[Tuple[float, float]] = None,
        time_range: Optional[Tuple[np.datetime64, np.datetime64]] = None,
    ) -> xr.Dataset | xr.DataArray:
        """
        Subset the area defined by `lat_range`, `lon_range` and `time_range`.

        Args:
            `data` (xr.Dataset or xr.DataArray): Xarray object
            `lat_range` (Tuple of two floats): lower and upper latitude
                bounds
            `lon_range` (Tuple of two floats): lower and upper longitude
                bounds
            `vert_range` (Tuple of two floats): lower and upper vertical
                bounds, either depth or height depending on the dataset
            `time_range` (Tuple of two datetimes): start and end of
                selected period

        Returns:
            `subset`: An object of the same type as `data` containing the
                specified range
        """
        # TODO: Check that `data` is actually gridded in lat/lon

        sel_dict = {}
        # check coordinate names
        coord_name_dict = get_grid_coord_names(data)

        if lat_range is not None:
            sel_dict[coord_name_dict["lat"]] = slice(lat_range[0], lat_range[1])
        if lon_range is not None:
            sel_dict[coord_name_dict["lon"]] = slice(lon_range[0], lon_range[1])
        if vert_range is not None:
            sel_dict[coord_name_dict["depth"]] = slice(vert_range[0], vert_range[1])
        if time_range is not None:
<<<<<<< HEAD
            sel_dict[coord_name_dict["time"]] = slice(
                time_range[0], time_range[1]
            )  # type: ignore
=======
            sel_dict[coord_name_dict["time"]] = slice(time_range[0], time_range[1])  # type: ignore
>>>>>>> add_tests

        return data.sel(sel_dict)

    @staticmethod
    def coarsen_grid(
        data: xr.Dataset | xr.DataArray,
        horizontal_window: Optional[int] = None,
        vertical_window: Optional[int] = None,
        time_window: Optional[int | str] = None,
    ) -> xr.Dataset | xr.DataArray:
        """
        Coarsens the grid's resolution by applyin *mean* along some dimension(s).

        Args:
            `data` (xr.Dataset or xr.DataArray):
            `horizontal_window` (int): size of the averaging window in
                the latitude and longitude dimensions.
            `vertical_window` (int): size of the averaging window in the
                depth dimension.
            `time_window` (int or str): if `int`, it represents the size
                of the averaging window in the time dimension; if `str`,
                represents a timedelta over which to resample the time
                dimension.
                See the following for more detail:
                https://docs.xarray.dev/en/stable/generated/xarray.DataArray.resample.html
                https://stackoverflow.com/questions/68448310/grouping-xarray-daily-data-into-monthly-means

        Returns:
            `coarsened_data`: an object of the same type as data with
                *mean* applied over its dimensions.

        """
        coarsen_dict = {}
        # Check dimension names
        dim_name_dict = get_grid_dim_names(data)

        temp = data
        if horizontal_window is not None:
            coarsen_dict[dim_name_dict["lat"]] = horizontal_window
            coarsen_dict[dim_name_dict["lon"]] = horizontal_window
        if vertical_window is not None:
            coarsen_dict[dim_name_dict["depth"]] = vertical_window
        if type(time_window) is int:
            coarsen_dict[dim_name_dict["time"]] = time_window
        elif type(time_window) is str:
            temp = temp.resample({
                dim_name_dict["time"]: time_window
            }).mean()

        return temp.coarsen(coarsen_dict, boundary="pad").mean()  # type: ignore

    @staticmethod
    def concatenate(
        ds1: xr.Dataset, ds2: xr.Dataset, dim: str
    ) -> xr.Dataset:
        """Concatenates two datasets according to the specified dimension.
    
        Args:
            ds1(xr.Dataset): first xarray dataset
            ds2(xr.Dataset): second dataset
        Return:
            (xr.Dataset): concatenated dataset
        """
        return xr.concat([ds1, ds2], dim)

