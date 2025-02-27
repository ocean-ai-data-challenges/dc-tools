#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Classes and functions for processing xarray datasets."""

import datetime

import numpy as np
import xarray as xr


class DataGridder:
    """Gridding the data in 2D."""

    @staticmethod
    def interpolate_to_2dgrid(
        ds_in: xr.Dataset, lon_res: float = 1, lat_res: float = 1
    ) -> xr.Dataset | None:
        """
        Interpolate data on a regular 2d-grid (lon-lat).

        Args:
            ds_in (xr.Dataset): Xarray dataset.
            lon_res (float): longitude resolution (degrees).
            lat_res (float): latitude resolution (degrees).

        Returns:
            xr.Dataset: dataset interpolated on the defined grid.
        """
        try:
            # Creating new xarray dataset
            """ds_grid = xr.Dataset(
                coords={
                    "lon": (
                        ["lon"],
                        np.arange(
                            ds_in["lon"].values.min(),
                            ds_in["lon"].values.max(),
                            lon_res,
                        ),
                    ),
                    "lat": (
                        ["lat"],
                        np.arange(
                            ds_in["lat"].values.min(),
                            ds_in["lat"].values.max(),
                            lat_res,
                        ),
                    ),
                }
            )
            # Regrid using xesmf
            # regridder = xe.Regridder(ds_in, ds_grid, "bilinear")
            # ds_grid = regridder(ds_in)"""

            # native xarray regridding
            grid_lons = np.arange(
                ds_in["lon"].values.min(),
                ds_in["lon"].values.max(),
                lon_res,
            )
            grid_lats = np.arange(
                ds_in["lat"].values.min(),
                ds_in["lat"].values.max(),
                lat_res,
            )
            target_grid = xr.Dataset(
                {"lat": (["lat"], grid_lats), "lon": (["lon"], grid_lons)}
            )
            ds_grid = ds_in.interp(
                lat=target_grid.lat, lon=target_grid.lon, method="nearest"
            )

            # Updating STAC metadata
            ds_grid.attrs.update(ds_in.attrs)
            ds_grid.attrs["stac_version"] = "1.0.0"
            ds_grid.attrs["interpolation_method"] = "linear"
            ds_grid.attrs["processed_date"] = datetime.datetime.now(
                datetime.timezone.utc
            ).isoformat()
            return ds_grid
        except Exception:
            # import traceback
            # print(traceback.format_exc())
            # print(f"Error while interpolating data : {e}")
            print("Error while interpolating data")
            return None
