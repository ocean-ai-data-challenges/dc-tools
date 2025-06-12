#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Classes and functions for loading xarray datasets."""

from typing import Any, List, Optional, Union

from fsspec import FSMap
from loguru import logger
import netCDF4
import traceback
import xarray as xr


def list_all_group_paths(nc_path: str) -> List[str]:
    def walk(grp, prefix=""):
        paths = []
        for name, subgrp in grp.groups.items():
            full = f"{prefix}/{name}" if prefix else name
            paths.append(full)
            paths.extend(walk(subgrp, full))
        return paths
    with netCDF4.Dataset(nc_path, "r") as nc:
        return walk(nc)


def open_and_concat_groups(
    source: Union[FSMap, str],
    group_paths: List[str] = None,
    **xr_kwargs
) -> xr.Dataset:
    """
    Ouvre récursivement les groupes NetCDF imbriqués et concatène leurs variables dans un seul Dataset.
    Les noms de variables sont préfixés par le chemin du groupe (avec '__' comme séparateur).

    Args:
        source (Union[FSMap, str]): Chemin du fichier NetCDF ou FSMap (fsspec).
        group_paths (List[str] or None): Liste des chemins complets des groupes à ouvrir (ex: ["ku", "group_data_01/ku"]).
        **xr_kwargs: Arguments additionnels pour xr.open_dataset.

    Returns:
        xr.Dataset: Dataset concaténé.
    """
    if not group_paths:
        return xr.open_dataset(source, **xr_kwargs)

    datasets = []
    for group_path in group_paths:
        ds = xr.open_dataset(source, group=group_path, **xr_kwargs)
        prefix = group_path.replace("/", "__")
        ds = ds.rename({var: f"{prefix}__{var}" for var in ds.data_vars})
        datasets.append(ds)

    ds_merged = xr.merge(datasets, compat="no_conflicts", join="outer")
    return ds_merged


class FileLoader:
    """Loading NetCDF or Zarr files."""

    @staticmethod
    def open_dataset_auto(
        path: str,
        manager: Any,
        groups: Optional[list[str]] = None,
        engine: Optional[str] = "netcdf4",
    ) -> xr.Dataset:
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
                group_paths = list_all_group_paths(path)
                if group_paths:
                    ds = open_and_concat_groups(
                        # manager.params.fs.get_mapper(path),
                        path,
                        group_paths=group_paths,
                        chunks='auto',
                        engine=engine,
                    )
                    return ds
                else:
                    # logger.info(f"Opening NetCDF dataset: {path}")
                    # logger.debug(f"Using engine: {engine}")
                    # logger.debug(f"Using fs: {manager.params.fs}")
                    # return xr.open_dataset(manager.params.fs.open(path), engine=engine)
                    return xr.open_dataset(path, engine=engine)
                # return collect_all_groups(manager.params.fs.open(path), engine="netcdf4")
        except Exception as exc:
            logger.error(f"Failed to open dataset {path}: {traceback.format_exc()}")
            raise

    @staticmethod
    def load_dataset(
        file_path: str,
        adaptive_chunking: bool = False,
        groups: Optional[list[str]] = None,
        engine: Optional[str] = "netcdf4",
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
            Loaded Xarray Dataset, or None if error while loading and `fail_on_error = True`.

        Raises
        ------
        ValueError
            If `file_path` does not point to a NetCDF of Zarr file.
        """
        try:
            if file_path.endswith(".nc"):
                # logger.debug(
                #    f"Loading dataset from NetCDF file: {file_path}"
                #)
                group_paths = list_all_group_paths(file_path)
                if group_paths:
                    ds = open_and_concat_groups(
                        file_path, group_paths=group_paths, chunks='auto',
                        engine=engine,
                    )
                else:
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
                f"Error when loading file {file_path}: {traceback.print_exc()}"
            )
            return None
