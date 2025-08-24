#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Classes and functions for loading xarray datasets."""

from typing import Any, List, Optional, Union

from fsspec import FSMap
from loguru import logger
from memory_profiler import profile
import netCDF4
import numpy as np
import traceback
import xarray as xr

# from dctools.utilities.misc_utils import to_float32



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
            # std_chunks = {"lat": 256, "lon": 256, "time": 1}
            open_kwargs = {"chunks": 'auto'}
            if path.endswith(".zarr"):
                #logger.info(f"Opening Zarr dataset: {path}")
                return xr.open_zarr(manager.params.fs.get_mapper(path), chunks="auto")
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
                    # return xr.open_dataset(manager.params.fs.open(path), engine=engine)
                    ds = xr.open_dataset(path, engine=engine, **open_kwargs)
                    return ds
        except Exception as exc:
            logger.error(f"Failed to open dataset {path}: {traceback.format_exc()}")
            raise


    @staticmethod
    def load_dataset(
        file_path: str,
        adaptive_chunking: bool = False,
        groups: Optional[list[str]] = None,
        engine: Optional[str] = "netcdf4",
        variables: Optional[list[str]] = None,
    ) -> xr.Dataset | None:
        try:
            open_kwargs = {"chunks": 'auto'} # {"lat": 256, "lon": 256, "time": 1}}
            if variables:
                open_kwargs["drop_variables"] = [v for v in ds.variables if v not in variables]
            if file_path.endswith(".nc"):
                group_paths = list_all_group_paths(file_path)
                if group_paths:
                    # Ouvre chaque groupe séparément, puis concatène
                    datasets = []
                    for group_path in group_paths:
                        ds = xr.open_dataset(file_path, group=group_path, **open_kwargs)
                        datasets.append(ds)
                    ds = xr.merge(datasets, compat="no_conflicts", join="outer")
                else:
                    ds = xr.open_dataset(file_path, **open_kwargs)
                # ds = to_float32(ds)
                return ds
            elif file_path.endswith(".zarr"):
                ds = xr.open_zarr(file_path, chunks='auto')  #{"lat": 256, "lon": 256, "time": 1})
                #ds = to_float32(ds)
                return ds
            else:
                raise ValueError(f"Unsupported file format {file_path}.")
        except Exception as error:
            logger.error(f"Error when loading file {file_path}: {error}")
            return None
