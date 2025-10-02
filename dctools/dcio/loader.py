#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Classes and functions for loading xarray datasets."""

import gc
import os
from typing import Any, Callable, List, Optional

import fsspec
from loguru import logger
from memory_profiler import profile
import netCDF4
import numpy as np
import traceback
import xarray as xr
import zarr

# Configuration pour la compatibilité Dask
#os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
#os.environ['NETCDF4_DEACTIVATE_MPI'] = '1'


def configure_xarray_for_dask():
    """
    Configure xarray pour une utilisation optimale avec Dask workers.
    À appeler une fois au début de l'application.
    """
    import dask
    
    # Configuration Dask pour xarray
    dask.config.set({
        'array.chunk-size': '256MB',
        'array.slicing.split_large_chunks': False,
        'distributed.worker.daemon': False,
        'distributed.comm.timeouts.tcp': 300,
        'distributed.comm.timeouts.connect': 180,
    })
    
    # Configuration xarray
    xr.set_options(
        file_cache_maxsize=1,  # Cache minimal pour éviter les conflits
        warn_for_unclosed_files=False,  # Évite les warnings dans les workers
    )


def choose_chunks_automatically(
    ds: xr.Dataset,
    target_chunk_mb: int = 32,
    min_chunk: int = 1,
) -> dict:
    """
    Propose un schéma de chunking adapté en fonction de la taille mémoire des variables.
    """
    target_bytes = target_chunk_mb * 1024**2
    element_size = np.dtype("float64").itemsize
    target_elems = target_bytes // element_size

    dim_sizes = dict(ds.sizes)
    suggested = {}

    for dim, size in dim_sizes.items():
        vars_using_dim = [v for v in ds.data_vars if dim in ds[v].dims]
        if not vars_using_dim:
            continue

        max_other = 1
        for v in vars_using_dim:
            other_dims = [d for d in ds[v].dims if d != dim]
            prod = np.prod([dim_sizes[d] for d in other_dims])
            max_other = max(max_other, prod)

        elems_if_full = size * max_other

        if elems_if_full <= target_elems:
            chunk = size
        else:
            chunk = max(min_chunk, target_elems // max_other)

        suggested[dim] = int(chunk)

    return suggested


def list_all_group_paths(nc_path: str) -> List[str]:
    """
    Liste tous les chemins de groupes dans un fichier NetCDF.
    Thread-safe pour utilisation avec Dask.
    """
    def walk(grp, prefix=""):
        paths = []
        for name, subgrp in grp.groups.items():
            full = f"{prefix}/{name}" if prefix else name
            paths.append(full)
            paths.extend(walk(subgrp, full))
        return paths
    try:
        # Utiliser mode 'r' avec format='NETCDF4' pour compatibilité Dask
        with netCDF4.Dataset(nc_path, "r", format='NETCDF4') as nc:
            groups = walk(nc)
        gc.collect()
        return groups
    except Exception as e:
        logger.warning(f"Could not read groups from {nc_path}: {e}")
        traceback.print_exc()
        return []


class FileLoader:
    @staticmethod
    def open_dataset_auto(
        file_path: str,
        adaptive_chunking: bool = False,
        groups: Optional[list[str]] = None,
        engine: Optional[str] = "h5netcdf",
        variables: Optional[list[str]] = None,
        dask_safe: Optional[bool] = True,
        target_chunk_mb: Optional[int] = 128,
        file_storage: Optional[Any] = None,
        reading_retries: Optional[int] = 3,
    ) -> xr.Dataset | None:
        """
        Load a dataset with Dask-safe configurations and optional adaptive chunking.

        Args:
            file_path (str): Path to the file.
            adaptive_chunking (bool): Whether to auto-tune chunks.
            groups (Optional[list[str]]): NetCDF groups to load.
            engine (Optional[str]): Engine to use.
            variables (Optional[list[str]]): Variables to keep.
            dask_safe (bool): Whether to use Dask-safe configurations.
            target_chunk_mb (int): Target chunk size in MB (for adaptive mode).

        Returns:
            xr.Dataset | None: Loaded dataset or None if error.
        """
        os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
        try:
            # open_kwargs = {"chunks": "auto", "engine": engine}
            open_kwargs = {"engine": engine}
            if dask_safe:
                open_kwargs.update({"lock": False, "cache": False})

            ds = None

            if file_path.endswith(".nc"):
                # Récupérer les groupes
                group_paths = list_all_group_paths(file_path)
                if group_paths:
                    datasets = []
                    for group_path in group_paths:
                        try:
                            sub_ds = xr.open_dataset(file_path, group=group_path, **open_kwargs)
                            prefix = group_path.replace("/", "__")
                            sub_ds = sub_ds.rename({var: f"{prefix}__{var}" for var in sub_ds.data_vars})
                            datasets.append(sub_ds)
                        except Exception as e:
                            logger.warning(f"Failed to open group {group_path}: {e}")
                            continue
                    if datasets:
                        ds = xr.merge(datasets, compat="no_conflicts", join="outer")
                    else:
                        logger.error(f"No valid groups found in {file_path}")
                        return None
                else:
                    ds = xr.open_dataset(file_path, **open_kwargs)

            elif file_path.endswith(".zarr"):
                zarr_kwargs = {
                    "chunks": "auto",
                    "consolidated": True,
                }
                if file_storage is not None:
                    for attempt in range(reading_retries):
                        try:
                            # Support for remote storage (e.g., S3)
                            store = file_storage.get_mapper(file_path)  # <-- mapping, pas file-like
                            kvstore = zarr.storage.KVStore(store)
                            ds = xr.open_zarr(kvstore, consolidated=True)  # **zarr_kwargs)
                        except Exception as e:
                            logger.warning(f"Reading attempt {attempt + 1} failed: {e}")
                            if attempt == reading_retries - 1:
                                raise
                else:
                    for attempt in range(reading_retries):
                        try:
                            ds = xr.open_zarr(file_path, **zarr_kwargs)
                        except Exception as e:
                            logger.warning(f"Reading attempt {attempt + 1} failed: {e}")
                            if attempt == reading_retries - 1:
                                raise   
            else:
                raise ValueError(f"Unsupported file format {file_path}.")

            # Filtrage des variables après ouverture
            if variables and ds is not None:
                available_vars = list(ds.variables.keys())
                vars_to_drop = [v for v in available_vars if v not in variables]
                if vars_to_drop:
                    ds = ds.drop_vars(vars_to_drop, errors="ignore")

            # Appliquer adaptive chunking si demandé
            if adaptive_chunking and ds is not None:
                chunks = choose_chunks_automatically(ds, target_chunk_mb=target_chunk_mb)
                if chunks:
                    ds = ds.chunk(chunks)
            return ds

        except Exception as error:
            logger.warning(f"Error when loading file {file_path}: {error}")
            traceback.print_exc()
            return None



'''    @staticmethod
    def load_dataset(
        file_path: str,
        adaptive_chunking: bool = False,
        groups: Optional[list[str]] = None,
        engine: Optional[str] = "h5netcdf",
        variables: Optional[list[str]] = None,
        dask_safe: bool = True,
    ) -> xr.Dataset | None:
        """
        Load a dataset with Dask-safe configurations.
        
        Args:
            file_path (str): Path to the file.
            adaptive_chunking (bool): Whether to use adaptive chunking.
            groups (Optional[list[str]]): NetCDF groups to load.
            engine (Optional[str]): Engine to use.
            variables (Optional[list[str]]): Variables to keep.
            dask_safe (bool): Whether to use Dask-safe configurations.
        
        Returns:
            xr.Dataset | None: Loaded dataset or None if error.
        """
        try:
            # Configuration de base pour Dask
            open_kwargs = {"chunks": 'auto', "engine": engine}
            
            if dask_safe:
                open_kwargs.update({
                    "lock": False,  # Important pour éviter les deadlocks
                    "cache": False,  # Évite les conflits de cache entre workers
                })
            
            # Note: variables filtering doit être fait APRÈS ouverture du dataset
            # pour éviter les erreurs avec des variables non définies
            
            if file_path.endswith(".nc"):
                group_paths = list_all_group_paths(file_path)
                if group_paths:
                    # Ouvre chaque groupe séparément, puis concatène
                    datasets = []
                    for group_path in group_paths:
                        try:
                            ds = xr.open_dataset(file_path, group=group_path, **open_kwargs)
                            datasets.append(ds)
                        except Exception as e:
                            logger.warning(f"Failed to open group {group_path}: {e}")
                            continue
                    
                    if datasets:
                        ds = xr.merge(datasets, compat="no_conflicts", join="outer")
                    else:
                        logger.error(f"No valid groups found in {file_path}")
                        return None
                else:
                    ds = xr.open_dataset(file_path, **open_kwargs)
                
            elif file_path.endswith(".zarr"):
                zarr_kwargs = {"chunks": 'auto'}
                ds = xr.open_zarr(file_path, **zarr_kwargs)
            else:
                raise ValueError(f"Unsupported file format {file_path}.")
            
            # Filtrage des variables après ouverture
            if variables and ds is not None:
                available_vars = list(ds.variables.keys())
                vars_to_drop = [v for v in available_vars if v not in variables]
                if vars_to_drop:
                    ds = ds.drop_vars(vars_to_drop, errors='ignore')
            
            return ds
            
        except Exception as error:
            logger.error(f"Error when loading file {file_path}: {error}")
            return None
'''