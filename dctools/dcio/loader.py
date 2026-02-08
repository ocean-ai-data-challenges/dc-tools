#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Classes and functions for loading xarray datasets."""

import gc
import os
from typing import Any, Dict, List, Optional

from loguru import logger
import netCDF4
import numpy as np
import traceback
import xarray as xr

# Dask configuration for compatibility
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
os.environ['NETCDF4_DEACTIVATE_MPI'] = '1'


def configure_xarray_for_dask():
    """Configure xarray for optimal usage with Dask workers.

    Call once at the beginning of the application.
    """
    import dask

    # Dask configuration for xarray
    dask.config.set({
        'array.chunk-size': '128MB',
        'array.slicing.split_large_chunks': False,
        'distributed.worker.daemon': False,
        'distributed.comm.timeouts.tcp': 300,
        'distributed.comm.timeouts.connect': 180,
    })

    # xarray configuration - Minimize file cache to avoid memory accumulation
    # This prevents file handles from being kept in memory between batches
    # Note: Some xarray versions don't allow 0, so we use 1 (minimal cache)
    xr.set_options(
        file_cache_maxsize=1,  # minimal: 1 file cached (effectively disabled)
        warn_for_unclosed_files=False,  # Avoid warnings in workers
    )


def choose_chunks_automatically(
    ds: xr.Dataset,
    target_chunk_mb: int = 32,
    min_chunk: int = 1,
) -> dict:
    """Propose a suitable chunking scheme based on variable memory size."""
    target_bytes = target_chunk_mb * 1024**2
    element_size = np.dtype("float64").itemsize
    target_elems = target_bytes // element_size

    dim_sizes = dict(ds.sizes)
    suggested: Dict[Any, Any] = {}

    for dim, size in dim_sizes.items():
        vars_using_dim = [v for v in ds.data_vars if dim in ds[v].dims]
        if not vars_using_dim:
            continue

        max_other = 1
        for v in vars_using_dim:
            other_dims = [d for d in ds[v].dims if d != dim]
            prod = int(np.prod([dim_sizes[d] for d in other_dims]))
            max_other = max(max_other, prod)

        elems_if_full = size * max_other

        if elems_if_full <= target_elems:
            chunk = size
        else:
            chunk = max(min_chunk, target_elems // max_other)

        suggested[dim] = int(chunk)

    return suggested


def list_all_group_paths(nc_path: str) -> List[str]:
    """List all group paths in a NetCDF file.

    Thread-safe for use with Dask.
    Tries h5netcdf first (faster), then netCDF4.
    """
    def walk(grp, prefix=""):
        paths: List[str] = []
        # h5netcdf and netCDF4 have slightly different APIs for groups
        # netcdf4: .groups (dict)
        # h5netcdf: .keys() but must check if it is a group
        items = getattr(grp, "groups", None)
        if items is None: # h5netcdf loop approach
             items = grp

        # Generic iteration
        for name in items:
             # h5netcdf key iter
             try:
                 item = grp[name]
             except Exception:
                 continue # skip if error

             # Check if it is a group
             # h5netcdf.Group ou netCDF4.Group
             is_group = False
             if hasattr(item, "groups") or "Group" in type(item).__name__:
                 is_group = True

             if is_group:
                full = f"{prefix}/{name}" if prefix else name
                paths.append(full)
                paths.extend(walk(item, full))
        return paths

    # Attempt with h5netcdf (often faster for listing)
    try:
        import h5netcdf
        with h5netcdf.File(nc_path, 'r') as nc:
             # The walk function must be adapted for h5netcdf if necessary
             # To keep it simple, we recreate a specific walk for h5netcdf
             def walk_h5(grp: Any, prefix: str = "") -> List[str]:
                paths: List[str] = []
                for name in grp.keys():
                    item = grp[name]
                    if isinstance(item, h5netcdf.Group):
                        full = f"{prefix}/{name}" if prefix else name
                        paths.append(full)
                        paths.extend(walk_h5(item, full))
                return paths

             groups = walk_h5(nc)
        gc.collect()
        return groups
    except (ImportError, OSError, Exception):
        # Silent fallback (or debug log) to netCDF4
        pass

    try:
        # Use mode 'r' with format='NETCDF4' for Dask compatibility
        with netCDF4.Dataset(nc_path, "r", format='NETCDF4') as nc:
            def walk_nc(grp: Any, prefix: str = "") -> List[str]:
                paths: List[str] = []
                for name, subgrp in grp.groups.items():
                    full = f"{prefix}/{name}" if prefix else name
                    paths.append(full)
                    paths.extend(walk_nc(subgrp, full))
                return paths
            groups = walk_nc(nc)
        gc.collect()
        return groups
    except Exception as e:
        logger.warning(f"Could not read groups from {nc_path}: {e}")
        # traceback.print_exc()
        return []


class FileLoader:
    """Utilities for loading datasets from various file formats."""

    @staticmethod
    def open_dataset_auto(
        file_path: str,
        adaptive_chunking: bool = False,
        groups: Optional[Optional[list[str]]] = None,
        engine: Optional[str] = "h5netcdf",
        variables: Optional[Optional[list[str]]] = None,
        dask_safe: Optional[bool] = True,
        target_chunk_mb: Optional[int] = 128,
        file_storage: Optional[Optional[Any]] = None,
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
        if reading_retries is None:
            reading_retries = 3
        if target_chunk_mb is None:
            target_chunk_mb = 128

        os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
        try:
            # force lazy loading: chunks={} tells xarray to use dask with file's chunks
            base_chunks: Dict[Any, Any] = {}

            # open_kwargs = {"chunks": "auto", "engine": engine}
            open_kwargs: Dict[str, Any] = {"engine": engine, "chunks": base_chunks}
            if dask_safe:
                open_kwargs.update({"lock": False, "cache": False})

            ds = None

            if file_path.endswith(".nc"):
                # Retrieve groups
                group_paths = list_all_group_paths(file_path)
                if group_paths:
                    datasets: List[Any] = []
                    for group_path in group_paths:
                        try:
                            sub_ds = xr.open_dataset(file_path, group=group_path, **open_kwargs)
                            prefix = group_path.replace("/", "__")
                            sub_ds = sub_ds.rename({
                                var: f"{prefix}__{var}"
                                for var in sub_ds.data_vars
                            })
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
                zarr_kwargs: Dict[str, Any] = {
                    "chunks": base_chunks,
                    "consolidated": True,
                }
                if file_storage is not None:
                    for attempt in range(reading_retries):
                        try:
                            # Support for remote storage (e.g., S3)
                            store = file_storage.get_mapper(file_path)  # <-- mapping, not file-like
                            # kvstore = zarr.storage.KVStore(store)
                            ds = xr.open_zarr(store, **zarr_kwargs)
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

            # Filtering variables after opening
            if variables and ds is not None:
                # available_vars = list(ds.variables.keys())
                available_data_vars = list(ds.data_vars.keys())
                vars_to_drop = [v for v in available_data_vars if v not in variables]
                if vars_to_drop:
                    ds = ds.drop_vars(vars_to_drop, errors="ignore")

            # Apply adaptive chunking if requested
            if adaptive_chunking and ds is not None:
                chunks = choose_chunks_automatically(ds, target_chunk_mb=target_chunk_mb)
                if chunks:
                    ds = ds.chunk(chunks)
            return ds

        except Exception as error:
            logger.warning(f"Error when loading file {file_path}: {error}")
            traceback.print_exc()
            return None
