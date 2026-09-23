#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Classes and functions for loading xarray datasets.

Moved from ``dctools.dcio.loader`` (dctools keeps a re-exporting shim). The fixes below were each
motivated by a real store on the EDITO S3: see the docstrings.
"""

import gc
import logging
import os
from typing import Any, Dict, List, Optional

import netCDF4
import numpy as np
import pandas as pd
import traceback
import xarray as xr

logger = logging.getLogger(__name__)

# Dask configuration for compatibility
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
os.environ["NETCDF4_DEACTIVATE_MPI"] = "1"
os.environ["NETCDF4_USE_FILE_LOCKING"] = "FALSE"
os.environ["HDF5_DISABLE_VERSION_CHECK"] = "1"
os.environ["ARGOPY_NETCDF_LOCKING"] = "FALSE"


def _nc_engine_for(file_path: str) -> str:
    """Return the best xarray engine for a local .nc file.

    Reads only the first 4 magic bytes to distinguish NetCDF3/CDF classic
    (``CDF\\x01`` / ``CDF\\x02``) from HDF5-based NetCDF4 (``\\x89HDF``).
    Falls back to ``h5netcdf`` for anything else (e.g. remote paths where
    the file cannot be opened for a byte peek).
    """
    try:
        with open(file_path, "rb") as f:
            magic = f.read(4)
        if magic[:3] == b"CDF":
            return "scipy"
    except OSError:
        pass
    return "h5netcdf"


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


def _fix_nanosecond_time(ds: xr.Dataset) -> xr.Dataset:
    """Convert time variables stored as 'nanoseconds since <epoch>' to datetime64[ns].

    xarray/cftime cannot decode 'nanoseconds' as a CF time unit.  When a zarr
    store is opened with ``decode_times=False`` this helper detects integer or
    floating-point variables whose ``units`` attribute starts with
    'nanoseconds since' and reconstructs proper datetime64[ns] values by
    parsing the reference epoch out of the ``units`` string and adding the raw
    values to it as a nanosecond offset.

    Some products (e.g. SWOT) encode this relative to the UNIX epoch
    (1970-01-01), while others (e.g. Jason-3/Saral) use a per-file reference
    timestamp close to the actual data. Parsing the epoch from ``units``
    (rather than assuming UNIX epoch) handles both cases correctly.
    """
    for var in list(ds.coords) + list(ds.data_vars):
        v = ds[var]
        units = v.attrs.get("units", "")
        if "nanoseconds since" in units and (
            np.issubdtype(v.dtype, np.integer) or np.issubdtype(v.dtype, np.floating)
        ):
            # Use scheduler="synchronous" to avoid routing through the distributed
            # client (which can be broken inside a cancelled/restarting worker and
            # would raise ClosedClientError when .values triggers __array__).
            raw_data = v.variable._data
            if hasattr(raw_data, "compute"):
                raw_np = raw_data.compute(scheduler="synchronous")
            else:
                raw_np = np.asarray(raw_data)
            try:
                ref_str = units.split("since", 1)[1].strip()
                ref_epoch = pd.Timestamp(ref_str).to_datetime64()
            except Exception:
                logger.warning(
                    f"Could not parse reference epoch from units '{units}' for "
                    f"variable '{var}'; assuming UNIX epoch (1970-01-01)."
                )
                ref_epoch = np.datetime64("1970-01-01T00:00:00.000000000")
            offsets_ns = np.round(raw_np).astype("int64").astype("timedelta64[ns]")
            new_values = ref_epoch + offsets_ns
            new_attrs = {k: val for k, val in v.attrs.items() if k not in ("units", "calendar")}
            ds[var] = xr.Variable(v.dims, new_values, new_attrs)
    return ds


def _zarr_vars_with_stringified_scale_attrs(store_or_path: Any) -> set:
    """Cheap, metadata-only check for a known conversion bug where CF
    ``scale_factor``/``add_offset`` attributes were written to a zarr store
    as strings instead of numbers (seen e.g. in some Sentinel SRAL zarr
    products). Returns the set of variable names affected.

    If left as strings, xarray's CF decoder does not fail at open time —
    it wraps the variable in a lazy decoder and only raises (a confusing
    ``UFuncTypeError``) the first time the data is actually computed, which
    can happen deep in the pipeline (e.g. bounding-box/geometry extraction)
    rather than at open time.
    """
    affected: set = set()
    try:
        import zarr

        root = zarr.open(store_or_path, mode="r")
        items = root.items() if hasattr(root, "items") else []
        for name, arr in items:
            attrs = getattr(arr, "attrs", {})
            for key in ("scale_factor", "add_offset"):
                if isinstance(attrs.get(key), str):
                    affected.add(name)
    except Exception:
        return set()
    return affected


def _fix_stringified_scale_attrs(ds: xr.Dataset, affected_vars: set, store_or_path: Any) -> xr.Dataset:
    """Work around a known conversion bug (seen in some Sentinel SRAL zarr
    products) where CF ``scale_factor``/``add_offset`` attributes were
    written as strings instead of numbers. xarray's CF decoder does not
    raise at open time for this — it lazily wraps the variable and only
    raises a confusing ``UFuncTypeError`` the first time the data is
    actually computed (e.g. deep in geometry/time-bounds extraction, or
    during metric computation).

    Empirically (checked across lat/lon, ssha, and mean_dyn_topo for the
    affected product), the raw on-disk values are already the final,
    physically-correct decoded values — the scale_factor/add_offset attrs
    are stale leftovers from the original packed-integer encoding and were
    never removed after the data was pre-scaled to floats during the
    NetCDF-to-zarr conversion. Applying them (even after just fixing the
    string->float type) would silently corrupt the data (e.g. turning a
    valid ~81 degree latitude into ~0.00008, or a valid ~0.1 m SSHA into
    ~0.0001 m). So for affected variables, we replace them with the raw
    (mask_and_scale=False) values and strip the stale attrs, rather than
    coercing-and-applying the scale/offset.
    """
    affected_vars = affected_vars & set(ds.variables)
    if not affected_vars:
        return ds
    try:
        raw_ds = xr.open_zarr(
            store_or_path, chunks={}, consolidated=False, mask_and_scale=False, decode_times=False
        )
        for name in affected_vars:
            raw_var = raw_ds[name].variable
            new_attrs = {
                k: v for k, v in raw_var.attrs.items() if k not in ("scale_factor", "add_offset")
            }
            ds[name] = xr.Variable(raw_var.dims, raw_var.data, new_attrs)
        logger.warning(
            f"Variable(s) {sorted(affected_vars)} had scale_factor/add_offset stored as "
            "strings; using raw (already physically-scaled) values instead of failing lazily."
        )
    except Exception as exc:
        logger.warning(f"Failed to fix stringified scale attrs for {affected_vars}: {exc}")
    return ds


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
        if items is None:  # h5netcdf loop approach
            items = grp

        # Generic iteration
        for name in items:
            # h5netcdf key iter
            try:
                item = grp[name]
            except Exception:
                continue  # skip if error

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

        with h5netcdf.File(nc_path, "r") as nc:
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
        with netCDF4.Dataset(nc_path, "r", format="NETCDF4") as nc:

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

        os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
        try:
            # force lazy loading: chunks={} tells xarray to use dask with file's chunks
            base_chunks: Dict[Any, Any] = {}

            ds = None

            if file_path.endswith(".nc"):
                # Pick engine based on file magic bytes so NetCDF3/CDF classic
                # files (e.g. SMOS .DBL.nc) are opened with scipy directly,
                # without the cost of an h5netcdf failure + retry.
                _engine = _nc_engine_for(file_path) if (engine is None or engine == "h5netcdf") else engine
                open_kwargs: Dict[str, Any] = {"engine": _engine, "chunks": base_chunks}
                if dask_safe:
                    open_kwargs["lock"] = False
                    # 'cache' kwarg is not supported by the scipy backend
                    if _engine != "scipy":
                        open_kwargs["cache"] = False
                # NetCDF3/CDF classic files cannot contain sub-groups; skip the
                # h5netcdf/netCDF4 group scan to avoid spurious warnings.
                group_paths = [] if _engine == "scipy" else list_all_group_paths(file_path)
                if group_paths:
                    datasets: List[Any] = []
                    for group_path in group_paths:
                        try:
                            sub_ds = xr.open_dataset(file_path, group=group_path, **open_kwargs)
                            prefix = group_path.replace("/", "__")
                            sub_ds = sub_ds.rename(
                                {var: f"{prefix}__{var}" for var in sub_ds.data_vars}
                            )
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
                    try:
                        ds = xr.open_dataset(file_path, **open_kwargs)
                    except Exception as e:
                        err_str = str(e).lower()
                        # Last-resort fallback: engine mismatch not caught by the
                        # magic-byte check (e.g. truncated header, remote file).
                        if (
                            "file signature not found" in err_str
                            or "unknown file format" in err_str
                        ) and open_kwargs.get("engine") != "scipy":
                            fallback_kw = {**open_kwargs, "engine": "scipy"}
                            logger.debug(
                                f"Engine mismatch for {file_path}; retrying with scipy"
                            )
                            ds = xr.open_dataset(file_path, **fallback_kw)
                        else:
                            raise

            elif file_path.endswith(".zarr"):
                zarr_kwargs: Dict[str, Any] = {
                    "chunks": base_chunks,
                    "consolidated": True,
                    # NOTE: do NOT force use_cftime=True here. Forcing cftime
                    # decodes every CF-encoded time variable (e.g. Sentinel
                    # SRAL "seconds since ..." time) into cftime.datetime
                    # objects (dtype=object) even when the dates are well
                    # within pandas' datetime64[ns] range. Downstream code
                    # (Class4Evaluator point-matching, time-tolerance
                    # windowing) expects/compares against numpy datetime64
                    # values (e.g. glonet's pred time), so an object-dtype
                    # cftime ref time silently matches zero points instead of
                    # raising — this caused every Sentinel batch to report
                    # 0 total pts. Letting xarray decide automatically only
                    # falls back to cftime when dates genuinely can't be
                    # represented as datetime64[ns] (e.g. out-of-range
                    # dates), which is already separately handled by the
                    # "overflow"/"unable to decode time" except branches
                    # below (decode_times=False fallback).
                    "decode_times": True,
                }
                _scale_check_target = (
                    file_storage.get_mapper(file_path) if file_storage is not None else file_path
                )
                _affected_scale_vars = _zarr_vars_with_stringified_scale_attrs(_scale_check_target)
                if file_storage is not None:
                    for attempt in range(reading_retries):
                        try:
                            # Support for remote storage (e.g., S3)
                            store = file_storage.get_mapper(file_path)  # <-- mapping, not file-like
                            # kvstore = zarr.storage.KVStore(store)
                            ds = xr.open_zarr(store, **zarr_kwargs)
                            break  # success — do NOT open the store again on next iteration
                        except Exception as e:
                            err_str = str(e).lower()
                            # Some SWOT files use 'nanoseconds since …' which is not a
                            # valid CF unit — fall back to decode_times=False and fix up.
                            if "nanoseconds" in err_str:
                                try:
                                    store = file_storage.get_mapper(file_path)
                                    no_decode_kw = {**zarr_kwargs, "decode_times": False}
                                    ds = _fix_nanosecond_time(xr.open_zarr(store, **no_decode_kw))
                                    break
                                except Exception as e2:
                                    logger.warning(f"Reading attempt {attempt + 1} failed (nanoseconds fallback): {e2}")
                                    if attempt == reading_retries - 1:
                                        raise
                            # If consolidated metadata is missing, retry without it
                            elif "zmetadata" in err_str or "consolidated" in err_str or "nosuchkey" in err_str:
                                try:
                                    store = file_storage.get_mapper(file_path)
                                    fallback_kw = {**zarr_kwargs, "consolidated": False}
                                    ds = xr.open_zarr(store, **fallback_kw)
                                    break
                                except Exception as e2:
                                    logger.warning(f"Reading attempt {attempt + 1} failed: {e2}")
                                    if attempt == reading_retries - 1:
                                        raise
                            # Time values outside 64-bit int range — open raw without
                            # time decoding.
                            elif (
                                "overflow" in err_str
                                or "unable to decode time" in err_str
                                or "failed to decode variable" in err_str
                            ):
                                try:
                                    store = file_storage.get_mapper(file_path)
                                    no_decode_kw = {**zarr_kwargs, "decode_times": False}
                                    ds = xr.open_zarr(store, **no_decode_kw)
                                    logger.warning(
                                        f"Opened {file_path} with decode_times=False due to "
                                        f"time overflow: {e}"
                                    )
                                    break
                                except Exception as e2:
                                    logger.warning(
                                        f"Reading attempt {attempt + 1} failed "
                                        f"(decode_times=False fallback): {e2}"
                                    )
                                    if attempt == reading_retries - 1:
                                        raise
                            else:
                                logger.warning(f"Reading attempt {attempt + 1} failed: {e}")
                                if attempt == reading_retries - 1:
                                    raise
                else:
                    for attempt in range(reading_retries):
                        try:
                            ds = xr.open_zarr(file_path, **zarr_kwargs)
                            break
                        except Exception as e:
                            err_str = str(e).lower()
                            # Some SWOT files use 'nanoseconds since …' which is not a
                            # valid CF unit — fall back to decode_times=False and fix up.
                            if "nanoseconds" in err_str:
                                try:
                                    no_decode_kw = {**zarr_kwargs, "decode_times": False}
                                    ds = _fix_nanosecond_time(xr.open_zarr(file_path, **no_decode_kw))
                                    break
                                except Exception as e2:
                                    logger.warning(f"Reading attempt {attempt + 1} failed (nanoseconds fallback): {e2}")
                                    if attempt == reading_retries - 1:
                                        raise
                            # If consolidated metadata is missing, retry without it
                            elif "zmetadata" in err_str or "consolidated" in err_str:
                                try:
                                    fallback_kw = {**zarr_kwargs, "consolidated": False}
                                    ds = xr.open_zarr(file_path, **fallback_kw)
                                    break
                                except Exception as e2:
                                    logger.warning(f"Reading attempt {attempt + 1} failed: {e2}")
                                    if attempt == reading_retries - 1:
                                        raise
                            # Time values outside 64-bit int range (fill values / far-future
                            # timestamps in zarr) — open raw without time decoding.
                            elif (
                                "overflow" in err_str
                                or "unable to decode time" in err_str
                                or "failed to decode variable" in err_str
                            ):
                                try:
                                    no_decode_kw = {**zarr_kwargs, "decode_times": False}
                                    ds = xr.open_zarr(file_path, **no_decode_kw)
                                    logger.warning(
                                        f"Opened {file_path} with decode_times=False due to "
                                        f"time overflow: {e}"
                                    )
                                    break
                                except Exception as e2:
                                    logger.warning(
                                        f"Reading attempt {attempt + 1} failed "
                                        f"(decode_times=False fallback): {e2}"
                                    )
                                    if attempt == reading_retries - 1:
                                        raise
                            else:
                                logger.warning(f"Reading attempt {attempt + 1} failed: {e}")
                                if attempt == reading_retries - 1:
                                    raise

                if ds is not None and _affected_scale_vars:
                    ds = _fix_stringified_scale_attrs(ds, _affected_scale_vars, _scale_check_target)
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
