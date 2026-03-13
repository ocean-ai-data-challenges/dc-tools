#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Driver-side batch preprocessing for observation datasets.

Processes all unique observation files on the driver into a single shared
Zarr store, eliminating redundant per-worker preprocessing when multiple
tasks share the same observation files (typical for SWOT/swath data with
wide ``time_tolerance``).

Public API
----------
- ``preprocess_batch_obs_files`` — main entry point.
"""

import atexit
import gc
import os
import shutil
import tempfile
import time as _time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger

from dctools.data.datasets.nan_filtering import _nan_mask_numpy
from dctools.data.datasets.preprocessing import preprocess_one_npoints


# ---------------------------------------------------------------------------
# Module-level temp-dir registry: cleaned on worker/process exit.
# Zarr stores written during SWOT preprocessing are registered here so that
# they are reliably removed even if an exception propagates.
# ---------------------------------------------------------------------------
_SWOT_TEMP_DIRS: List[str] = []


def _atexit_cleanup_swot_dirs() -> None:  # noqa: D401
    """Remove all temporary zarr directories created during preprocessing."""
    for _d in _SWOT_TEMP_DIRS:
        shutil.rmtree(_d, ignore_errors=True)


atexit.register(_atexit_cleanup_swot_dirs)


# ---------------------------------------------------------------------------
# Module-level helpers for process-pool based batch preprocessing.
# These must be at module scope so they are picklable by ProcessPoolExecutor.
# ---------------------------------------------------------------------------

def _open_local_zarr_simple(path: str, _alias: Any = None) -> xr.Dataset:
    """Open a local zarr file — prefer consolidated metadata.

    Module-level function (picklable for multiprocessing).
    """
    try:
        return xr.open_zarr(path, consolidated=True, chunks={})  # type: ignore[no-any-return]
    except Exception:
        return xr.open_zarr(path, consolidated=False, chunks={})  # type: ignore[no-any-return]


# Shared dummy dataframe for add_time_dim fallback (SWOT files contain
# their own per-point time coordinate — this is the fallback sentinel).
_DUMMY_TIME_DF = pd.DataFrame({
    "path": [""],
    "date_start": [pd.Timestamp("2000-01-01")],
    "date_end": [pd.Timestamp("2100-01-01")],
})


def _process_file_to_zarr(args: Tuple) -> Tuple[Optional[str], int]:
    """Process one observation file --> compute --> NaN-filter --> write mini zarr.

    Designed for ``ProcessPoolExecutor``: all arguments are picklable, each
    invocation is self-contained, and the result is just a ``(path, n_pts)``
    tuple.

    **Single-pass I/O** — the file is read/decompressed exactly once
    (the old code did it twice: once for the dask NaN mask, once for the
    final ``.compute()``).
    """
    (path, file_idx, is_swath, n_points_dim, alias,
     keep_vars, coordinates, output_dir) = args
    try:
        # Force synchronous dask scheduler — avoid any interaction with
        # a distributed Client that may have been inherited via fork().
        import dask as _dask_mod
        _dask_mod.config.set(scheduler="synchronous")

        # Auto-detect swath structure when not provided (pipeline mode).
        # Costs: one zarr open (metadata only) — negligible vs. compute.
        if is_swath is None:
            try:
                _probe_ds = _open_local_zarr_simple(path)
                is_swath = {"num_lines", "num_pixels"}.issubset(_probe_ds.dims)
                _probe_ds.close()
            except Exception:
                is_swath = False

        result = preprocess_one_npoints(
            path, is_swath, n_points_dim,
            _DUMMY_TIME_DF, 0,
            alias, _open_local_zarr_simple,
            keep_vars, None, coordinates,
            None, False,
        )
        if result is None:
            return (None, 0)

        # -- Single-pass compute -------------------------------------
        # Read + decompress data ONCE into memory.
        result = result.compute(scheduler="synchronous")

        # -- R4: float32 reduction (50% RAM + I/O saving) ---------------
        # SWOT SSH precision is ~1 cm; float32 (7 decimal digits) is more
        # than sufficient.  Downcast data vars only (keep time/coords as-is).
        for _vname in list(result.data_vars):
            if result[_vname].dtype == np.float64:
                result[_vname] = result[_vname].astype(np.float32)

        # NaN filter on in-memory numpy arrays (negligible cost).
        _nmask = _nan_mask_numpy(result, n_points_dim)
        if _nmask is not None:
            if int(_nmask.sum()) == 0:
                del result
                return (None, 0)
            result = result.isel({n_points_dim: _nmask})

        n_pts = result.sizes.get(n_points_dim, 0)
        if n_pts == 0:
            del result
            return (None, 0)

        # Write mini zarr directly from this worker process.
        zarr_path = os.path.join(output_dir, f"file_{file_idx}.zarr")
        result.to_zarr(zarr_path, mode="w")
        del result
        gc.collect()
        return (zarr_path, n_pts)
    except Exception as exc:
        logger.debug(f"Batch preproc ({alias}) file {file_idx}: {exc}")
        return (None, 0)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def preprocess_batch_obs_files(
    local_paths: List[str],
    alias: str,
    keep_vars: Optional[List[str]],
    coordinates: Dict[str, str],
    n_points_dim: str = "n_points",
    output_zarr_dir: Optional[str] = None,
    *,
    max_shared_obs_files: int = 0,
    prep_workers: int = 0,
    prep_use_processes: bool = True,
    dask_client: Optional[Any] = None,
) -> Optional[str]:
    """Preprocess all unique observation files on the driver into a single zarr.

    Eliminates redundant per-worker preprocessing when multiple tasks share
    the same observation files (typical for SWOT/swath data with wide
    time_tolerance).  Each unique file is processed exactly once:
    ``open --> swath_to_points --> NaN-mask --> compute``.

    When *dask_client* is provided (R5 — distributed preprocessing), file
    preprocessing is dispatched to the Dask cluster instead of a local
    ProcessPoolExecutor, using the full cluster's CPU/IO bandwidth.

    The resulting zarr contains all valid ocean points with a ``time``
    coordinate along *n_points_dim*, enabling per-worker time filtering.

    Parameters
    ----------
    local_paths : list of str
        Unique local file paths (post-prefetch).
    alias : str
        Dataset alias (e.g. ``"swot"``).
    keep_vars : list of str or None
        Variables to retain in the output.
    coordinates : dict
        Coordinate name mapping, must contain ``"time"`` key.
    n_points_dim : str
        Name of the points dimension (default ``"n_points"``).
    output_zarr_dir : str or None
        Directory for temp zarr files.  Created automatically when *None*.
    max_shared_obs_files : int
        Skip shared zarr build above this count (0 = use env/default 5000).
    prep_workers : int
        Override max preprocessing workers (0 = auto).
    prep_use_processes : bool
        Use ProcessPoolExecutor (True) vs ThreadPoolExecutor.

    Returns
    -------
    str or None
        Absolute path to the shared batch zarr, or *None* on failure.
    """
    if not local_paths:
        return None

    # -- File-count guard ----------------------------------------------
    # For very large observation sets (e.g. SWOT with 9 000+ files over
    # a full year), building the shared Zarr on the driver can take over
    # an hour (one mini-flush every ~12 s × 300 flushes).  Skip the
    # shared build when the number of unique files exceeds the limit;
    # workers will fall back to processing only their ~25 files each.
    _MAX_SHARED_OBS_FILES = max_shared_obs_files or int(
        os.environ.get("DCTOOLS_SHARED_OBS_MAX_FILES", "5000")
    )
    if len(local_paths) > _MAX_SHARED_OBS_FILES:
        logger.info(
            f"Shared batch preprocessing ({alias}): {len(local_paths)} "
            f"files exceeds limit {_MAX_SHARED_OBS_FILES} — skipping. "
            f"Workers will process files individually."
        )
        return None

    t_start = _time.perf_counter()

    if output_zarr_dir is None:
        output_zarr_dir = tempfile.mkdtemp(prefix=f"dctools_batch_{alias}_")
    os.makedirs(output_zarr_dir, exist_ok=True)
    output_zarr_path = os.path.join(output_zarr_dir, "batch_shared.zarr")

    # If a previous run already produced this zarr, reuse it.
    if os.path.isdir(output_zarr_path):
        try:
            _probe = xr.open_zarr(output_zarr_path, consolidated=False)
            if _probe.sizes.get(n_points_dim, 0) > 0:
                logger.debug(
                    f"Shared batch zarr ({alias}): reusing existing "
                    f"({_probe.sizes[n_points_dim]:,} pts)"
                )
                _probe.close()
                return output_zarr_path
            _probe.close()
        except Exception:
            pass  # stale zarr — rebuild

    # -- Probe first file to detect swath structure ----------------------
    # These zarr stores often lack per-variable _ARRAY_DIMENSIONS
    # attributes; dimension info lives only in the consolidated
    # .zmetadata file.  Use consolidated=True first (with fallback).
    try:
        try:
            _first = xr.open_zarr(local_paths[0], consolidated=True, chunks={})
        except Exception:
            _first = xr.open_zarr(local_paths[0], consolidated=False, chunks={})
        is_swath = {"num_lines", "num_pixels"}.issubset(_first.dims)
        _first.close()
        del _first
    except Exception as exc:
        logger.error(f"Cannot probe first obs file for batch preproc: {exc}")
        return None

    # -- Per-file parallel preprocessing ------------------------------
    # Architecture overview (v2 — ProcessPoolExecutor):
    #   • Each unique obs file is processed by a *separate OS process*
    #     --> open_zarr --> swath_to_points --> single-pass compute --> NaN mask
    #     --> write mini zarr.  True CPU parallelism (no GIL).
    #   • Single-pass I/O: data is read/decompressed ONCE per file
    #     (v1 read twice: once for the dask NaN mask, once for compute).
    #   • No mini-batch accumulation or serial flush — each process
    #     writes its own zarr directly.
    #   • Falls back to ThreadPoolExecutor if process creation fails
    #     (e.g. container restrictions, fork issues).
    _mini_zarr_paths: List[str] = []
    n_ok = 0
    n_pts_total = 0

    _cpu_count = os.cpu_count() or 4
    _env_max = prep_workers or int(os.environ.get("DCTOOLS_PREP_WORKERS", "0"))

    # -- Memory-aware worker cap ---------------------------------------
    # Each preprocessing worker opens one SWOT/obs zarr file, flattens it
    # to n_points and materialises it in memory before writing a mini-zarr.
    # Budget per process (configurable via DCTOOLS_PREP_MEM_PER_WORKER_MB):
    #   • Default 1500 MB — conservative estimate accounting for xarray +
    #     zarr decompression buffers + dask overhead on top of raw data.
    #   • SWOT passes at full resolution: ~200-800 MB resident each.
    # A hard ceiling of 8 workers avoids runaway parallelism even when
    # the machine reports lots of free RAM (e.g. Dask cluster pages to swap).
    _MEM_PER_WORKER_MB = int(
        os.environ.get("DCTOOLS_PREP_MEM_PER_WORKER_MB", "1500")
    )
    _DRIVER_HEADROOM_GB = float(
        os.environ.get("DCTOOLS_PREP_DRIVER_HEADROOM_GB", "4.0")
    )
    _HARD_MAX_WORKERS = int(
        os.environ.get("DCTOOLS_PREP_MAX_WORKERS", "8")
    )
    _default_max = min(_cpu_count, _HARD_MAX_WORKERS)
    if _env_max <= 0:
        try:
            import psutil as _psutil_prep
            _avail_gb = _psutil_prep.virtual_memory().available / (1024 ** 3)
            _max_from_mem = max(
                1,
                int((_avail_gb - _DRIVER_HEADROOM_GB) / (_MEM_PER_WORKER_MB / 1024))
            )
            _default_max = min(_default_max, _max_from_mem)
        except Exception:
            pass  # psutil unavailable --> keep cpu/_HARD_MAX_WORKERS cap

    _MAX_PREP_WORKERS = min(
        _env_max if _env_max > 0 else _default_max,
        len(local_paths),
    )

    _args_list = [
        (p, idx, is_swath, n_points_dim, alias,
         keep_vars, dict(coordinates) if not isinstance(coordinates, dict) else coordinates,
         output_zarr_dir)
        for idx, p in enumerate(local_paths)
    ]

    _use_processes = prep_use_processes
    # Env var override (backward compat for direct script usage)
    _env_thr = os.environ.get("DCTOOLS_PREP_THREADS_ONLY", "")
    if _env_thr.lower() in ("1", "true", "yes"):
        _use_processes = False

    logger.debug(
        f"Shared batch preprocessing ({alias}): "
        f"{len(local_paths)} unique files, "
        f"{_MAX_PREP_WORKERS} {'dask-distributed' if dask_client else ('processes' if _use_processes else 'threads')}"
    )

    # -- R5: Distributed preprocessing via Dask cluster -----------------
    # When a Dask client is available, dispatch _process_file_to_zarr to
    # the cluster workers.  This uses ALL cluster CPUs instead of only
    # the driver's local ProcessPoolExecutor (limited to 8 workers).
    if dask_client is not None:
        try:
            from dask.distributed import as_completed as _dask_ac
            _dask_futs = [
                dask_client.submit(_process_file_to_zarr, a, pure=False)
                for a in _args_list
            ]
            for fut in _dask_ac(_dask_futs):
                try:
                    zarr_path, n_pts = fut.result()
                except Exception as _fe:
                    logger.debug(f"Batch preproc ({alias}): dask task failed: {_fe!r}")
                    continue
                if zarr_path is not None:
                    _mini_zarr_paths.append(zarr_path)
                    n_ok += 1
                    n_pts_total += n_pts
            # Cancel any remaining futures and free cluster resources
            try:
                dask_client.cancel(_dask_futs, force=True)
            except Exception:
                pass
            del _dask_futs
        except Exception as _dask_exc:
            logger.warning(
                f"Batch preproc ({alias}): Dask distributed failed "
                f"({_dask_exc!r}), falling back to local pool"
            )
            dask_client = None  # fall through to local pool below

    if dask_client is None:
        try:
            if _use_processes:
                from concurrent.futures import ProcessPoolExecutor as _FilePool
            else:
                raise RuntimeError("threads-only requested via env")  # --> fallback

            with _FilePool(max_workers=_MAX_PREP_WORKERS) as pool:
                from concurrent.futures import as_completed as _proc_ac
                _futs = [pool.submit(_process_file_to_zarr, a) for a in _args_list]
                _fut_to_path = {f: a[0] for f, a in zip(_futs, _args_list, strict=False)}
                for fut in _proc_ac(_futs):
                    try:
                        zarr_path, n_pts = fut.result()
                    except Exception as _fe:
                        _fpath = os.path.basename(str(_fut_to_path.get(fut, "unknown")))
                        logger.warning(
                            f"Batch preproc ({alias}): skipping file {_fpath}: {_fe!r}"
                        )
                        continue
                    if zarr_path is not None:
                        _mini_zarr_paths.append(zarr_path)
                        n_ok += 1
                        n_pts_total += n_pts

        except (RuntimeError, OSError, BrokenPipeError) as _pool_exc:
            # Fallback: ThreadPoolExecutor (GIL-limited but portable).
            if _use_processes:
                logger.warning(
                    f"Batch preproc ({alias}): ProcessPoolExecutor failed "
                    f"({_pool_exc!r}), falling back to threads"
                )
            from concurrent.futures import ThreadPoolExecutor as _ThrPool
            from concurrent.futures import as_completed as _thr_ac

            with _ThrPool(max_workers=_MAX_PREP_WORKERS) as tpool:
                _futs = [tpool.submit(_process_file_to_zarr, a) for a in _args_list]
                _fut_to_path = {f: a[0] for f, a in zip(_futs, _args_list, strict=False)}
                for fut in _thr_ac(_futs):
                    try:
                        zarr_path, n_pts = fut.result()
                    except Exception as _fe:
                        _fpath = os.path.basename(str(_fut_to_path.get(fut, "unknown")))
                        logger.warning(
                            f"Batch preproc ({alias}): skipping file {_fpath}: {_fe!r}"
                        )
                        continue
                    if zarr_path is not None:
                        _mini_zarr_paths.append(zarr_path)
                        n_ok += 1
                        n_pts_total += n_pts

    if not _mini_zarr_paths:
        logger.warning(f"Shared batch preprocessing ({alias}): no valid data")
        shutil.rmtree(output_zarr_dir, ignore_errors=True)
        return None

    # -- Streaming append-to-zarr (R1+R2: no concat, no global sort) ------
    # Instead of xr.concat (explosion of dask graph for 158+ parts) followed
    # by a catastrophic isel(sort_idx) that materialises everything, we:
    #   1. Sort mini-zarrs by file-level t0 (cheap metadata scan).
    #   2. Append each mini-zarr sequentially to the output zarr.
    # Peak RAM = 1 materialized mini-zarr at a time (constant).
    # The output is time-monotone at file granularity (intra-file order is
    # preserved).  Workers use searchsorted for fast contiguous slicing.
    _time_key = coordinates.get("time", "time")
    _mz_t0: List[Any] = []
    _mz_n_pts: List[int] = []
    for _mp in _mini_zarr_paths:
        _tmn: Any = None
        _n_p = 0
        try:
            _mz_probe = xr.open_zarr(_mp, consolidated=False)
            _n_p = _mz_probe.sizes.get(n_points_dim, 0)
            if _time_key in _mz_probe.variables:
                _tv_raw = np.asarray(_mz_probe[_time_key].values)
                if not np.issubdtype(_tv_raw.dtype, np.datetime64):
                    try:
                        _tv_raw = _tv_raw.astype("datetime64[ns]")
                    except Exception:
                        _tv_raw = np.array([], dtype="datetime64[ns]")
                if len(_tv_raw) > 0:
                    _tmn = _tv_raw.min()
                del _tv_raw
            _mz_probe.close()
            del _mz_probe
        except Exception:
            pass
        _mz_n_pts.append(_n_p)
        _mz_t0.append(
            float("inf") if _tmn is None
            else float(np.datetime64(_tmn, "ns").view(np.int64))
        )

    # Sort mini-zarrs by start time for near-monotone output.
    if sorted(_mz_t0) != _mz_t0:
        _mz_order = np.argsort(_mz_t0, kind="mergesort")
        _mini_zarr_paths = [_mini_zarr_paths[i] for i in _mz_order]
        _mz_n_pts = [_mz_n_pts[i] for i in _mz_order]
        del _mz_order
    del _mz_t0

    # -- Streaming append: one mini-zarr at a time → constant RAM ----------
    _time_is_sorted = True
    _global_time_parts: List[np.ndarray] = []
    _first_written = False

    for _si, _mp in enumerate(_mini_zarr_paths):
        if _mz_n_pts[_si] == 0:
            continue
        try:
            _ds_part = xr.open_zarr(_mp, consolidated=False)
            # Materialize this single file (bounded RAM).
            _ds_part = _ds_part.compute(scheduler="synchronous")

            # Drop inherited encoding to avoid chunk-size validation errors.
            _time_enc = None
            if "time" in _ds_part.variables:
                _time_enc = dict(_ds_part["time"].encoding)
            for _v in _ds_part.variables:
                _ds_part[_v].encoding.clear()
            if _time_enc and "time" in _ds_part.variables:
                for _ek in ("units", "calendar", "dtype"):
                    if _ek in _time_enc:
                        _ds_part["time"].encoding[_ek] = _time_enc[_ek]

            if n_points_dim in _ds_part.coords:
                _ds_part = _ds_part.drop_vars(n_points_dim)

            # Collect time values for the sidecar index.
            if _time_key in _ds_part.variables:
                _tv_p = np.asarray(_ds_part[_time_key].values)
                if not np.issubdtype(_tv_p.dtype, np.datetime64):
                    try:
                        _tv_p = _tv_p.astype("datetime64[ns]")
                    except Exception:
                        _tv_p = np.array([], dtype="datetime64[ns]")
                _global_time_parts.append(_tv_p)
                del _tv_p

            if not _first_written:
                _ds_part.to_zarr(output_zarr_path, mode="w")
                _first_written = True
            else:
                _ds_part.to_zarr(
                    output_zarr_path,
                    append_dim=n_points_dim,
                    mode="a",
                )

            del _ds_part
            if _si % 20 == 0:
                gc.collect()
        except Exception as _exc_app:
            logger.warning(
                f"Batch streaming ({alias}): failed to append file {_si}: {_exc_app!r}"
            )
            continue

    _global_time: Optional[np.ndarray] = (
        np.concatenate(_global_time_parts) if _global_time_parts else None
    )
    del _global_time_parts

    # Check if the appended output is actually time-sorted.
    if _global_time is not None and len(_global_time) > 1:
        _time_is_sorted = bool(np.all(_global_time[:-1] <= _global_time[1:]))

    # -- Save time-sort metadata for workers ----------------------------
    # Workers read this to decide between searchsorted (sorted) and
    # boolean-mask (unsorted) time filtering.
    _meta_path = os.path.join(output_zarr_dir, "batch_metadata.json")
    try:
        import json as _json_meta
        with open(_meta_path, "w") as _mf:
            _json_meta.dump({"time_sorted": _time_is_sorted}, _mf)
    except Exception:
        pass

    # -- Save time index as sidecar .npy for zero-copy worker access ---
    # _global_time was already built file-by-file above (no extra zarr read).
    # Workers memory-map it (mmap_mode='r') for zero-copy searchsorted
    # (sorted) or boolean masking (unsorted).
    _time_npy_path = os.path.join(output_zarr_dir, "time_index.npy")
    if _global_time is not None:
        try:
            _tv_save = _global_time
            if np.issubdtype(_tv_save.dtype, np.integer):
                _tv_save = _tv_save.astype("datetime64[ns]")
            np.save(_time_npy_path, _tv_save)
            logger.debug(
                f"Shared batch ({alias}): saved time index "
                f"({len(_tv_save):,} pts) to {_time_npy_path}"
            )
            del _tv_save
        except Exception as _exc_npy:
            logger.debug(f"Could not save time index .npy: {_exc_npy}")
    del _global_time

    # -- Free driver memory ---------------------------------------------
    gc.collect()

    # Consolidate metadata so worker threads don't each stat hundreds
    # of small .zarray/.zattrs files simultaneously.
    try:
        import zarr as _zarr_mod
        _zarr_mod.consolidate_metadata(output_zarr_path)
    except Exception:
        pass  # non-critical — workers fall back to consolidated=False

    # Cleanup mini zarrs (lazy_parts already closed above)
    for p in _mini_zarr_paths:
        shutil.rmtree(p, ignore_errors=True)

    # Register for cleanup at exit
    _SWOT_TEMP_DIRS.append(output_zarr_dir)

    elapsed = _time.perf_counter() - t_start
    logger.debug(
        f"Shared batch preprocessing ({alias}): "
        f"{n_ok}/{len(local_paths)} files --> "
        f"{n_pts_total:,} points in {elapsed:.1f}s"
    )

    return output_zarr_path


# ---------------------------------------------------------------------------
# Combined download + preprocess pipeline
# ---------------------------------------------------------------------------

def download_and_preprocess_obs_pipeline(
    remote_paths: List[str],
    cache_dir: str,
    fs: Any,
    alias: str,
    keep_vars: Optional[List[str]],
    coordinates: Dict[str, str],
    n_points_dim: str = "n_points",
    output_zarr_dir: Optional[str] = None,
    *,
    download_workers: int = 8,
    max_shared_obs_files: int = 0,
    prep_workers: int = 0,
    prep_use_processes: bool = True,
) -> Tuple[Dict[str, str], Optional[str]]:
    """Download observation files and preprocess them simultaneously.

    Instead of the sequential *download-all → preprocess-all* pattern, this
    pipeline starts the preprocessing pool immediately and feeds it files as
    soon as each one finishes downloading.  Download threads and preprocessing
    workers run concurrently, so both pools are active at the same time.

    This function is designed to be called AFTER pred/ref background threads
    have already been started, so all three operations (obs download,
    obs preprocessing, pred/ref downloads) proceed in parallel.

    Parameters
    ----------
    remote_paths : list of str
        Remote S3/Wasabi file paths.
    cache_dir : str
        Local cache directory for downloaded files.
    fs : fsspec-compatible filesystem handle.
    alias : str
        Dataset alias (e.g. ``"swot"``).
    keep_vars, coordinates, n_points_dim, output_zarr_dir
        Forwarded to preprocessing — same semantics as
        :func:`preprocess_batch_obs_files`.
    download_workers : int
        Max concurrent download threads (default: 8).
    max_shared_obs_files : int
        Skip shared zarr build above this count (0 = env / default 5000).
    prep_workers, prep_use_processes
        Forwarded to preprocessing worker configuration.

    Returns
    -------
    (path_map, shared_zarr_path)
        *path_map* maps remote_path → local_path.
        *shared_zarr_path* is the merged observation zarr or *None*.
    """
    import threading as _pl_thr
    import shutil as _pl_sh
    from concurrent.futures import (
        ProcessPoolExecutor as _PPool,
        ThreadPoolExecutor as _TPool,
        as_completed as _pl_ac,
    )

    if not remote_paths:
        return {}, None

    unique_paths = list(dict.fromkeys(remote_paths))
    _MAX_FILES = max_shared_obs_files or int(
        os.environ.get("DCTOOLS_SHARED_OBS_MAX_FILES", "5000")
    )
    _skip_shared = len(unique_paths) > _MAX_FILES

    t_pl = _time.perf_counter()
    if output_zarr_dir is None:
        output_zarr_dir = tempfile.mkdtemp(prefix=f"dctools_pipeline_{alias}_")
    os.makedirs(output_zarr_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    # Worker count (memory-aware, same formula as preprocess_batch_obs_files)
    _cpu = os.cpu_count() or 4
    _env_max = prep_workers or int(os.environ.get("DCTOOLS_PREP_WORKERS", "0"))
    _MEM_MB = int(os.environ.get("DCTOOLS_PREP_MEM_PER_WORKER_MB", "1500"))
    _HEAD = float(os.environ.get("DCTOOLS_PREP_DRIVER_HEADROOM_GB", "4.0"))
    _HARD_MAX = int(os.environ.get("DCTOOLS_PREP_MAX_WORKERS", "8"))
    _dmax = min(_cpu, _HARD_MAX)
    if _env_max <= 0:
        try:
            import psutil as _psutil_pl
            _av = _psutil_pl.virtual_memory().available / 1024 ** 3
            _dmax = min(_dmax, max(1, int((_av - _HEAD) / (_MEM_MB / 1024))))
        except Exception:
            pass
    _MAX_PREP = min(_env_max if _env_max > 0 else _dmax, len(unique_paths))

    _use_proc = prep_use_processes
    if os.environ.get("DCTOOLS_PREP_THREADS_ONLY", "").lower() in ("1", "true", "yes"):
        _use_proc = False

    _coord_d = dict(coordinates) if not isinstance(coordinates, dict) else coordinates
    _DL_W = min(download_workers, len(unique_paths), 64)

    logger.info(
        f"Obs pipeline ({alias}): {len(unique_paths)} files, "
        f"{_MAX_PREP} {'processes' if _use_proc else 'threads'} prep "
        f"\u00d7 {_DL_W} download threads"
    )

    # Shared state (thread-safe)
    path_map: Dict[str, str] = {}
    _pm_lock = _pl_thr.Lock()
    _file_idx: List[int] = [0]
    _idx_lock = _pl_thr.Lock()
    _proc_futures: List[Any] = []   # list of (file_idx, Future)
    _pf_lock = _pl_thr.Lock()

    _PoolCls = _PPool if _use_proc else _TPool
    try:
        with _PoolCls(max_workers=_MAX_PREP) as _prep_pool:

            def _dl_and_submit(rpath: str) -> bool:
                """Download one file; on success immediately submit preprocessing."""
                from pathlib import Path as _DLPath
                filename = _DLPath(rpath).name
                local_path = os.path.join(cache_dir, filename)
                ext = _DLPath(rpath).suffix
                try:
                    if ext == ".zarr":
                        if not (os.path.isdir(local_path) and os.listdir(local_path)):
                            tid = _pl_thr.current_thread().ident
                            s3key = rpath
                            if s3key.startswith("s3://"):
                                s3key = s3key[5:]
                            tmp = local_path + f".dl.{tid}"
                            if os.path.isdir(tmp):
                                _pl_sh.rmtree(tmp, ignore_errors=True)
                            fs.get(s3key, tmp, recursive=True)
                            # Unwrap single-level nested directory
                            _items = os.listdir(tmp)
                            if (
                                len(_items) == 1
                                and os.path.isdir(os.path.join(tmp, _items[0]))
                                and not any(f.startswith(".z") for f in _items)
                            ):
                                _nested = os.path.join(tmp, _items[0])
                                _unwrap = tmp + "_unwrap"
                                os.rename(_nested, _unwrap)
                                _pl_sh.rmtree(tmp, ignore_errors=True)
                                os.rename(_unwrap, tmp)
                            if os.path.isdir(local_path):
                                _pl_sh.rmtree(local_path, ignore_errors=True)
                            try:
                                os.rename(tmp, local_path)
                            except OSError as _re:
                                import errno as _errno_dl
                                if _re.errno == _errno_dl.ENOTEMPTY and os.path.isdir(local_path):
                                    _pl_sh.rmtree(tmp, ignore_errors=True)
                                else:
                                    raise
                    else:  # single file (.nc, etc.)
                        if not os.path.isfile(local_path):
                            tid = _pl_thr.current_thread().ident
                            tmp = local_path + f".tmp.{tid}"
                            with fs.open(rpath, "rb") as rf:
                                with open(tmp, "wb") as lf:
                                    lf.write(rf.read())
                            os.rename(tmp, local_path)

                    with _pm_lock:
                        path_map[rpath] = local_path

                    # Immediately submit to preprocessing pool.
                    if not _skip_shared and os.path.exists(local_path):
                        with _idx_lock:
                            fidx = _file_idx[0]
                            _file_idx[0] += 1
                        args = (
                            local_path, fidx, None,  # is_swath=None → auto-detect
                            n_points_dim, alias, keep_vars, _coord_d, output_zarr_dir,
                        )
                        pf = _prep_pool.submit(_process_file_to_zarr, args)
                        with _pf_lock:
                            _proc_futures.append((fidx, pf))
                    return True

                except Exception as _exc_dl:
                    logger.debug(f"Pipeline ({alias}) dl {filename}: {_exc_dl!r}")
                    with _pm_lock:
                        path_map[rpath] = local_path  # may not exist; callers check
                    return False

            # All downloads run concurrently.  Each thread submits preprocessing
            # as soon as its file is ready → download and preprocessing overlap.
            with _TPool(max_workers=_DL_W) as _dl_pool:
                _dl_futs = [_dl_pool.submit(_dl_and_submit, rp) for rp in unique_paths]
                for _ in _pl_ac(_dl_futs):
                    pass  # errors handled inside _dl_and_submit
            # _prep_pool.__exit__ blocks until all submitted tasks complete. ↑

    except (RuntimeError, OSError, BrokenPipeError) as _pl_exc:
        if _use_proc:
            logger.warning(
                f"Pipeline ({alias}): ProcessPool failed ({_pl_exc!r}), "
                "retrying with threads"
            )
            return download_and_preprocess_obs_pipeline(
                remote_paths=remote_paths, cache_dir=cache_dir, fs=fs,
                alias=alias, keep_vars=keep_vars, coordinates=coordinates,
                n_points_dim=n_points_dim, output_zarr_dir=output_zarr_dir,
                download_workers=download_workers,
                max_shared_obs_files=max_shared_obs_files,
                prep_workers=prep_workers, prep_use_processes=False,
            )
        raise

    # Download-only case (too many files for shared zarr).
    if _skip_shared:
        elapsed = _time.perf_counter() - t_pl
        logger.info(
            f"Pipeline ({alias}): {len(path_map)}/{len(unique_paths)} files "
            f"downloaded in {elapsed:.1f}s "
            f"(shared zarr skipped — {len(unique_paths)} > {_MAX_FILES})"
        )
        return path_map, None

    # Collect preprocessing results (futures already complete — pool exited).
    _mini_zarr_paths: List[str] = []
    n_ok = 0
    n_pts_total = 0
    for _fidx, _fut in _proc_futures:
        try:
            zarr_path, n_pts = _fut.result()
        except Exception as _fe:
            logger.debug(f"Pipeline preproc ({alias}) idx {_fidx}: {_fe!r}")
            continue
        if zarr_path is not None:
            _mini_zarr_paths.append(zarr_path)
            n_ok += 1
            n_pts_total += n_pts

    if not _mini_zarr_paths:
        logger.warning(f"Pipeline ({alias}): no valid data after preprocessing")
        return path_map, None

    # ── Streaming append-to-zarr (same as preprocess_batch_obs_files) ──
    output_zarr_path = os.path.join(output_zarr_dir, "batch_shared.zarr")
    _time_key = coordinates.get("time", "time")

    # Sort mini-zarrs by file-level t0 for near-monotone output.
    _mz_t0_pl: List[Any] = []
    _mz_n_pts_pl: List[int] = []
    for _mp in _mini_zarr_paths:
        _tmn_p: Any = None
        _n_p = 0
        try:
            _pr = xr.open_zarr(_mp, consolidated=False)
            _n_p = _pr.sizes.get(n_points_dim, 0)
            if _time_key in _pr.variables:
                _tv_raw = np.asarray(_pr[_time_key].values)
                if not np.issubdtype(_tv_raw.dtype, np.datetime64):
                    try:
                        _tv_raw = _tv_raw.astype("datetime64[ns]")
                    except Exception:
                        _tv_raw = np.array([], dtype="datetime64[ns]")
                if len(_tv_raw) > 0:
                    _tmn_p = _tv_raw.min()
                del _tv_raw
            _pr.close()
            del _pr
        except Exception:
            pass
        _mz_n_pts_pl.append(_n_p)
        _mz_t0_pl.append(
            float("inf") if _tmn_p is None
            else float(np.datetime64(_tmn_p, "ns").view(np.int64))
        )
    if sorted(_mz_t0_pl) != _mz_t0_pl:
        _ord = np.argsort(_mz_t0_pl, kind="mergesort")
        _mini_zarr_paths = [_mini_zarr_paths[i] for i in _ord]
        _mz_n_pts_pl = [_mz_n_pts_pl[i] for i in _ord]
        del _ord
    del _mz_t0_pl

    # -- Streaming append: one mini-zarr at a time → constant RAM ----------
    _time_is_sorted_pl = True
    _global_time_parts_pl: List[np.ndarray] = []
    _first_written_pl = False

    for _si_pl, _mp in enumerate(_mini_zarr_paths):
        if _mz_n_pts_pl[_si_pl] == 0:
            continue
        try:
            _ds_part = xr.open_zarr(_mp, consolidated=False)
            _ds_part = _ds_part.compute(scheduler="synchronous")

            # Drop inherited encoding to avoid chunk-size validation errors.
            _time_enc_pl = None
            if "time" in _ds_part.variables:
                _time_enc_pl = dict(_ds_part["time"].encoding)
            for _v in _ds_part.variables:
                _ds_part[_v].encoding.clear()
            if _time_enc_pl and "time" in _ds_part.variables:
                for _ek in ("units", "calendar", "dtype"):
                    if _ek in _time_enc_pl:
                        _ds_part["time"].encoding[_ek] = _time_enc_pl[_ek]

            if n_points_dim in _ds_part.coords:
                _ds_part = _ds_part.drop_vars(n_points_dim)

            # Collect time values for the sidecar index.
            if _time_key in _ds_part.variables:
                _tv_p = np.asarray(_ds_part[_time_key].values)
                if not np.issubdtype(_tv_p.dtype, np.datetime64):
                    try:
                        _tv_p = _tv_p.astype("datetime64[ns]")
                    except Exception:
                        _tv_p = np.array([], dtype="datetime64[ns]")
                _global_time_parts_pl.append(_tv_p)
                del _tv_p

            if not _first_written_pl:
                _ds_part.to_zarr(output_zarr_path, mode="w")
                _first_written_pl = True
            else:
                _ds_part.to_zarr(
                    output_zarr_path,
                    append_dim=n_points_dim,
                    mode="a",
                )

            del _ds_part
            if _si_pl % 20 == 0:
                gc.collect()
        except Exception as _exc_app:
            logger.warning(
                f"Pipeline streaming ({alias}): failed to append file {_si_pl}: {_exc_app!r}"
            )
            continue

    _global_time_pl: Optional[np.ndarray] = (
        np.concatenate(_global_time_parts_pl) if _global_time_parts_pl else None
    )
    del _global_time_parts_pl

    # Check if the appended output is actually time-sorted.
    if _global_time_pl is not None and len(_global_time_pl) > 1:
        _time_is_sorted_pl = bool(np.all(_global_time_pl[:-1] <= _global_time_pl[1:]))

    # Save time-sort metadata for workers.
    _meta_path_pl = os.path.join(output_zarr_dir, "batch_metadata.json")
    try:
        import json as _json_meta_pl
        with open(_meta_path_pl, "w") as _mf_pl:
            _json_meta_pl.dump({"time_sorted": _time_is_sorted_pl}, _mf_pl)
    except Exception:
        pass

    # Sidecar time index for worker zero-copy filtering
    _time_npy_pl = os.path.join(output_zarr_dir, "time_index.npy")
    if _global_time_pl is not None:
        try:
            _tv_s = _global_time_pl
            if np.issubdtype(_tv_s.dtype, np.integer):
                _tv_s = _tv_s.astype("datetime64[ns]")
            np.save(_time_npy_pl, _tv_s)
            del _tv_s
        except Exception as _exc_n:
            logger.debug(f"Could not save time index .npy: {_exc_n}")
    del _global_time_pl

    gc.collect()

    try:
        import zarr as _zarr_pl
        _zarr_pl.consolidate_metadata(output_zarr_path)
    except Exception:
        pass

    # Parallel mini-zarr cleanup
    with _TPool(max_workers=min(16, len(_mini_zarr_paths))) as _clp:
        for _p in _mini_zarr_paths:
            _clp.submit(shutil.rmtree, _p, True)

    _SWOT_TEMP_DIRS.append(output_zarr_dir)

    elapsed = _time.perf_counter() - t_pl
    logger.info(
        f"Obs pipeline ({alias}): "
        f"{n_ok}/{len(unique_paths)} files → "
        f"{n_pts_total:,} pts in {elapsed:.1f}s"
    )
    return path_map, output_zarr_path
