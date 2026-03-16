#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Driver-side batch preprocessing for observation datasets.

Processes all unique observation files on the driver into a single shared
Zarr store, eliminating redundant per-worker preprocessing when multiple
tasks share the same observation files (typical for swath data with
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


def _force_gc_and_trim() -> None:
    """Force garbage collection and return freed memory to the OS (Linux)."""
    gc.collect()
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Module-level temp-dir registry: cleaned on worker/process exit.
# Zarr stores written during data preprocessing are registered here so that
# they are reliably removed even if an exception propagates.
# ---------------------------------------------------------------------------
_DATA_TEMP_DIRS: List[str] = []


def _atexit_cleanup_data_dirs() -> None:  # noqa: D401
    """Remove all temporary zarr directories created during preprocessing."""
    for _d in _DATA_TEMP_DIRS:
        shutil.rmtree(_d, ignore_errors=True)


atexit.register(_atexit_cleanup_data_dirs)


# ---------------------------------------------------------------------------
# Shared streaming-append helper — DEPRECATED (no longer called).
#
# The serial merge approach (append N mini-zarrs into one big zarr) was
# replaced by the manifest-based approach in ``_write_manifest``.  Workers
# now open only the mini-zarrs that overlap their time window directly,
# which distributes the I/O across all Dask CPUs instead of serialising
# it on the driver.  This function is preserved for reference only.
# ---------------------------------------------------------------------------

def _streaming_append_mini_zarrs(  # noqa: C901  (deprecated, not called)
    mini_zarr_paths: List[str],
    output_zarr_path: str,
    output_zarr_dir: str,
    n_points_dim: str,
    time_key: str,
    alias: str,
    *,
    prefetch: int = 2,
    pre_n_pts: Optional[List[int]] = None,
    pre_t_min: Optional[List[float]] = None,
) -> None:
    """Sort mini-zarrs by time, then stream-append into a single Zarr store.

    Also writes sidecar metadata (``batch_metadata.json``, ``time_index.npy``)
    that workers use for fast time-based slicing.

    Parameters
    ----------
    mini_zarr_paths : list[str]
        Paths to per-file mini-zarr stores (will be cleaned up after merge).
    output_zarr_path : str
        Target Zarr store path.
    output_zarr_dir : str
        Parent directory for sidecar files.
    n_points_dim, time_key, alias
        Dimension / variable names and dataset alias for logging.
    prefetch : int
        Number of mini-zarrs to pre-read in the background thread.
    """
    import json as _json_merge
    import queue as _q_merge
    import threading as _thr_merge

    # -- Probe mini-zarr start times for sorting --------------------------
    # When pre-computed metadata is available (from _process_file_to_zarr),
    # skip the probe phase entirely — saves N extra zarr opens.
    _use_pre = pre_n_pts is not None and pre_t_min is not None
    if _use_pre:
        _mz_n_pts: List[int] = list(pre_n_pts)  # type: ignore[arg-type]
        _mz_t0: List[Any] = list(pre_t_min)  # type: ignore[arg-type]
    else:
        _mz_t0 = []
        _mz_n_pts = []
        for _mp in mini_zarr_paths:
            _tmn: Any = None
            _n_p = 0
            try:
                _mz_probe = xr.open_zarr(_mp, consolidated=False)
                _n_p = _mz_probe.sizes.get(n_points_dim, 0)
                if time_key in _mz_probe.variables:
                    _tv_raw = np.asarray(_mz_probe[time_key].values)
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

    if sorted(_mz_t0) != _mz_t0:
        _mz_order = np.argsort(_mz_t0, kind="mergesort")
        mini_zarr_paths = [mini_zarr_paths[i] for i in _mz_order]
        _mz_n_pts = [_mz_n_pts[i] for i in _mz_order]
        del _mz_order
    del _mz_t0

    # -- Producer-consumer: background read, main-thread write ------------
    # Write time values incrementally to a temporary binary file instead of
    # accumulating in a Python list.  For large batches (100M+ points) the
    # old list + np.concatenate approach consumed 800 MB+ of driver RAM.
    _time_bin_path = os.path.join(output_zarr_dir, "_time_parts.bin")
    _time_bin_fd = open(_time_bin_path, "wb")  # noqa: SIM115
    _time_bin_count = 0
    _first_written = False

    _read_q: _q_merge.Queue = _q_merge.Queue(maxsize=prefetch)
    _SENTINEL = object()

    def _reader_thread():
        for _si, _mp in enumerate(mini_zarr_paths):
            if _mz_n_pts[_si] == 0:
                continue
            try:
                # Prefer consolidated metadata (single JSON read)
                # when mini-zarrs were consolidated by _process_file_to_zarr.
                # Keep dataset LAZY (dask-backed) — materialization happens
                # chunk-by-chunk during to_zarr() in the main thread.
                try:
                    _ds = xr.open_zarr(_mp, consolidated=True)
                except Exception:
                    _ds = xr.open_zarr(_mp, consolidated=False)
                _read_q.put((_si, _ds))
            except Exception as _exc_r:
                logger.debug(f"Merge reader ({alias}) file {_si}: {_exc_r!r}")
        _read_q.put(_SENTINEL)

    _rthr = _thr_merge.Thread(target=_reader_thread, daemon=True)
    _rthr.start()

    # -- Batched append: accumulate several mini-zarrs in RAM before
    # writing a single zarr append.  Each individual to_zarr(append_dim=...)
    # re-reads and rewrites zarr metadata, which becomes the dominant cost
    # when done 500+ times.  Grouping into macro-batches of ~20 reduces
    # the number of appends from N to N/20, cutting metadata overhead ~20×.
    _APPEND_BATCH_SIZE = int(os.environ.get("DCTOOLS_APPEND_BATCH_SIZE", "20"))
    _pending: list = []
    _pending_count = 0

    # Force synchronous dask scheduler for the streaming merge.
    # Datasets are lazy (dask-backed); to_zarr() will materialise and
    # write data chunk-by-chunk, bounding peak RAM to ~1 chunk instead
    # of accumulating full datasets in memory.
    import dask as _dask_stream
    _prev_sched = _dask_stream.config.get("scheduler", None)
    _dask_stream.config.set(scheduler="synchronous")

    def _prepare_part(_ds_part, _si_local):
        """Clean encoding and extract time from one mini-zarr part."""
        nonlocal _time_bin_count
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
        # Write time values incrementally to disk (not RAM).
        if time_key in _ds_part.variables:
            _tv_p = np.asarray(_ds_part[time_key].values)
            if not np.issubdtype(_tv_p.dtype, np.datetime64):
                try:
                    _tv_p = _tv_p.astype("datetime64[ns]")
                except Exception:
                    _tv_p = np.array([], dtype="datetime64[ns]")
            if len(_tv_p) > 0:
                if _tv_p.dtype != np.dtype("datetime64[ns]"):
                    _tv_p = _tv_p.astype("datetime64[ns]")
                _time_bin_fd.write(_tv_p.tobytes())
                _time_bin_count += len(_tv_p)
            del _tv_p
        return _ds_part

    def _flush_pending():
        """Concatenate pending parts and write a single zarr append."""
        nonlocal _first_written, _pending, _pending_count
        if not _pending:
            return
        if len(_pending) == 1:
            _macro = _pending[0]
        else:
            _macro = xr.concat(_pending, dim=n_points_dim,
                               coords="minimal", compat="override",
                               join="override")
        # Unify chunks along n_points_dim: concat of lazy datasets with
        # different sizes produces heterogeneous dask chunks that zarr
        # rejects.  Rechunk into a single contiguous chunk per flush —
        # memory stays bounded (≤ APPEND_BATCH_SIZE mini-zarrs).
        # Note: Dataset.chunks raises ValueError when variables have
        # inconsistent chunks, so we apply .chunk() unconditionally.
        try:
            # Revert Point 3: Dask rechunking from irregular to regular chunks causes
            # catastrophic graph explosion and 100% CPU lockup. We MUST use -1
            # (which unifies the irregular chunks into one valid zarr block).
            _macro = _macro.chunk({n_points_dim: -1})
        except Exception:
            # Already unchunked (numpy-backed) — nothing to do.
            pass
        if not _first_written:
            _macro.to_zarr(output_zarr_path, mode="w",
                           safe_chunks=False)
            _first_written = True
        else:
            _macro.to_zarr(output_zarr_path,
                           append_dim=n_points_dim, mode="a",
                           safe_chunks=False)
        del _macro
        for _p in _pending:
            del _p
        _pending = []
        _pending_count = 0

    _iter_count = 0
    while True:
        item = _read_q.get()
        if item is _SENTINEL:
            break
        _si, _ds_part = item
        try:
            _ds_part = _prepare_part(_ds_part, _si)
            _pending.append(_ds_part)
            _pending_count += 1
            if _pending_count >= _APPEND_BATCH_SIZE:
                _flush_pending()
            _iter_count += 1
            # Throttle GC: collect every 50 iterations instead of every one.
            if _iter_count % 50 == 0:
                gc.collect()
        except Exception as _exc_app:
            logger.warning(
                f"Streaming ({alias}): failed to append file {_si}: {_exc_app!r}"
            )
            continue

    # Flush any remaining pending parts.
    _flush_pending()

    # Restore previous dask scheduler.
    if _prev_sched is not None:
        _dask_stream.config.set(scheduler=_prev_sched)
    else:
        try:
            del _dask_stream.config.config["scheduler"]
        except (KeyError, AttributeError):
            pass

    _rthr.join()
    _time_bin_fd.close()

    # -- Sidecar metadata --------------------------------------------------
    # Read the time index back from the incremental binary file using
    # memory-mapping so that only the pages needed for the sorted-check
    # and np.save are resident — never the full 800 MB+ array at once.
    _global_time: Optional[np.ndarray] = None
    if _time_bin_count > 0:
        try:
            _global_time = np.memmap(
                _time_bin_path, dtype="datetime64[ns]",
                mode="r", shape=(_time_bin_count,),
            )
        except Exception:
            # Fallback: read fully into RAM (should not happen).
            _global_time = np.fromfile(
                _time_bin_path, dtype="datetime64[ns]",
                count=_time_bin_count,
            )

    _time_is_sorted = True
    if _global_time is not None and len(_global_time) > 1:
        # Check sortedness in chunks to limit peak RAM from temporaries.
        _CHUNK = 2_000_000
        for _ci in range(0, len(_global_time) - 1, _CHUNK):
            _seg = _global_time[_ci : _ci + _CHUNK + 1]
            if not bool(np.all(_seg[:-1] <= _seg[1:])):
                _time_is_sorted = False
                break
            del _seg

    _meta_path = os.path.join(output_zarr_dir, "batch_metadata.json")
    try:
        with open(_meta_path, "w") as _mf:
            _json_merge.dump({"time_sorted": _time_is_sorted}, _mf)
    except Exception:
        pass

    _time_npy_path = os.path.join(output_zarr_dir, "time_index.npy")
    if _global_time is not None:
        try:
            # Chunked write: stream from mmap in 2M-element chunks to
            # avoid materialising the full 800 MB+ array in driver RAM.
            import numpy.lib.format as _npy_fmt
            with open(_time_npy_path, "wb") as _npy_f:
                _npy_fmt.write_array_header_1_0(
                    _npy_f,
                    {"descr": np.dtype("datetime64[ns]").str,
                     "fortran_order": False,
                     "shape": (_time_bin_count,)},
                )
                _CHUNK_NPY = 2_000_000
                for _ci in range(0, _time_bin_count, _CHUNK_NPY):
                    _seg = np.array(_global_time[_ci:_ci + _CHUNK_NPY])
                    _npy_f.write(_seg.tobytes())
                    del _seg
            logger.debug(
                f"Shared batch ({alias}): saved time index "
                f"({_time_bin_count:,} pts) to {_time_npy_path}"
            )
        except Exception as _exc_npy:
            logger.debug(f"Could not save time index .npy: {_exc_npy}")
    # Free mmap and remove temporary binary file.
    del _global_time
    try:
        os.unlink(_time_bin_path)
    except OSError:
        pass

    # Consolidate Zarr metadata.
    try:
        import zarr as _zarr_mod
        _zarr_mod.consolidate_metadata(output_zarr_path)
    except Exception:
        pass

    # Cleanup mini-zarrs.
    for p in mini_zarr_paths:
        shutil.rmtree(p, ignore_errors=True)


# ---------------------------------------------------------------------------
# Module-level helpers for process-pool based batch preprocessing.
# These must be at module scope so they are picklable by ProcessPoolExecutor.
# ---------------------------------------------------------------------------

def _open_local_zarr_simple(path: str, _alias: Any = None) -> xr.Dataset:
    """Open a local observation dataset from Zarr or NetCDF.

    Module-level function (picklable for multiprocessing).
    """
    from dctools.data.datasets.loader import FileLoader

    ds = FileLoader.open_dataset_auto(path, dask_safe=True)
    if ds is None:
        raise FileNotFoundError(f"Cannot open observation dataset: {path}")
    return ds


# Shared dummy dataframe for add_time_dim fallback (SWATH files contain
# their own per-point time coordinate — this is the fallback sentinel).
_DUMMY_TIME_DF = pd.DataFrame({
    "path": [""],
    "date_start": [pd.Timestamp("2000-01-01")],
    "date_end": [pd.Timestamp("2100-01-01")],
})


def _process_file_to_zarr(args: Tuple) -> Tuple[Optional[str], int, float, float]:
    """Process one observation file --> compute --> NaN-filter --> write mini zarr.

    Designed for ``ProcessPoolExecutor``: all arguments are picklable, each
    invocation is self-contained, and the result is a
    ``(path, n_pts, t_min_ns, t_max_ns)`` tuple where the timestamps
    (int64 view of datetime64[ns]) are used by the manifest for
    time-range filtering without re-opening mini-zarrs.

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

        # Cap C-level thread pools (pyinterp, BLAS, OpenMP) to 1 thread.
        # Without this, each preprocessing task on a Dask worker spawns
        # dozens of C++ threads (pyinterp releases the GIL), causing
        # massive CPU oversubscription and thrashing — the primary cause
        # of the 0% progress stall on SWOT swath data.
        try:
            from dctools.metrics.worker_cleanup import _cap_worker_threads
            _cap_worker_threads(1)
        except Exception:
            pass

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
            return (None, 0, float("inf"), float("inf"))

        # -- R1: Single-pass compute + NaN filter --------------------------
        # Compute the boolean NaN mask eagerly (cheap — just notnull flags),
        # then compute the full dataset and index with the mask.
        # Note: true lazy boolean indexing is not possible with dask
        # (unknown output size), so we compute the mask first, then
        # filter the materialised data.  Still a single data read.
        result = result.compute(scheduler="synchronous")

        _nmask = _nan_mask_numpy(result, n_points_dim)
        if _nmask is not None:
            if int(_nmask.sum()) == 0:
                del result
                return (None, 0, float("inf"), float("inf"))
            result = result.isel({n_points_dim: _nmask})

        n_pts = result.sizes.get(n_points_dim, 0)
        if n_pts == 0:
            del result
            return (None, 0, float("inf"), float("inf"))

        # Extract time min/max for manifest-based time filtering.
        _t_min_ns = float("inf")
        _t_max_ns = float("-inf")
        _time_key_local = coordinates.get("time", "time")
        for _tc in (_time_key_local, "time"):
            if _tc in result.coords:
                _tv = np.asarray(result.coords[_tc].values)
                if not np.issubdtype(_tv.dtype, np.datetime64):
                    try:
                        _tv = _tv.astype("datetime64[ns]")
                    except Exception:
                        continue
                if len(_tv) > 0:
                    _t_min_ns = float(
                        np.datetime64(_tv.min(), "ns").view(np.int64)
                    )
                    _t_max_ns = float(
                        np.datetime64(_tv.max(), "ns").view(np.int64)
                    )
                break

        # Write mini zarr directly from this worker process.
        zarr_path = os.path.join(output_dir, f"file_{file_idx}.zarr")
        result.to_zarr(zarr_path, mode="w")
        # Consolidate metadata so the merge reader can use
        # consolidated=True (single JSON read vs N .zarray stat calls).
        try:
            import zarr as _zarr_cons
            _zarr_cons.consolidate_metadata(zarr_path)
        except Exception:
            pass
        del result
        _force_gc_and_trim()
        return (zarr_path, n_pts, _t_min_ns, _t_max_ns)
    except Exception as exc:
        logger.warning(f"Batch preproc ({alias}) file {file_idx}: {exc}")
        return (None, 0, float("inf"), float("inf"))


# ---------------------------------------------------------------------------
# Manifest writing (replaces serial merge).
# ---------------------------------------------------------------------------

def _write_manifest(
    manifest_dir: str,
    mini_zarr_paths: List[str],
    n_pts_list: List[int],
    t_min_list: List[float],
    t_max_list: List[float],
    n_points_dim: str,
    time_key: str,
    alias: str,
) -> str:
    """Write a JSON manifest listing mini-zarr paths and time metadata.

    Workers use the manifest to open only the mini-zarrs that overlap
    their time window, eliminating the serial merge phase on the driver.

    Returns the path to the manifest file.
    """
    import json as _json_man

    manifest_path = os.path.join(manifest_dir, "manifest.json")
    files = []
    for zpath, npts, tmin, tmax in zip(
        mini_zarr_paths, n_pts_list, t_min_list, t_max_list
    ):
        files.append({
            "path": os.path.abspath(zpath),
            "n_points": npts,
            "t_min_ns": tmin,
            "t_max_ns": tmax,
        })

    manifest = {
        "version": 1,
        "n_points_dim": n_points_dim,
        "time_key": time_key,
        "alias": alias,
        "total_points": sum(n_pts_list),
        "files": files,
    }
    with open(manifest_path, "w") as f:
        _json_man.dump(manifest, f)

    logger.debug(
        f"Manifest ({alias}): {len(files)} mini-zarrs, "
        f"{sum(n_pts_list):,} total points → {manifest_path}"
    )
    return manifest_path


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
    the same observation files (typical for swath data with wide
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

    # If a previous run already produced a manifest, reuse it.
    _existing_manifest = os.path.join(output_zarr_dir, "manifest.json")
    if os.path.isfile(_existing_manifest):
        try:
            import json as _json_reuse
            with open(_existing_manifest) as _mf:
                _em = _json_reuse.load(_mf)
            if _em.get("total_points", 0) > 0:
                logger.debug(
                    f"Shared batch ({alias}): reusing existing manifest "
                    f"({_em['total_points']:,} pts)"
                )
                return output_zarr_dir
        except Exception:
            pass  # stale manifest — rebuild

    # -- Probe first file to detect swath structure ----------------------
    # These zarr stores often lack per-variable _ARRAY_DIMENSIONS
    # attributes; dimension info lives only in the consolidated
    # .zmetadata file.  Use consolidated=True first (with fallback).
    try:
        _first = _open_local_zarr_simple(local_paths[0])
        is_swath = {"num_lines", "num_pixels"}.issubset(_first.dims)
        # Point 2: Dynamically calculate raw footprint (used for memory limits below)
        _first_nbytes = sum(v.nbytes for v in _first.variables.values())
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
    _pre_n_pts: List[int] = []
    _pre_t_min: List[float] = []
    _pre_t_max: List[float] = []
    n_ok = 0
    n_pts_total = 0

    _cpu_count = os.cpu_count() or 4
    _env_max = prep_workers or int(os.environ.get("DCTOOLS_PREP_WORKERS", "0"))

    # -- Memory-aware worker cap ---------------------------------------
    # Each preprocessing worker opens one obs zarr file, flattens it
    # to n_points and materialises it in memory before writing a mini-zarr.
    # Budget per process (configurable via DCTOOLS_PREP_MEM_PER_WORKER_MB):
    # The worker count is capped by available memory (via psutil) and CPU
    # count.  An optional env var DCTOOLS_PREP_MAX_WORKERS can further
    # restrict it (e.g. in constrained containers).
    #
    # Memory multiplier is type-aware:
    #   • SWOT (swath): swath_to_points + decompression + NaN mask + mini-zarr
    #     write peaks at ~4× the raw uncompressed footprint; floor 1500 MB.
    #   • Nadir/Argo: simpler pipeline, ~2× raw footprint; floor 500 MB.
    _mem_multiplier = 4.0 if is_swath else 2.0
    _mem_floor_mb = 1500 if is_swath else 500
    _dyn_mem_mb = max(int((_first_nbytes * _mem_multiplier) / (1024 ** 2)), _mem_floor_mb)
    
    _env_mem = os.environ.get("DCTOOLS_PREP_MEM_PER_WORKER_MB", "")
    _MEM_PER_WORKER_MB = int(_env_mem) if _env_mem else _dyn_mem_mb
    _DRIVER_HEADROOM_GB = float(
        os.environ.get("DCTOOLS_PREP_DRIVER_HEADROOM_GB", "4.0")
    )
    _HARD_MAX_WORKERS = int(
        os.environ.get("DCTOOLS_PREP_MAX_WORKERS", str(_cpu_count))
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

    if not _use_processes:
        _MAX_PREP_WORKERS = min(_MAX_PREP_WORKERS, 4)

    logger.debug(
        f"Shared batch preprocessing ({alias}): "
        f"{len(local_paths)} unique files, "
        f"{_MAX_PREP_WORKERS} {'dask-distributed' if dask_client else ('processes' if _use_processes else 'threads')}"
    )

    # -- R5: Distributed preprocessing via Dask cluster -----------------
    # When a Dask client is available, dispatch _process_file_to_zarr to
    # the cluster workers.  This uses ALL cluster CPUs instead of only
    # the driver's local ProcessPoolExecutor (limited to 8 workers).
    #
    # THROTTLED SUBMISSION: Submit at most _max_inflight tasks at a time
    # to prevent memory explosion on workers.  Each SWOT file peaks at
    # ~5 GB during swath_to_points; submitting all at once can pause or
    # kill workers before any task completes → 0% progress stall.
    if dask_client is not None:
        try:
            from dask.distributed import as_completed as _dask_ac

            # Determine max in-flight tasks from cluster capacity.
            try:
                _ncores_prep = dask_client.ncores()
                _max_inflight = max(len(_ncores_prep), 2)
            except Exception:
                _max_inflight = max(_MAX_PREP_WORKERS, 2)

            _dask_futs = []
            _all_dask_futs = []
            _next_prep = 0
            _n_seed = min(_max_inflight, len(_args_list))

            # Seed initial batch
            for _si in range(_n_seed):
                _f = dask_client.submit(
                    _process_file_to_zarr, _args_list[_si], pure=False
                )
                _dask_futs.append(_f)
                _all_dask_futs.append(_f)
                _next_prep += 1

            _ac_prep = _dask_ac(_dask_futs)
            for fut in _ac_prep:
                try:
                    zarr_path, n_pts, t_min_ns, t_max_ns = fut.result()
                except Exception as _fe:
                    logger.warning(f"Batch preproc ({alias}): dask task failed: {_fe!r}")
                    # Submit next task even on failure to keep pipeline flowing
                    if _next_prep < len(_args_list):
                        _f_new = dask_client.submit(
                            _process_file_to_zarr,
                            _args_list[_next_prep],
                            pure=False,
                        )
                        _all_dask_futs.append(_f_new)
                        _ac_prep.add(_f_new)
                        _next_prep += 1
                    continue
                if zarr_path is not None:
                    _mini_zarr_paths.append(zarr_path)
                    _pre_n_pts.append(n_pts)
                    _pre_t_min.append(t_min_ns)
                    _pre_t_max.append(t_max_ns)
                    n_ok += 1
                    n_pts_total += n_pts
                # Submit next task as a slot becomes available
                if _next_prep < len(_args_list):
                    _f_new = dask_client.submit(
                        _process_file_to_zarr,
                        _args_list[_next_prep],
                        pure=False,
                    )
                    _all_dask_futs.append(_f_new)
                    _ac_prep.add(_f_new)
                    _next_prep += 1
            # Cancel any remaining futures and free cluster resources
            try:
                dask_client.cancel(_all_dask_futs, force=True)
            except Exception:
                pass
            del _all_dask_futs, _dask_futs
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
                        zarr_path, n_pts, t_min_ns, t_max_ns = fut.result()
                    except Exception as _fe:
                        _fpath = os.path.basename(str(_fut_to_path.get(fut, "unknown")))
                        logger.warning(
                            f"Batch preproc ({alias}): skipping file {_fpath}: {_fe!r}"
                        )
                        continue
                    if zarr_path is not None:
                        _mini_zarr_paths.append(zarr_path)
                        _pre_n_pts.append(n_pts)
                        _pre_t_min.append(t_min_ns)
                        _pre_t_max.append(t_max_ns)
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

            # Point 4: Cap thread pool at a small absolute limit to avoid thrashing GIL
            _fallback_threads = min(_MAX_PREP_WORKERS, 4)
            with _ThrPool(max_workers=_fallback_threads) as tpool:
                _futs = [tpool.submit(_process_file_to_zarr, a) for a in _args_list]
                _fut_to_path = {f: a[0] for f, a in zip(_futs, _args_list, strict=False)}
                for fut in _thr_ac(_futs):
                    try:
                        zarr_path, n_pts, t_min_ns, t_max_ns = fut.result()
                    except Exception as _fe:
                        _fpath = os.path.basename(str(_fut_to_path.get(fut, "unknown")))
                        logger.warning(
                            f"Batch preproc ({alias}): skipping file {_fpath}: {_fe!r}"
                        )
                        continue
                    if zarr_path is not None:
                        _mini_zarr_paths.append(zarr_path)
                        _pre_n_pts.append(n_pts)
                        _pre_t_min.append(t_min_ns)
                        _pre_t_max.append(t_max_ns)
                        n_ok += 1
                        n_pts_total += n_pts

    # -- Error rate check --------------------------------------------------
    # Abort if too many files failed preprocessing.  A high failure rate
    # usually indicates a systematic issue (wrong alias, corrupt files,
    # incompatible schema) rather than isolated bad files.
    n_failed = len(local_paths) - n_ok
    _FAILURE_THRESHOLD = float(
        os.environ.get("DCTOOLS_PREP_MAX_FAILURE_RATE", "0.5")
    )
    if len(local_paths) > 0 and n_failed / len(local_paths) > _FAILURE_THRESHOLD:
        logger.error(
            f"Shared batch preprocessing ({alias}): "
            f"{n_failed}/{len(local_paths)} files failed "
            f"({n_failed / len(local_paths):.0%} > {_FAILURE_THRESHOLD:.0%} threshold). "
            f"Aborting — check warnings above for per-file errors."
        )
        shutil.rmtree(output_zarr_dir, ignore_errors=True)
        return None

    if not _mini_zarr_paths:
        logger.warning(f"Shared batch preprocessing ({alias}): no valid data")
        shutil.rmtree(output_zarr_dir, ignore_errors=True)
        return None

    logger.info(
        f"Shared batch preprocessing ({alias}): "
        f"{n_ok}/{len(local_paths)} files OK, {n_failed} failed"
    )

    # -- Write manifest (replaces serial merge) ---------------------------
    # Instead of merging all mini-zarrs into one big zarr (serial I/O
    # bottleneck), write a lightweight JSON manifest.  Workers open only
    # the mini-zarrs that overlap their time window — the work is
    # distributed across all Dask CPUs instead of serialised on the driver.
    _time_key = coordinates.get("time", "time")
    _write_manifest(
        output_zarr_dir, _mini_zarr_paths,
        _pre_n_pts, _pre_t_min, _pre_t_max,
        n_points_dim, _time_key, alias,
    )

    # Register for cleanup at exit
    _DATA_TEMP_DIRS.append(output_zarr_dir)

    # -- Free driver memory ---------------------------------------------
    _force_gc_and_trim()

    elapsed = _time.perf_counter() - t_start
    logger.debug(
        f"Shared batch preprocessing ({alias}): "
        f"{n_ok}/{len(local_paths)} files --> "
        f"{n_pts_total:,} points in {elapsed:.1f}s"
    )

    return output_zarr_dir


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
    dask_client: Optional[Any] = None,
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
    _HARD_MAX = int(os.environ.get("DCTOOLS_PREP_MAX_WORKERS", str(_cpu)))
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

    # Point 4: Limit thread pool limit as doing CPU work on threads blocks GIL
    if not _use_proc:
        _MAX_PREP = min(_MAX_PREP, 4)

    _coord_d = dict(coordinates) if not isinstance(coordinates, dict) else coordinates
    _DL_W = min(download_workers, len(unique_paths), 64)

    # -- Distributed preprocessing via Dask cluster ----------------------
    # When a dask_client is provided, download all files first (threaded),
    # then delegate preprocessing to the Dask cluster via
    # preprocess_batch_obs_files(dask_client=...).  This avoids competing
    # ProcessPool + Dask for driver RAM and uses the full cluster bandwidth.
    if dask_client is not None:
        logger.info(
            f"Obs pipeline ({alias}): {len(unique_paths)} files, "
            f"distributed Dask preprocessing × {_DL_W} download threads"
        )
        path_map: Dict[str, str] = {}
        from concurrent.futures import ThreadPoolExecutor as _DLPool, as_completed as _dl_ac

        def _download_one(rpath: str) -> Optional[str]:
            from pathlib import Path as _DLPath
            import shutil as _dl_sh
            filename = _DLPath(rpath).name
            local_path = os.path.join(cache_dir, filename)
            ext = _DLPath(rpath).suffix
            try:
                if ext == ".zarr":
                    if not (os.path.isdir(local_path) and os.listdir(local_path)):
                        import threading as _dl_thr
                        tid = _dl_thr.current_thread().ident
                        s3key = rpath[5:] if rpath.startswith("s3://") else rpath
                        tmp = local_path + f".dl.{tid}"
                        if os.path.isdir(tmp):
                            _dl_sh.rmtree(tmp, ignore_errors=True)
                        fs.get(s3key, tmp, recursive=True)
                        _items = os.listdir(tmp)
                        if (
                            len(_items) == 1
                            and os.path.isdir(os.path.join(tmp, _items[0]))
                            and not any(f.startswith(".z") for f in _items)
                        ):
                            _nested = os.path.join(tmp, _items[0])
                            _unwrap = tmp + "_unwrap"
                            os.rename(_nested, _unwrap)
                            _dl_sh.rmtree(tmp, ignore_errors=True)
                            os.rename(_unwrap, tmp)
                        if os.path.isdir(local_path):
                            _dl_sh.rmtree(local_path, ignore_errors=True)
                        try:
                            os.rename(tmp, local_path)
                        except OSError as _re:
                            import errno as _errno_dl
                            if _re.errno == _errno_dl.ENOTEMPTY and os.path.isdir(local_path):
                                _dl_sh.rmtree(tmp, ignore_errors=True)
                            else:
                                raise
                else:
                    if not os.path.isfile(local_path):
                        import threading as _dl_thr
                        tid = _dl_thr.current_thread().ident
                        tmp = local_path + f".tmp.{tid}"
                        with fs.open(rpath, "rb") as rf:
                            with open(tmp, "wb") as lf:
                                lf.write(rf.read())
                        os.rename(tmp, local_path)
                return local_path
            except Exception as _exc:
                logger.debug(f"Pipeline ({alias}) dl {filename}: {_exc!r}")
                return None

        with _DLPool(max_workers=_DL_W) as dl_pool:
            futs = {dl_pool.submit(_download_one, rp): rp for rp in unique_paths}
            for fut in _dl_ac(futs):
                rp = futs[fut]
                try:
                    lp = fut.result()
                    if lp is not None:
                        path_map[rp] = lp
                except Exception:
                    pass

        if not path_map:
            return {}, None

        local_unique = list(dict.fromkeys(path_map.values()))
        shared_zarr = preprocess_batch_obs_files(
            local_paths=local_unique,
            alias=alias,
            keep_vars=keep_vars,
            coordinates=_coord_d,
            n_points_dim=n_points_dim,
            output_zarr_dir=output_zarr_dir,
            max_shared_obs_files=max_shared_obs_files,
            prep_workers=prep_workers,
            prep_use_processes=prep_use_processes,
            dask_client=dask_client,
        )
        elapsed = _time.perf_counter() - t_pl
        logger.info(
            f"Obs pipeline ({alias}): "
            f"{len(path_map)}/{len(unique_paths)} files → "
            f"distributed Dask in {elapsed:.1f}s"
        )
        return path_map, shared_zarr

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

    # -- P2.1: probe is_swath once (not once per worker) ----------------
    # _process_file_to_zarr determines is_swath with a metadata-only Zarr
    # open when is_swath=None.  All files in a batch share the same
    # structure, so probing once from the first downloaded file and
    # sharing the result via this mutable slot eliminates N-1 redundant
    # probes (fast, but still O(N) unnecessary I/O for large batches).
    _is_swath_known: List[Any] = [None]  # slot: None | True | False
    _is_swath_lock = _pl_thr.Lock()

    _PoolCls = _PPool if _use_proc else _TPool
    
    # Point 1: Backpressure / throttle downloads to avoid swamping memory/disk when CPU is bottlenecked
    _prep_throttle = _pl_thr.Semaphore(max(_MAX_PREP * 2, 4))

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
                        # Determine is_swath once via thread-safe double-check
                        # and reuse for all subsequent workers.
                        _task_is_swath = _is_swath_known[0]
                        if _task_is_swath is None:
                            with _is_swath_lock:
                                if _is_swath_known[0] is None:
                                    try:
                                        _probe = _open_local_zarr_simple(local_path)
                                        _is_swath_known[0] = {"num_lines", "num_pixels"}.issubset(_probe.dims)
                                        _probe.close()
                                    except Exception:
                                        _is_swath_known[0] = False
                                _task_is_swath = _is_swath_known[0]
                        args = (
                            local_path, fidx, _task_is_swath,  # known, not None
                            n_points_dim, alias, keep_vars, _coord_d, output_zarr_dir,
                        )
                        pf = _prep_pool.submit(_process_file_to_zarr, args)
                        pf.add_done_callback(lambda _: _prep_throttle.release())
                        with _pf_lock:
                            _proc_futures.append((fidx, pf))
                    else:
                        _prep_throttle.release()
                    return True

                except Exception as _exc_dl:
                    logger.debug(f"Pipeline ({alias}) dl {filename}: {_exc_dl!r}")
                    _prep_throttle.release()
                    return False

            # All downloads run concurrently.  Each thread submits preprocessing
            # as soon as its file is ready → download and preprocessing overlap.
            with _TPool(max_workers=_DL_W) as _dl_pool:
                def _throttled_submit(rp):
                    _prep_throttle.acquire()
                    return _dl_and_submit(rp)

                _dl_futs = [_dl_pool.submit(_throttled_submit, rp) for rp in unique_paths]
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
    _pre_n_pts: List[int] = []
    _pre_t_min: List[float] = []
    _pre_t_max: List[float] = []
    n_ok = 0
    n_pts_total = 0
    for _fidx, _fut in _proc_futures:
        try:
            zarr_path, n_pts, t_min_ns, t_max_ns = _fut.result()
        except Exception as _fe:
            logger.debug(f"Pipeline preproc ({alias}) idx {_fidx}: {_fe!r}")
            continue
        if zarr_path is not None:
            _mini_zarr_paths.append(zarr_path)
            _pre_n_pts.append(n_pts)
            _pre_t_min.append(t_min_ns)
            _pre_t_max.append(t_max_ns)
            n_ok += 1
            n_pts_total += n_pts

    if not _mini_zarr_paths:
        logger.warning(f"Pipeline ({alias}): no valid data after preprocessing")
        return path_map, None

    if len(path_map) != len(unique_paths):
        logger.warning(
            f"Pipeline ({alias}): only {len(path_map)}/{len(unique_paths)} files "
            "downloaded successfully; disabling shared obs manifest for this batch"
        )
        _force_gc_and_trim()
        return path_map, None

    # -- Write manifest (replaces serial merge) ---------------------------
    _time_key = coordinates.get("time", "time")
    _write_manifest(
        output_zarr_dir, _mini_zarr_paths,
        _pre_n_pts, _pre_t_min, _pre_t_max,
        n_points_dim, _time_key, alias,
    )

    _DATA_TEMP_DIRS.append(output_zarr_dir)
    _force_gc_and_trim()

    elapsed = _time.perf_counter() - t_pl
    logger.info(
        f"Obs pipeline ({alias}): "
        f"{n_ok}/{len(unique_paths)} files → "
        f"{n_pts_total:,} pts in {elapsed:.1f}s"
    )
    return path_map, output_zarr_dir
