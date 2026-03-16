"""Metrics evaluator module for distributed evaluation."""

import gc
import json
import os
import time
import traceback
import ctypes
from argparse import Namespace
from typing import Any, Callable, Dict, List, Optional

from collections import OrderedDict
import threading

import dask
import numpy as np
import pandas as pd
import xarray as xr
from dask.distributed import as_completed, wait
from loguru import logger
from oceanbench.core.distributed import DatasetProcessor
from tqdm import tqdm

from dctools.data.connection.connection_manager import (
    clean_for_serialization,
    create_worker_connect_config,
)
from dctools.data.datasets.dataloader import (
    EvaluationDataloader,
    ObservationDataViewer,
    filter_by_time,
)


_WORKER_DATASET_CACHE_LOCK = threading.Lock()
_WORKER_DATASET_CACHE: "OrderedDict[str, xr.Dataset]" = OrderedDict()


def _parse_memory_limit(value: Any) -> int:
    """Parse a human-readable memory string (e.g. ``"6GB"``) into bytes.

    Supports units: B, KB, MB, GB, TB (case-insensitive).
    If *value* is already numeric it is returned as-is.
    """
    if isinstance(value, (int, float)):
        return int(value)
    import re
    _s = str(value).strip().upper()
    _m = re.match(r"^([\d.]+)\s*(TB|GB|MB|KB|B)?$", _s)
    if not _m:
        raise ValueError(f"Cannot parse memory limit: {value!r}")
    _num = float(_m.group(1))
    _unit = (_m.group(2) or "B")
    _multipliers = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3, "TB": 1024**4}
    return int(_num * _multipliers[_unit])


def _open_dataset_worker_cached(
    open_func: Callable[[str], Optional[xr.Dataset]],
    source: str,
) -> tuple[Optional[xr.Dataset], bool]:
    """Open a dataset with a small per-worker LRU cache.

    This primarily targets remote Zarr datasets (S3/Wasabi) where repeated
    open calls are expensive (metadata reads). Cache size can be tuned via
    `DCTOOLS_WORKER_DATASET_CACHE_SIZE`.
    """
    try:
        cache_size = int(os.environ.get("DCTOOLS_WORKER_DATASET_CACHE_SIZE", "4"))
    except Exception:
        cache_size = 4

    if cache_size <= 0:
        return open_func(source), False

    with _WORKER_DATASET_CACHE_LOCK:
        cached = _WORKER_DATASET_CACHE.get(source)
        if cached is not None:
            _WORKER_DATASET_CACHE.move_to_end(source)
            return cached, True

    ds = open_func(source)
    if ds is None:
        return None, False

    with _WORKER_DATASET_CACHE_LOCK:
        existing = _WORKER_DATASET_CACHE.get(source)
        if existing is not None:
            _WORKER_DATASET_CACHE.move_to_end(source)
            return existing, True

        _WORKER_DATASET_CACHE[source] = ds
        _WORKER_DATASET_CACHE.move_to_end(source)
        while len(_WORKER_DATASET_CACHE) > cache_size:
            _, evicted = _WORKER_DATASET_CACHE.popitem(last=False)
            try:
                if hasattr(evicted, "close"):
                    evicted.close()
            except Exception:
                pass
    return ds, False


def _compute_with_timeout(
    arr: "xr.Dataset",
    timeout_s: int = 90,
    **kwargs: Any,
) -> "xr.Dataset":
    """Run ``arr.compute()`` in a daemon thread with a hard Python timeout.

    When the Dask synchronous scheduler is used from inside a Dask worker,
    the underlying ``aiobotocore`` (asyncio) ``read_timeout`` may never fire:
    its cancellation coroutine is registered in an event loop that is not
    progressing while the calling thread is blocked.  This wrapper guarantees
    the call cannot hang indefinitely regardless of the S3 client's internal
    timeout settings.
    """
    _result: list = [None]
    _exc: list = [None]
    _done = threading.Event()

    def _run() -> None:
        try:
            _result[0] = arr.compute(**kwargs)
        except Exception as _e:  # noqa: BLE001
            _exc[0] = _e
        finally:
            _done.set()

    _t = threading.Thread(target=_run, daemon=True)
    _t.start()
    if not _done.wait(timeout=timeout_s):
        # The daemon thread is still blocking on S3 — the OS will kill it at
        # process exit.  Raise so the task is marked as failed quickly.
        raise RuntimeError(
            f"arr.compute() timed out after {timeout_s}s. "
            "Likely cause: aiobotocore async read_timeout not firing in "
            "synchronous Dask-scheduler context (stale S3 connection). "
            "The task will be skipped by the watchdog and retried next run."
        )
    if _exc[0] is not None:
        raise _exc[0]
    return _result[0]  # type: ignore[no-any-return]


from dctools.data.datasets.dataset_manager import MultiSourceDatasetManager  # noqa: E402
from dctools.metrics.metrics import MetricComputer  # noqa: E402
from dctools.utilities.format_converter import convert_format1_to_format2  # noqa: E402
from dctools.utilities.misc_utils import (  # noqa: E402
    deep_copy_object,
    serialize_structure,
    to_float32,
)


def worker_memory_cleanup():
    """
    Manual memory cleanup to be run on workers.

    Performs aggressive garbage collection and memory trimming.
    """
    # Single gc.collect() is sufficient — 3× adds overhead per call
    gc.collect()

    # Linux-specific memory trimming (release to OS)
    try:
        ctypes.CDLL('libc.so.6').malloc_trim(0)
    except Exception:
        pass


def _clear_xarray_file_cache() -> bool:
    """Best-effort clearing of xarray's global file cache on the current process."""
    try:
        import xarray as xr

        # Use default xarray file cache (128) — setting to 1 forces
        # constant file re-opening which kills I/O throughput on swath data.
        # xr.set_options(file_cache_maxsize=1)

        try:
            # Clear any existing cached file handles
            # Not part of xarray's public API, but widely used and necessary
            xr.backends.file_manager.FILE_CACHE.clear()
        except Exception:
            pass
        return True
    except Exception:
        return False


def _worker_full_cleanup() -> bool:
    """Full cleanup routine to run on workers via client.run()."""
    import os
    # Ensure HDF5/NetCDF env vars are set in worker
    env_vars = {
        "HDF5_USE_FILE_LOCKING": "FALSE",
        "NETCDF4_DEACTIVATE_MPI": "1",
        "NETCDF4_USE_FILE_LOCKING": "FALSE",
        "HDF5_DISABLE_VERSION_CHECK": "1",
        "ARGOPY_NETCDF_LOCKING": "FALSE",
    }
    for key, value in env_vars.items():
        os.environ[key] = value

    # Cap library-level threads to prevent CPU oversubscription
    # (pyinterp, BLAS, OpenMP, torch, etc.)
    for tvar in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
        "PYINTERP_NUM_THREADS", "GOTO_NUM_THREADS", "BLOSC_NTHREADS",
        # dc_catalog ThreadPoolExecutor — defaults to 16, caps it here to 1
        "DCTOOLS_CATALOG_THREADS",
        # PyTorch inter-op pool — NOT covered by OMP_NUM_THREADS;
        # defaults to cpu_count() = 22 (measured: get_num_interop_threads()=16)
        "TORCH_NUM_THREADS", "TORCH_NUM_INTEROP_THREADS",
    ):
        os.environ[tvar] = "1"
    # Torch inter-op threads at runtime (env var only works at import time)
    try:
        import torch as _torch_init
        _torch_init.set_num_threads(1)
        _torch_init.set_num_interop_threads(1)
    except RuntimeError:
        pass  # set_num_interop_threads already called — safe
    except Exception:
        pass

    # threadpoolctl: resize already-running BLAS/OpenMP pools.
    try:
        import threadpoolctl
        threadpoolctl.threadpool_limits(limits=1)
    except Exception:
        pass
    # allow 2 threads for decent decompression speed on Blosc-compressed SWOT HDF5 chunks.
    try:
        import blosc  # type: ignore[import-not-found]
        blosc.set_nthreads(2)
    except Exception:
        pass
    try:
        from numcodecs import blosc as _nc_blosc  # type: ignore[import-untyped]
        _nc_blosc.set_nthreads(2)
    except Exception:
        pass

    # Clear the per-worker dataset LRU cache so the next batch always opens
    # fresh S3 connections.  Stale connections (closed by the S3 server after
    # inactivity) cause aiobotocore's async read_timeout not to fire when
    # called from a synchronous Dask scheduler context — resulting in 20+ min
    # hangs.  Clearing here ensures each batch starts with live connections.
    try:
        with _WORKER_DATASET_CACHE_LOCK:
            for _evicted_ds in _WORKER_DATASET_CACHE.values():
                try:
                    if hasattr(_evicted_ds, "close"):
                        _evicted_ds.close()
                except Exception:
                    pass
            _WORKER_DATASET_CACHE.clear()
    except Exception:
        pass

    # Patch xr.open_dataset to use scipy engine for in-memory data
    # (netCDF4 C library fails with EPERM on BytesIO / raw bytes)
    try:
        import io
        import xarray as xr
        if not hasattr(xr, '_original_open_dataset'):
            xr._original_open_dataset = xr.open_dataset  # type: ignore[attr-defined]

            def _open_dataset_scipy_for_inmem(filename_or_obj, *args, **kwargs):
                if isinstance(filename_or_obj, (bytes, io.BytesIO, io.BufferedIOBase)):
                    kwargs.setdefault('engine', 'scipy')
                if kwargs.get('engine') == 'scipy':
                    _bk = kwargs.get('backend_kwargs')
                    if _bk is None:
                        _bk = {}
                    else:
                        _bk = dict(_bk)
                    _bk.setdefault('mmap', False)
                    kwargs['backend_kwargs'] = _bk
                return xr._original_open_dataset(filename_or_obj, *args, **kwargs)  # type: ignore[attr-defined]

            xr.open_dataset = _open_dataset_scipy_for_inmem
    except Exception:
        pass

    _clear_xarray_file_cache()
    worker_memory_cleanup()
    return True


def _cap_worker_threads(max_threads: int = 1) -> None:
    """Limit per-worker thread parallelism for BLAS/OpenMP/pyinterp.

    When Dask workers spawn CPU-bound C/C++ code (pyinterp, scipy, BLAS),
    each library may itself create threads.  With N workers × T Dask
    threads × K library threads the machine can be massively
    oversubscribed, causing 100% CPU on all cores and thrashing.

    This function uses **two complementary mechanisms**:

    1. Environment variables — honoured by libraries that have NOT yet
       initialised their thread pool (i.e. first call).
    2. ``threadpoolctl`` — directly resizes already-running BLAS / OpenMP
       thread pools at the C level, even if the env vars were set after
       library initialisation.

    Calling this at the top of each task ensures that only
    *max_threads* additional threads are created per worker task.
    """
    _t = str(max_threads)
    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "GOTO_NUM_THREADS",
        "BLOSC_NTHREADS",
        # Our custom env var read by oceanbench's pyinterp wrappers
        "PYINTERP_NUM_THREADS",
        # Cap Python ThreadPoolExecutor in dc_catalog (default 16 -> saturates all CPUs)
        "DCTOOLS_CATALOG_THREADS",
        # PyTorch: TORCH_NUM_INTEROP_THREADS controls the inter-op pool
        # (cpu_count() by default = 22 on this machine; NOT covered by
        # OMP_NUM_THREADS or threadpoolctl)
        "TORCH_NUM_THREADS", "TORCH_NUM_INTEROP_THREADS",
    ):
        os.environ[var] = _t

    # ── PyTorch: call the runtime setters as well ─────────────────────────
    # The env vars above work only if torch hasn't been imported yet.
    # set_num_interop_threads() can only be called once (before any forward
    # pass); subsequent calls raise RuntimeError — absorb silently.
    try:
        import torch as _torch_rt
        _torch_rt.set_num_threads(max_threads)
        _torch_rt.set_num_interop_threads(max_threads)
    except RuntimeError:
        pass  # already set
    except Exception:
        pass

    # ── C-level thread pool cap (belt-and-suspenders) ──
    # threadpoolctl talks directly to the shared libraries already loaded
    # in the process (libopenblas, libgomp, libiomp5, …) and resizes
    # their internal pools.  This works even if the env vars were set
    # *after* the library created its default thread pool.
    try:
        import threadpoolctl
        threadpoolctl.threadpool_limits(limits=max_threads)
    except ImportError:
        pass
    except Exception:
        pass

    # ── Blosc: NOT covered by threadpoolctl — cap explicitly ──────────
    # The Blosc compression library uses its own internal thread pool
    # (independent of OpenMP and BLAS).  SWOT NetCDF/Zarr files are
    # typically Blosc-compressed; allow a small number of threads (2)
    # for decent decompression throughput without oversubscribing CPUs.
    _blosc_threads = max(2, max_threads)
    try:
        import blosc
        blosc.set_nthreads(_blosc_threads)
    except Exception:
        pass
    try:
        from numcodecs import blosc as _nc_blosc
        _nc_blosc.set_nthreads(_blosc_threads)
    except Exception:
        pass


def compute_metric(
    entry: Dict[str, Any],
    pred_source_config: Namespace,
    ref_source_config: Namespace,
    model: str,
    list_metrics: List[MetricComputer],
    pred_transform: Callable,
    ref_transform: Callable,
    argo_index: Optional[Optional[Any]] = None,
    reduce_precision: bool = False,
    results_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compute metrics for a single prediction-reference pair entry.

    Args:
        entry (Dict[str, Any]): Dictionary containing data and metadata for the evaluation.
        pred_source_config (Namespace): Configuration for the prediction source.
        ref_source_config (Namespace): Configuration for the reference source.
        model (str): Name of the model being evaluated.
        list_metrics (List[MetricComputer]): List of metric computers to apply.
        pred_transform (Callable): Tranformation function for prediction data.
        ref_transform (Callable): Transformation function for reference data.
        argo_index (Optional[Any], optional): Index for Argo data, if applicable. Defaults to None.
        reduce_precision (bool, optional): Whether to reduce floating point precision to float32.
            Defaults to False.

    Returns:
        Dict[str, Any]: Dictionary containing the evaluation results.
    """
    try:
        # ── Prevent CPU oversubscription ──
        # Observation datasets (SWOT, saral, …) trigger pyinterp
        # bilinear interpolation which is CPU-bound C++ code that
        # releases the GIL.  With 8 workers × 2 threads × 4 pyinterp
        # threads the machine (16 physical cores) gets massively
        # oversubscribed -> 100 % CPU thrashing.
        # Cap library-level threads to 1 so only the Dask thread
        # concurrency controls parallelism.
        _cap_worker_threads(1)
        _t0_total = time.perf_counter()
        forecast_reference_time = entry.get("forecast_reference_time")
        lead_time = entry.get("lead_time")
        valid_time = entry.get("valid_time")
        pred_coords = entry.get("pred_coords")
        ref_coords = entry.get("ref_coords")
        ref_alias = entry.get("ref_alias")
        ref_is_observation = entry.get("ref_is_observation")

        # Logging debug: Confirm task start on worker
        # logger.debug(
        #     f"Start processing: {forecast_reference_time} "
        #     f"(Valid: {valid_time}) on worker"
        # )

        pred_protocol = pred_source_config.protocol
        ref_protocol = ref_source_config.protocol

        pred_source = entry["pred_data"]
        ref_source = entry["ref_data"]

        open_pred_func = create_worker_connect_config(
            pred_source_config,
            argo_index,
        )
        open_ref_func = create_worker_connect_config(
            ref_source_config,
            argo_index,
        )

        pred_data_from_cache = False
        pred_data_base = None
        if isinstance(pred_source, str):
            pred_data_base, pred_data_from_cache = _open_dataset_worker_cached(
                open_pred_func,
                pred_source,
            )
            pred_data = pred_data_base
        else:
            pred_data = pred_source

        # Simple nearest-neighbor time selection (original, fast approach)
        pred_data_selected = pred_data.sel(time=valid_time, method="nearest")  # type: ignore[union-attr]

        # Ensure dimension "time" is preserved or restored (needed for concatenation later)
        if "time" not in pred_data_selected.dims:
            pred_data = pred_data_selected.expand_dims("time")
            pred_data = pred_data.assign_coords(time=[valid_time])
        else:
            pred_data = pred_data_selected.assign_coords(time=[valid_time])

        # ── Prediction: keep lazy until after transforms ───────────────
        # For gridded datasets, transforms often subset/regrid the data.
        # Computing *before* transforms can load far more data than needed
        # and can blow worker memory (leading to worker restarts / scheduler
        # errors). We keep it lazy here, apply transforms, then compute once.

        # Drop unused variables early when possible (reduces IO + memory).
        _pred_keep = getattr(pred_source_config, "keep_variables", None)
        if not _pred_keep:
            _pred_keep = getattr(pred_source_config, "eval_variables", None)
        if _pred_keep and hasattr(pred_data, "data_vars"):
            _pred_keep_set = [v for v in list(_pred_keep) if v in pred_data.data_vars]
            if _pred_keep_set:
                pred_data = pred_data[_pred_keep_set]

        if reduce_precision:
            pred_data = to_float32(pred_data)

        if ref_source is not None:
            if ref_is_observation:
                # Observation entry is a dict with connection metadata;
                # we must reconstruct the actual xr.Dataset on the worker.
                raw_ref_df = ref_source["source"]
                keep_vars = ref_source["keep_vars"]
                target_dimensions = ref_source["target_dimensions"]
                time_bounds = ref_source["time_bounds"]
                metadata = ref_source["metadata"]

                ref_df = raw_ref_df.get_dataframe()
                t0, t1 = time_bounds
                ref_df = filter_by_time(ref_df, t0, t1)

                # ── Substitute remote paths with prefetched local copies ──
                # When the driver has pre-downloaded observation files to
                # local disk (see prefetch_obs_files_to_local), replace
                # the S3 paths in the dataframe so the worker opens local
                # files instead of issuing remote requests.
                _prefetched_map = (
                    ref_source.get("prefetched_local_paths")
                    if isinstance(ref_source, dict) else None
                )
                if _prefetched_map and "path" in ref_df.columns:
                    ref_df = ref_df.copy()
                    ref_df["path"] = ref_df["path"].map(
                        lambda p: _prefetched_map.get(p, p)
                    )

                if ref_df.empty:
                    logger.warning(
                        f"No {ref_alias} data for time interval: {t0}/{t1}"
                    )
                    return {
                        "ref_alias": ref_alias,
                        "result": None,
                    }

                n_points_dim = "n_points"
                if ref_coords is not None and hasattr(ref_coords.coordinates, "n_points"):
                    n_points_dim = ref_coords.coordinates["n_points"]

                # ── ARGO shared-Zarr fast path ────────────────────────────
                # The driver has merged ALL batch time-windows into a
                # single shared, time-sorted Zarr (see
                # ArgoManager.prefetch_batch_shared_zarr).  Each worker
                # opens this Zarr and filters by its specific time_bounds
                # using np.searchsorted -> reads only contiguous chunks.
                # This replaces the old per-window Zarr approach.
                _argo_shared_zarr = (
                    ref_source.get("prefetched_argo_shared_zarr")
                    if isinstance(ref_source, dict) else None
                )
                # Legacy per-window path (kept for backward compat)
                _argo_legacy_zarr = (
                    ref_source.get("prefetched_zarr_path")
                    if isinstance(ref_source, dict) else None
                )
                _used_prefetch = False

                if (
                    ref_alias == "argo_profiles"
                    and _argo_shared_zarr
                ):
                    try:
                        import xarray as _xr_argo
                        # Support either a single shared store (str/path)
                        # or multiple stores (list[str]) for month-boundary
                        # windows.
                        if isinstance(_argo_shared_zarr, (list, tuple)):
                            _zarr_paths = [str(p) for p in _argo_shared_zarr]
                        else:
                            _zarr_paths = [str(_argo_shared_zarr)]

                        _zarr_paths = [p for p in _zarr_paths if p]
                        if not _zarr_paths or not all(os.path.exists(p) for p in _zarr_paths):
                            raise FileNotFoundError(
                                "One or more shared ARGO Zarr paths are missing"
                            )

                        _ds_parts = [
                            _xr_argo.open_zarr(
                                p,
                                consolidated=True,
                                chunks=None,
                            )
                            for p in _zarr_paths
                        ]

                        if len(_ds_parts) == 1:
                            ref_data = _ds_parts[0]
                        else:
                            # Concatenate in time order (paths injected already sorted)
                            ref_data = _xr_argo.concat(_ds_parts, dim="obs")

                        # Normalise dimension name
                        _obs_dim = "obs"
                        if _obs_dim in ref_data.dims and n_points_dim not in ref_data.dims:
                            ref_data = ref_data.rename({_obs_dim: n_points_dim})
                        elif _obs_dim not in ref_data.dims and n_points_dim in ref_data.dims:
                            pass  # already named correctly
                        else:
                            _obs_dim = n_points_dim  # no rename needed

                        # Detect time coordinate name (ARGO uses uppercase TIME)
                        _time_name = None
                        for _tc in ("TIME", "time", "JULD"):
                            if _tc in ref_data.coords or _tc in ref_data.data_vars:
                                _time_name = _tc
                                break

                        if _time_name is not None:
                            if _time_name in ref_data.coords:
                                _t_vals = np.asarray(ref_data.coords[_time_name].values)
                            else:
                                _t_vals = np.asarray(ref_data[_time_name].values)
                            if not np.issubdtype(_t_vals.dtype, np.datetime64):
                                try:
                                    _t_vals = pd.to_datetime(_t_vals).values
                                except Exception:
                                    _t_vals = _t_vals.astype("datetime64[ns]")

                            _t0_np = np.datetime64(pd.Timestamp(time_bounds[0]))
                            _t1_np = np.datetime64(pd.Timestamp(time_bounds[1]))

                            # Fast path: data is sorted -> contiguous slice
                            _is_sorted = (
                                len(_t_vals) <= 1
                                or bool(np.all(_t_vals[:-1] <= _t_vals[1:]))
                            )
                            if _is_sorted:
                                _i0 = int(np.searchsorted(_t_vals, _t0_np, side="left"))
                                _i1 = int(np.searchsorted(_t_vals, _t1_np, side="right"))
                                ref_data = ref_data.isel({n_points_dim: slice(_i0, _i1)})
                            else:
                                _mask = (_t_vals >= _t0_np) & (_t_vals <= _t1_np)
                                ref_data = ref_data.isel({n_points_dim: _mask})

                        # With chunks=None the data is already NumPy;
                        # .compute() is only needed if still dask-backed.
                        if any(
                            hasattr(ref_data[v].data, "dask")
                            for v in ref_data.variables
                        ):
                            ref_data = ref_data.compute(
                                scheduler="synchronous"
                            )

                        if ref_data.sizes.get(n_points_dim, 0) == 0:
                            ref_data = None

                        _used_prefetch = True
                    except Exception as exc:
                        logger.warning(
                            f"Cannot use shared ARGO Zarr "
                            f"'{_argo_shared_zarr}': {exc!r} "
                            "— falling back"
                        )
                        import traceback as _tb_argo
                        _tb_argo.print_exc()

                # ── Legacy per-window ARGO fast path (backward compat) ────
                if (
                    not _used_prefetch
                    and ref_alias == "argo_profiles"
                    and _argo_legacy_zarr
                    and os.path.exists(str(_argo_legacy_zarr))
                ):
                    try:
                        import xarray as _xr_local
                        ref_data = _xr_local.open_zarr(
                            str(_argo_legacy_zarr), consolidated=True
                        )
                        if "obs" in ref_data.dims and n_points_dim not in ref_data.dims:
                            ref_data = ref_data.rename({"obs": n_points_dim})
                        _used_prefetch = True
                    except Exception as exc:
                        logger.warning(
                            f"Cannot open prefetched ARGO Zarr "
                            f"'{_argo_legacy_zarr}': {exc!r} "
                            "— falling back to live download"
                        )

                # ── Shared obs zarr fast path (SWOT, saral, …) ────────────
                # The driver has already preprocessed ALL unique observation
                # files into a single shared zarr (see
                # preprocess_batch_obs_files).  We open it, filter by this
                # task's time_bounds, and skip per-file preprocessing
                # entirely.
                _shared_obs_zarr = (
                    ref_source.get("prefetched_obs_zarr_path")
                    if isinstance(ref_source, dict)
                    else None
                )
                if (
                    not _used_prefetch
                    and _shared_obs_zarr
                    and os.path.exists(str(_shared_obs_zarr))
                ):
                    # Retry up to 3 times — with threaded workers,
                    # concurrent zarr reads can hit transient I/O
                    # or metadata-discovery races.
                    _zarr_last_exc = None
                    for _zarr_attempt in range(3):
                        try:
                            import xarray as _xr_obs
                            from pathlib import Path as _OPath

                            # Prefer consolidated metadata (single
                            # JSON read vs hundreds of small files).
                            try:
                                ref_data = _xr_obs.open_zarr(
                                    str(_shared_obs_zarr),
                                    consolidated=True,
                                )
                            except Exception:
                                ref_data = _xr_obs.open_zarr(
                                    str(_shared_obs_zarr),
                                    consolidated=False,
                                )

                            # Filter by this task's time window.
                            t0_tb, t1_tb = time_bounds
                            _t0 = np.datetime64(pd.Timestamp(t0_tb))
                            _t1 = np.datetime64(pd.Timestamp(t1_tb))

                            # ── Zero-copy time index via sidecar .npy ─────
                            # The driver saves the sorted time array as a
                            # contiguous .npy file during shared-zarr build.
                            # Workers memory-map it (mmap_mode='r') so the OS
                            # shares the same physical pages across all workers
                            # — ~0 MB additional RSS instead of ~860 MB per
                            # worker for a 107 M-point zarr.
                            _time_npy = str(
                                _OPath(str(_shared_obs_zarr)).parent
                                / "time_index.npy"
                            )
                            _time_vals = None
                            if os.path.exists(_time_npy):
                                _time_vals = np.load(
                                    _time_npy, mmap_mode="r"
                                )
                                if np.issubdtype(
                                    _time_vals.dtype, np.integer
                                ):
                                    # mmap is read-only — copy to
                                    # allow dtype cast.
                                    _time_vals = np.array(
                                        _time_vals
                                    ).astype("datetime64[ns]")

                            if _time_vals is None:
                                # Fallback: load time from zarr (expensive).
                                _time_var = ref_data.coords.get("time")
                                if _time_var is not None:
                                    if hasattr(_time_var, "compute"):
                                        _time_vals = _time_var.compute(
                                            scheduler="synchronous"
                                        ).values
                                    else:
                                        _time_vals = _time_var.values
                                    _time_vals = np.asarray(_time_vals)
                                    if np.issubdtype(
                                        _time_vals.dtype, np.integer
                                    ):
                                        _time_vals = _time_vals.astype(
                                            "datetime64[ns]"
                                        )
                                    elif not np.issubdtype(
                                        _time_vals.dtype, np.datetime64
                                    ):
                                        _time_vals = pd.to_datetime(
                                            _time_vals
                                        ).values

                            if _time_vals is not None:
                                # The shared zarr is always time-sorted
                                # (see preprocess_batch_obs_files), so
                                # use contiguous slice indexing — reads
                                # ONLY the relevant chunks instead of
                                # scanning all 107 M points.
                                _i0 = int(
                                    np.searchsorted(
                                        _time_vals, _t0, side="left",
                                    )
                                )
                                _i1 = int(
                                    np.searchsorted(
                                        _time_vals, _t1, side="right",
                                    )
                                )
                                ref_data = ref_data.isel(
                                    {
                                        n_points_dim: slice(_i0, _i1)
                                    }
                                )

                            ref_data = ref_data.compute(
                                scheduler="synchronous"
                            )

                            if ref_data.sizes.get(
                                n_points_dim, 0
                            ) == 0:
                                logger.debug(
                                    f"[{ref_alias}] valid_time={valid_time}: "
                                    f"0 observation points in the shared zarr "
                                    f"for time window {time_bounds[0]}–{time_bounds[1]}. "
                                    f"This may indicate missing source files in the "
                                    f"catalog for this date (incomplete download or "
                                    f"data gap from the provider) — result will be null."
                                )
                                ref_data = None

                            _used_prefetch = True
                            _zarr_last_exc = None
                            break  # success
                        except Exception as exc:
                            _zarr_last_exc = exc
                            ref_data = None
                            if _zarr_attempt < 2:
                                import time as _time_retry
                                _time_retry.sleep(
                                    1.0 * (1 + _zarr_attempt)
                                )
                                continue

                    if _zarr_last_exc is not None:
                        import traceback as _tb_zarr
                        logger.warning(
                            f"Cannot use shared obs zarr "
                            f"'{_shared_obs_zarr}' after 3 "
                            f"attempts: {_zarr_last_exc!r}\n"
                            f"{_tb_zarr.format_exc()}"
                        )

                if not _used_prefetch:
                    # ── Memory guard: cap files processed per worker ──────
                    # Without the shared Zarr, each worker opens ALL matching
                    # files independently.  For SWOT, a single time_tolerance
                    # window (24 h) typically matches ~25 swath files at
                    # ~15 MB each (≈375 MB), well within the 6 GB budget.
                    # Cap to 50 files to handle SWOT safely while still
                    # preventing runaway loads.
                    _MAX_FILES_PER_WORKER = int(
                        os.environ.get("DCTOOLS_MAX_OBS_FILES_PER_WORKER", "50")
                    )
                    _n_ref_files = len(ref_df)
                    logger.debug(
                        f"Worker fallback ({ref_alias}): processing "
                        f"{_n_ref_files} obs files for time window "
                        f"{time_bounds[0]}–{time_bounds[1]}"
                    )
                    if _n_ref_files > _MAX_FILES_PER_WORKER:
                        logger.warning(
                            f"Worker fallback ({ref_alias}): {_n_ref_files} files "
                            f"exceed per-worker limit {_MAX_FILES_PER_WORKER}. "
                            f"Truncating to avoid OOM.  Set "
                            f"DCTOOLS_MAX_OBS_FILES_PER_WORKER=0 to disable."
                        )
                        if _MAX_FILES_PER_WORKER > 0:
                            ref_df = ref_df.head(_MAX_FILES_PER_WORKER)
                    ref_raw_data = ObservationDataViewer(
                        ref_df,
                        open_ref_func, str(ref_alias or ""),
                        keep_vars, target_dimensions, metadata,
                        time_bounds,
                        n_points_dim=n_points_dim,
                        dataset_processor=None,
                    )
                    ref_data = ref_raw_data.preprocess_datasets(ref_df)
            else:
                # Non-observation references are passed as a single source
                # (typically a path string or datetime for CMEMS). Do not pass
                # `ref_alias` as a positional arg: BaseConnectionManager.open
                # expects (path, mode="rb").
                ref_data_from_cache = False
                ref_data_base = None
                if ref_protocol == "cmems":
                    with dask.config.set(scheduler="synchronous"):
                        ref_data = open_ref_func(ref_source)
                else:
                    if isinstance(ref_source, str):
                        ref_data_base, ref_data_from_cache = _open_dataset_worker_cached(
                            open_ref_func,
                            ref_source,
                        )
                        ref_data = ref_data_base
                    else:
                        ref_data = open_ref_func(ref_source)

                # ── Ensure reference is sliced by time (like prediction) ──
                # For large gridded reference datasets (GLORYS, DUACS), failing
                # to slice by time loads the entire multi-year dataset into memory
                # during .compute(), causing immediate worker kills.
                if ref_data is not None:
                    _time_dim = None
                    for _td in ("time", "TIME", "JULD"):
                        if _td in ref_data.coords or _td in ref_data.dims:
                            _time_dim = _td
                            break

                    if _time_dim:
                        try:
                            # Use nearest neighbor to match valid_time
                            _ref_sel = ref_data.sel({_time_dim: valid_time}, method="nearest")

                            # Preserve time dimension (needed for concatenation)
                            if _time_dim not in _ref_sel.dims:
                                ref_data = _ref_sel.expand_dims(_time_dim).assign_coords({_time_dim: [valid_time]})  # noqa: E501
                            else:
                                ref_data = _ref_sel.assign_coords({_time_dim: [valid_time]})
                        except Exception as _exc_slice:
                            logger.warning(
                                f"Could not slice reference {ref_alias} by time: {_exc_slice!r}"
                            )
        else:
            ref_data = None

        # Grid-to-Track Logic Handling
        run_grid_to_track = False
        from torchvision import transforms as output_transforms

        # Determine if we should optimize for Grid-to-Track
        if ref_is_observation and pred_transform is not None:
            # modified_transforms = []

            # Helper to inspect and filter transforms
            def inspect_transform(t):
                # If it's a Compose, recurse
                if isinstance(t, output_transforms.Compose):
                    sub_list: List[Any] = []
                    for sub_t in t.transforms:
                        res = inspect_transform(sub_t)
                        if res:
                            sub_list.append(res)
                    return output_transforms.Compose(sub_list) if sub_list else None

                # Check for interpolation transform
                name = getattr(t, "transform_name", "")
                if name == "glorys_to_glonet":
                    # This is the Grid-to-Grid interpolation we want to avoid!
                    return None
                return t

            new_transform_structure = inspect_transform(pred_transform)

            # If structure changed, it means we removed the interpolation
            # -> we must add Grid-to-Track
            if new_transform_structure != pred_transform:
                pred_transform = new_transform_structure
                run_grid_to_track = True


        if pred_transform is not None:
            pred_data = pred_transform(pred_data)
        if pred_data is None:
            logger.debug(
                f"[{ref_alias}] valid_time={valid_time}: no prediction data "
                f"for time window {time_bounds[0]}–{time_bounds[1]} "
                f"— skipping metric computation (result=null)."
            )
            return {
                "ref_alias": ref_alias,
                "result": None,
                "n_points": 0,
                "duration_s": 0.0,
                "preprocess_s": 0.0,
                "forecast_reference_time": forecast_reference_time,
                "lead_time": lead_time,
                "valid_time": valid_time,
            }
        # Compute once after transforms (safe & memory-friendlier).
        if hasattr(pred_data, 'chunks') and pred_data.chunks:
            # Use a hard Python-level timeout instead of bare .compute().
            # With scheduler='synchronous', aiobotocore's asyncio read_timeout
            # never fires (event loop blocked by the synchronous scheduler).
            # _compute_with_timeout() runs the compute in a daemon thread and
            # raises after DCTOOLS_S3_COMPUTE_TIMEOUT seconds (default 90 s).
            _s3_timeout = int(os.environ.get("DCTOOLS_S3_COMPUTE_TIMEOUT", "90"))
            pred_data = _compute_with_timeout(
                pred_data, timeout_s=_s3_timeout, scheduler="synchronous"
            )

        # Grid-to-Track interpolation is now handled internally by Class4 metrics
        # if run_grid_to_track and ref_data is not None:
        #    pass

        if ref_data is not None and ref_transform is not None:
            ref_data = ref_transform(ref_data)
            # Re-materialize if transform produced new lazy arrays
            if hasattr(ref_data, 'chunks') and ref_data.chunks:
                _s3_timeout = int(os.environ.get("DCTOOLS_S3_COMPUTE_TIMEOUT", "90"))
                ref_data = _compute_with_timeout(
                    ref_data, timeout_s=_s3_timeout, scheduler="synchronous"
                )

        if reduce_precision:
            # pred_data is already float32.
            if ref_data is not None:
                ref_data = to_float32(ref_data)

        # Force reloading if memory becomes an issue? No, trust Dask, but do explicit GC
        _clear_xarray_file_cache()
        gc.collect()

        _preprocess_time = time.perf_counter() - _t0_total
        t_start = time.perf_counter()

        results: Any = None
        _per_bins_data = None  # populated by the class4 observation branch

        if ref_is_observation:
            if ref_data is None:
                logger.debug(
                    f"[{ref_alias}] valid_time={valid_time}: no observation data "
                    f"for time window {time_bounds[0]}–{time_bounds[1]} "
                    f"— skipping metric computation (result=null)."
                )
                return {
                    "ref_alias": ref_alias,
                    "result": None,
                    "n_points": 0,
                    "duration_s": 0.0,
                    "preprocess_s": _preprocess_time,
                    "forecast_reference_time": forecast_reference_time,
                    "lead_time": lead_time,
                    "valid_time": valid_time,
                }

            n_points_raw = 0
            _npt_candidates = ('n_points', 'N_POINTS', 'obs')
            if isinstance(ref_data, list):
                # Count points (metadata only, fast)
                for ds in ref_data:
                    if hasattr(ds, 'sizes'):
                        for _npt in _npt_candidates:
                            if _npt in ds.sizes:
                                n_points_raw += ds.sizes[_npt]
                                break
            elif hasattr(ref_data, 'sizes'):
                for _npt in _npt_candidates:
                    if _npt in ref_data.sizes:
                        n_points_raw = ref_data.sizes[_npt]
                        break

            '''logger.debug(
                f"[{ref_alias}] valid_time={valid_time} — preprocessing done, "
                f"{n_points_raw} obs points — starting metric computation…"
            )'''

            _cap_worker_threads(1)

            with dask.config.set(scheduler='synchronous'):
                results = list_metrics[0].compute(
                    pred_data, ref_data,
                    pred_coords, ref_coords,
                )

            t_end = time.perf_counter()
            duration = t_end - t_start
            logger.debug(
                f"[{ref_alias}] valid_time={valid_time}: "
                f"metrics done in {duration:.1f}s ({n_points_raw} obs pts)"
            )

            # Class4Evaluator.run() now returns {"results": DataFrame, "per_bins": {...}}
            # instead of a bare DataFrame.  Unpack before converting to records.
            if isinstance(results, dict) and "results" in results:
                _per_bins_data = results.get("per_bins")
                results = results["results"]
            if isinstance(results, pd.DataFrame):
                results = results.to_dict('records')
        else:
            # results = {}
            results = {}
            # ── Re-cap thread pools before metric computation (grid path) ──
            _cap_worker_threads(1)

            # Context manager for the loop
            with dask.config.set(scheduler='synchronous'):
                for metric in list_metrics:
                    logger.debug(
                        f"[compute_metric] Computing metric={metric.metric_name} "
                        f"type(metric)={type(metric).__name__} "
                        f"is_class4={getattr(metric, 'is_class4', None)} "
                        f"for ref_alias={ref_alias}"
                    )
                    return_res = metric.compute(
                        pred_data, ref_data,
                        pred_coords, ref_coords,
                    )

                    # Unwrap {"results": ..., "per_bins": ...} wrapper(s).
                    # May be nested multiple times (oceanbench_metrics wraps,
                    # then _compute_spatial_per_bins wraps again).
                    while isinstance(return_res, dict) and "results" in return_res:
                        _metric_per_bins = return_res.get("per_bins") or {}
                        if _metric_per_bins:
                            if _per_bins_data is None:
                                _per_bins_data = {}
                            _per_bins_data.update(_metric_per_bins)
                        return_res = return_res["results"]

                    if return_res is None:
                        return {
                            "ref_alias": ref_alias,
                            "result": None,
                            "n_points": 0,
                            "duration_s": 0.0,
                            "preprocess_s": _preprocess_time,
                            "forecast_reference_time": forecast_reference_time,
                            "lead_time": lead_time,
                            "valid_time": valid_time,
                        }

                    # Convert return_res to res_dict {var_depth_label: scalar_value}
                    res_dict: Dict[Any, Any] = {}
                    _lead_day_label = None
                    if lead_time is not None:
                        try:
                            _lead_day_label = f"Lead day {int(lead_time) + 1}"
                        except Exception:
                            _lead_day_label = None

                    if isinstance(return_res, dict):
                        # Plain dict from metric — extract scalar per key
                        import numpy as _np
                        for k, v in return_res.items():
                            if isinstance(v, (int, float)):
                                res_dict[k] = v
                            elif isinstance(v, _np.ndarray):
                                res_dict[k] = float(v.flat[0]) if v.size > 0 else None
                            elif isinstance(v, (list, tuple)) and len(v) > 0:
                                # Pick lead-day index if possible
                                _idx = 0
                                if lead_time is not None:
                                    try:
                                        _idx = int(lead_time)
                                    except Exception:
                                        _idx = 0
                                if _idx < len(v):
                                    res_dict[k] = float(v[_idx])
                                else:
                                    res_dict[k] = float(v[0])
                            else:
                                res_dict[k] = v
                    elif hasattr(return_res, 'index') and len(return_res) > 0:
                        # DataFrame path (original logic)
                        for var_depth_label in return_res.index:
                            metric_values = return_res.loc[var_depth_label].to_dict()
                            if _lead_day_label and _lead_day_label in metric_values:
                                res_dict[var_depth_label] = metric_values[_lead_day_label]
                            elif 'Lead day 1' in metric_values:
                                res_dict[var_depth_label] = metric_values['Lead day 1']
                            else:
                                try:
                                    res_dict[var_depth_label] = next(iter(metric_values.values()))
                                except StopIteration:
                                    res_dict[var_depth_label] = None
                    elif len(return_res) == 0:
                        return {
                            "ref_alias": ref_alias,
                            "result": None,
                            "n_points": 0,
                            "duration_s": 0.0,
                            "preprocess_s": _preprocess_time,
                            "forecast_reference_time": forecast_reference_time,
                            "lead_time": lead_time,
                            "valid_time": valid_time,
                        }

                    results[metric.get_metric_name()] = res_dict

            # Convert from nested Format1 to Format2
            results = convert_format1_to_format2(results)
            t_end = time.perf_counter()
            duration = t_end - t_start
            # logger.debug(f"Fcast {forecast_reference_time} (LT {lead_time}): {duration:.2f}s")

            # Count grid points for reporting (was missing -> always showed 0)
            n_points_raw = 0
            if ref_data is not None and hasattr(ref_data, 'sizes'):
                _skip = {'time', 'forecast_reference_time', 'lead_time'}
                _grid_dims = [d for d in ref_data.sizes if d not in _skip]
                if _grid_dims:
                    n_points_raw = 1
                    for _gd in _grid_dims:
                        n_points_raw *= ref_data.sizes[_gd]

        res = {
            "ref_alias": ref_alias,
            "result": results,
            "duration_s": duration,
            "preprocess_s": _preprocess_time,
            "n_points": n_points_raw if 'n_points_raw' in locals() else 0
        }
        # Add forecast fields if present
        if forecast_reference_time is not None:
            res["forecast_reference_time"] = forecast_reference_time
        if lead_time is not None:
            res["lead_time"] = lead_time
        if valid_time is not None:
            res["valid_time"] = valid_time
        # Carry the observation flag so downstream writers can tag per_bins.
        res["ref_is_observation"] = bool(ref_is_observation)
        # Attach per_bins from the class4 observation branch (if present).
        if _per_bins_data is not None:
            res["per_bins"] = _per_bins_data

        return res

    except Exception as exc:  # noqa: E722
        logger.error(
            f"Error computing metrics for dataset {ref_alias} and date {forecast_reference_time}: "
            f"{repr(exc)}"
        )
        traceback.print_exc()
        return {
            "ref_alias": ref_alias,
            "result": None,
            "error": repr(exc),
        }

    finally:
        # NOTE: We intentionally do not restore the dask scheduler
        # here.  The driver sets scheduler="synchronous" once before
        # task dispatch.  Restoring "distributed" in the finally
        # block of one thread created a race condition that undid the
        # fix for other threads still running -> deadlock.

        # Aggressive memory release at end of task
        if 'pred_data' in locals():
            try:
                _skip_close = (
                    bool(locals().get("pred_data_from_cache"))
                    and ("pred_data_base" in locals())
                    and (pred_data is locals().get("pred_data_base"))
                )
                if not _skip_close and hasattr(pred_data, 'close'):
                    pred_data.close()  # type: ignore[union-attr]
                del pred_data
            except Exception:
                pass

        if 'ref_data' in locals():
            try:
                # Handle list of datasets (e.g. from observation data viewer)
                if isinstance(ref_data, list):
                    for ds in ref_data:
                        try:
                            if hasattr(ds, 'close'):
                                ds.close()
                        except Exception:
                            pass
                else:
                    _skip_close = (
                        bool(locals().get("ref_data_from_cache"))
                        and ("ref_data_base" in locals())
                        and (ref_data is locals().get("ref_data_base"))
                    )
                    if (not _skip_close) and hasattr(ref_data, 'close'):
                        ref_data.close()
                del ref_data
            except Exception:
                pass
        # pred_data_selected variable does not exist in the scope, removing it
        # if 'pred_data_selected' in locals():
        #     try:
        #         del pred_data_selected
        #     except Exception: pass

        worker_memory_cleanup()


class Evaluator:
    """Class to evaluate metrics on datasets."""

    def __init__(
        self,
        dataset_manager: MultiSourceDatasetManager,
        metrics: Dict[str, List[MetricComputer]],
        dataloader: EvaluationDataloader,
        ref_aliases: List[str],
        dataset_processor: DatasetProcessor,
        dask_cfgs_by_dataset: Optional[Dict[str, Dict[str, Any]]] = None,
        results_dir: Optional[str] = None,
        reduce_precision: bool = False,
        restart_workers_per_batch: bool = False,
        restart_frequency: int = 1,
        max_p_memory_increase: float = 0.2, # 20% increase default
        max_worker_memory_fraction: float = 0.85,
    ):
        """
        Initializes the evaluator.

        Args:
            dataset_manager (MultiSourceDatasetManager): Multi-source dataset manager.
            metrics (Dict[str, List[MetricComputer]]):
                Dictionary {ref_alias: [MetricComputer, ...]}.
            dataloader (EvaluationDataloader):
                Dataloader for evaluation.
            ref_aliases (List[str]): List of reference aliases.
            dataset_processor (DatasetProcessor): Dataset processor for distribution.
            dask_cfgs_by_dataset (Dict[str, Dict[str, Any]], optional):
                Per-dataset Dask configuration (n_workers, threads_per_worker,
                memory_limit) extracted from the YAML config sources.
                Defaults to None.
            results_dir (str, optional): Folder to save results. Defaults to None.
            reduce_precision (bool, optional): Reduce float precision (float32).
                Defaults to False.
            restart_workers_per_batch (bool, optional): Restart workers after each batch.
                Defaults to False.
            restart_frequency (int, optional): Frequency (nb of batches) cleanup/restart.
                Defaults to 1.
            max_p_memory_increase (float, optional): RAM increase threshold before
                restart. Defaults to 0.5 (50%).
            max_worker_memory_fraction (float, optional): Absolute threshold (fraction of
                Dask memory_limit) beyond which restart is triggered.
                Defaults to 0.85 (85%).
        """
        self.dataset_manager = dataset_manager
        self.dataset_processor = dataset_processor
        self.metrics = metrics
        self.dataloader = dataloader
        self.dask_cfgs_by_dataset = dask_cfgs_by_dataset or {}
        self.reduce_precision = reduce_precision
        self.restart_workers_per_batch = restart_workers_per_batch
        self.restart_frequency = restart_frequency
        self.max_p_memory_increase = max_p_memory_increase
        self.max_worker_memory_fraction = max_worker_memory_fraction
        # self.results = []
        self.ref_aliases = ref_aliases
        self.results_dir = results_dir
        # Track the current cluster sizing so we know when to reconfigure.
        self._current_cluster_ref: Optional[str] = None

        (
            self.ref_managers,
            self.ref_catalogs,
            self.ref_connection_params,
        ) = dataset_manager.get_config()

    # ------------------------------------------------------------------
    # Cluster reconfiguration when switching observation datasets
    # ------------------------------------------------------------------
    def _reconfigure_cluster_for_ref(self, ref_alias: str) -> None:
        """Resize the Dask cluster if *ref_alias* needs a different config.

        Each observation dataset can declare its own
        ``n_parallel_workers / nthreads_per_worker / memory_limit_per_worker``
        in the YAML config.  When the evaluator switches from one obs
        dataset to another, this method tears down the current
        ``DatasetProcessor`` and spins up a new one matching the target
        config.  If no per-dataset config exists for *ref_alias*, or the
        existing cluster already matches, this is a no-op.
        """
        if self._current_cluster_ref == ref_alias:
            return  # already configured for this dataset

        desired = self.dask_cfgs_by_dataset.get(ref_alias)
        if not desired:
            # No per-dataset override -> keep current cluster as-is.
            self._current_cluster_ref = ref_alias
            return

        # Read desired sizing.
        d_workers = int(desired.get("n_workers", 1))
        d_threads = int(desired.get("threads_per_worker", 1))
        d_memory = desired.get("memory_limit", "4GB")

        # Read current cluster sizing for comparison.
        _client = getattr(self.dataset_processor, "client", None)
        if _client is not None:
            try:
                _info = _client.scheduler_info()
                _ws = _info.get("workers", {})
                _cur_n = len(_ws)
                # threads & memory from first worker
                _any_w: dict = next(iter(_ws.values()), {})
                _cur_threads = _any_w.get("nthreads", 1)
                _cur_mem = _any_w.get("memory_limit", 0)
                # Parse desired memory into bytes for comparison.
                _d_mem_bytes = _parse_memory_limit(d_memory)
                if (
                    _cur_n == d_workers
                    and _cur_threads == d_threads
                    and _cur_mem == _d_mem_bytes
                ):
                    # Already matches -> nothing to do.
                    self._current_cluster_ref = ref_alias
                    return
            except Exception:
                pass  # cannot query -> proceed with reconfiguration

        logger.info(
            f"Reconfiguring Dask cluster for '{ref_alias}': "
            f"Workers={d_workers}, Threads={d_threads}, MemLimit={d_memory}"
        )

        # Tear down existing cluster.
        # Silence distributed.worker during teardown: workers that still have
        # an in-flight heartbeat will log a CommClosedError when the scheduler
        # stream is closed.  This is expected and non-fatal.
        # We suppress the logger *and* wait a short time after close() so that
        # any in-flight async heartbeats complete while the logger is still
        # silenced.
        import logging as _logging
        import time as _time
        _dist_logger = _logging.getLogger("distributed")
        _dist_worker_logger = _logging.getLogger("distributed.worker")
        _dist_level = _dist_logger.level
        _dist_worker_level = _dist_worker_logger.level
        _dist_logger.setLevel(_logging.CRITICAL)
        _dist_worker_logger.setLevel(_logging.CRITICAL)
        try:
            self.dataset_processor.close()
            # Give in-flight heartbeat RPCs time to fail silently.
            _time.sleep(1.0)
        except Exception:
            pass
        finally:
            _dist_logger.setLevel(_dist_level)
            _dist_worker_logger.setLevel(_dist_worker_level)

        # Create a fresh DatasetProcessor.
        self.dataset_processor = DatasetProcessor(
            distributed=True,
            n_workers=d_workers,
            threads_per_worker=d_threads,
            memory_limit=d_memory,
        )

        # Propagate HDF5/NetCDF env vars to new workers.
        from dctools.utilities.init_dask import configure_dask_workers_env
        try:
            configure_dask_workers_env(self.dataset_processor.client)
        except Exception:
            pass

        self._current_cluster_ref = ref_alias
        # Reset baseline memory after cluster rebuild.
        self.baseline_memory = None
        logger.debug(
            f"Dask cluster reconfigured for '{ref_alias}': "
            f"dashboard={getattr(self.dataset_processor.client, 'dashboard_link', 'N/A')}"
        )

    def log_cluster_memory_usage(self, batch_idx: int):
        """Log memory usage of each Dask worker."""
        if not hasattr(self.dataset_processor, "client") or self.dataset_processor.client is None:
            return

        try:
            info = self.dataset_processor.client.scheduler_info()
            workers = info.get('workers', {})

            logger.debug(f"=== Memory Usage Start Batch {batch_idx} ===")
            for w_addr, w_info in workers.items():
                # Some versions of dask put 'metrics' in the info
                mem_used = w_info.get('metrics', {}).get('memory', w_info.get('memory', 0))
                mem_limit = w_info.get('memory_limit', 0)

                if mem_limit > 0:
                    percent = (mem_used / mem_limit) * 100
                    logger.debug(
                        f"Worker {w_info.get('name', w_addr)}: "
                        f"{percent:.1f}% ({mem_used / 1024**3:.2f}GB / {mem_limit / 1024**3:.2f}GB)"
                    )
                else:
                    logger.debug(
                        f"Worker {w_info.get('name', w_addr)}: "
                        f"{mem_used / 1024**3:.2f}GB used (no limit)"
                    )

        except Exception as e:
            logger.warning(f"Could not log cluster memory usage: {e}")

    def get_max_memory_usage(self) -> float:
        """Get the maximum memory usage across all workers (in bytes)."""
        if not hasattr(self.dataset_processor, "client") or self.dataset_processor.client is None:
            return 0.0

        try:
            info = self.dataset_processor.client.scheduler_info()
            workers = info.get('workers', {})
            max_mem = 0.0
            for w_info in workers.values():
                mem_used = w_info.get('metrics', {}).get('memory', w_info.get('memory', 0))
                if mem_used > max_mem:
                    max_mem = mem_used
            return max_mem
        except Exception:
            return 0.0

    def get_max_memory_fraction(self) -> float:
        """Get max(memory_used / memory_limit) across workers.

        Returns:
            float: Fraction in [0, +inf). Returns 0.0 if unavailable.
        """
        if not hasattr(self.dataset_processor, "client") or self.dataset_processor.client is None:
            return 0.0

        try:
            info = self.dataset_processor.client.scheduler_info()
            workers = info.get("workers", {})
            max_frac = 0.0
            for w_info in workers.values():
                mem_used = w_info.get("metrics", {}).get("memory", w_info.get("memory", 0))
                mem_limit = w_info.get("memory_limit", 0) or 0
                if mem_limit and mem_limit > 0:
                    max_frac = max(max_frac, float(mem_used) / float(mem_limit))
            return max_frac
        except Exception:
            return 0.0

    def evaluate(self) -> List[Dict[str, Any]]:
        """
        Evaluates metrics on dataloader data for each reference.

        Returns:
            List[Dict[str, Any]]: Metric results for each batch and each reference.
        """
        self.scattered_argo_indexes: Dict[str, Any] = {}
        self.scattered_ref_catalogs: Dict[str, Any] = {}

        # Baseline memory usage (will be set at first batch or after restart)
        self.baseline_memory = None

        # ── ARGO pre-fetch cache dir ──────────────────────────────────────
        # Zarr files created by prefetch_batch_shared_zarr persist across
        # batches (a window fetched for batch N is reused by batch N+1 if
        # the same time window appears again).  The cache is stored under
        # data_directory (parent of results_dir) so it persists across runs
        # and avoids re-downloading the same ARGO months every evaluation.
        from pathlib import Path as _PfPath
        _results_path = _PfPath(self.results_dir) if self.results_dir else _PfPath("/tmp")
        # results_dir == data_directory/results_batches  →  parent == data_directory
        _data_dir = _results_path.parent if _results_path.name == "results_batches" else _results_path  # noqa: E501
        self._argo_zarr_cache_dir = str(_data_dir / "argo_batch_cache")

        # ── Purge stale obs batch zarr from any previous run ─────────────
        # The shared obs zarr (obs_batch_shared/{alias}/batch_shared.zarr) is
        # written to a fixed path that persists across runs.  If a previous
        # run was aborted after only 1-2 batches, the zarr covers only a
        # fraction of the year.  Subsequent runs would silently reuse it,
        # causing all tasks outside that time range to return 0 pts with no
        # error.  Deleting the directory at the start of each run forces a
        # clean rebuild from the full set of observation files.
        _obs_shared_root = _results_path / "obs_batch_shared"
        if _obs_shared_root.exists():
            import shutil as _shutil_obs
            try:
                _shutil_obs.rmtree(_obs_shared_root, ignore_errors=True)
                logger.debug(f"Purged stale obs_batch_shared cache: {_obs_shared_root}")
            except Exception:
                pass

        try:
            # ── Pre-materialise batches to know total count ───────────
            # Batches are lightweight metadata dicts (no actual data),
            # so materialising upfront is cheap and lets us display
            # clear "Batch X/N" progress throughout the run.
            self._lookahead_cache = getattr(self, '_lookahead_cache', {})

            _all_batches: List[List[Dict[str, Any]]] = list(self.dataloader)
            _total_batches = len(_all_batches)
            logger.debug(
                f"Evaluation plan: {_total_batches} batch(es), "
                f"{sum(len(b) for b in _all_batches)} total tasks"
            )

            _prev_ref_alias: Optional[str] = None
            _ref_aliases_ordered: List[str] = list(
                dict.fromkeys(
                    b[0].get("ref_alias") for b in _all_batches if b and b[0].get("ref_alias")  # type: ignore[misc]
                )
            )

            for batch_idx, batch in enumerate(_all_batches):
                _next_raw = _all_batches[batch_idx + 1] if batch_idx + 1 < _total_batches else None

                pred_alias = self.dataloader.pred_alias
                ref_alias = batch[0].get("ref_alias")

                # Print the reference banner the first time a new reference is encountered.
                if ref_alias != _prev_ref_alias:
                    _n_ref_total = len(_ref_aliases_ordered)
                    _n_ref_current = (
                        _ref_aliases_ordered.index(ref_alias) + 1
                        if ref_alias in _ref_aliases_ordered
                        else "?"
                    )
                    _sep_ref = "─" * 60
                    print(f"    ┌{_sep_ref}┐")
                    print(
                        f"    │  ◆  Reference dataset ({_n_ref_current}/{_n_ref_total}) :  {str(ref_alias).upper():<28}│"  # noqa: E501
                    )
                    print(f"    └{_sep_ref}┘")
                    _prev_ref_alias = ref_alias

                # ── Reconfigure cluster if this ref dataset needs
                #    different sizing (workers / threads / memory) ──
                self._reconfigure_cluster_for_ref(ref_alias)  # type: ignore[arg-type]

                # Extract necessary information
                pred_connection_params = self.dataloader.pred_connection_params
                ref_connection_params = self.dataloader.ref_connection_params[ref_alias]  # type: ignore[index]
                pred_transform = self.dataloader.pred_transform
                if self.dataloader.ref_transforms is not None:
                    ref_transform = self.dataloader.ref_transforms[ref_alias]  # type: ignore[index]

                argo_index = None
                if hasattr(self.dataloader.ref_managers[ref_alias], 'argo_index'):  # type: ignore[index]
                    argo_index = self.dataloader.ref_managers[ref_alias].get_argo_index()  # type: ignore[index]

                # Build look-ahead context for the NEXT batch (if any).
                # _evaluate_batch will launch the background download during
                # its as_completed loop (workers busy -> driver has spare CPU).
                _la_next = None
                if _next_raw is not None:
                    _la_next = {
                        'batch': _next_raw,
                        'ref_alias': _next_raw[0].get("ref_alias") if _next_raw else None,
                    }

                batch_results = self._evaluate_batch(
                    batch, pred_alias, ref_alias,  # type: ignore[arg-type]
                    pred_connection_params, ref_connection_params,
                    pred_transform, ref_transform,
                    argo_index=argo_index,
                    _lookahead_next=_la_next,
                    _batch_idx=batch_idx,
                    _total_batches=_total_batches,
                )
                if batch_results is None:
                    continue
                serial_results = [
                    serialize_structure(res)
                    for res in batch_results
                    if res is not None
                ]

                # Save batch by batch
                batch_file = os.path.join(
                    self.results_dir or ".", f"results_{pred_alias}_batch_{batch_idx}.json"
                )
                with open(batch_file, "w") as f:
                    json.dump(serial_results, f, indent=2, ensure_ascii=False)

                # CRITICAL: Explicit cleanup
                del batch_results
                del serial_results
                gc.collect()

                # ── Memory-triggered worker restart ───────────────────────
                # Two criteria (either one triggers a restart):
                #   1. Relative: current max fraction increased by more than
                #      max_p_memory_increase compared to the post-batch-0
                #      baseline (accumulation over batches).
                #   2. Absolute: any worker exceeds max_worker_memory_fraction
                #      of its Dask memory_limit.
                if self.restart_workers_per_batch:
                    _client = getattr(self.dataset_processor, "client", None)
                    if _client is not None:
                        _cur_frac = self.get_max_memory_fraction()

                        # Initialise baseline on the very first batch.
                        if self.baseline_memory is None:
                            self.baseline_memory = _cur_frac
                            logger.debug(
                                f"Memory baseline set after batch {batch_idx}: "
                                f"{_cur_frac:.2%}"
                            )
                        else:
                            _rel_increase = (
                                (_cur_frac - self.baseline_memory)
                                / max(self.baseline_memory, 1e-6)
                            )
                            _abs_exceeded = (
                                _cur_frac
                                >= float(self.max_worker_memory_fraction or 1.0)
                            )
                            _rel_exceeded = (
                                _rel_increase
                                >= float(self.max_p_memory_increase or 1.0)
                            )

                            if _abs_exceeded or _rel_exceeded:
                                _reason = (
                                    f"absolute {_cur_frac:.2%} >= "
                                    f"{self.max_worker_memory_fraction:.2%}"
                                    if _abs_exceeded
                                    else f"relative +{_rel_increase:.0%} >= "
                                    f"+{self.max_p_memory_increase:.0%} "
                                    f"(baseline={self.baseline_memory:.2%})"
                                )
                                logger.info(
                                    f"Restarting Dask workers after batch "
                                    f"{batch_idx} ({_reason})"
                                )
                                import logging as _log_restart
                                _dist_lvl = _log_restart.getLogger(
                                    "distributed"
                                ).level
                                _log_restart.getLogger(
                                    "distributed"
                                ).setLevel(_log_restart.CRITICAL)
                                try:
                                    _client.restart()
                                except Exception as _exc_restart:
                                    logger.warning(
                                        f"Worker restart failed: {_exc_restart!r}"
                                    )
                                finally:
                                    _log_restart.getLogger(
                                        "distributed"
                                    ).setLevel(_dist_lvl)
                                # Reset baseline after the cluster is clean.
                                self.baseline_memory = None
                                gc.collect()

            # Cleanup scattered data
            self.scattered_argo_indexes.clear()
            self.scattered_ref_catalogs.clear()

        except Exception as exc:
            logger.error(f"Evaluation failed: {repr(exc)}")
            raise

        finally:
            # ── ARGO pre-fetch Zarr cache: kept on disk for reuse ─────────
            # The cache lives under data_directory/argo_batch_cache and is
            # intentionally NOT deleted so subsequent runs can reuse already-
            # downloaded months.  Each month's zarr file has a deterministic
            # name (argo_full_month_YYYY-MM.zarr) so there is no staleness risk.
            _cache = getattr(self, "_argo_zarr_cache_dir", None)
            if _cache:
                logger.debug(f"ARGO Zarr cache preserved for future runs: {_cache}")

        return []

    def clean_namespace(self, namespace: Namespace) -> Namespace:
        """Clean namespace by removing unpicklable objects."""
        ns = Namespace(**vars(namespace))
        # Removes unpicklable attributes
        for key in ['dask_cluster', 'fs', 'dataset_processor', 'client', 'session']:
            if hasattr(ns, key):
                delattr(ns, key)
        # Also cleans objects in ns.params if present
        if hasattr(ns, "params"):
            for key in ['fs', 'client', 'session', 'dataset_processor']:
                if hasattr(ns.params, key):
                    delattr(ns.params, key)
        return ns

    def _evaluate_batch(
        self, batch: List[Dict[str, Any]],
        pred_alias: str, ref_alias: str,
        pred_connection_params: Dict[str, Any], ref_connection_params: Dict[str, Any],
        pred_transform: Any, ref_transform: Any,
        argo_index: Optional[Any] = None,
        _lookahead_next: Optional[Dict[str, Any]] = None,
        _batch_idx: int = 0,
        _total_batches: int = 1,
    ) -> List[Dict[str, Any]]:
        _phase_t0 = time.time()
        if batch:
            raw_dates = [e.get("forecast_reference_time") for e in batch]
            dates = [d for d in raw_dates if d is not None]
            if dates:
                logger.debug(f"Process batch forecasts: {min(dates)} to {max(dates)}")

        ref_alias = batch[0].get("ref_alias") or ref_alias

        pred_connection_params = deep_copy_object(
            pred_connection_params, skip_list=['dataset_processor', 'fs']
        )
        pred_connection_params = clean_for_serialization(pred_connection_params)
        pred_connection_params = self.clean_namespace(pred_connection_params)

        if hasattr(pred_transform, 'dataset_processor'):
            delattr(pred_transform, 'dataset_processor')
        if hasattr(ref_transform, 'dataset_processor'):
            delattr(ref_transform, 'dataset_processor')

        ref_connection_params = deep_copy_object(
            ref_connection_params, skip_list=['dataset_processor', 'fs']
        )
        ref_connection_params = clean_for_serialization(ref_connection_params)
        ref_connection_params = self.clean_namespace(ref_connection_params)

        # argo_index is now passed as a Future (already scattered) or None.
        # No need to scatter it again per batch.
        scattered_argo_index = argo_index

        metric_list = self.metrics.get(ref_alias)
        if not metric_list:
            err = (
                f"No metric configuration found for reference alias '{ref_alias}' "
                f"(available: {list(self.metrics.keys())})."
            )
            logger.error(err)
            return [{
                "forecast_reference_time": batch[0].get("forecast_reference_time") if batch else None,  # noqa: E501
                "model": pred_alias,
                "reference": ref_alias,
                "result": None,
                "n_points": 0,
                "duration_s": 0.0,
                "error": err,
            }]

        try:
            # Use map_tasks for direct task submission (no delayed graph overhead)
            from functools import partial
            fn = partial(
                compute_metric,
                pred_source_config=pred_connection_params,
                ref_source_config=ref_connection_params,
                model=pred_alias,
                list_metrics=metric_list,
                pred_transform=pred_transform,
                ref_transform=ref_transform,
                argo_index=scattered_argo_index,
                reduce_precision=self.reduce_precision,
                results_dir=self.results_dir,
            )
            fn.__name__ = "compute_metric"  # type: ignore[attr-defined]  # prevent full repr in tqdm progress bar

            batch_t0 = time.time()
            num_tasks = len(batch)

            # ── Throttle observation batches to prevent CPU oversubscription ──
            # Observation datasets (satellite) trigger heavy
            # CPU-bound interpolation (pyinterp) on each worker.  Submitting
            # all tasks at once lets Dask schedule them across all workers
            # simultaneously, each of which may spawn internal C++ threads
            # -> total thread count far exceeds physical cores -> 100 % CPU
            # thrashing.  We split large observation batches into smaller
            # sub-batches so that at most *max_concurrent_obs* tasks run in
            # parallel, leaving headroom for the OS and driver.
            is_obs_batch = batch and batch[0].get("ref_is_observation", False)
            # All tasks run with _cap_worker_threads(1) so internal C++
            # libraries (pyinterp, BLAS, Blosc) create only 1 thread each.
            # No need to throttle concurrency below n_workers.
            # max_concurrent_obs = num_tasks

            # with threads_per_worker set to 1, C libraries are capped to 1.
            # Dask limits concurrency to n_workers × threads_per_worker.
            # However, submitting all tasks upfront can overwhelm the scheduler
            # task queue or cause memory fragmentation if tasks are large.
            # Limit concurrent tasks to avoid scheduler overhead/pauses.
            _client = self.dataset_processor.client
            _ncores = _client.ncores()  # {worker_addr: nthreads}
            _N = len(_ncores)
            _total_slots = int(sum(_ncores.values())) if _ncores else _N
            # Cap in-flight futures to the actual execution capacity.
            # This keeps a small queue (Dask will queue the rest anyway),
            # avoids scheduler overload, and keeps behaviour consistent
            # with dc2.yaml (n_parallel_workers × nthreads_per_worker).
            # For observation batches without a shared Zarr (i.e. when the
            # driver-side preprocessing was skipped because the file count
            # exceeded the limit), each task independently processes all
            # matching files on its worker — extremely memory-intensive.
            # In that case, limit concurrency to n_workers (not total_slots)
            # so that at most one heavy task runs per physical worker,
            # preventing concurrent threads from doubling memory pressure.
            _has_shared_zarr = (
                is_obs_batch
                and batch
                and isinstance(batch[0].get("ref_data"), dict)
                and batch[0]["ref_data"].get("prefetched_obs_zarr_path") is not None
            )
            if is_obs_batch and not _has_shared_zarr:
                # Heavy fallback path: cap to n_workers (1 task/worker).
                max_concurrent_obs = max(_N, 2)
                logger.debug(
                    f"{ref_alias}: no shared obs Zarr — throttling to "
                    f"{max_concurrent_obs} concurrent tasks to limit "
                    f"per-worker memory pressure"
                )
            elif is_obs_batch:
                # With shared zarr, each task still .compute()s its
                # slice + prediction data.  With 2 threads/worker,
                # concurrent tasks on the same worker double memory.
                # Cap to n_workers so at most 1 obs task per worker.
                max_concurrent_obs = max(_N, 2)
            else:
                max_concurrent_obs = max(_total_slots, 4)

            # ── Single clean progress bar ─────────────────────────────────
            # One overall bar on the driver. No per-worker bars, no
            # worker-side tqdm, no monkey-patched metrics bar.  Each
            # completed task prints a one-line summary via tqdm.write()
            # which is designed to coexist with the progress bar.
            import sys as _sys_bars

            # Clean duplicate definition
            # _client = self.dataset_processor.client
            # _N = len(_client.ncores())

            # ── ARGO pipeline: shared batch Zarr prefetch ─────────────
            # Instead of per-window download (N separate HTTP sessions,
            # heavy profile overlap, N separate Zarr writes), we now:
            #   1. Merge ALL time windows into one global bounding interval
            #   2. Download ALL profiles in a single pass (one HTTP session
            #      with connection pooling -> connection reuse)
            #   3. Write ONE shared time-sorted Zarr for the entire batch
            #   4. Workers read the shared Zarr + filter by their specific
            #      time_bounds via searchsorted (contiguous chunk reads)
            #
            # Typical savings for a 10-entry batch with time_tolerance=12h:
            #   Downloads:   10 × ~1 day -> 1 × ~11 days (overlap removed)
            #   Zarr writes: 10 -> 1
            #   Worker I/O:  each reads only its slice (searchsorted)
            _shared_argo_zarr: Optional[str] = None
            _argo_pipeline = (
                is_obs_batch
                and ref_alias == "argo_profiles"
                and hasattr(
                    self.dataloader.ref_managers.get(ref_alias, None),
                    "prefetch_batch_shared_zarr",
                )
            )

            if _argo_pipeline:
                # --- Collect all time windows from the batch ---------------
                _all_time_bounds: List[tuple] = []
                for _entry in batch:
                    _ref_d = _entry.get("ref_data")
                    if isinstance(_ref_d, dict) and "time_bounds" in _ref_d:
                        _tb = _ref_d["time_bounds"]
                        _all_time_bounds.append(
                            (pd.Timestamp(_tb[0]), pd.Timestamp(_tb[1]))
                        )

                if _all_time_bounds:
                    _mgr = self.dataloader.ref_managers.get(ref_alias)
                    if _mgr is not None:
                        from pathlib import Path as _PfPath
                        _cache_dir = _PfPath(
                            getattr(
                                self,
                                "_argo_zarr_cache_dir",
                                str(
                                    _PfPath(
                                        getattr(self, "results_dir", None)
                                        or "/tmp"
                                    ) / "argo_batch_cache"
                                ),
                            )
                        )
                        # Prefer partitioned monthly prefetch when available.
                        _partitions = None
                        if hasattr(_mgr, "prefetch_batch_shared_zarr_partitioned"):
                            try:
                                _partitions = _mgr.prefetch_batch_shared_zarr_partitioned(
                                    time_bounds_list=_all_time_bounds,
                                    cache_dir=_cache_dir,
                                )
                            except Exception as exc:
                                logger.warning(
                                    f"ARGO partitioned shared prefetch failed: {exc!r} — falling back"  # noqa: E501
                                )

                        if _partitions:
                            # Inject per-entry path(s). Each entry gets either
                            # a single path (same-month) or a list of paths
                            # (month boundary).
                            for _entry in batch:
                                _ref_d = _entry.get("ref_data")
                                if not isinstance(_ref_d, dict) or "time_bounds" not in _ref_d:
                                    continue
                                _tb = _ref_d["time_bounds"]
                                _t0_e = pd.Timestamp(_tb[0])
                                _t1_e = pd.Timestamp(_tb[1])
                                if _t1_e < _t0_e:
                                    _t0_e, _t1_e = _t1_e, _t0_e
                                _paths = []
                                for _p in _partitions:
                                    try:
                                        _p0 = pd.Timestamp(_p.get("t0"))
                                        _p1 = pd.Timestamp(_p.get("t1"))
                                        _pp = _p.get("zarr_path")
                                    except Exception:
                                        continue
                                    if not _pp:
                                        continue
                                    if not (_p1 < _t0_e or _p0 > _t1_e):
                                        _paths.append((_p0, str(_pp)))
                                _paths_sorted = [p for _, p in sorted(_paths, key=lambda x: x[0])]
                                if len(_paths_sorted) == 1:
                                    _ref_d["prefetched_argo_shared_zarr"] = _paths_sorted[0]
                                elif len(_paths_sorted) > 1:
                                    _ref_d["prefetched_argo_shared_zarr"] = _paths_sorted
                        else:
                            _shared_argo_zarr = _mgr.prefetch_batch_shared_zarr(
                                time_bounds_list=_all_time_bounds,
                                cache_dir=_cache_dir,
                            )

                            # --- Inject shared Zarr path into every batch entry --------
                            if _shared_argo_zarr:
                                for _entry in batch:
                                    _ref_d = _entry.get("ref_data")
                                    if isinstance(_ref_d, dict):
                                        _ref_d["prefetched_argo_shared_zarr"] = _shared_argo_zarr

            # ── Use look-ahead prefetched data if available ────────────
            _la_cache = getattr(self, '_lookahead_cache', {})
            _la_key = id(batch)  # use batch object identity
            _la_data = _la_cache.pop(_la_key, None)
            if _la_data:
                logger.debug("Look-ahead: using pre-downloaded data from previous batch")
            #   • A single visible progress bar for the download phase
            #   • No concurrent S3 requests from multiple workers
            #   • Workers open local files -> fast, no bandwidth contention
            _obs_path_map: Dict[str, str] = _la_data.get('obs_map', {}) if _la_data else {}
            _obs_prefetch = (
                is_obs_batch
                and ref_alias != "argo_profiles"
                and not _argo_pipeline
            )
            _t_obs_dl = time.time()
            if _obs_prefetch:
                if _obs_path_map:
                    logger.debug("Obs download: using look-ahead cache")
                    # Inject mapping into batch entries
                    for _entry in batch:
                        _ref_d = _entry.get("ref_data")
                        if isinstance(_ref_d, dict):
                            _ref_d["prefetched_local_paths"] = _obs_path_map
                else:
                    _ref_mgr = self.dataloader.ref_managers.get(ref_alias)
                    _has_fs = (
                        _ref_mgr is not None
                        and hasattr(_ref_mgr, "params")
                        and hasattr(_ref_mgr.params, "fs")
                        and _ref_mgr.params.fs is not None
                    )
                    if _has_fs:
                        # Collect all unique remote paths across the batch
                        _all_remote_paths: List[str] = []
                        for _entry in batch:
                            _ref_d = _entry.get("ref_data")
                            if isinstance(_ref_d, dict) and "source" in _ref_d:
                                _cat = _ref_d["source"]
                                _tb = _ref_d.get("time_bounds")
                                if _tb is not None and hasattr(_cat, "get_dataframe"):
                                    _cat_df = _cat.get_dataframe()
                                    _filt = filter_by_time(
                                        _cat_df,
                                        pd.Timestamp(_tb[0]),
                                        pd.Timestamp(_tb[1]),
                                    )
                                    # Cost proxy for scheduling: how many catalog rows
                                    # (usually files) overlap this observation window.
                                    # Large windows tend to produce stragglers.
                                    try:
                                        _entry["_obs_cost"] = int(len(_filt))
                                    except Exception:
                                        _entry["_obs_cost"] = 0
                                    _paths = _filt["path"].tolist()
                                    _all_remote_paths.extend(_paths)  # type: ignore[arg-type]
                        # De-duplicate
                        _unique_remote = list(dict.fromkeys(_all_remote_paths))
                        if _unique_remote:
                            from pathlib import Path as _PfPath
                            from dctools.data.connection.connection_manager import (
                                prefetch_obs_files_to_local,
                            )
                            _obs_cache_dir = str(
                                _PfPath(
                                    getattr(self, "results_dir", None) or "/tmp"
                                ) / "obs_prefetch_cache" / str(ref_alias)
                            )
                            logger.debug(
                                f"Observation prefetch ({ref_alias}): "
                                f"{len(_unique_remote)} files to download/verify"
                            )
                            _obs_path_map = prefetch_obs_files_to_local(
                                remote_paths=_unique_remote,
                                cache_dir=_obs_cache_dir,
                                fs=_ref_mgr.params.fs,  # type: ignore[union-attr]
                                ref_alias=str(ref_alias),
                            )
                            # Inject mapping into every batch entry
                            for _entry in batch:
                                _ref_d = _entry.get("ref_data")
                                if isinstance(_ref_d, dict):
                                    _ref_d["prefetched_local_paths"] = _obs_path_map
            _t_obs_dl = time.time() - _t_obs_dl

            # ── Prediction data prefetch — define + launch in background ──
            # Runs concurrently with obs preprocessing below (different
            # S3 endpoints -> no contention).  Joined before task dispatch.
            import threading as _pred_thr

            # ── Reference grid data prefetch (Wasabi/S3 Zarr) ─────────────
            # For non-observation gridded references stored as Zarr on S3/Wasabi
            # workers can deadlock/hang due
            # to many concurrent small S3 requests. We prefetch the required Zarr
            # stores to local disk on the driver and remap paths in the batch so
            # workers open local files only.
            _ref_prefetched = False
            _ref_result: Dict[str, str] = {}

            def _do_ref_prefetch():
                nonlocal _ref_prefetched
                if not batch:
                    return
                if is_obs_batch:
                    return

                if os.environ.get("DCTOOLS_ENABLE_REF_PREFETCH", "1") not in ("1", "true", "True"):
                    return

                _ref_protocol = getattr(ref_connection_params, "protocol", None)
                if _ref_protocol not in ("wasabi", "s3"):
                    return

                _sample_ref = batch[0].get("ref_data")
                _ref_is_remote = (
                    isinstance(_sample_ref, str)
                    and _sample_ref.endswith(".zarr")
                    and _sample_ref.startswith(("https://", "http://", "s3://"))
                )
                if not _ref_is_remote:
                    return

                _ref_mgr = self.dataloader.ref_managers.get(ref_alias)
                _ref_fs_params = getattr(_ref_mgr, "params", None) if _ref_mgr is not None else None
                _ref_s3fs = getattr(_ref_fs_params, "fs", None)
                if _ref_s3fs is None:
                    return

                from pathlib import Path as _PfRef
                import shutil as _sh_ref
                from concurrent.futures import ThreadPoolExecutor as _RefPool
                import threading as _ref_dl_threading

                _ref_cache_dir = str(
                    _PfRef(getattr(self, "results_dir", None) or "/tmp")
                    / "ref_prefetch_cache"
                    / str(ref_alias)
                )
                os.makedirs(_ref_cache_dir, exist_ok=True)

                _ref_endpoint = getattr(_ref_fs_params, "endpoint_url", "") or ""
                _unique_ref_paths = list(
                    dict.fromkeys(
                        e["ref_data"]
                        for e in batch
                        if isinstance(e.get("ref_data"), str)
                        and str(e.get("ref_data")).endswith(".zarr")
                    )
                )
                if not _unique_ref_paths:
                    return

                _counters = {"dl": 0, "hit": 0}
                _ref_lock = _ref_dl_threading.Lock()

                def _dl_one_ref(_rp: str) -> None:
                    _fname = _PfRef(_rp).name
                    _local_zarr = os.path.join(_ref_cache_dir, _fname)
                    if os.path.isdir(_local_zarr) and os.listdir(_local_zarr):
                        with _ref_lock:
                            _ref_result[_rp] = _local_zarr
                            _counters["hit"] += 1
                        return
                    try:
                        # logger.debug(f"Prefetching reference: {_fname}")
                        _tid = _ref_dl_threading.current_thread().ident
                        _tmp_zarr = _local_zarr + f".downloading.{_tid}"
                        if os.path.isdir(_tmp_zarr):
                            _sh_ref.rmtree(_tmp_zarr, ignore_errors=True)

                        _s3_key = _rp
                        if _ref_endpoint and _s3_key.startswith(_ref_endpoint):
                            _s3_key = _s3_key[len(_ref_endpoint):].lstrip("/")
                        elif _s3_key.startswith("s3://"):
                            _s3_key = _s3_key[len("s3://"):]

                        _ref_s3fs.get(_s3_key, _tmp_zarr, recursive=True)

                        if os.path.isdir(_local_zarr):
                            _sh_ref.rmtree(_local_zarr, ignore_errors=True)
                        os.rename(_tmp_zarr, _local_zarr)
                        with _ref_lock:
                            _ref_result[_rp] = _local_zarr
                            _counters["dl"] += 1
                    except Exception as _exc_rf:
                        logger.warning(
                            f"Reference prefetch failed for {_fname}: {_exc_rf!r}"
                        )

                # Be conservative: reference stores can be large.
                _N_REF_DL = min(2, len(_unique_ref_paths))
                with _RefPool(max_workers=_N_REF_DL) as _rp:
                    list(_rp.map(_dl_one_ref, _unique_ref_paths))

                if _ref_result:
                    logger.debug(
                        f"Reference prefetch ({ref_alias}): "
                        f"{_counters['dl']} downloaded, "
                        f"{_counters['hit']} cached "
                        f"({len(_unique_ref_paths)} unique files)"
                    )
                    _ref_prefetched = True

            _pred_prefetched = False
            _pred_result: Dict[str, str] = {}

            def _do_pred_prefetch():
                """Download prediction zarr stores in parallel (background thread)."""
                nonlocal _pred_prefetched
                if not batch:
                    return
                _sample_pred = batch[0].get("pred_data")
                _pred_is_remote = (
                    isinstance(_sample_pred, str)
                    and (
                        _sample_pred.startswith("https://")
                        or _sample_pred.startswith("http://")
                        or _sample_pred.startswith("s3://")
                    )
                )
                if not _pred_is_remote:
                    return

                from pathlib import Path as _PfPred
                import shutil as _sh_pred
                from concurrent.futures import ThreadPoolExecutor as _PredPool
                import threading as _pred_dl_threading

                _pred_cache_dir = str(
                    _PfPred(
                        getattr(self, "results_dir", None) or "/tmp"
                    )
                    / "pred_prefetch_cache"
                    / str(pred_alias)
                )
                os.makedirs(_pred_cache_dir, exist_ok=True)

                _pred_fs = getattr(
                    getattr(
                        self.dataloader, "pred_manager", None
                    ),
                    "params", None,
                )
                _pred_s3fs = getattr(_pred_fs, "fs", None)
                _pred_endpoint = getattr(
                    _pred_fs, "endpoint_url", ""
                ) or ""

                _unique_pred_paths = list(dict.fromkeys(
                    e["pred_data"]
                    for e in batch
                    if isinstance(e.get("pred_data"), str)
                ))

                _counters = {"dl": 0, "hit": 0}
                _pred_lock = _pred_dl_threading.Lock()

                def _dl_one_pred(_rp):
                    _fname = _PfPred(_rp).name
                    _local_zarr = os.path.join(
                        _pred_cache_dir, _fname
                    )
                    if os.path.isdir(_local_zarr) and os.listdir(
                        _local_zarr
                    ):
                        with _pred_lock:
                            _pred_result[_rp] = _local_zarr
                            _counters["hit"] += 1
                        return
                    try:
                        logger.debug(
                            f"Prefetching prediction: {_fname}"
                        )
                        _tid = _pred_dl_threading.current_thread().ident
                        _tmp_zarr = _local_zarr + f".downloading.{_tid}"
                        if os.path.isdir(_tmp_zarr):
                            _sh_pred.rmtree(
                                _tmp_zarr, ignore_errors=True
                            )

                        _s3_key = _rp
                        if _pred_endpoint and _s3_key.startswith(
                            _pred_endpoint
                        ):
                            _s3_key = _s3_key[
                                len(_pred_endpoint):
                            ].lstrip("/")
                        elif _s3_key.startswith("s3://"):
                            _s3_key = _s3_key[len("s3://"):]

                        if _pred_s3fs is not None:
                            _pred_s3fs.get(
                                _s3_key,
                                _tmp_zarr,
                                recursive=True,
                            )
                        else:
                            import xarray as _xr_prefetch
                            with dask.config.set(
                                scheduler="synchronous"
                            ):
                                _ds_pf = _xr_prefetch.open_zarr(
                                    _rp, chunks={}
                                )
                                _ds_pf = _ds_pf.compute()
                            _ds_pf.to_zarr(
                                _tmp_zarr, mode="w",
                                consolidated=True,
                            )
                            _ds_pf.close()
                            del _ds_pf

                        if os.path.isdir(_local_zarr):
                            _sh_pred.rmtree(
                                _local_zarr, ignore_errors=True
                            )
                        os.rename(_tmp_zarr, _local_zarr)
                        with _pred_lock:
                            _pred_result[_rp] = _local_zarr
                            _counters["dl"] += 1
                    except Exception as _exc_pf:
                        logger.warning(
                            f"Prediction prefetch failed for "
                            f"{_fname}: {_exc_pf!r}"
                        )

                _N_PRED_DL = min(4, len(_unique_pred_paths))
                with _PredPool(max_workers=_N_PRED_DL) as _pp:
                    list(_pp.map(_dl_one_pred, _unique_pred_paths))

                if _pred_result:
                    logger.debug(
                        f"Prediction prefetch ({pred_alias}): "
                        f"{_counters['dl']} downloaded, "
                        f"{_counters['hit']} cached "
                        f"({len(_unique_pred_paths)} unique files)"
                    )
                    _pred_prefetched = True

            # ── Start prediction prefetch in background ──────────────
            _t_pred_dl = time.time()

            _t_ref_dl = 0.0
            _ref_thread = None
            _ref_protocol = getattr(ref_connection_params, "protocol", None)
            _sample_ref = batch[0].get("ref_data") if batch else None
            _need_ref_prefetch = (
                (not is_obs_batch)
                and (os.environ.get("DCTOOLS_ENABLE_REF_PREFETCH", "1") in ("1", "true", "True"))
                and (_ref_protocol in ("wasabi", "s3"))
                and isinstance(_sample_ref, str)
                and _sample_ref.endswith(".zarr")
                and _sample_ref.startswith(("https://", "http://", "s3://"))
            )
            if _need_ref_prefetch:
                _t_ref_dl = time.time()
                _ref_thread = _pred_thr.Thread(
                    target=_do_ref_prefetch, daemon=True, name="ref-prefetch"
                )
                _ref_thread.start()

            _pred_thread = _pred_thr.Thread(
                target=_do_pred_prefetch, daemon=True, name="pred-prefetch"
            )
            _pred_thread.start()

            # ── Shared batch preprocessing for swath/track obs ────────────
            # When files have been prefetched, preprocess ALL unique files
            # once on the driver into a single shared zarr.  Workers then
            # open this zarr and filter by their time_bounds — instead of
            # each worker independently opening/preprocessing 40+ files
            # (most of which are shared with other tasks).
            # Speedup: N_tasks × files_per_task  ->  1 × unique_files.
            _shared_obs_zarr: Optional[str] = None
            _t_obs_prep = time.time()
            if _obs_prefetch and _obs_path_map:
                _ref_d0 = batch[0].get("ref_data") if batch else None
                if isinstance(_ref_d0, dict):
                    try:
                        _md = _ref_d0.get("metadata", {})
                        _coord_sys = (
                            _md.get("coord_system")
                            if isinstance(_md, dict)
                            else getattr(_md, "coord_system", None)
                        )
                        _coords = (
                            getattr(_coord_sys, "coordinates", None)
                            if _coord_sys is not None
                            else None
                        )
                        if _coords is None:
                            _coords = {"time": "time"}

                        _kv = _ref_d0.get("keep_vars")
                        _n_pts_dim = "n_points"
                        _rc0 = batch[0].get("ref_coords")
                        if (
                            _rc0 is not None
                            and hasattr(_rc0, "coordinates")
                            and isinstance(
                                getattr(_rc0.coordinates, "get", None),
                                type(dict.get),
                            )
                        ):
                            _n_pts_dim = _rc0.coordinates.get(
                                "n_points", "n_points"
                            )

                        _local_unique = list(
                            dict.fromkeys(_obs_path_map.values())
                        )

                        from dctools.data.datasets.dataloader import (
                            preprocess_batch_obs_files,
                        )
                        from pathlib import Path as _PfPath2

                        _shared_zarr_dir = str(
                            _PfPath2(
                                getattr(self, "results_dir", None)
                                or "/tmp"
                            )
                            / "obs_batch_shared"
                            / str(ref_alias)
                            / f"batch_{_batch_idx}"
                        )

                        _shared_obs_zarr = preprocess_batch_obs_files(
                            local_paths=_local_unique,
                            alias=str(ref_alias),
                            keep_vars=_kv,
                            coordinates=(
                                dict(_coords)
                                if not isinstance(_coords, dict)
                                else _coords
                            ),
                            n_points_dim=_n_pts_dim,
                            output_zarr_dir=_shared_zarr_dir,
                        )

                        if _shared_obs_zarr:
                            for _entry in batch:
                                _rd = _entry.get("ref_data")
                                if isinstance(_rd, dict):
                                    _rd[
                                        "prefetched_obs_zarr_path"
                                    ] = _shared_obs_zarr
                    except Exception as _exc_shared:
                        logger.warning(
                            f"Shared batch preprocessing failed "
                            f"({ref_alias}): {_exc_shared!r}"
                        )
                        import traceback as _tb_shared

                        _tb_shared.print_exc()

            _t_obs_prep = time.time() - _t_obs_prep

            # ── Wait for prediction prefetch thread ──────────────────
            if _ref_thread is not None:
                _ref_thread.join()
                _t_ref_dl = time.time() - _t_ref_dl
            _pred_thread.join()
            _t_pred_dl = time.time() - _t_pred_dl

            _t_prefetch_total = time.time() - _phase_t0
            logger.debug(
                f"Prefetch done in {_t_prefetch_total:.1f}s "
                f"(obs_dl={_t_obs_dl:.1f}s  obs_prep={_t_obs_prep:.1f}s  "
                f"ref_dl={_t_ref_dl:.1f}s  pred_dl={_t_pred_dl:.1f}s) — dispatching {num_tasks} tasks"  # noqa: E501
            )

            # Apply reference path remapping (from current batch prefetch)
            if _ref_result:
                for _entry in batch:
                    _rd = _entry.get("ref_data")
                    if isinstance(_rd, str) and _rd in _ref_result:
                        _entry["ref_data"] = _ref_result[_rd]

            # Apply prediction path remapping (from current batch prefetch)
            if _pred_result:
                for _entry in batch:
                    _pd = _entry.get("pred_data")
                    if (
                        isinstance(_pd, str)
                        and _pd in _pred_result
                    ):
                        _entry["pred_data"] = _pred_result[_pd]

            # Apply look-ahead prediction paths (downloaded during previous batch)
            _la_pred_map = _la_data.get('pred_map', {}) if _la_data else {}
            if _la_pred_map:
                for _entry in batch:
                    _pd = _entry.get("pred_data")
                    if (
                        isinstance(_pd, str)
                        and _pd in _la_pred_map
                    ):
                        _entry["pred_data"] = _la_pred_map[_pd]

            # Always create the progress bar, even if no lookahead
            _overall_bar = tqdm(
                total=num_tasks,
                desc=f"[Batch {_batch_idx+1}/{_total_batches}] {ref_alias}",
                leave=True,
                unit="task",
                dynamic_ncols=True,
                file=_sys_bars.stderr,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {desc}"
            )

            logger.debug(
                f"{ref_alias}: {num_tasks} tasks on {_N} workers"
            )

            # ── Observation scheduling: submit heavy windows first ──
            # We only keep ~slots tasks in-flight. If the batch list is in
            # chronological order and cost varies a lot, the heaviest tasks
            # can end up being submitted last, creating a long end-of-batch
            # tail. Sorting by a simple cost proxy reduces the tail (LPT).
            _task_order: List[int] = list(range(num_tasks))
            if is_obs_batch:
                _task_order.sort(
                    key=lambda i: int(batch[i].get("_obs_cost", 0) or 0),
                    reverse=True,
                )
                _max_cost = max((int(e.get("_obs_cost", 0) or 0) for e in batch), default=0)
                _min_cost = min((int(e.get("_obs_cost", 0) or 0) for e in batch), default=0)
                if _max_cost != _min_cost:
                    logger.debug(
                        f"{ref_alias}: scheduling obs tasks by descending cost "
                        f"(_obs_cost rows): min={_min_cost}, max={_max_cost}"
                    )

            # NOTE: If the batch has <= workers tasks, Dask cannot keep all
            # workers busy for the whole batch. As soon as faster tasks finish,
            # their workers go idle while a few slow/IO-bound tasks ("stragglers")
            # continue running. This typically shows up as a CPU usage drop around
            # the middle/last third of the batch even though tasks remain.
            if num_tasks <= _N:
                logger.debug(
                    f"{ref_alias}: batch has {num_tasks} tasks for {_N} workers; "
                    "CPU may drop mid-batch due to stragglers. "
                    "If this is undesirable, increase batch_size (more tasks than workers) "
                    "or reduce n_workers."
                )

            # ── CRITICAL: force synchronous dask scheduler ────────
            # With processes=False the cluster uses threads that all
            # share the same process.  An active distributed Client
            # makes .compute() submit sub-tasks to the cluster.
            # If all worker slots are busy -> deadlock.
            # Set once here, never restore — this is intentional.
            # _client.submit() is a Client method and is not affected
            # by this config setting, so task dispatch still works.
            # dask.config.set(scheduler="synchronous")

            _active: Dict[Any, int] = {}
            _results: Dict[int, Any] = {}
            _all_futures: List[Any] = []
            # Driver-observed (submit -> result) wall times help diagnose
            # end-of-batch slowdowns (queueing, pauses, IO waits) vs pure
            # compute time reported by workers.
            _submitted_at: Dict[Any, float] = {}
            _wall_times_s: List[float] = []
            _wall_tail_s: List[float] = []
            _wall_by_idx: Dict[int, float] = {}
            _wall_tail_by_idx: Dict[int, float] = {}
            _next = 0
            _n_collected = 0
            _ac = as_completed([])

            # Submit all tasks at once — ARGO data is already prefetched
            # to local Zarr, so workers won't block on HTTP.
            # For non-ARGO batches, same behaviour as before.
            _n_seed = min(max_concurrent_obs, num_tasks)
            while _next < _n_seed:
                _task_i = _task_order[_next]
                _f = _client.submit(fn, batch[_task_i], retries=1, pure=False)
                _all_futures.append(_f)
                _active[_f] = _task_i
                _submitted_at[_f] = time.monotonic()
                _ac.add(_f)
                _next += 1

            logger.debug(
                f"{ref_alias}: {_n_seed}/{num_tasks} tasks submitted to "
                f"{_N} workers — waiting for first result…"
            )
            if _overall_bar is not None:
                _overall_bar.set_postfix_str("workers busy…")

            # ── Look-ahead: download next batch's data during as_completed ──
            # While workers are busy computing, the driver has spare CPU/IO.
            # Use it to prefetch obs+pred files for the next batch.
            import threading as _la_thr

            _la_thread = None
            if _lookahead_next is not None:
                _la_batch = _lookahead_next.get('batch')
                _la_ref_alias = _lookahead_next.get('ref_alias')
                _la_is_obs = (
                    _la_batch
                    and _la_batch[0].get("ref_is_observation", False)
                )

                def _do_lookahead():
                    """Download obs+pred files for the next batch."""
                    _result: Dict[str, Any] = {}
                    try:
                        # ── Obs download ──────────────────────────────
                        if (
                            _la_is_obs
                            and _la_ref_alias != "argo_profiles"
                        ):
                            _la_mgr = self.dataloader.ref_managers.get(
                                _la_ref_alias  # type: ignore[arg-type]
                            )
                            _la_has_fs = (
                                _la_mgr is not None
                                and hasattr(_la_mgr, "params")
                                and hasattr(_la_mgr.params, "fs")
                                and _la_mgr.params.fs is not None
                            )
                            if _la_has_fs:
                                _la_paths: List[str] = []
                                for _e in _la_batch:  # type: ignore[union-attr]
                                    _rd = _e.get("ref_data")
                                    if (
                                        isinstance(_rd, dict)
                                        and "source" in _rd
                                    ):
                                        _cat = _rd["source"]
                                        _tb = _rd.get("time_bounds")
                                        if (
                                            _tb is not None
                                            and hasattr(
                                                _cat, "get_dataframe"
                                            )
                                        ):
                                            _cdf = _cat.get_dataframe()
                                            _ff = filter_by_time(
                                                _cdf,
                                                pd.Timestamp(_tb[0]),
                                                pd.Timestamp(_tb[1]),
                                            )
                                            _la_paths.extend(
                                                _ff["path"].tolist()
                                            )
                                _la_uniq = list(
                                    dict.fromkeys(_la_paths)
                                )
                                if _la_uniq:
                                    from pathlib import Path as _PfLA
                                    from dctools.data.connection.connection_manager import (
                                        prefetch_obs_files_to_local,
                                    )
                                    _la_cache = str(
                                        _PfLA(
                                            getattr(
                                                self, "results_dir",
                                                None,
                                            )
                                            or "/tmp"
                                        )
                                        / "obs_prefetch_cache"
                                        / str(_la_ref_alias)
                                    )
                                    logger.debug(
                                        f"Look-ahead: downloading "
                                        f"{len(_la_uniq)} obs files "
                                        f"for {_la_ref_alias}"
                                    )
                                    _la_obs_map = (
                                        prefetch_obs_files_to_local(
                                            remote_paths=_la_uniq,
                                            cache_dir=_la_cache,
                                            fs=_la_mgr.params.fs,  # type: ignore[union-attr]
                                            ref_alias=(
                                                f"LA:{_la_ref_alias}"
                                            ),
                                        )
                                    )
                                    if _la_obs_map:
                                        _result['obs_map'] = _la_obs_map

                        # ── Pred download ─────────────────────────────
                        if _la_batch:
                            _sample = _la_batch[0].get("pred_data")
                            _is_remote = isinstance(_sample, str) and (
                                _sample.startswith("https://")
                                or _sample.startswith("http://")
                                or _sample.startswith("s3://")
                            )
                            if _is_remote:
                                from pathlib import Path as _PfLA2
                                import shutil as _sh_la
                                _la_pred_cache = str(
                                    _PfLA2(
                                        getattr(
                                            self, "results_dir",
                                            None,
                                        )
                                        or "/tmp"
                                    )
                                    / "pred_prefetch_cache"
                                    / str(
                                        self.dataloader.pred_alias
                                    )
                                )
                                os.makedirs(
                                    _la_pred_cache, exist_ok=True
                                )
                                _la_pfs = getattr(
                                    getattr(
                                        self.dataloader,
                                        "pred_manager",
                                        None,
                                    ),
                                    "params",
                                    None,
                                )
                                _la_s3fs = getattr(
                                    _la_pfs, "fs", None
                                )
                                _la_ep = getattr(
                                    _la_pfs, "endpoint_url", ""
                                ) or ""
                                _la_upreds = list(
                                    dict.fromkeys(
                                        e["pred_data"]
                                        for e in _la_batch
                                        if isinstance(
                                            e.get("pred_data"), str
                                        )
                                    )
                                )
                                _la_pred_map: Dict[str, str] = {}
                                for _rp in _la_upreds:
                                    _fn = _PfLA2(_rp).name
                                    _lz = os.path.join(
                                        _la_pred_cache, _fn
                                    )
                                    if (
                                        os.path.isdir(_lz)
                                        and os.listdir(_lz)
                                    ):
                                        _la_pred_map[_rp] = _lz
                                        continue
                                    try:
                                        _s3k = _rp
                                        if (
                                            _la_ep
                                            and _s3k.startswith(
                                                _la_ep
                                            )
                                        ):
                                            _s3k = _s3k[
                                                len(_la_ep):
                                            ].lstrip("/")
                                        elif _s3k.startswith(
                                            "s3://"
                                        ):
                                            _s3k = _s3k[5:]
                                        _tmpz = (
                                            _lz + ".downloading.la"
                                        )
                                        if os.path.isdir(_tmpz):
                                            _sh_la.rmtree(
                                                _tmpz,
                                                ignore_errors=True,
                                            )
                                        if _la_s3fs is not None:
                                            _la_s3fs.get(
                                                _s3k,
                                                _tmpz,
                                                recursive=True,
                                            )
                                        if os.path.isdir(_lz):
                                            _sh_la.rmtree(
                                                _lz,
                                                ignore_errors=True,
                                            )
                                        os.rename(_tmpz, _lz)
                                        _la_pred_map[_rp] = _lz
                                    except Exception:
                                        pass
                                if _la_pred_map:
                                    _result['pred_map'] = _la_pred_map

                    except Exception as _exc_la:
                        logger.debug(
                            f"Look-ahead prefetch error: {_exc_la!r}"
                        )

                    # Store results for the next _evaluate_batch call
                    if _result:
                        _la_cache_dict = getattr(
                            self, '_lookahead_cache', {}
                        )
                        _la_cache_dict[id(_la_batch)] = _result
                        self._lookahead_cache = _la_cache_dict
                        logger.debug(
                            "Look-ahead: pre-downloaded "
                            f"{len(_result.get('obs_map', {}))} obs + "
                            f"{len(_result.get('pred_map', {}))} pred "
                            f"files for next batch"
                        )

                _la_thread = _la_thr.Thread(
                    target=_do_lookahead,
                    daemon=True,
                    name="lookahead-dl",
                )
                _la_thread.start()

            # ── Background heartbeat + stall-watchdog ────────────────────
            # The heartbeat updates the progress bar every 30 s.
            # The watchdog detects when NO task has completed for more than
            # STALL_TIMEOUT seconds (e.g. all workers blocked on a stalled
            # S3 / HTTP connection) and cancels the stuck futures to unblock
            # the as_completed loop.  Cancelled futures are yielded as errors.
            import threading as _threading_hb

            _hb_stop = _threading_hb.Event()
            _hb_t0 = time.time()
            # Track the last time a task completed (initialised to start).
            _last_progress: list = [time.time()]
            # After this many seconds with zero new completions -> cancel all.
            _STALL_TIMEOUT = 1200  # 20 minutes — SWOT per-worker preprocessing can be slow

            # Log the inevitable "tail" once: when pending tasks <= execution slots,
            # the cluster cannot keep all slots busy (under-subscription).
            _tail_logged: list = [False]
            _last_state_log_s: list = [0.0]

            def _maybe_log_cluster_state(elapsed_s: float, pending: int):
                if elapsed_s - _last_state_log_s[0] < 60.0:
                    return
                _last_state_log_s[0] = elapsed_s
                try:
                    info = _client.scheduler_info()
                    workers = info.get("workers", {})
                    paused = 0
                    max_frac = 0.0
                    for w_info in workers.values():
                        if w_info.get("status") == "paused":
                            paused += 1
                        mem_used = w_info.get("metrics", {}).get(
                            "memory", w_info.get("memory", 0)
                        )
                        mem_limit = w_info.get("memory_limit", 0) or 0
                        if mem_limit and mem_limit > 0:
                            max_frac = max(max_frac, float(mem_used) / float(mem_limit))

                    # Only speak up when it is actionable.
                    if paused or max_frac >= float(self.max_worker_memory_fraction or 0.0):
                        logger.warning(
                            f"{ref_alias}: state pending={pending}, active={len(_active)}, "
                            f"paused_workers={paused}/{len(workers)}, "
                            f"max_mem_frac={max_frac:.2f}"
                        )
                except Exception as _exc_state:
                    logger.debug(f"{ref_alias}: cannot query scheduler state: {_exc_state!r}")

            def _heartbeat_fn():
                while not _hb_stop.is_set():
                    _hb_stop.wait(30)
                    if _hb_stop.is_set():
                        break
                    _elapsed = time.time() - _hb_t0
                    _pending = max(num_tasks - _n_collected, 0)
                    pct = 100.0 * (_n_collected / num_tasks) if num_tasks else 0.0
                    _overall_bar.set_postfix_str(
                        f"{pct:.1f}% done, {_elapsed:.0f}s elapsed"
                    )

                    if (not _tail_logged[0]) and _pending <= _total_slots:
                        _tail_logged[0] = True
                        logger.debug(
                            f"{ref_alias}: entering end-of-batch tail: "
                            f"pending={_pending} <= slots={_total_slots}. "
                            "CPU drop is expected here; remaining time is dominated by the slowest tasks."  # noqa: E501
                        )

                    _maybe_log_cluster_state(_elapsed, _pending)

                    # ── Watchdog: cancel futures if no progress ──────────
                    _stall_s = time.time() - _last_progress[0]
                    if _stall_s >= _STALL_TIMEOUT and _active:
                        logger.error(
                            f"{ref_alias}: NO task completed in the last "
                            f"{_stall_s:.0f}s — workers appear deadlocked "
                            f"(likely S3/network timeout).  Cancelling "
                            f"{len(_active)} stuck futures to unblock."
                        )
                        for _stuck_f in list(_active.keys()):
                            try:
                                _stuck_f.cancel()
                            except Exception:
                                pass
                        # Reset timer so we don't cancel again immediately.
                        _last_progress[0] = time.time()

            _hb_thread = _threading_hb.Thread(
                target=_heartbeat_fn, daemon=True
            )
            _hb_thread.start()

            try:
                for _done in _ac:
                    _idx = _active.pop(_done)
                    _t_submit = _submitted_at.pop(_done, None)
                    if _t_submit is not None:
                        _wall = time.monotonic() - _t_submit
                        _wall_times_s.append(_wall)
                        _wall_by_idx[int(_idx)] = float(_wall)
                        _tail_remaining = max(num_tasks - _n_collected, 0)
                        if _tail_remaining <= _total_slots:
                            _wall_tail_s.append(_wall)
                            _wall_tail_by_idx[int(_idx)] = float(_wall)
                    try:
                        _res = _done.result()
                    except Exception as _exc:
                        logger.warning(
                            f"Task {_idx} ({ref_alias}) raised: {_exc!r}"
                        )
                        _res = {
                            "ref_alias": ref_alias,
                            "result": None,
                            "n_points": 0,
                            "duration_s": 0.0,
                            "error": repr(_exc),
                        }
                    _results[_idx] = _res
                    _n_collected += 1
                    _last_progress[0] = time.time()  # reset stall watchdog

                    # Log timing for the very first completed task
                    if _n_collected == 1:
                        _first_elapsed = time.time() - _hb_t0
                        logger.debug(
                            f"{ref_alias}: first result received "
                            f"after {_first_elapsed:.1f}s"
                        )
                        _overall_bar.set_postfix_str("")

                    # Only log errors; normal completions are silent
                    # (the tqdm bar + batch summary are enough).
                    _err = _res.get("error") if isinstance(_res, dict) else None
                    if _err:
                        _vt = _res.get("valid_time", "") if isinstance(_res, dict) else ""
                        _overall_bar.write(f"  \u2717 {_vt}  ERROR: {_err}")

                    _overall_bar.update(1)

                    # Submit remaining tasks as slots become available
                    if _next < num_tasks:
                        _task_i = _task_order[_next]
                        _f_new = _client.submit(
                            fn, batch[_task_i], retries=1, pure=False
                        )
                        _all_futures.append(_f_new)
                        _active[_f_new] = _task_i
                        _submitted_at[_f_new] = time.monotonic()
                        _ac.add(_f_new)
                        _next += 1

                    if _n_collected >= num_tasks:
                        break
            finally:
                _hb_stop.set()
                _hb_thread.join(timeout=2)
                try:
                    _overall_bar.close()
                except Exception:
                    pass

            # ── Wait for look-ahead thread (non-blocking if already done) ──
            if _la_thread is not None:
                _la_thread.join(timeout=5)

            # ── Explicit batch cleanup on client/workers ───────────────────
            try:
                if _all_futures:
                    _client.cancel(_all_futures, force=True)
                    wait(_all_futures)
            except Exception:
                pass
            try:
                _client.run(_worker_full_cleanup)
            except Exception:
                pass

            # Restore original batch order
            batch_results: List[Any] = [_results[i] for i in range(num_tasks)]

            batch_duration = time.time() - batch_t0

            def _pct(values: List[float], q: float) -> float:
                if not values:
                    return 0.0
                if q <= 0:
                    return float(min(values))
                if q >= 1:
                    return float(max(values))
                xs = sorted(values)
                pos = q * (len(xs) - 1)
                lo = int(pos)
                hi = min(lo + 1, len(xs) - 1)
                if hi == lo:
                    return float(xs[lo])
                w = pos - lo
                return float(xs[lo] * (1.0 - w) + xs[hi] * w)

            if _wall_times_s:
                logger.debug(
                    f"{ref_alias}: wall-times submit->result: "
                    f"p50={_pct(_wall_times_s, 0.50):.1f}s "
                    f"p90={_pct(_wall_times_s, 0.90):.1f}s "
                    f"p99={_pct(_wall_times_s, 0.99):.1f}s "
                    f"max={max(_wall_times_s):.1f}s"
                )
                if _wall_tail_s:
                    logger.debug(
                        f"{ref_alias}: tail wall-times (when pending<=slots): "
                        f"p50={_pct(_wall_tail_s, 0.50):.1f}s "
                        f"p90={_pct(_wall_tail_s, 0.90):.1f}s "
                        f"max={max(_wall_tail_s):.1f}s"
                    )

                # Pin down extreme stragglers (often the reason the tail feels like a stall).
                _slow_idxs = sorted(
                    _wall_by_idx.keys(),
                    key=lambda i: _wall_by_idx.get(i, 0.0),
                    reverse=True,
                )[: min(5, len(_wall_by_idx))]
                if _slow_idxs:
                    _lines: List[str] = []
                    for _i in _slow_idxs:
                        _entry = batch[_i] if 0 <= _i < len(batch) else {}
                        _vt = _entry.get("valid_time")
                        _frt = _entry.get("forecast_reference_time")
                        _cost = _entry.get("_obs_cost")
                        _tb = None
                        _rd = _entry.get("ref_data")
                        if isinstance(_rd, dict) and "time_bounds" in _rd:
                            _tb = _rd.get("time_bounds")

                        _res_i = batch_results[_i] if 0 <= _i < len(batch_results) else None
                        _np = _res_i.get("n_points") if isinstance(_res_i, dict) else None
                        _pp = _res_i.get("preprocess_s") if isinstance(_res_i, dict) else None
                        _mt = _res_i.get("duration_s") if isinstance(_res_i, dict) else None

                        _w = _wall_by_idx.get(_i, 0.0)
                        _is_tail = _i in _wall_tail_by_idx
                        _tail_tag = " tail" if _is_tail else ""
                        _lines.append(
                            f"idx={_i} wall={_w:.1f}s{_tail_tag} "
                            f"preproc={_pp!s} metrics={_mt!s} pts={_np!s} "
                            f"_obs_cost={_cost!s} valid_time={_vt} forecast_ref={_frt} time_bounds={_tb}"  # noqa: E501
                        )

                    '''logger.debug(
                        f"{ref_alias}: slowest tasks (driver wall-time):\n  "
                        + "\n  ".join(_lines)
                    )'''

            # Analyze task timings
            _valid = [r for r in batch_results if r and isinstance(r, dict)]
            times = [r.get('duration_s', 0) for r in _valid]
            preprocs = [r.get('preprocess_s', 0) for r in _valid]
            points = [r.get('n_points', 0) for r in _valid]

            if times:
                total_pts = sum(points)
                avg_pp = sum(preprocs) / len(preprocs)
                avg_mt = sum(times) / len(times)
                logger.info(
                    f"Batch done: {len(batch_results)}/{num_tasks} tasks "
                    f"in {batch_duration:.1f}s | "
                    f"Avg preproc={avg_pp:.1f}s  metrics={avg_mt:.1f}s | "
                    f"{total_pts:,} total pts"
                )
            else:
                logger.info(f"Batch done in {batch_duration:.1f}s (no valid results)")

            return batch_results
        except Exception as exc:
            logger.error(f"Error processing batch: {repr(exc)}")
            traceback.print_exc()
            return [{
                "forecast_reference_time": batch[0].get("forecast_reference_time") if batch else None,  # noqa: E501
                "model": pred_alias,
                "reference": ref_alias,
                "result": None,
                "n_points": 0,
                "duration_s": 0.0,
                "error": repr(exc),
            }]
