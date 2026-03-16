"""Worker-side memory, thread, and cache cleanup utilities.

This module contains functions that run **inside Dask workers** to manage
memory pressure, cap library-level thread parallelism, and clear stale
caches between evaluation batches.
"""

import ctypes
import gc
import os
import threading
from collections import OrderedDict
from typing import Any

import xarray as xr

# ---------------------------------------------------------------------------
# Per-worker LRU dataset cache (thread-safe)
# ---------------------------------------------------------------------------
_WORKER_DATASET_CACHE_LOCK = threading.Lock()
_WORKER_DATASET_CACHE: "OrderedDict[str, xr.Dataset]" = OrderedDict()


# ---------------------------------------------------------------------------
# Memory cleanup
# ---------------------------------------------------------------------------
def worker_memory_cleanup() -> None:
    """Manual memory cleanup to be run on workers.

    Performs aggressive garbage collection and memory trimming.
    """
    # Single gc.collect() is sufficient — 3× adds overhead per call
    gc.collect()

    # Linux-specific memory trimming (release to OS)
    try:
        ctypes.CDLL('libc.so.6').malloc_trim(0)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Xarray file-cache clearing
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Full worker cleanup (env vars + thread caps + caches + GC)
# ---------------------------------------------------------------------------
def _worker_full_cleanup() -> bool:
    """Full cleanup routine to run on workers via client.run()."""
    import os

    def _blosc_thread_count(default: str = "4") -> int:
        try:
            return max(1, int(os.environ.get("BLOSC_NTHREADS", default)))
        except (TypeError, ValueError):
            return max(1, int(default))

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
        "PYINTERP_NUM_THREADS", "GOTO_NUM_THREADS",
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
    _blosc_threads = _blosc_thread_count()
    os.environ["BLOSC_NTHREADS"] = str(_blosc_threads)

    try:
        import blosc  # type: ignore[import-not-found]
        blosc.set_nthreads(_blosc_threads)
    except Exception:
        pass
    try:
        from numcodecs import blosc as _nc_blosc  # type: ignore[import-untyped]
        _nc_blosc.set_nthreads(_blosc_threads)
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


# ---------------------------------------------------------------------------
# Stale S3 connection clearing (for in-task retry)
# ---------------------------------------------------------------------------
def _clear_stale_s3_state() -> None:
    """Clear all S3-related cached state so the next open gets fresh connections.

    This must be called before re-opening a dataset on retry after an S3
    timeout, because the stale aiobotocore sessions are baked into the
    cached filesystem objects.
    """
    # 1. Close & clear worker dataset cache
    with _WORKER_DATASET_CACHE_LOCK:
        for _ds in _WORKER_DATASET_CACHE.values():
            try:
                if hasattr(_ds, "close"):
                    _ds.close()
            except Exception:
                pass
        _WORKER_DATASET_CACHE.clear()

    # 2. Clear xarray file-handle cache
    _clear_xarray_file_cache()

    # 3. Clear fsspec/s3fs filesystem instance caches so new opens
    #    create fresh aiobotocore sessions instead of reusing stale ones.
    try:
        import s3fs
        if hasattr(s3fs.S3FileSystem, "_cache"):
            s3fs.S3FileSystem._cache.clear()
    except Exception:
        pass
    try:
        import fsspec
        fsspec.filesystem.cache.clear()  # type: ignore[attr-defined]
    except Exception:
        pass

    gc.collect()


# ---------------------------------------------------------------------------
# Thread-pool capping
# ---------------------------------------------------------------------------
_THREADS_CAPPED: bool = False


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
    global _THREADS_CAPPED
    if _THREADS_CAPPED and max_threads == 1:
        return
    _t = str(max_threads)

    def _blosc_thread_count() -> int:
        try:
            return max(1, int(os.environ.get("BLOSC_NTHREADS", _t)))
        except (TypeError, ValueError):
            return max_threads

    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "GOTO_NUM_THREADS",
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

    # -- PyTorch: call the runtime setters as well -------------------------
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

    # -- C-level thread pool cap (belt-and-suspenders) --
    # threadpoolctl talks directly to the shared libraries already loaded
    # in the process (libopenblas, libgomp, libiomp5, …) and resizes
    # their internal pools.  This works even if the env vars were set
    # *after* the library created its default thread pool.
    try:
        import threadpoolctl
        threadpoolctl.threadpool_limits(limits=max_threads)
    except ImportError:
        pass

    _blosc_threads = _blosc_thread_count()
    os.environ["BLOSC_NTHREADS"] = str(_blosc_threads)
    try:
        import blosc  # type: ignore[import-not-found]
        blosc.set_nthreads(_blosc_threads)
    except Exception:
        pass
    try:
        from numcodecs import blosc as _nc_blosc  # type: ignore[import-untyped]
        _nc_blosc.set_nthreads(_blosc_threads)
    except Exception:
        pass
    if max_threads == 1:
        _THREADS_CAPPED = True
