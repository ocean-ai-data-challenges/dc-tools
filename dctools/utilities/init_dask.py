"""Dask initialization and configuration functions."""

from argparse import Namespace
import logging
import multiprocessing
import os
import psutil
from typing import Optional
import warnings

import dask
from dask.distributed import LocalCluster
from loguru import logger


def get_optimal_workers():
    """Determine the optimal number of workers based on system resources."""
    num_cores = multiprocessing.cpu_count()
    available_memory = psutil.virtual_memory().available / 1e9  # Available memory (GB)
    worker_memory = float(get_optimal_memory_limit().rstrip("GB"))
    # Adjust worker count — cap to avoid memory contention on large machines
    max_workers_by_memory = int(available_memory // worker_memory)
    optimal_workers = min(4, num_cores, max_workers_by_memory)

    return max(1, optimal_workers)  # At least 1 worker


def get_optimal_memory_limit():
    """Calculate a memory limit based on available RAM."""
    total_memory = psutil.virtual_memory().total / 1e9  # Total RAM in GB
    # Limit to 4GB max per worker to avoid memory leaks
    available_memory = min(6, int(total_memory * 0.25))  # Reduction to 30% and max 8GB
    return f"{available_memory}GB"


def get_hdf5_env_vars():
    """Get dictionary of all HDF5/NetCDF environment variables needed."""
    return {
        "HDF5_USE_FILE_LOCKING": "FALSE",
        "NETCDF4_DEACTIVATE_MPI": "1",
        "NETCDF4_USE_FILE_LOCKING": "FALSE",
        "HDF5_DISABLE_VERSION_CHECK": "1",
        # Additional variables for argopy compatibility
        "ARGOPY_NETCDF_LOCKING": "FALSE",
    }


def apply_h5py_monkey_patch():
    """Patch xr.open_dataset to use scipy engine for in-memory data.

    The netCDF4 C library cannot open in-memory (bytes/BytesIO) NetCDF files
    on some systems due to HDF5 file-locking restrictions.  The scipy engine
    handles NetCDF3 natively without any locking.

    This function can be called in workers to ensure in-memory reads succeed.
    """
    try:
        import io
        import xarray as xr

        if not hasattr(xr, "_original_open_dataset"):
            xr._original_open_dataset = xr.open_dataset  # type: ignore[attr-defined]

            def _open_dataset_scipy_for_inmem(filename_or_obj, *args, **kwargs):
                if isinstance(filename_or_obj, (bytes, io.BytesIO, io.BufferedIOBase)):
                    kwargs.setdefault("engine", "scipy")
                if kwargs.get("engine") == "scipy":
                    _bk = kwargs.get("backend_kwargs")
                    if _bk is None:
                        _bk = {}
                    else:
                        _bk = dict(_bk)
                    _bk.setdefault("mmap", False)
                    kwargs["backend_kwargs"] = _bk
                return xr._original_open_dataset(filename_or_obj, *args, **kwargs)  # type: ignore[attr-defined]

            xr.open_dataset = _open_dataset_scipy_for_inmem
        return True
    except Exception:
        return False


def configure_dask_workers_env(client):
    """Configure Dask workers with HDF5/NetCDF environment and h5py patch.

    Args:
        client: Dask distributed client

    Returns:
        bool: True if successful, False otherwise
    """

    def set_worker_env_and_patch():
        import io
        import os

        # Set environment variables
        env_vars = {
            "HDF5_USE_FILE_LOCKING": "FALSE",
            "NETCDF4_DEACTIVATE_MPI": "1",
            "NETCDF4_USE_FILE_LOCKING": "FALSE",
            "HDF5_DISABLE_VERSION_CHECK": "1",
            "ARGOPY_NETCDF_LOCKING": "FALSE",
        }
        for key, value in env_vars.items():
            os.environ[key] = value

        # Set env vars for libraries that READ them at init time.
        # This is a no-op for libraries already initialised, but ensures
        # any library first loaded AFTER this call uses the cap.
        for tvar in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "PYINTERP_NUM_THREADS",
            "GOTO_NUM_THREADS",
        ):
            os.environ[tvar] = "1"
        # Blosc gets its own higher cap (see below)
        os.environ["BLOSC_NTHREADS"] = "2"

        # ── threadpoolctl: resize C-level pools that are ALREADY running ──
        # env vars are only read at library *initialisation*; for libraries
        # that have already started their thread pool (OpenBLAS, libgomp,
        # libiomp5 …) we must resize the pool at the C level directly.
        # threadpoolctl does exactly this via dlsym/GetProcAddress.
        try:
            import threadpoolctl

            threadpoolctl.threadpool_limits(limits=1)
        except Exception:
            pass

        # ── Blosc: allow 2 threads for decent decompression speed ────
        # Blosc uses its OWN thread pool (not OpenMP); it is not covered
        # by threadpoolctl.  SWOT files are Blosc-compressed HDF5 chunks;
        # 2 threads gives good throughput without oversubscribing CPUs.
        try:
            import blosc  # type: ignore[import-not-found]

            blosc.set_nthreads(2)
        except Exception:
            pass
        # numcodecs Blosc (used by zarr)
        try:
            from numcodecs import blosc as nc_blosc  # type: ignore[import-untyped]

            nc_blosc.set_nthreads(2)
        except Exception:
            pass

        # Patch xr.open_dataset to use scipy engine for in-memory data
        # (netCDF4 C library fails with EPERM on BytesIO / raw bytes)
        import xarray as xr

        if not hasattr(xr, "_original_open_dataset"):
            xr._original_open_dataset = xr.open_dataset  # type: ignore[attr-defined]

            def _open_dataset_scipy_for_inmem(filename_or_obj, *args, **kwargs):
                if isinstance(filename_or_obj, (bytes, io.BytesIO, io.BufferedIOBase)):
                    kwargs.setdefault("engine", "scipy")
                if kwargs.get("engine") == "scipy":
                    _bk = kwargs.get("backend_kwargs")
                    if _bk is None:
                        _bk = {}
                    else:
                        _bk = dict(_bk)
                    _bk.setdefault("mmap", False)
                    kwargs["backend_kwargs"] = _bk
                return xr._original_open_dataset(filename_or_obj, *args, **kwargs)  # type: ignore[attr-defined]

            xr.open_dataset = _open_dataset_scipy_for_inmem

        return True

    try:
        result = client.run(set_worker_env_and_patch)
        logger.debug(
            f"Successfully configured HDF5/NetCDF env vars and h5py patch on {len(result)} workers"
        )
        return True
    except Exception as e:
        logger.warning(f"Could not configure workers: {e}")
        return False


def setup_dask(args: Optional[Namespace] = None):
    """Automatically configure Dask based on available resources."""
    # Determine available memory and CPU - REDUCED to avoid NetCDF conflicts
    num_workers = get_optimal_workers()
    memory_limit = get_optimal_memory_limit()

    # Configuration to avoid NetCDF/HDF5 conflicts
    env_vars = get_hdf5_env_vars()
    for key, value in env_vars.items():
        os.environ[key] = value

    # Use threads instead of processes to avoid conflicts
    dask.config.set(scheduler="threads")
    dask.config.set({"temporary-directory": "/tmp/dask"})

    cluster = LocalCluster(
        n_workers=num_workers,
        threads_per_worker=1,
        memory_limit=memory_limit,
        processes=False,  # Force threads to avoid process overhead/conflicts
        silence_logs=False,  # Keep logs for debugging
        env=env_vars,
    )

    # Also push env vars to any spawned workers just in case
    client = dask.distributed.Client(cluster)

    def set_worker_env():
        import io
        import os

        # Set all HDF5/NetCDF environment variables in worker
        env_vars = {
            "HDF5_USE_FILE_LOCKING": "FALSE",
            "NETCDF4_DEACTIVATE_MPI": "1",
            "NETCDF4_USE_FILE_LOCKING": "FALSE",
            "HDF5_DISABLE_VERSION_CHECK": "1",
            "ARGOPY_NETCDF_LOCKING": "FALSE",
        }
        for key, value in env_vars.items():
            os.environ[key] = value

        # Patch xr.open_dataset to use scipy for in-memory data
        import xarray as xr

        if not hasattr(xr, "_original_open_dataset"):
            xr._original_open_dataset = xr.open_dataset  # type: ignore[attr-defined]

            def _open_dataset_scipy_for_inmem(filename_or_obj, *args, **kwargs):
                if isinstance(filename_or_obj, (bytes, io.BytesIO, io.BufferedIOBase)):
                    kwargs.setdefault("engine", "scipy")
                if kwargs.get("engine") == "scipy":
                    _bk = kwargs.get("backend_kwargs")
                    if _bk is None:
                        _bk = {}
                    else:
                        _bk = dict(_bk)
                    _bk.setdefault("mmap", False)
                    kwargs["backend_kwargs"] = _bk
                return xr._original_open_dataset(filename_or_obj, *args, **kwargs)  # type: ignore[attr-defined]

            xr.open_dataset = _open_dataset_scipy_for_inmem

    try:
        client.run(set_worker_env)
        logger.info("Successfully propagated HDF5/NetCDF env vars to Dask workers")
    except Exception as e:
        logger.warning(f"Could not propagate env vars to workers: {e}")

    logger.info(
        f"Dask is running on {num_workers} CPU workers, each with {memory_limit} of memory."
    )

    configure_dask_logging()

    return cluster


def configure_dask_logging():
    """Configure Dask logs to be quiet."""
    # Remove noisy Dask-specific loggers
    dask_loggers = [
        "distributed",
        "distributed.core",
        "distributed.worker",
        "distributed.scheduler",
        "distributed.nanny",
        "distributed.comm",
        "distributed.utils",
        "distributed.client",
        "tornado.application",
    ]

    for logger_name in dask_loggers:
        logging.getLogger(logger_name).setLevel(logging.ERROR)

    # Silence Dask warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="distributed")
    warnings.filterwarnings("ignore", message=".*Event loop was unresponsive.*")

    # Global Dask configuration
    dask.config.set(
        {
            "distributed.worker.daemon": False,
            "distributed.comm.timeouts.tcp": "60s",
            "distributed.comm.timeouts.connect": "60s",
            # Memory management thresholds — keep generous to avoid premature
            # spill-to-disk which kills swath (SWOT/SARAL) throughput.
            "distributed.worker.memory.target": 0.8,
            "distributed.worker.memory.spill": 0.9,
            "distributed.worker.memory.pause": 0.95,
            "distributed.worker.memory.terminate": False,
            "logging": {
                "distributed": {
                    "": "error",  # root "distributed" logger
                    "worker": "error",  # sub-logger: distributed.worker
                }
            },
        }
    )
