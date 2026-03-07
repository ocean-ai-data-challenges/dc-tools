"""Dask initialization and configuration functions."""

import logging
import warnings

import dask
from loguru import logger


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
