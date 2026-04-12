"""Dask initialization and configuration functions."""

import logging
import logging.config as _logging_config
import warnings

import dask
from loguru import logger


# ---------------------------------------------------------------------------
# Permanent noise filter for distributed.* loggers
# ---------------------------------------------------------------------------

_DISTRIBUTED_NOISE_LOGGERS = (
    "distributed",
    "distributed.core",
    "distributed.comm",
    "distributed.nanny",
    "distributed.scheduler",
    "distributed.worker",
    "distributed.utils",
    "distributed.client",
    "tornado.application",
)

_NOISE_SUBSTRINGS = (
    "has been closed",
    "Connection to tcp://",
    "Closing dangling",
    "Event loop was unresponsive",
    "Worker exceeded",
    "Scheduler is unable to accept new work",
    "Timed out trying to connect",
)


class _DistributedNoiseFilter(logging.Filter):
    """Drop benign distributed INFO chatter regardless of level changes."""

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        if record.levelno >= logging.WARNING:
            return True
        msg = record.getMessage()
        return not any(s in msg for s in _NOISE_SUBSTRINGS)


def _install_distributed_noise_filter() -> None:
    """Attach _DistributedNoiseFilter to all distributed.* loggers and set ERROR level."""
    _filter = _DistributedNoiseFilter()
    for name in _DISTRIBUTED_NOISE_LOGGERS:
        _lg = logging.getLogger(name)
        _lg.setLevel(logging.WARNING)
        if not any(isinstance(f, _DistributedNoiseFilter) for f in _lg.filters):
            _lg.addFilter(_filter)


# Monkey-patch logging.config.dictConfig so that whenever distributed calls it
# (during LocalCluster creation/teardown) our filter is re-applied afterwards.
# distributed calls dictConfig to reset log levels from dask config, which
# wipes any filters we added.  The patch ensures they're always restored.
_orig_dictConfig = _logging_config.dictConfig


def _noise_aware_dictConfig(config: dict) -> None:  # type: ignore[type-arg]
    _orig_dictConfig(config)
    try:
        _install_distributed_noise_filter()
    except Exception:
        pass


_logging_config.dictConfig = _noise_aware_dictConfig


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
    # Set all distributed.* loggers to WARNING (filter handles the rest).
    _install_distributed_noise_filter()

    # Silence Dask warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="distributed")
    warnings.filterwarnings("ignore", message=".*Event loop was unresponsive.*")

    # Global Dask configuration.
    # The "logging" key is read by distributed's setup_logging() when it calls
    # logging.config.dictConfig — use a flat {loggerName: level} mapping.
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
                "distributed": "error",
                "distributed.core": "error",
                "distributed.comm": "error",
                "distributed.nanny": "error",
                "distributed.scheduler": "error",
                "distributed.worker": "error",
                "distributed.utils": "error",
                "distributed.client": "error",
                "tornado.application": "error",
            },
        }
    )
