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
    worker_memory = float(get_optimal_memory_limit().rstrip('GB'))
    # Adjust worker count
    max_workers_by_memory = int(available_memory // worker_memory)
    optimal_workers = min(num_cores, max_workers_by_memory)

    return max(1, optimal_workers)  # At least 1 worker


def get_optimal_memory_limit():
    """Calculate a memory limit based on available RAM."""
    total_memory = psutil.virtual_memory().total / 1e9  # Total RAM in GB
    # Limit to 4GB max per worker to avoid memory leaks
    available_memory = min(6, int(total_memory * 0.25))  # Reduction to 30% and max 8GB
    return f"{available_memory}GB"

def setup_dask(args: Optional[Namespace] = None):
    """Automatically configure Dask based on available resources."""
    # Determine available memory and CPU - REDUCED to avoid NetCDF conflicts
    num_workers = get_optimal_workers()
    memory_limit = get_optimal_memory_limit()

    # Configuration to avoid NetCDF/HDF5 conflicts
    os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
    os.environ['NETCDF4_USE_FILE_LOCKING'] = 'FALSE'
    os.environ['HDF5_DISABLE_VERSION_CHECK'] = '1'

    # Use threads instead of processes to avoid conflicts
    dask.config.set(scheduler='threads')
    dask.config.set({"temporary-directory": "/tmp/dask"})

    cluster = LocalCluster(
        n_workers=num_workers,
        threads_per_worker=1,
        memory_limit=memory_limit,
        processes=False,  # Force threads to avoid process overhead/conflicts
        silence_logs=False,  # Keep logs for debugging
    )
    logger.info(
        f"Dask is running on {num_workers} CPU workers, each with {memory_limit} of memory."
    )

    configure_dask_logging()

    return cluster


def configure_dask_logging():
    """Configure Dask logs to be quiet."""
    # Remove noisy Dask-specific loggers
    dask_loggers = [
        'distributed',
        'distributed.core',
        'distributed.worker',
        'distributed.scheduler',
        'distributed.nanny',
        'distributed.comm',
        'distributed.utils',
        'distributed.client',
        'tornado.application'
    ]

    for logger_name in dask_loggers:
        logging.getLogger(logger_name).setLevel(logging.ERROR)

    # Silence Dask warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='distributed')
    warnings.filterwarnings('ignore', message='.*Event loop was unresponsive.*')

    # CRITICAL: Minimize xarray file cache to prevent memory accumulation
    # Note: Some xarray versions don't allow 0, so we use 1 (minimal cache)
    import xarray as xr
    xr.set_options(file_cache_maxsize=1)

    # Global Dask configuration
    dask.config.set({
        'distributed.worker.daemon': False,
        'distributed.comm.timeouts.tcp': '60s',
        'distributed.comm.timeouts.connect': '60s',
        # Memory management thresholds - trigger automatic cleanups
        'distributed.worker.memory.target': 0.6,  # Start cleanup at 60%
        'distributed.worker.memory.spill': 0.7,   # Spill to disk at 70%
        'distributed.worker.memory.pause': 0.8,   # Pause at 80%
        'distributed.worker.memory.terminate': False,
        # Force immediate transition to old generation for easier GC
        'distributed.worker.memory.recent-to-old-time': '0s',

        'logging': {
            'distributed': {
                '': 'error',            # root "distributed" logger
                'worker': 'error'       # sub-logger: distributed.worker
            }
        }
    })

