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
    """Détermine le nombre optimal de workers basé sur les ressources système."""
    num_cores = multiprocessing.cpu_count()
    available_memory = psutil.virtual_memory().available / 1e9  # Mémoire dispo (Go)
    worker_memory = float(get_optimal_memory_limit().rstrip('GB'))
    # On ajuste le nombre de workers 
    max_workers_by_memory = int(available_memory // worker_memory)
    optimal_workers = 4 #min(num_cores, max_workers_by_memory)

    return max(1, optimal_workers)  # Au moins 1 worker


def get_optimal_memory_limit():
    """Calcule une limite de mémoire basée sur la RAM disponible."""
    total_memory = psutil.virtual_memory().total / 1e9  # RAM totale en Go
    # Limiter à 4GB max par worker pour éviter les fuites mémoire
    available_memory = min(6, int(total_memory * 0.25))  # Réduction à 30% et max 8GB
    return f"{available_memory}GB"

def setup_dask(args: Optional[Namespace] = None):
    """Configure automatiquement Dask en fonction des ressources disponibles."""
    # Déterminer mémoire et CPU disponibles - RÉDUIT pour éviter les conflits NetCDF
    num_workers = get_optimal_workers()
    memory_limit = get_optimal_memory_limit()

    # Configuration pour éviter les conflits NetCDF/HDF5
    os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
    os.environ['NETCDF4_USE_FILE_LOCKING'] = 'FALSE'
    os.environ['HDF5_DISABLE_VERSION_CHECK'] = '1'
    
    # Utiliser threads au lieu de processes pour éviter les conflits
    dask.config.set(scheduler='threads')
    dask.config.set({"temporary-directory": "/tmp/dask"})
    
    cluster = LocalCluster(
        n_workers=num_workers,
        threads_per_worker=1,
        memory_limit=memory_limit,
        processes=False,  # Forcer l'utilisation de threads pour éviter overhead
        silence_logs=False,  # Garder les logs pour debugging
    )
    logger.info(f"Dask tourne sur CPU avec {num_workers} workers et {memory_limit} de mémoire")

    configure_dask_logging()

    return cluster


def configure_dask_logging():
    """Configure les logs Dask pour être silencieux."""
    
    # Supprimer les logs Dask spécifiques
    dask_loggers = [
        'distributed',
        'distributed.core',
        'distributed.worker',
        'distributed.scheduler',
        'distributed.comm',
        'distributed.utils',
        'distributed.client',
        'tornado.application'
    ]
    
    for logger_name in dask_loggers:
        logging.getLogger(logger_name).setLevel(logging.ERROR)
    
    # Supprimer les warnings Dask
    warnings.filterwarnings('ignore', category=UserWarning, module='distributed')
    warnings.filterwarnings('ignore', message='.*Event loop was unresponsive.*')
    
    # Configuration Dask globale

    dask.config.set({
        'distributed.worker.daemon': False,
        'distributed.comm.timeouts.tcp': '60s',
        'distributed.comm.timeouts.connect': '60s',
        'distributed.worker.memory.target': 0.8,
        'distributed.worker.memory.spill': 0.9,
        'distributed.worker.memory.pause': 0.95,
        'distributed.worker.memory.terminate': False,
        'logging': {
            'distributed': {
                '': 'error',            # root "distributed" logger
                'worker': 'error'       # sub-logger: distributed.worker
            }
        }
    })