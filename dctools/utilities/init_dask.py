from argparse import Namespace
import json
import multiprocessing
import os
import psutil

import dask
from dask.distributed import Client, LocalCluster #, LocalCUDACluster
from loguru import logger


# Détection automatique du nombre de workers
def get_optimal_workers():
    """Détermine le nombre optimal de workers basé sur les ressources système."""
    num_cores = multiprocessing.cpu_count()
    available_memory = psutil.virtual_memory().available / 1e9  # Mémoire dispo (Go)

    # On ajuste le nombre de workers 
    max_workers_by_memory = int(available_memory // 2)
    optimal_workers = min(num_cores, max_workers_by_memory)
    
    return max(1, optimal_workers)  # Au moins 1 worker


# Détection automatique de la mémoire disponible

def get_optimal_memory_limit():
    """Calcule une limite de mémoire basée sur la RAM disponible."""
    total_memory = psutil.virtual_memory().total / 1e9  # RAM totale en Go
    return f"{int(total_memory * 0.8)}GB"  # On garde 20% de marge

def get_best_scheduler():
    """Choisit le scheduler optimal selon la charge CPU/GPU détectée."""
    num_cores = multiprocessing.cpu_count()
    
    if num_cores > 8:
        return "processes"  # Charge lourde, on privilégie le multiprocessing
    else:
        return "threads"  # Petite charge, le threading est suffisant

def setup_dask(args: Namespace):
    """Configure automatiquement Dask en fonction des ressources disponibles."""

    # Déterminer mémoire et CPU disponibles
    num_workers = get_optimal_workers()
    memory_limit = get_optimal_memory_limit()

    os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
    #dask.config.set(get=dask.async.get_sync())
    dask.config.set(scheduler='threads')

    cluster = LocalCluster(n_workers=num_workers,
                        threads_per_worker=1,
                        #memory_target_fraction=0.5,
                        memory_limit=memory_limit)
    
    logger.info(f"Dask tourne sur CPU avec {num_workers} workers et {memory_limit} de mémoire")


    return cluster
