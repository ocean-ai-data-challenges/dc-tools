from argparse import Namespace
import multiprocessing
import os
import psutil
from typing import Optional

import dask
from dask.distributed import LocalCluster
from loguru import logger


def get_optimal_workers():
    """Détermine le nombre optimal de workers basé sur les ressources système."""
    num_cores = multiprocessing.cpu_count()
    available_memory = psutil.virtual_memory().available / 1e9  # Mémoire dispo (Go)

    # On ajuste le nombre de workers 
    max_workers_by_memory = int(available_memory // 5)
    optimal_workers = min(num_cores, max_workers_by_memory)

    return max(1, optimal_workers)  # Au moins 1 worker


def get_optimal_memory_limit():
    """Calcule une limite de mémoire basée sur la RAM disponible."""
    total_memory = psutil.virtual_memory().total / 1e9  # RAM totale en Go
    return f"{int(total_memory * 0.7)}GB"  # On garde 30% de marge


def setup_dask(args: Optional[Namespace] = None):
    """Configure automatiquement Dask en fonction des ressources disponibles."""
    # Déterminer mémoire et CPU disponibles
    num_workers = 1 ######################## TODO : REMOVE get_optimal_workers()
    memory_limit = get_optimal_memory_limit()

    os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
    #dask.config.set(get=dask.async.get_sync())
    dask.config.set(scheduler='threads')
    dask.config.set({"temporary-directory": "/tmp/dask",
                     "distributed.worker.memory.target": 0.6,
                     "distributed.worker.memory.pause": 0.7,
                     "distributed.worker.memory.spill": 0.8,
                     "distributed.worker.memory.terminate": 0.95,

                    # Réduire le parallélisme pour éviter trop de requêtes simultanées
                    "distributed.comm.timeouts.tcp": '300s',
                    
                    # Paramètres spécifiques pour S3
                    "distributed.worker.connections.outgoing": 2,  # Limite les connexions
                    "distributed.worker.connections.incoming": 2,
                    
                    # Chunking plus gros pour réduire le nombre de requêtes
                    "array.chunk-size": "256MB",  # Plus gros chunks = moins de requêtes
    })
    dask.config.set({"distributed.nanny.pre-spawn-environ.MALLOC_TRIM_THRESHOLD_": 1})
    cluster = LocalCluster(n_workers=num_workers,
        threads_per_worker=1,
        memory_limit=memory_limit,
    )
    logger.info(f"Dask tourne sur CPU avec {num_workers} workers et {memory_limit} de mémoire")

    return cluster
