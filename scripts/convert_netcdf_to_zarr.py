#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
Script pour convertir des fichiers NetCDF vers Zarr.
Utilise dask pour le parallélisme et optimise la mémoire.
"""

import os
import sys
from pathlib import Path
import pandas as pd
from typing import Any, List, Optional
import argparse
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing

import dask
import dask.array as da
from dask.distributed import Client, LocalCluster
from loguru import logger
from oceanbench.core.distributed import DatasetProcessor
import xarray as xr
import yaml
import zarr
from tqdm import tqdm

from dctools.data.datasets.dataset_manager import MultiSourceDatasetManager
from dctools.data.connection.connection_manager import clean_for_serialization, create_worker_connect_config
from dctools.data.coordinates import TARGET_DIM_RANGES, TARGET_DEPTH_VALS
from dctools.data.datasets.dataset import get_dataset_from_config
from dctools.data.datasets.dataset_manager import MultiSourceDatasetManager
from dctools.dcio.saver import progressive_zarr_save
from dctools.utilities.misc_utils import deep_copy_object
from dctools.utilities.xarray_utils import netcdf_to_zarr


def parse_args():
    parser = argparse.ArgumentParser(description="Interpole les variables CMEMS et sauvegarde en Zarr journalier.")
    '''default_config = os.path.join(os.path.dirname(__file__), "interpolate_config.yaml")
    parser.add_argument(
        "--config", type=str, default=default_config,
        help=f"Chemin du fichier de configuration YAML (défaut: {default_config})"
    )'''
    parser.add_argument("--source", type=str, required=True, help="Nom de la source à traiter (ex: glorys)")
    parser.add_argument("--data_dir", type=str, required=True, help="Répertoire de stockage des fichiers intermédiaires")
    parser.add_argument("--output_dir", type=str, required=True, help="Répertoire de sortie des fichiers Zarr")

    return parser.parse_args()


def convert_single_file(
    ds: xr.Dataset, 
    output_path: str, 
    chunk_size: dict = None,
    compression: str = 'zlib',
    compression_level: int = 3
) -> bool:
    """
    Convertit un seul fichier NetCDF vers Zarr.
    
    Args:
        input_path: Chemin vers le fichier NetCDF
        output_path: Chemin vers le fichier Zarr de sortie
        chunk_size: Dictionnaire des tailles de chunks par dimension
        compression: Type de compression ('gzip', 'lz4', 'blosc')
        compression_level: Niveau de compression (1-9)
    
    Returns:
        bool: True si succès, False sinon
    """
    try:
        # Ouvrir avec chunks optimisés
            
        # Configuration de l'encodage Zarr optimisé
        encoding = {}
        for var in ds.data_vars:
            chunks = ds[var].chunks if hasattr(ds[var], 'chunks') else None
            if chunks is not None:
                # Si chunks est une tuple de tuples, aplatir
                if isinstance(chunks, tuple) and isinstance(chunks[0], tuple):
                    chunks = tuple(c[0] for c in chunks)
            encoding[var] = {
                'compressor': zarr.Blosc(cname=compression, clevel=compression_level),
                'chunks': chunks
            }
        
        # Conversion vers Zarr
        ds.to_zarr(
            output_path,
            mode='w',
            encoding=encoding,
            consolidated=True,  # Métadonnées consolidées pour de meilleures performances
            )
            
        return True
        
    except Exception as exc:
        logger.error(f"Erreur lors de la conversion: {exc}")
        return False

def convert_to_zarr_worker(
    source_config: dict,
    file_path: str,
    variables: list,
    output_dir: str,
    argo_index: Any = None,
):
    """
    Interpole les variables CMEMS sur une grille cible et sauvegarde chaque jour en Zarr.
    Args:
        cmems_params (dict): paramètres pour CMEMSManager (dataset_id, credentials, etc.)
        variables (list): liste des variables à interpoler
        out_grid (dict): dict des coordonnées cibles (lat, lon, [depth])
        output_dir (str): répertoire de sortie
        start_date (str): date de début (YYYY-MM-DD)
        end_date (str): date de fin (YYYY-MM-DD)
        weights_filepath (str): chemin vers le fichier de poids xESMF (optionnel)
    """
    protocol = source_config.protocol

    open_func = create_worker_connect_config(
        source_config,
        argo_index,
    )

    if protocol == "cmems":
        # cmems not compatible with Dask workers (pickling errors)
        with dask.config.set(scheduler='synchronous'):
            ds = open_func(file_path)
    else:
        # Sélectionne les variables à interpoler
        ds = open_func(file_path)

    input_name = Path(file_path).name
    output_name = str(Path(input_name).with_suffix(".zarr"))
    output_path = os.path.join(output_dir, output_name)
    try:
        zarr_path = netcdf_to_zarr(ds, output_path, overwrite=True)
    except Exception as e:
        pass

    # logger.info(f"Converted to zarr: {zarr_path}")


def estimate_optimal_chunks(sample_file: str) -> dict:
    """Estime la taille optimale des chunks basée sur un fichier d'exemple."""
    
    try:
        with xr.open_dataset(sample_file) as ds:
            chunks = {}
            
            for dim, size in ds.sizes.items():
                if dim == 'time':
                    chunks[dim] = min(size, 90)
                elif dim in ['lat', 'latitude', 'y']:
                    chunks[dim] = min(size, 200)
                elif dim in ['lon', 'longitude', 'x']:
                    chunks[dim] = min(size, 200)
                elif dim in ['depth', 'lev', 'level']:
                    # Pour la profondeur, garder toutes les couches ensemble
                    chunks[dim] = size
                else:
                    # Pour autres dimensions, chunks raisonnables
                    chunks[dim] = min(size, 100)
            
            logger.info(f"Chunks estimés: {chunks}")
            return chunks
            
    except Exception as exc:
        logger.warning(f"Impossible d'estimer les chunks: {exc}")
        return None


def find_netcdf_files(directory: str, pattern: str = "**/*.nc") -> List[str]:
    """Trouve tous les fichiers NetCDF dans un répertoire."""
    
    path = Path(directory)
    files = list(path.glob(pattern))
    logger.info(f"Trouvé {len(files)} fichiers NetCDF dans {directory}")
    
    return [str(f) for f in files]



def build_dataset_from_config(args, source_config, dataset_processor, root_data_folder, root_catalog_folder, file_cache=None):
    """
    Construit un dictionnaire de datasets à partir de la config YAML.
    """
    max_samples = args.max_samples
    use_catalog = True
    filter_values = {
        "start_time": args.start_time,
        "end_time": args.end_time,
        "min_lon": args.min_lon if args.min_lon is not None else -180,
        "max_lon": args.max_lon if args.max_lon is not None else 180,
        "min_lat": args.min_lat if args.min_lat is not None else -90,
        "max_lat": args.max_lat if args.max_lat is not None else 90,
    }
    #dataset_name = config.get("dataset")
    dataset = get_dataset_from_config(
        source=source_config,
        root_data_folder=root_data_folder,
        root_catalog_folder=root_catalog_folder,
        dataset_processor=dataset_processor,
        max_samples=max_samples,
        use_catalog=use_catalog,
        file_cache=file_cache,
        filter_values=filter_values,
    )
    return dataset

def clean_namespace(namespace: argparse.Namespace) -> argparse.Namespace:
    ns = argparse.Namespace(**vars(namespace))
    # Supprime les attributs non picklables
    for key in ['dask_cluster', 'fs', 'dataset_processor', 'client', 'session']:
        if hasattr(ns, key):
            delattr(ns, key)
    # Nettoie aussi les objets dans ns.params si présent
    if hasattr(ns, "params"):
        for key in ['fs', 'client', 'session', 'dataset_processor']:
            if hasattr(ns.params, key):
                delattr(ns.params, key)
    return ns

def main():
    config_name = "convert_to_zarr_config"
    config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"{config_name}.yaml",
    )

    args = parse_args()

    if config_path:
        config = None
        with open(config_path, 'r') as fp:
            config = yaml.safe_load(fp)
        for key, value in config.items():
            vars(args)[key] = value

    source_name = args.source
    output_dir = args.output_dir
    start_date = args.start_time
    end_date = args.end_time
    os.makedirs(output_dir, exist_ok=True)

    # Trouver la source demandée dans la config
    source_cfg = next((s for s in args.sources if s["dataset"] == source_name), None)
    assert source_cfg is not None, f"Source '{source_name}' not found in config file."

    root_data_folder = args.data_dir
    root_catalog_folder = os.path.join(root_data_folder, "catalogs")
    os.makedirs(root_catalog_folder, exist_ok=True)

    # Créer le DatasetProcessor (distributed=True pour parallélisation)
    dataset_processor = DatasetProcessor(distributed=True, n_workers=args.n_parallel_workers,
                                        threads_per_worker=args.nthreads_per_worker,
                                        memory_limit=args.memory_limit_per_worker)

    # Construire tous les datasets à partir de la config
    dataset = build_dataset_from_config(
        args, source_cfg, dataset_processor, root_data_folder, root_catalog_folder
    )

    # Ajouter au MultiSourceDatasetManager
    manager = MultiSourceDatasetManager(
        dataset_processor=dataset_processor,
        target_dimensions=TARGET_DIM_RANGES,
        time_tolerance=pd.Timedelta(hours=args.delta_time),
        list_references=[source_name],
        max_cache_files=args.max_cache_files
    )
    manager.add_dataset(source_name, dataset)

    manager.filter_all_by_date(
        start=pd.to_datetime(start_date),
        end=pd.to_datetime(end_date),
    )

    # Construire le catalogue
    manager.build_catalogs()

    all_managers,_, all_connection_params = manager.get_config()
    connection_params = all_connection_params.get(source_name, None)
    connection_manager = all_managers.get(source_name, None)
    connection_params = deep_copy_object(
        connection_params, skip_list=['dataset_processor', 'fs']
    )
    connection_params = clean_for_serialization(connection_params)
    connection_params = clean_namespace(connection_params)
    connection_params.dataset_processor = None

    argo_index = None
    if hasattr(connection_manager, 'argo_index'):
        argo_index = connection_manager.get_argo_index()
    if argo_index is not None:
        scattered_argo_index = dataset_processor.scatter_data(
            argo_index,
            broadcast_item = True,
        )
    else:
        scattered_argo_index = None

    # Récupérer le catalogue du dataset
    catalog_df = dataset.get_catalog().get_dataframe()
    variables = source_cfg.get("keep_variables", None)

    # Préparer les tâches pour les workers
    delayed_tasks = []
    for idx, row in catalog_df.iterrows():
        file_path = row["path"]

        # Créer la tâche delayed pour le worker
        task = dataset_processor.client.submit(
            convert_to_zarr_worker,
            connection_params,
            file_path,
            variables,
            output_dir,
            scattered_argo_index,
        )
        delayed_tasks.append(task)

    # Exécuter les tâches en parallèle et attendre les résultats
    results = dataset_processor.client.gather(delayed_tasks)
    print(f"  terminée pour {len(results)} fichiers.")

if __name__ == "__main__":
    main()