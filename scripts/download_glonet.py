#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
Script pour lister, télécharger et consolider des fichiers Zarr depuis un bucket S3 (Wasabi/MinIO).
Les fichiers sont sauvegardés progressivement dans un répertoire local pour limiter l'empreinte mémoire.
"""

import os
import sys
import yaml
import argparse
from datetime import datetime, timedelta
from pathlib import Path

import dask
from dask.distributed import Client, as_completed
from loguru import logger
import fsspec
import zarr
import xarray as xr

from oceanbench.core.distributed import DatasetProcessor



def parse_args():
    parser = argparse.ArgumentParser(description="Télécharge et consolide des fichiers Zarr depuis S3.")
    parser.add_argument("--config", type=str, required=True, help="Chemin du fichier YAML de config (dc2.yaml)")
    parser.add_argument("--dataset", type=str, required=True, help="Nom du dataset à traiter (ex: glonet)")
    parser.add_argument("--output_dir", type=str, required=True, help="Répertoire de sortie pour les fichiers Zarr")
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def get_s3_params(config, dataset_name):
    source_cfg = next((s for s in config["sources"] if s["dataset"] == dataset_name), None)
    assert source_cfg is not None, f"Dataset '{dataset_name}' not found in config file."
    s3_bucket = source_cfg["s3_bucket"]
    s3_folder = source_cfg["s3_folder"]
    endpoint_url = source_cfg.get("url") or source_cfg.get("endpoint_url")
    key = source_cfg.get("s3_key", None)
    secret = source_cfg.get("s3_secret_key", None)
    return s3_bucket, s3_folder, endpoint_url, key, secret


def create_fs(endpoint_url, s3_bucket, s3_folder):
    client_kwargs = {'endpoint_url': endpoint_url} if endpoint_url else None
    fs = fsspec.filesystem(
        's3',
        anon=True,
        client_kwargs=client_kwargs,
    )
    return fs


def list_zarr_files(endpoint_url, s3_bucket, s3_folder, start_date):
    # Génère la liste des fichiers Zarr par date (ex: .../20240103.zarr)
    date = datetime.strptime(start_date, "%Y%m%d")
    #end_dt = datetime.strptime(end_date, "%Y%m%d")
    list_f = []
    while True:
        if date.year < 2025:
            date_str = date.strftime("%Y%m%d")
            '''list_f.append(
                f"{endpoint_url}/{s3_bucket}/{s3_folder}/{date_str}.zarr"
                #f"https://minio.dive.edito.eu/project-glonet/public/glonet_reforecast_2024/{date_str}.zarr"
                #f"s3://project-glonet/public/glonet_reforecast_2024/{date_str}.zarr"
            )'''
            list_f.append(
                f"s3://{s3_bucket}/{s3_folder}/{date_str}.zarr"
            )
            date = date + timedelta(days=7)
        else:
            break
    logger.info(f"{len(list_f)} fichiers Zarr trouvés")
    return list_f



def download_and_consolidate_zarr(remote_path, fs, output_dir):
    file_name = Path(remote_path).name
    local_path = str(Path(output_dir) / file_name)
    logger.info(f"Téléchargement et consolidation de {remote_path} → {local_path}")

    # Ouvre le store distant en lecture via fsspec
    #store = fs.get_mapper(remote_path)
    try:
        #ds = xr.open_zarr(store, consolidated=True)
        #ds.to_zarr(local_path, mode="w", consolidated=True)
        fs.get(remote_path, local_path, recursive=True)
        logger.info(f"Fichier sauvegardé : {local_path}")

    except Exception as exc:
        logger.error(f"Erreur lors du traitement de {remote_path}: {exc}")



'''def download_and_consolidate_zarr(remote_path, fs, output_dir):
    # Télécharge et consolide un fichier Zarr distant dans le répertoire local
    file_name = Path(remote_path).name
    local_path = str(Path(output_dir) / file_name)
    logger.info(f"Téléchargement et consolidation de {remote_path} → {local_path}")

    # Ouvre le store distant en lecture
    #   store = fs.get_mapper(remote_path)
    try:
        #ds = xr.open_zarr(store, consolidated=True)
        ds = xr.open_zarr(remote_path)
        # Sauvegarde progressive pour limiter la RAM
        ds.to_zarr(local_path, mode="w", consolidated=True)
        logger.info(f"Fichier sauvegardé : {local_path}")
    except Exception as exc:
        logger.error(f"Erreur lors du traitement de {remote_path}: {exc}")'''

def main():
    args = parse_args()
    config = load_config(args.config)
    os.makedirs(args.output_dir, exist_ok=True)

    s3_bucket, s3_folder, endpoint_url, key, secret = get_s3_params(config, args.dataset)
    fs = create_fs(endpoint_url, s3_bucket, s3_folder)

    start_date = "20240103"

    zarr_files = list_zarr_files(
        endpoint_url,
        s3_bucket, s3_folder,
        start_date,
    )
    # 'https://minio.dive.edito.eu/project-oceanbench/public/glonet_full_2024/20240103.zarr'
    # 'https://minio.dive.edito.eu/project-oceanbench/public/glonet_full_2024/20240103.zarr'


    # Démarrer le client Dask local
    dataset_processor = DatasetProcessor(
        distributed=True, n_workers=6,
        threads_per_worker=1,
        memory_limit="8GB"
    )
    # Soumettre les tâches en parallèle
    delayed_tasks = []
    for remote_path in zarr_files:
        delayed_tasks.append(dask.delayed(download_and_consolidate_zarr)(
            remote_path, fs, args.output_dir
        ))
    _ = dataset_processor.compute_delayed_tasks(delayed_tasks)


    logger.info("Traitement terminé.")

if __name__ == "__main__":
    main()
