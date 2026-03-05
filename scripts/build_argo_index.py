#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
Script pour construire l'index mensuel ARGO et le stocker sur S3/Wasabi.

Ce script utilise la nouvelle classe ArgoInterface pour :
1. Récupérer les fichiers ARGO mensuels via argopy
2. Créer des références Kerchunk compressées (JSON + Zstd)
3. Construire un index temporel optimisé (epoch int64)
4. Stocker les index sur S3/Wasabi pour un accès ultra-rapide

Usage:
    python scripts/build_argo_index.py --year 2024 --config dc/config/dc2.yaml
"""

import argparse
import os
import sys
from pathlib import Path

# Set environment variables before imports
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
os.environ["NETCDF4_DEACTIVATE_MPI"] = "1"
os.environ["NETCDF4_USE_FILE_LOCKING"] = "FALSE"
os.environ["HDF5_DISABLE_VERSION_CHECK"] = "1"
os.environ["ARGOPY_NETCDF_LOCKING"] = "FALSE"

from loguru import logger
import yaml

from dctools.data.connection.config import ARGOConnectionConfig
from dctools.data.connection.argo_data import ArgoInterface


def load_argo_config(config_path: str) -> dict:
    """
    Charge la configuration ARGO depuis un fichier YAML.

    Args:
        config_path: Chemin vers le fichier de configuration

    Returns:
        dict: Configuration ARGO extraite
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Trouver la section argo_profiles
    argo_config = None
    for source in config.get("sources", []):
        if source.get("dataset") == "argo_profiles":
            argo_config = source
            break

    if not argo_config:
        raise ValueError("No 'argo_profiles' dataset found in config file")

    # Ajouter catalog_connection pour les clés S3
    catalog_conn = config.get("catalog_connection", {})

    # Construire les paramètres pour ARGOConnectionConfig
    params = {
        "init_type": "from_scratch",
        "local_root": config.get("data_directory", "/tmp/argo"),
        "s3_bucket": argo_config.get("s3_bucket") or catalog_conn.get("s3_bucket"),
        "s3_folder": argo_config.get("s3_folder"),
        "s3_key": argo_config.get("s3_key") or catalog_conn.get("s3_key"),
        "s3_secret_key": argo_config.get("s3_secret_key") or catalog_conn.get("s3_secret_key"),
        "endpoint_url": argo_config.get("url") or catalog_conn.get("url"),
        "variables": argo_config.get("variables"),
        "depth_values": (config.get("target_dimensions") or {}).get("depth"),
        "chunks": argo_config.get("chunks", {"N_PROF": 2000}),
        "file_cache": None,
        "dataset_processor": None,
        "filter_values": {
            "start_time": config.get("start_time", "2024-01-01"),
            "end_time": config.get("end_time", "2024-12-31"),
            "min_lon": config.get("min_lon", -180),
            "max_lon": config.get("max_lon", 180),
            "min_lat": config.get("min_lat", -90),
            "max_lat": config.get("max_lat", 90),
        },
    }

    return params


def build_index(args):
    """
    Construit l'index mensuel ARGO.

    Args:
        args: Arguments de ligne de commande
    """
    logger.info("=" * 80)
    logger.info("ARGO Index Builder")
    logger.info("=" * 80)

    # Charger la configuration
    logger.info(f"Loading configuration from {args.config}")
    params = load_argo_config(args.config)

    # Override local_root if output_dir is provided
    if args.output_dir:
        logger.info(f"Overriding local_root with: {args.output_dir}")
        params["local_root"] = args.output_dir

    # Créer l'interface ARGO
    logger.info("Creating ARGO interface...")
    config = ARGOConnectionConfig(params)
    argo_interface = ArgoInterface.from_config(config)

    logger.info(f"Base path: {argo_interface.base_path}")
    logger.info(f"Variables: {argo_interface.variables}")
    logger.info(f"S3 endpoint: {params.get('endpoint_url')}")

    # Construire les index mensuels
    logger.info(f"Building monthly indexes for {args.start_year} to {args.end_year}")
    logger.info(f"Temporary directory: {args.temp_dir}")
    logger.info(f"Number of workers: {args.workers}")

    try:
        argo_interface.build_multi_year_monthly(
            start_year=args.start_year,
            end_year=args.end_year,
            temp_dir=args.temp_dir,
            n_workers=args.workers,
        )

        logger.info("=" * 80)
        logger.info("✓ Index building completed successfully!")
        logger.info("=" * 80)
        logger.info(f"Master index stored at: {argo_interface.base_path}/master_index.json")
        logger.info(f"Monthly indexes stored at: {argo_interface.base_path}/YYYY_MM.json.zst")

    except Exception as e:
        logger.error(f"Failed to build ARGO index: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


def main():
    """Point d'entrée principal."""
    parser = argparse.ArgumentParser(
        description="Build monthly ARGO index with Kerchunk references"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="dc/config/dc2.yaml",
        help="Path to configuration file (default: dc/config/dc2.yaml)",
    )

    parser.add_argument(
        "--start-year", type=int, default=2024, help="Start year for indexing (default: 2024)"
    )

    parser.add_argument(
        "--end-year", type=int, default=2024, help="End year for indexing (default: 2024)"
    )

    parser.add_argument(
        "--temp-dir",
        type=str,
        default="/tmp/argo_refs",
        help="Temporary directory for intermediate files (default: /tmp/argo_refs)",
    )

    parser.add_argument(
        "--workers", type=int, default=8, help="Number of parallel workers (default: 8)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for the index (overrides config)",
    )

    args = parser.parse_args()

    # Créer le répertoire temporaire si nécessaire
    Path(args.temp_dir).mkdir(parents=True, exist_ok=True)

    build_index(args)


if __name__ == "__main__":
    main()
