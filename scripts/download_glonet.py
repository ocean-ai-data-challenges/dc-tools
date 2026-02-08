#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
Script to list, download and consolidate Zarr files from an S3 bucket (Wasabi/MinIO).

Files are saved progressively in a local directory
to limit memory footprint.
"""

import os
import yaml
import argparse
from datetime import datetime, timedelta
from pathlib import Path

import dask
from loguru import logger
import fsspec

from oceanbench.core.distributed import DatasetProcessor



def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Download and consolidate Zarr files from S3."
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to the YAML config file (dc2.yaml)"
    )
    parser.add_argument(
        "--dataset", type=str, required=True,
        help="Nom du dataset à traiter (ex: glonet)"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory for Zarr files"
    )
    return parser.parse_args()

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def get_s3_params(config, dataset_name):
    """Extract S3 parameters from configuration."""
    source_cfg = next((s for s in config["sources"] if s["dataset"] == dataset_name), None)
    assert source_cfg is not None, f"Dataset '{dataset_name}' not found in config file."
    s3_bucket = source_cfg["s3_bucket"]
    s3_folder = source_cfg["s3_folder"]
    endpoint_url = source_cfg.get("url") or source_cfg.get("endpoint_url")
    key = source_cfg.get("s3_key", None)
    secret = source_cfg.get("s3_secret_key", None)
    return s3_bucket, s3_folder, endpoint_url, key, secret


def create_fs(endpoint_url, s3_bucket, s3_folder):
    """Create filesystem object for S3 access."""
    client_kwargs = {'endpoint_url': endpoint_url} if endpoint_url else None
    fs = fsspec.filesystem(
        's3',
        anon=True,
        client_kwargs=client_kwargs,
    )
    return fs


def list_zarr_files(endpoint_url, s3_bucket, s3_folder, start_date):
    """
    Generate list of Zarr files by date.

    (e.g: .../20240103.zarr)
    """
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
    logger.info(f"{len(list_f)} Zarr files found")
    return list_f



def download_and_consolidate_zarr(remote_path, fs, output_dir):
    """Download and consolidate remote Zarr file to local directory."""
    file_name = Path(remote_path).name
    local_path = str(Path(output_dir) / file_name)
    logger.info(f"Downloading and consolidating {remote_path} → {local_path}")

    # Open the remote store for reading via fsspec
    #store = fs.get_mapper(remote_path)
    try:
        #ds = xr.open_zarr(store, consolidated=True)
        #ds.to_zarr(local_path, mode="w", consolidated=True)
        fs.get(remote_path, local_path, recursive=True)
        logger.info(f"File saved: {local_path}")

    except Exception as exc:
        logger.error(f"Error while processing {remote_path}: {exc}")



'''def download_and_consolidate_zarr(remote_path, fs, output_dir):
    # Download and consolidate a remote Zarr file into the local directory
    file_name = Path(remote_path).name
    local_path = str(Path(output_dir) / file_name)
    logger.info(f"Downloading and consolidating {remote_path} → {local_path}")

    # Open the remote store for reading
    #   store = fs.get_mapper(remote_path)
    try:
        #ds = xr.open_zarr(store, consolidated=True)
        ds = xr.open_zarr(remote_path)
        # Progressive save to limit RAM
        ds.to_zarr(local_path, mode="w", consolidated=True)
        logger.info(f"File saved: {local_path}")
    except Exception as exc:
        logger.error(f"Error while processing {remote_path}: {exc}")'''

def main():
    """Main execution function."""
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


    # Start the local Dask client
    dataset_processor = DatasetProcessor(
        distributed=True, n_workers=6,
        threads_per_worker=1,
        memory_limit="8GB"
    )
    # Submit tasks in parallel
    delayed_tasks = []
    for remote_path in zarr_files:
        delayed_tasks.append(dask.delayed(download_and_consolidate_zarr)(
            remote_path, fs, args.output_dir
        ))
    _ = dataset_processor.compute_delayed_tasks(delayed_tasks)


    logger.info("Processing completed.")

if __name__ == "__main__":
    main()
