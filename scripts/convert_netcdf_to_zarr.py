#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
Script to convert NetCDF files to Zarr.

Uses dask for parallelism and optimizes memory usage.
"""

import os
from pathlib import Path
import pandas as pd
from typing import Any, List
import argparse

import dask
from loguru import logger
from oceanbench.core.distributed import DatasetProcessor
import xarray as xr
import yaml
import zarr

from dctools.data.datasets.dataset_manager import MultiSourceDatasetManager
from dctools.data.connection.connection_manager import (
    clean_for_serialization,
    create_worker_connect_config
)
from dctools.data.coordinates import TARGET_DIM_RANGES
from dctools.data.datasets.dataset import get_dataset_from_config
from dctools.utilities.misc_utils import deep_copy_object
from dctools.utilities.xarray_utils import netcdf_to_zarr


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Interpolate CMEMS variables and save per-day Zarr outputs."
    )
    '''default_config = os.path.join(os.path.dirname(__file__), "interpolate_config.yaml")
    parser.add_argument(
        "--config", type=str, default=default_config,
        help=f"Chemin du fichier de configuration YAML (défaut: {default_config})"
    )'''
    parser.add_argument(
        "--source", type=str, required=True,
        help="Name of the source to process (e.g., glorys)"
    )
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Directory to store intermediate files"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output directory for Zarr files"
    )

    return parser.parse_args()


def convert_single_file(
    ds: xr.Dataset,
    output_path: str,
    chunk_size: dict = None,
    compression: str = 'zlib',
    compression_level: int = 3
) -> bool:
    """Convert a single NetCDF file to Zarr.

    Args:
    ds: xarray Dataset to convert
    output_path: Path to the output Zarr store
    chunk_size: Dict of chunk sizes per dimension
    compression: Compression type ('gzip', 'lz4', 'blosc')
    compression_level: Compression level (1-9)

    Returns:
        bool: True on success, False otherwise
    """
    try:
        # Open with optimized chunks

    # Optimized Zarr encoding configuration
        encoding = {}
        for var in ds.data_vars:
            chunks = ds[var].chunks if hasattr(ds[var], 'chunks') else None
            if chunks is not None:
                # If chunks is a tuple of tuples, flatten it
                if isinstance(chunks, tuple) and isinstance(chunks[0], tuple):
                    chunks = tuple(c[0] for c in chunks)
            encoding[var] = {
                'compressor': zarr.Blosc(cname=compression, clevel=compression_level),
                'chunks': chunks
            }

        # Convert to Zarr
        ds.to_zarr(
            output_path,
            mode='w',
            encoding=encoding,
            consolidated=True,  # Consolidated metadata for better performance
            )

        return True

    except Exception as exc:
        logger.error(f"Error during conversion: {exc}")
        return False

def convert_to_zarr_worker(
    source_config: dict,
    file_path: str,
    variables: list,
    output_dir: str,
    argo_index: Any = None,
):
    """Interpolate CMEMS variables onto a target grid and save each day to Zarr.

    Args:
        source_config (dict): parameters for CMEMSManager (dataset_id, credentials, etc.)
        file_path (str): path to the input file
        variables (list): list of variables to interpolate
        output_dir (str): output directory
        argo_index (Any): Argo index (optional)
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
        # Select variables to interpolate
        ds = open_func(file_path)

    input_name = Path(file_path).name
    output_name = str(Path(input_name).with_suffix(".zarr"))
    output_path = os.path.join(output_dir, output_name)
    try:
        netcdf_to_zarr(ds, output_path, overwrite=True)
    except Exception:
        pass

    # logger.info(f"Converted to zarr: {output_path}")


def estimate_optimal_chunks(sample_file: str) -> dict:
    """Estimate an optimal chunk size based on an example file."""
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
                    # For depth, keep all levels together
                    chunks[dim] = size
                else:
                    # For other dimensions, keep reasonable chunks
                    chunks[dim] = min(size, 100)

            logger.info(f"Estimated chunks: {chunks}")
            return chunks

    except Exception as exc:
        logger.warning(f"Couldn't estimate chunks: {exc}")
        return None


def find_netcdf_files(directory: str, pattern: str = "**/*.nc") -> List[str]:
    """Find all NetCDF files in a directory."""
    path = Path(directory)
    files = list(path.glob(pattern))
    logger.info(f"Found {len(files)} NetCDF files in {directory}")

    return [str(f) for f in files]



def build_dataset_from_config(
        args, source_config, dataset_processor, root_data_folder,
        root_catalog_folder, file_cache=None
):
    """Build a dataset dict from the YAML config.

    Wrapper around get_dataset_from_config used in scripts.
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
    """Clean namespace from unpicklable objects before sending to workers."""
    ns = argparse.Namespace(**vars(namespace))
    # Remove unpicklable attributes
    for key in ['dask_cluster', 'fs', 'dataset_processor', 'client', 'session']:
        if hasattr(ns, key):
            delattr(ns, key)
    # Also clean objects inside ns.params if present
    if hasattr(ns, "params"):
        for key in ['fs', 'client', 'session', 'dataset_processor']:
            if hasattr(ns.params, key):
                delattr(ns.params, key)
    return ns

def main():
    """Main execution function."""
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

    # Find the requested source in the config
    source_cfg = next((s for s in args.sources if s["dataset"] == source_name), None)
    assert source_cfg is not None, f"Source '{source_name}' not found in config file."

    root_data_folder = args.data_dir
    root_catalog_folder = os.path.join(root_data_folder, "catalogs")
    os.makedirs(root_catalog_folder, exist_ok=True)

    # Create DatasetProcessor (distributed=True for parallelism)
    dataset_processor = DatasetProcessor(distributed=True, n_workers=args.n_parallel_workers,
                                        threads_per_worker=args.nthreads_per_worker,
                                        memory_limit=args.memory_limit_per_worker)

    # Build the dataset from the config
    dataset = build_dataset_from_config(
        args, source_cfg, dataset_processor, root_data_folder, root_catalog_folder
    )

    # Add to the MultiSourceDatasetManager
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

    # Build the catalog
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

    # Retrieve the dataset catalog
    catalog_df = dataset.get_catalog().get_dataframe()
    variables = source_cfg.get("keep_variables", None)

    # Prepare tasks for workers
    delayed_tasks = []
    for _idx, row in catalog_df.iterrows():
        file_path = row["path"]

    # Create the task for the worker
        task = dataset_processor.client.submit(
            convert_to_zarr_worker,
            connection_params,
            file_path,
            variables,
            output_dir,
            scattered_argo_index,
        )
        delayed_tasks.append(task)

    # Run tasks in parallel and wait for results
    results = dataset_processor.client.gather(delayed_tasks)
    print(f"  completed for {len(results)} files.")

if __name__ == "__main__":
    main()
