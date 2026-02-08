"""Grid interpolation script for spatial data processing."""

import argparse
from argparse import Namespace
import os
import dask
import pandas as pd
from typing import Any

from oceanbench.core.distributed import DatasetProcessor
import yaml

from dctools.data.connection.connection_manager import clean_for_serialization
from dctools.data.coordinates import TARGET_DIM_RANGES, TARGET_DEPTH_VALS
from dctools.data.datasets.dataset import get_dataset_from_config
from dctools.data.datasets.dataset_manager import MultiSourceDatasetManager
from dctools.metrics.evaluator import (
    CONNECTION_CONFIG_REGISTRY, CONNECTION_REGISTRY,
)
from dctools.processing.interpolation import interpolate_dataset
from dctools.utilities.misc_utils import deep_copy_object



def clean_namespace(namespace: Namespace) -> Namespace:
    """Clean namespace by removing unpicklable attributes."""
    ns = Namespace(**vars(namespace))
    # Removes unpicklable attributes
    for key in ['dask_cluster', 'fs', 'dataset_processor', 'client', 'session']:
        if hasattr(ns, key):
            delattr(ns, key)
    # Also clean objects in ns.params if present
    if hasattr(ns, "params"):
        for key in ['fs', 'client', 'session', 'dataset_processor']:
            if hasattr(ns.params, key):
                delattr(ns.params, key)
    return ns

def interpolate_cmems_to_zarr(
    source_config: dict,
    file_path: str,
    variables: list,
    out_grid: dict,
    output_dir: str,
    start_date: str,
    end_date: str,
    transform: Any,
    weights_filepath: str = None,
    argo_index: Any = None,
):
    """Interpolates CMEMS variables on a target grid and saves daily to Zarr.

    Args:
        cmems_params (dict): parameters for CMEMSManager (dataset_id, credentials, etc.)
        variables (list): list of variables to interpolate
        out_grid (dict): dict of target coordinates (lat, lon, [depth])
        output_dir (str): output directory
        start_date (str): start date (YYYY-MM-DD)
        end_date (str): end date (YYYY-MM-DD)
        weights_filepath (str): path to xESMF weights file (optional)
    """
    #os.makedirs(output_dir, exist_ok=True)
    #dates = pd.date_range(start=start_date, end=end_date, freq="D")
    protocol = source_config.protocol
    delattr(source_config, "protocol")

    config_cls = CONNECTION_CONFIG_REGISTRY[protocol]
    connection_cls = CONNECTION_REGISTRY[protocol]
    config = config_cls(**vars(source_config))

    if protocol == 'cmems':
        if hasattr(config.params, 'fs'):
            try:
                if hasattr(config.params.fs, '_session'):
                    if hasattr(config.params.fs._session, 'close'):
                        config.params.fs._session.close()
            except Exception:
                pass
            delattr(config.params, "fs")


    if protocol == "argo":
        connection_manager = connection_cls(config, argo_index=argo_index)
    else:
        connection_manager = connection_cls(config)
    open_func = connection_manager.open

    if protocol == "cmems":
        # cmems not compatible with Dask workers (pickling errors)
        with dask.config.set(scheduler='synchronous'):
            ds = open_func(file_path)
            # Selects variables to interpolate
            ds_sel = ds[variables].copy()
            ds_transform = transform(ds_sel)
            # Interpolation
            ds_interp = interpolate_dataset(
                ds=ds_transform,
                target_grid=out_grid,
                dataset_processor=None,
                weights_filepath=weights_filepath,
                interpolation_lib='xesmf',
            )
    else:
        # Selects variables to interpolate
        ds = open_func(file_path)
        ds_sel = ds[variables].copy()
        ds_transform = transform(ds_sel)
        # Interpolation
        ds_interp = interpolate_dataset(
            ds=ds_transform,
            target_grid=out_grid,
            dataset_processor=None,
            weights_filepath=weights_filepath,
            interpolation_lib='xesmf',
        )

    if ds_interp is None:
        return None
    # Zarr file name
    if isinstance(file_path, pd.Timestamp):
        file_name = file_path.strftime("%Y%m%d")
    elif not isinstance(file_path, str):
        file_path = str(file_path)
    else:
        pass

    file_name = file_name.replace(" ", "_").replace(":", "_")
    zarr_path = os.path.join(output_dir, f"{file_name}.zarr")



    # Save
    with dask.config.set(scheduler='synchronous'):
        ds_interp.to_zarr(zarr_path, mode="w", consolidated=True)
    ds.close()
    print(f"Saved standardized dataset to {zarr_path}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Interpolates CMEMS variables and saves daily to Zarr."
    )
    default_config = os.path.join(
        os.path.dirname(__file__), "interpolate_config.yaml"
    )
    parser.add_argument(
        "--config", type=str, default=default_config,
        help=f"Path to YAML configuration file (default: {default_config})"
    )
    parser.add_argument(
        "--source", type=str, required=True,
        help="Name of source to process (e.g. glorys)"
    )
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Directory for intermediate files"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output directory for Zarr files"
    )

    default_weights_filepath = os.path.join(os.path.dirname(__file__), "glorys_weights.nc")
    parser.add_argument("--weights_filepath", type=str, default=default_weights_filepath,
        help="Path to xESMF weights file (optional)")
    return parser.parse_args()


def build_dataset_from_config(
    args, source_config, dataset_processor, root_data_folder,
    root_catalog_folder, file_cache=None
):
    """Builds a dictionary of datasets from YAML config."""
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

def main():
    """Main entry point for grid interpolation."""
    # args = parse_args()

    config_name = "interpolate_config"
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
    weights_filepath = args.weights_filepath
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
        max_cache_files=args.max_cache_files,
    )
    manager.add_dataset(source_name, dataset)

    manager.filter_all_by_date(
        start=pd.to_datetime(start_date),
        end=pd.to_datetime(end_date),
    )

    # Build the catalog
    manager.build_catalogs()

    # Initialize the standardize_glorys transform
    transform = manager.get_transform(
        "standardize",
        dataset_alias=source_name,
        interp_ranges=TARGET_DIM_RANGES,
        weights_path=weights_filepath,
        depth_coord_vals=TARGET_DEPTH_VALS,
    )

    if hasattr(transform, 'dataset_processor'):
        delattr(transform, 'dataset_processor')

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
    out_grid = {
        "lat": TARGET_DIM_RANGES["lat"],
        "lon": TARGET_DIM_RANGES["lon"],
    }
    if "depth" in TARGET_DIM_RANGES:
        out_grid["depth"] = TARGET_DIM_RANGES["depth"]

    # Prepare tasks for workers
    delayed_tasks = []
    for _idx, row in catalog_df.iterrows():
        file_path = row["path"]

        # Create delayed task for worker
        task = dataset_processor.client.submit(
            interpolate_cmems_to_zarr,
            connection_params,
            file_path,
            variables,
            out_grid,
            output_dir,
            start_date,
            end_date,
            transform,
            weights_filepath,
            scattered_argo_index,
        )
        delayed_tasks.append(task)

    # Execute tasks in parallel and wait for results
    results = dataset_processor.client.gather(delayed_tasks)
    print(f"Interpolation completed for {len(results)} files.")

if __name__ == "__main__":
    main()
