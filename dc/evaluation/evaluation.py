#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Evaluator class."""

from argparse import Namespace
from glob import glob
import json
import os
# from typing import Any, Optional
import gc
from datetime import timedelta
import warnings

import geopandas as gpd
import pandas as pd
from dask.distributed import get_client
from loguru import logger
from oceanbench.core.distributed import DatasetProcessor
from shapely import geometry

from dctools.data.coordinates import (
    get_standardized_var_name,
    TARGET_DIM_RANGES,
)
from dctools.data.datasets.dataset import get_dataset_from_config
from dctools.data.datasets.dataloader import EvaluationDataloader
from dctools.data.datasets.dataset_manager import MultiSourceDatasetManager
from dctools.metrics.evaluator import Evaluator
from dctools.metrics.metrics import MetricComputer
from dctools.metrics.oceanbench_metrics import get_variable_alias
from dctools.utilities.file_utils import empty_folder
from dctools.utilities.init_dask import configure_dask_logging
from dctools.utilities.misc_utils import (
    make_serializable,
    nan_to_none,
    transform_in_place,
)

warnings.simplefilter("ignore", UserWarning)

class DC2Evaluation:
    """Class that manages evaluation of Data Challenge 2."""

    def __init__(self, arguments: Namespace) -> None:
        """Init class.

        Args:
            arguments (str): Namespace with config.
        """
        self.args = arguments

        # Configure silence for Dask
        configure_dask_logging()

        '''self.dataset_references = {
            "glonet_edito": [
                "jason3_edito", "saral_edito", "swot_edito", "glorys_edito", "argo_profiles", "argo_velocities_edito",
                "SSS_fields_edito", "SST_fields",
            ]
        }'''
        self.dataset_references = {
            "glonet": [
                "jason3", "saral", "swot", "glorys", "argo_profiles", "argo_velocities",
                "SSS_fields", "SST_fields",
            ]
        }
        self.all_datasets = list(set(
            list(self.dataset_references.keys()) +
            [item for sublist in self.dataset_references.values() for item in sublist]
        ))
        memory_limit_per_worker = self.args.memory_limit_per_worker
        n_parallel_workers = self.args.n_parallel_workers
        nthreads_per_worker = self.args.nthreads_per_worker

        logger.info(
            f"Init DatasetProcessor with: Workers={n_parallel_workers}, "
            f"Threads={nthreads_per_worker}, MemLimit={memory_limit_per_worker}"
        )

        self.dataset_processor = DatasetProcessor(
            distributed=True, n_workers=n_parallel_workers,
            threads_per_worker=nthreads_per_worker,
            memory_limit=memory_limit_per_worker
        )

    def filter_data(
        self, manager: MultiSourceDatasetManager,
        filter_region: gpd.GeoSeries,
    ):
        """Filter data by time and region.

        Args:
            manager (MultiSourceDatasetManager): Dataset manager.
            filter_region (gpd.GeoSeries): Filter region.

        Returns:
             MultiSourceDatasetManager: Filtered dataset manager.
        """
        # Apply time filters
        manager.filter_all_by_date(
            start=pd.to_datetime(self.args.start_time),
            end=pd.to_datetime(self.args.end_time),
        )
        # Apply spatial filters
        manager.filter_all_by_region(
            region=filter_region
        )
        return manager

    def setup_transforms(
        self,
        dataset_manager: MultiSourceDatasetManager,
        aliases: list[str],
    ):
        """Fixture to configure transforms."""
        transforms_dict = {}
        for alias in aliases:
            kwargs = {
                "reduce_precision": self.args.reduce_precision
            }
            # Only add weights if needed (e.g. for glorys interpolation)
            # The logic for using these weights is now in transforms.py
            if alias == "glorys_cmems":
                kwargs["regridder_weights"] = self.args.regridder_weights

            transforms_dict[alias] = dataset_manager.get_transform(
                dataset_alias=alias,
                **kwargs
            )
        return transforms_dict


    def check_dataloader(
        self,
        dataloader: EvaluationDataloader,
    ):
        """Check dataloader integrity.

        Args:
            dataloader (EvaluationDataloader): Dataloader to check.
        """
        for batch in dataloader:
            logger.debug(f"Batch: {batch}")
            # Verify that batch contains expected keys
            assert "pred_data" in batch[0]
            assert "ref_data" in batch[0]
            # Verify that data are of type str (paths)
            assert isinstance(batch[0]["pred_data"], str)
            if batch[0]["ref_data"]:
                assert isinstance(batch[0]["ref_data"], str)

    def get_catalog(
        self,
        dataset_name: str, local_catalog_dir: str,
        catalog_cfg: dict,
    ):
        """Get dataset catalog, downloading if necessary.

        Args:
            dataset_name (str): Dataset name.
            local_catalog_dir (str): Local directory.
            catalog_cfg (dict): Catalog config.
        """
        import fsspec

        def download_catalog_file(
                remote_path: str,
                local_path: str,
            ):

            def create_fs(catalog_cfg):
                key = catalog_cfg.get("s3_key", None)
                secret_key = catalog_cfg.get("s3_secret_key", None)
                endpoint_url = catalog_cfg.get("url", None)

                client_kwargs={'endpoint_url': endpoint_url}
                if key is None or secret_key is None:
                    fs = fsspec.filesystem('s3', anon=True, client_kwargs=client_kwargs)
                else:
                    fs = fsspec.filesystem(
                        "s3", key=key, secret=secret_key, client_kwargs=client_kwargs
                    )
                return fs
            """Downloads remote file (S3) to local file."""
            fs = create_fs(catalog_cfg)
            if not fs.exists(remote_path):
                logger.warning(
                    f"Remote catalog file not found: {remote_path}"
                )
                return False
            data = fs.cat_file(remote_path)   # reads whole file into memory
            with open(local_path, "wb") as local_file:
                local_file.write(data)
            return True

        # check if local file exists
        local_catalog_path = os.path.join(local_catalog_dir, f"{dataset_name}.json")
        if os.path.isfile(local_catalog_path) and os.path.getsize(local_catalog_path) > 0:
            return
        else:
            # Get catalog from server if no local file exists
            remote_catalog_path = f"s3://{catalog_cfg['s3_bucket']}/{catalog_cfg['s3_folder']}/{dataset_name}.json"

            if not download_catalog_file(remote_catalog_path, local_catalog_path):
                return

    def setup_dataset_manager(self, list_all_references: list[str]) -> None:
        """Setup dataset manager and datasets.

        Args:
            list_all_references (list[str]): List of reference datasets.
        """
        manager = MultiSourceDatasetManager(
            dataset_processor=self.dataset_processor,
            target_dimensions=TARGET_DIM_RANGES,
            time_tolerance=pd.Timedelta(hours=self.args.delta_time),
            list_references=list_all_references,
            max_cache_files=self.args.max_cache_files,
        )
        datasets = {}
        for source in sorted(self.args.sources, key=lambda x: x["dataset"], reverse=False):
            source_name = source['dataset']
            if source_name not in self.all_datasets:
                logger.warning(f"Dataset {source_name} is not supported yet, skipping.")
                continue
            logger.info(
                f"\n\n\n=========  START EVALUATION FOR CANDIDATE : {source_name}  ========="
            )
            # Explicit memory cleanup on Dask workers after each batch
            try:
                client = get_client()
                client.run(gc.collect)
                logger.info("Memory cleanup (gc.collect) executed on all Dask workers.")
            except Exception as e:
                logger.warning(f"Could not execute memory cleanup on Dask workers: {e}")
            #"glorys", "argo_profiles", "argo_velocities",
            #"jason1", "jason2", "jason3",
            #"saral", "swot", "SSS_fields", "SST_fields",
            '''if (source_name != "glonet_edito" and source_name != "swot_edito" and
                    source_name != "saral_edito" and source_name != "jason3_edito" and
                    source_name != "glorys_edito"):'''
            if (source_name != "glonet" and source_name != "swot" and
                    source_name != "saral" and source_name != "jason3" and
                    source_name != "glorys"):
                logger.warning(f"Dataset {source_name} is not supported yet, skipping.")
                continue

            # Download dataset index file (catalog) if needed
            self.get_catalog(
                source_name,
                self.args.catalog_dir,
                self.args.catalog_connection,
            )

            kwargs = {}
            kwargs["source"] = source
            kwargs["root_data_folder"] = self.args.data_directory
            kwargs["root_catalog_folder"] = self.args.catalog_dir
            kwargs["dataset_processor"] = self.dataset_processor
            kwargs["max_samples"] = self.args.max_samples
            kwargs["file_cache"] = manager.file_cache
            kwargs["filter_values"] = {
                "start_time": self.args.start_time,
                "end_time": self.args.end_time,
                "min_lon": self.args.min_lon,
                "max_lon": self.args.max_lon,
                "min_lat": self.args.min_lat,
                "max_lat": self.args.max_lat,
            }

            logger.info(f"\n========= Setup dataset {source_name} =========\n")
            datasets[source_name] = get_dataset_from_config(
                **kwargs
            )
            # Add datasets with aliases
            manager.add_dataset(source_name, datasets[source_name])

        filter_region = geometry.Polygon(
            [(self.args.min_lon,self.args.min_lat),
            (self.args.min_lon,self.args.max_lat),
            (self.args.max_lon,self.args.max_lat),
            (self.args.max_lon,self.args.min_lat)]
        )
        filter_region_gs = gpd.GeoSeries([filter_region], crs="EPSG:4326")

        # Apply spatio-temporal filters
        manager = self.filter_data(manager, filter_region_gs)

        return manager


    def run_eval(self) -> None:
        """Proceed to evaluation."""
        dataset_manager = self.setup_dataset_manager(self.all_datasets)
        aliases = dataset_manager.datasets.keys()

        dataloaders = {}
        metrics_names = {}
        metrics = {}
        metrics_kwargs = {}
        evaluators = {}
        models_results = {}
        transforms_dict = self.setup_transforms(dataset_manager, aliases)

        for alias in self.dataset_references.keys():
            dataset_json_path = os.path.join(self.args.data_directory, f"results_{alias}.json")
            results_files_dir = os.path.join(self.args.data_directory, "results_batches")

            # Check if directory exists
            if os.path.isdir(results_files_dir):
                # Check if it is empty
                if os.listdir(results_files_dir):
                    logger.info("Results dir exists. Removing old results files.")
                    empty_folder(results_files_dir, extension=".json")
            else:
                os.makedirs(results_files_dir, exist_ok=True)

            dataset_manager.build_forecast_index(
                alias,
                init_date=self.args.start_time,
                end_date=self.args.end_time,
                n_days_forecast=int(self.args.n_days_forecast),
                n_days_interval=int(self.args.n_days_interval),
            )
            list_references = [
                ref for ref in self.dataset_references[alias] if ref in dataset_manager.datasets
            ]
            pred_source_dict = next((s for s in self.args.sources if s.get("dataset") == alias), {})
            metrics_names[alias] = pred_source_dict.get("metrics", ["rmsd"])

            metrics_kwargs[alias] = {}
            ref_transforms = {}
            metrics[alias] = {}
            pred_transform = transforms_dict.get(alias)
            for ref_alias in list_references:
                # Verify that reference dataset exists
                if ref_alias not in dataset_manager.datasets:
                    logger.warning(
                        f"Reference dataset '{ref_alias}' not found in dataset manager. "
                        "Skipping."
                    )
                    continue

                ref_source_dict = next(
                    (s for s in self.args.sources if s.get("dataset") == ref_alias), {}
                )
                ref_transforms[ref_alias] = transforms_dict.get(ref_alias)
                metrics_names[ref_alias] = ref_source_dict.get("metrics", ["rmsd"])
                ref_is_observation = dataset_manager.datasets[
                    ref_alias
                ].get_global_metadata()["is_observation"]
                pred_eval_vars = dataset_manager.datasets[alias].get_eval_variables()
                ref_eval_vars = dataset_manager.datasets[
                    ref_alias
                ].get_eval_variables()

                # Common variables
                common_vars = [
                    get_standardized_var_name(var)
                    for var in pred_eval_vars if var in ref_eval_vars
                ]
                if not common_vars:
                    logger.warning(
                        "No common variables found between pred_data and "
                        "ref_data for evaluation."
                    )
                    continue

                oceanbench_eval_variables = [   # Oceanbench lib format
                    get_variable_alias(var) for var in common_vars
                ] if common_vars else None

                # common metrics
                common_metrics = [
                    metric for metric in metrics_names[alias]
                    if metric in metrics_names[ref_alias]
                ]
                metrics_kwargs[alias][ref_alias] = {
                    "add_noise": False
                }
                if not ref_is_observation:
                    metrics[alias][ref_alias] = [
                        MetricComputer(
                            common_vars,
                            oceanbench_eval_variables,
                            metric_name=metric,
                            **metrics_kwargs[alias][ref_alias],
                        )
                        for metric in common_metrics
                    ]
                else:
                    interpolation_method = ref_source_dict.get(
                        "interpolation_method", "pyinterp"
                    )
                    time_tolerance = ref_source_dict.get("time_tolerance", None)
                    time_tolerance = timedelta(hours=time_tolerance)
                    class4_kwargs={
                        "interpolation_method": interpolation_method,
                        "list_scores": common_metrics,
                        "time_tolerance": time_tolerance,
                    }
                    metrics[alias][ref_alias] = [
                        MetricComputer(
                            common_vars,
                            oceanbench_eval_variables,
                            metric_name=metric,
                            is_class4=True,
                            class4_kwargs=class4_kwargs,
                            **metrics_kwargs[alias][ref_alias]
                        ) for metric in common_metrics
                    ]
            forecast_mode = False
            if self.args.n_days_forecast > 1:
                forecast_mode = True
            dataloaders[alias] = dataset_manager.get_dataloader(
                pred_alias=alias,
                ref_aliases=list_references,
                batch_size=self.args.batch_size,
                pred_transform=pred_transform,
                ref_transforms=ref_transforms,
                forecast_mode=forecast_mode,
                n_days_forecast=self.args.n_days_forecast,
                lead_time_unit='days',
            )

            # Check the dataloader
            # self.check_dataloader(dataloaders[alias])

            evaluators[alias] = Evaluator(
                dataset_manager=dataset_manager,
                metrics=metrics[alias],
                dataloader=dataloaders[alias],
                ref_aliases=list_references,
                dataset_processor=self.dataset_processor,
                results_dir=results_files_dir,
                reduce_precision=getattr(self.args, "reduce_precision", False),
                restart_workers_per_batch=getattr(self.args, "restart_workers_per_batch", False),
                restart_frequency=getattr(self.args, "restart_frequency", 1),
                max_worker_memory_fraction=getattr(
                    self.args, "max_worker_memory_fraction", 0.85
                ),
            )
            logger.info(f"\n\n\n=========  START EVALUATION FOR CANDIDATE : {alias}  =========")
            models_results[alias] = evaluators[alias].evaluate()


            # Eval has finished. Process results and write JSON
            try:
                # Search all batch files
                batch_files = glob(os.path.join(results_files_dir, "results_*_batch_*.json"))
                results_dict = {}
                for batch_file in batch_files:

                    with open(batch_file, "r") as f:
                        batch_results = json.load(f)
                        # Transform to make serializable
                        transform_in_place(batch_results, make_serializable)
                        serializable_result = nan_to_none(batch_results)
                    if alias not in results_dict:
                        results_dict[alias] = []
                    results_dict[alias].extend(serializable_result)
                # Save final JSON
                with open(dataset_json_path, 'w') as json_file:
                    # Clear file if it already exists
                    json_file.write('')
                    logger.info(f"Cleared contents of {json_file}")
                    json.dump({
                        "dataset": alias,
                        "results": results_dict,
                        "metadata": {
                            "evaluation_date": pd.Timestamp.now().isoformat(),
                            "total_entries": sum(len(v) for v in results_dict.values()),
                            "config": {
                                "start_time": self.args.start_time,
                                "end_time": self.args.end_time,
                                "n_days_forecast": self.args.n_days_forecast,
                                "n_days_interval": self.args.n_days_interval,
                            }
                        }
                    }, json_file, indent=2, ensure_ascii=False)

            except Exception as exc:
                logger.error(f"Failed to write JSON results: {exc}")
                raise
            # finally:
            #     self.dataset_processor.close()



        dataset_manager.file_cache.clear()


    def close(self):
        """Close the dataset processor and release resources."""
        if self.dataset_processor:
            self.dataset_processor.close()


