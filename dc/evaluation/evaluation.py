#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Evaluator class."""

from argparse import Namespace
import json
import os
# from typing import Any, Optional


from dask.distributed import Client
from datetime import timedelta
import geopandas as gpd
from loguru import logger
from oceanbench.core.distributed import DatasetProcessor
import pandas as pd
from shapely import geometry

from dctools.data.coordinates import get_standardized_var_name
from dctools.data.datasets.dataset import get_dataset_from_config
from dctools.data.datasets.dataloader import EvaluationDataloader
from dctools.data.datasets.dataset_manager import MultiSourceDatasetManager

from dctools.metrics.evaluator import Evaluator
from dctools.metrics.metrics import MetricComputer
from dctools.metrics.oceanbench_metrics import get_variable_alias
# from dctools.processing.distributed import ParallelExecutor
from dctools.utilities.init_dask import setup_dask
from dctools.data.coordinates import (
    TARGET_DIM_RANGES,
    TARGET_DEPTH_VALS,
)
from dctools.utilities.misc_utils import (
    make_serializable,
    nan_to_none,
    transform_in_place,
)

class DC2Evaluation:
    """Class to evaluate models on Glorys forecasts."""

    def __init__(self, arguments: Namespace) -> None:
        """Init class.

        Args:
            arguments (str): Namespace with config.
        """
        self.args = arguments

        self.dataset_references = {
            "glonet": [
                "jason3", "glorys", "argo_profiles", "argo_velocities",
                "saral", "swot", "SSS_fields", "SST_fields",
            ]
        }
        self.all_datasets = list(set(
            list(self.dataset_references.keys()) + 
            [item for sublist in self.dataset_references.values() for item in sublist]
        ))
        memory_limit_per_worker = self.args.memory_limit_per_worker
        n_parallel_workers = self.args.n_parallel_workers
        nthreads_per_worker = self.args.nthreads_per_worker
        self.dataset_processor = DatasetProcessor(
            distributed=True, n_workers=n_parallel_workers,
            threads_per_worker=nthreads_per_worker,
            memory_limit=memory_limit_per_worker
        )

    def filter_data(
        self, manager: MultiSourceDatasetManager,
        filter_region: gpd.GeoSeries,
    ):
        # Appliquer les filtres temporels
        manager.filter_all_by_date(
            start=pd.to_datetime(self.args.start_time),
            end=pd.to_datetime(self.args.end_time),
        )
        # Appliquer les filtres spatiaux
        '''manager.filter_all_by_region(
            region=filter_region
        )'''
        return manager

    def setup_transforms(
        self,
        dataset_manager: MultiSourceDatasetManager,
        aliases: list[str],
    ):
        """Fixture pour configurer les transformations."""
        transforms_dict = {}
        for alias in aliases:
            if alias == "glorys":
                transforms_dict["glorys"] = dataset_manager.get_transform(
                    "standardize_glorys",
                    dataset_alias="glorys",
                    interp_ranges=TARGET_DIM_RANGES,
                    weights_path=self.args.regridder_weights,
                    depth_coord_vals=TARGET_DEPTH_VALS,
                )
            else:
                transforms_dict[alias] = dataset_manager.get_transform(
                    "standardize",
                    dataset_alias=alias,
                )
        return transforms_dict


    def check_dataloader(
        self,
        dataloader: EvaluationDataloader,
    ):
        for batch in dataloader:
            logger.debug(f"Batch: {batch}")
            # Vérifier que le batch contient les clés attendues
            assert "pred_data" in batch[0]
            assert "ref_data" in batch[0]
            # Vérifier que les données sont de type str (paths)
            assert isinstance(batch[0]["pred_data"], str)
            if batch[0]["ref_data"]:
                assert isinstance(batch[0]["ref_data"], str)

    def setup_dataset_manager(self, list_all_references: list[str]) -> None:

        manager = MultiSourceDatasetManager(
            dataset_processor=self.dataset_processor,
            target_dimensions=TARGET_DIM_RANGES,
            time_tolerance=pd.Timedelta(hours=self.args.delta_time),
            list_references=list_all_references,
            max_cache_files=self.args.max_cache_files,
        )
        datasets = {}
        for source in sorted(self.args.sources, key=lambda x: x["dataset"], reverse=True):
            source_name = source['dataset']
            if source_name not in self.all_datasets:
                continue
            #"glorys", "argo_profiles", "argo_velocities",
            #"jason1", "jason2", "jason3",
            #"saral", "swot", "SSS_fields", "SST_fields",
            if source_name != "glonet" and source_name != "glorys":  # and source_name != "jason3" and source_name != "saral" and source_name != "glorys":
                continue
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

            logger.debug(f"\n\nSetup dataset {source_name}\n")
            datasets[source_name] = get_dataset_from_config(
                **kwargs
            )
            # Ajouter les datasets avec des alias
            manager.add_dataset(source_name, datasets[source_name])

        filter_region = geometry.Polygon(
            [(self.args.min_lon,self.args.min_lat),
            (self.args.min_lon,self.args.max_lat),
            (self.args.max_lon,self.args.max_lat),
            (self.args.max_lon,self.args.min_lat)]
        )

        # Appliquer les filtres spatio-temporels
        # manager = self.filter_data(manager, filter_region) ##  TODO : check filtering validity

        return manager


    def run_eval(self) -> None:
        """Proceed to evaluation."""

        dataset_manager = self.setup_dataset_manager( self.all_datasets)
        aliases = dataset_manager.datasets.keys()

        dataloaders = {}
        metrics_names = {}
        metrics = {}
        metrics_kwargs = {}
        evaluators = {}
        models_results = {}
        transforms_dict = self.setup_transforms(dataset_manager, aliases, )

        json_path=os.path.join(self.args.catalog_dir, f"all_test_results.json")
        for alias in self.dataset_references.keys():
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
            for ref_alias in  list_references:
                # Vérifier que le dataset de référence existe
                if ref_alias not in dataset_manager.datasets:
                    logger.warning(f"Reference dataset '{ref_alias}' not found in dataset manager. Skipping.")
                    continue
                    
                ref_source_dict = next((s for s in self.args.sources if s.get("dataset") == ref_alias), {})
                ref_transforms[ref_alias] = transforms_dict.get(ref_alias)
                metrics_names[ref_alias] = ref_source_dict.get("metrics", ["rmsd"])
                ref_is_observation = dataset_manager.datasets[ref_alias].get_global_metadata()["is_observation"]
                pred_eval_vars = dataset_manager.datasets[alias].get_eval_variables()
                ref_eval_vars = dataset_manager.datasets[ref_alias].get_eval_variables()

                # variables communes
                common_vars = [get_standardized_var_name(var) for var in pred_eval_vars if var in ref_eval_vars]
                if not common_vars:
                    logger.warning("No common variables found between pred_data and ref_data for evaluation.")
                    continue
                
                oceanbench_eval_variables = [
                    get_variable_alias(var) for var in common_vars
                ] if common_vars else None

                common_metrics = [metric for metric in metrics_names[alias] if metric in metrics_names[ref_alias]]
                metrics_kwargs[alias][ref_alias] = {
                    "add_noise": False,
                    #"eval_variables": pred_eval_vars,
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
                        "interpolation_method", "kdtree"
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

            # Vérifier le dataloader
            # self.check_dataloader(dataloaders[alias])

            evaluators[alias] = Evaluator(
                dataset_manager=dataset_manager,
                metrics=metrics[alias],
                dataloader=dataloaders[alias],
                ref_aliases=list_references,
                dataset_processor=self.dataset_processor,
            )
            logger.info(f"\n\n\n=========  START EVALUATION FOR CANDIDATE : {alias}  =========")
            models_results[alias] = evaluators[alias].evaluate()


        try:
            # Sérialiser tous les résultats
            serialized_results = {}
            for dataset_alias, results in models_results.items():
                logger.info(f"Processing results for {dataset_alias}: {len(results)} entries")
                
                # Sérialiser chaque résultat individuellement
                serialized_entries = []
                for result in results:
                    # Vérifier que le résultat contient les champs attendus
                    if "result" not in result:
                        logger.warning(f"Missing 'result' field in entry: {result}")
                        continue
                        
                    # Transformer pour rendre sérialisable
                    transform_in_place(result, make_serializable)
                    serializable_result = nan_to_none(result)
                    serialized_entries.append(serializable_result)

                serialized_results[dataset_alias] = serialized_entries

            # Écrire le JSON final
            with open(json_path, 'w') as json_file:
                json.dump(serialized_results, json_file, indent=2, ensure_ascii=False)

            logger.info(f"Successfully wrote {len(serialized_results)} datasets results to {json_path}")

            for dataset_alias, results in serialized_results.items():
                dataset_json_path = os.path.join(self.args.catalog_dir, f"results_{dataset_alias}.json")
                with open(dataset_json_path, 'w') as json_file:
                    # Vider le fichier s'il existe déjà
                    json_file.write('')
                    logger.info(f"Cleared contents of {json_file}")
                    json.dump({
                        "dataset": dataset_alias,
                        "results": results,
                        "metadata": {
                            "evaluation_date": pd.Timestamp.now().isoformat(),
                            "total_entries": len(results),
                            "config": {
                                "start_time": self.args.start_time,
                                "end_time": self.args.end_time,
                                "n_days_forecast": self.args.n_days_forecast,
                                "n_days_interval": self.args.n_days_interval,
                            }
                        }
                    }, json_file, indent=2, ensure_ascii=False)
                logger.info(f"Created individual results file: {json_file}")

        except Exception as exc:
            logger.error(f"Failed to write JSON results: {exc}")
            raise
        finally:
            self.dataset_processor.close()

