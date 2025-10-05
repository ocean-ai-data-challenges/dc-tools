
from argparse import Namespace
import gc
import os
from typing import Any, Callable, Dict, List, Optional

import dask

from loguru import logger
from oceanbench.core.distributed import DatasetProcessor
import traceback

import json


# from dctools.data.datasets.dataloader import DatasetLoader
from dctools.data.connection.connection_manager import (
    create_worker_connect_config
)
from dctools.data.connection.connection_manager import clean_for_serialization
from dctools.data.datasets.dataloader import EvaluationDataloader, ObservationDataViewer, filter_by_time
from dctools.data.datasets.dataset_manager import MultiSourceDatasetManager
from dctools.metrics.metrics import MetricComputer
from dctools.utilities.misc_utils import (
    deep_copy_object, make_serializable,
    nan_to_none, transform_in_place,
    serialize_structure
)


def compute_metric(
    entry: Dict[str, Any],
    pred_source_config: Namespace,
    ref_source_config: Namespace,
    model: str,
    list_metrics: list[MetricComputer],
    pred_transform: Callable,
    ref_transform: Callable,
    argo_index: Optional[Any] = None,
) -> Dict[str, Any]:
    try:
        forecast_reference_time = entry.get("forecast_reference_time")
        lead_time = entry.get("lead_time")
        valid_time = entry.get("valid_time")
        pred_coords = entry.get("pred_coords")
        ref_coords = entry.get("ref_coords")
        ref_alias = entry.get("ref_alias")
        ref_is_observation = entry.get("ref_is_observation")
        logger.info(f"Process forecast: {forecast_reference_time}, lead time: {lead_time}")

        pred_protocol = pred_source_config.protocol
        ref_protocol = ref_source_config.protocol

        pred_source = entry["pred_data"]
        ref_source = entry["ref_data"]
    
        open_pred_func = create_worker_connect_config(
            pred_source_config,
            argo_index,
        )
        open_ref_func = create_worker_connect_config(
            ref_source_config,
            argo_index,
        )

        if isinstance(pred_source, str):
            if pred_protocol == "cmems":
                # cmems not compatible with Dask workers (pickling errors)
                with dask.config.set(scheduler='synchronous'):
                    pred_data = open_pred_func(pred_source)
            else:
                pred_data = open_pred_func(pred_source)
        else:
            pred_data = pred_source


        pred_data_selected = pred_data.sel(time=valid_time, method="nearest")
        
        # Si la dimension time a été supprimée (cas scalaire), la restaurer
        if "time" not in pred_data_selected.dims:
            pred_data = pred_data_selected.expand_dims("time")
            # Assigner la valeur valid_time à la coordonnée time
            pred_data = pred_data.assign_coords(time=[valid_time])
        else:
            # La dimension time existe déjà, mais s'assurer qu'elle a la bonne valeur
            pred_data = pred_data_selected.assign_coords(time=[valid_time])

        if ref_source is not None:
            if ref_is_observation:
                ref_source = entry["ref_data"]
                raw_ref_df = ref_source["source"]
                keep_vars = ref_source["keep_vars"]
                target_dimensions = ref_source["target_dimensions"]
                time_bounds = ref_source["time_bounds"]
                metadata = ref_source["metadata"]

                ref_df = raw_ref_df.get_dataframe()
                t0, t1 = time_bounds
                ref_df = filter_by_time(ref_df, t0, t1)

                if ref_df.empty:
                    logger.warning(f"No {ref_alias} Data for time interval: {t0}/{t1}]")
                    return {
                        "model": model,
                        "ref_alias": ref_alias,
                        "result": None,
                    }
                n_points_dim = "n_points"   # default
                if hasattr(ref_coords.coordinates, "n_points"):
                        n_points_dim=ref_coords.coordinates["n_points"]
                ref_raw_data = ObservationDataViewer(
                    ref_df,
                    open_ref_func, ref_alias,
                    keep_vars, target_dimensions, metadata,
                    time_bounds,
                    n_points_dim = n_points_dim,
                    dataset_processor=None,
                )
                # load immediately before increasing Dask graph size
                ref_data = ref_raw_data.preprocess_datasets(
                    ref_df,
                    load_to_memory=False,
                )
            else:
                if ref_protocol == "cmems":
                    with dask.config.set(scheduler='synchronous'):
                        ref_data = open_ref_func(ref_source, ref_alias)
                else:
                    ref_data = open_ref_func(ref_source, ref_alias)
        else:
            ref_data = None

        if pred_transform:
            if ref_protocol == "cmems":
                with dask.config.set(scheduler='synchronous'):
                    pred_data = pred_transform(pred_data)
            else:
                pred_data = pred_transform(pred_data)
        if ref_data is not None and ref_transform is not None:
            if ref_protocol == "cmems":
                with dask.config.set(scheduler='synchronous'):
                    ref_data = ref_transform(ref_data)
            else:
                ref_data = ref_transform(ref_data)

        if ref_is_observation:
            if ref_data is None:
                return {
                    "model": model,
                    "ref_alias": ref_alias,
                    "result": None,
                }
            results = list_metrics[0].compute(
                pred_data, ref_data,
                pred_coords, ref_coords,
            )
        else:
            results = {}
            for metric in list_metrics:
                return_res = metric.compute(
                    pred_data, ref_data,
                    pred_coords, ref_coords,
                )

                res_dict = {}

                if len(return_res) == 0:
                    return {
                        "model": model,
                        "ref_alias": ref_alias,
                        "result": None,
                    }
                # Convertir chaque ligne du DataFrame en dictionnaire
                for var_depth_label in return_res.index:
                    # Nettoyer le nom de la variable/profondeur pour en faire une clé valide
                    clean_key = var_depth_label.lower().replace(" ", "_")
                    
                    # Extraire les valeurs des métriques pour tous les lead days
                    metric_values = return_res.loc[var_depth_label].to_dict()
                    
                    # Structure : {variable: {lead_day: metric_value}}
                    res_dict[clean_key] = metric_values['Lead day 1']

                results[metric.get_metric_name()] = res_dict

        res = {
            "model": model,
            "ref_alias": ref_alias,
            "result": results.compute() if hasattr(results, "compute") else results,
        }
        # Ajoute les champs forecast si présents
        if forecast_reference_time is not None:
            res["forecast_reference_time"] = forecast_reference_time
        if lead_time is not None:
            res["lead_time"] = lead_time
        if valid_time is not None:
            res["valid_time"] = valid_time

        # Libérer les objets volumineux
        '''if ref_data is not None:
            if hasattr(ref_data, "close"):
                ref_data.close()
            if isinstance(ref_data, list):
                for item in ref_data:
                    if hasattr(item, "close"):
                        item.close()
            # del ref_data'''

        gc.collect()
        return res
    except Exception as exc:
        logger.error(f"Error computing metrics for dataset {ref_alias} and date {forecast_reference_time}: {repr(exc)}")
        import traceback
        traceback.print_exc()
        return {
            "model": model,
            "ref_alias": ref_alias,
            "result": None,
        }


class Evaluator:
    def __init__(
        self,
        dataset_manager: MultiSourceDatasetManager,
        metrics: Dict[str, List[MetricComputer]],
        dataloader: Dict[str, EvaluationDataloader],
        ref_aliases: List[str],
        dataset_processor: DatasetProcessor,
        results_dir: str = None,
    ):
        """
        Initialise l'évaluateur.

        Args:
            dask_client (object): Client Dask pour la parallélisation.
            metrics (Dict[str, List[MetricComputer]]): Dictionnaire {ref_alias: [MetricComputer, ...]}.
            dataloader (Dict[str, EvaluationDataloader]): Dictionnaire {ref_alias: EvaluationDataloader}.
        """
        self.dataset_manager = dataset_manager
        self.dataset_processor = dataset_processor
        self.metrics = metrics
        self.dataloader = dataloader
        # self.results = []
        self.ref_aliases = ref_aliases
        self.results_dir = results_dir

        self.ref_managers, self.ref_catalogs, self.ref_connection_params = dataset_manager.get_config()

    def evaluate(self) -> List[Dict[str, Any]]:
        """
        Évalue les métriques sur les données du dataloader pour chaque référence.

        Returns:
            List[Dict[str, Any]]: Résultats des métriques pour chaque lot et chaque référence.
        """
        try:
            for batch_idx, batch in enumerate(self.dataloader):
                pred_alias =self.dataloader.pred_alias
                ref_alias = batch[0].get("ref_alias")
                # Extraire les informations nécessaires
                pred_connection_params = self.dataloader.pred_connection_params
                ref_connection_params = self.dataloader.ref_connection_params[ref_alias]
                pred_transform = self.dataloader.pred_transform
                if self.dataloader.ref_transforms is not None:
                    ref_transform = self.dataloader.ref_transforms[ref_alias]
                argo_index = None
                if hasattr(self.dataloader.ref_managers[ref_alias], 'argo_index'):
                    argo_index = self.dataloader.ref_managers[ref_alias].get_argo_index()
                batch_results = self._evaluate_batch(
                    batch, pred_alias, ref_alias,
                    pred_connection_params, ref_connection_params,
                    pred_transform, ref_transform,
                    argo_index=argo_index,
                )
                if batch_results is None:
                    continue
                serial_results = [serialize_structure(res) for res in batch_results if res is not None]

                # Sauvegarde batch par batch
                batch_file = os.path.join(self.results_dir, f"results_{pred_alias}_batch_{batch_idx}.json")
                with open(batch_file, "w") as f:
                    json.dump(serial_results, f, indent=2, ensure_ascii=False)

        except Exception as exc:
            logger.error(f"Evaluation failed: {traceback.format_exc()}")
            raise

    def clean_namespace(self, namespace: Namespace) -> Namespace:
        ns = Namespace(**vars(namespace))
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

    def _evaluate_batch(
        self, batch: List[Dict[str, Any]],
        pred_alias: str, ref_alias: str,
        pred_connection_params: Dict[str, Any], ref_connection_params: Dict[str, Any],
        pred_transform: Any, ref_transform: Any,
        argo_index: Any = None,
    ) -> List[Dict[str, Any]]:
        delayed_tasks = []

        ref_alias = batch[0].get("ref_alias")

        pred_connection_params = deep_copy_object(
            pred_connection_params, skip_list=['dataset_processor', 'fs']
        )
        pred_connection_params = clean_for_serialization(pred_connection_params)
        pred_connection_params = self.clean_namespace(pred_connection_params)

        if hasattr(pred_transform, 'dataset_processor'):
            delattr(pred_transform, 'dataset_processor')
        if hasattr(ref_transform, 'dataset_processor'):
            delattr(ref_transform, 'dataset_processor')

        ref_connection_params = deep_copy_object(
            ref_connection_params, skip_list=['dataset_processor', 'fs']
        )
        ref_connection_params = clean_for_serialization(ref_connection_params)
        ref_connection_params = self.clean_namespace(ref_connection_params)

        if argo_index is not None:
            scattered_argo_index = self.dataset_processor.scatter_data(
                argo_index,
                broadcast_item = True,
            )
        else:
            scattered_argo_index = None

        try:
            for entry in batch:
                delayed_tasks.append(dask.delayed(compute_metric)(
                    entry,
                    pred_connection_params,
                    ref_connection_params,
                    pred_alias,
                    self.metrics[ref_alias],
                    pred_transform=pred_transform,
                    ref_transform=ref_transform,
                    argo_index=scattered_argo_index,
                ))

            batch_results = self.dataset_processor.compute_delayed_tasks(delayed_tasks)
            return batch_results
        except Exception as exc:
            logger.error(f"Error processing entry {entry}: {repr(exc)}")
            traceback.print_exc()
            return None
