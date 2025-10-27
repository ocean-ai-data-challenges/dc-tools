from argparse import Namespace
import gc
from typing import Any, Callable, Dict, List, Optional

import dask

from loguru import logger
from oceanbench.core.distributed import DatasetProcessor
import traceback

from dctools.data.connection.config import (
    ARGOConnectionConfig, GlonetConnectionConfig,
    WasabiS3ConnectionConfig, S3ConnectionConfig, 
    FTPConnectionConfig, CMEMSConnectionConfig,
    LocalConnectionConfig
)
from dctools.data.connection.connection_manager import (
    ArgoManager, GlonetManager,
    LocalConnectionManager, S3WasabiManager,
    S3Manager, FTPManager, CMEMSManager, clean_for_serialization,
)
from dctools.data.datasets.dataloader import EvaluationDataloader
from dctools.data.datasets.dataset_manager import MultiSourceDatasetManager
from dctools.metrics.metrics import MetricComputer
from dctools.utilities.misc_utils import deep_copy_object, make_fully_serializable


CONNECTION_REGISTRY = {
    "argo": ArgoManager,
    "cmems": CMEMSManager,
    "ftp": FTPManager,
    "glonet": GlonetManager,
    "local": LocalConnectionManager,
    "s3": S3Manager,
    "wasabi": S3WasabiManager,
}

CONNECTION_CONFIG_REGISTRY = {
    "argo": ARGOConnectionConfig,
    "cmems": CMEMSConnectionConfig,
    "ftp": FTPConnectionConfig,
    "glonet": GlonetConnectionConfig,
    "local": LocalConnectionConfig,
    "s3": S3ConnectionConfig,
    "wasabi": WasabiS3ConnectionConfig,
}


def compute_metric(
    entry: Dict[str, Any],
    pred_source_config: Namespace,
    ref_source_config: Namespace,
    model: str,
    list_metrics: list[MetricComputer],
    pred_transform: Callable,
    ref_transforms: List[Callable],
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

        if ref_transforms and ref_alias in ref_transforms:
            ref_transform = ref_transforms[ref_alias]
        pred_source = entry["pred_data"]
        ref_source = entry["ref_data"]
    
        pred_protocol = pred_source_config.protocol
        ref_protocol = ref_source_config.protocol


        if ref_protocol == 'cmems':
            if hasattr(ref_source_config, 'fs') and hasattr(ref_source_config.fs, '_session'):
                try:
                    if hasattr(ref_source_config.fs._session, 'close'):
                        ref_source_config.fs._session.close()
                except:
                    pass
                ref_source_config.fs = None

        pred_source_config.dataset_processor = None,
        ref_source_config.dataset_processor = None,

        # Recrée l'objet de lecture dans le worker
        pred_config_cls = CONNECTION_CONFIG_REGISTRY[pred_protocol]
        pred_connection_cls = CONNECTION_REGISTRY[pred_protocol]
        delattr(pred_source_config, "protocol")
        pred_config = pred_config_cls(**vars(pred_source_config))

        # remove fsspec handler 'fs' from Config, otherwise: serialization
        if pred_protocol == 'cmems': 
            if hasattr(
                pred_config.params, 'fs') and hasattr(pred_config.params.fs, '_session'
            ):
                try:
                    if hasattr(pred_config.params.fs._session, 'close'):
                        pred_config.params.fs._session.close()
                except:
                    pass
                pred_config.params.fs = None


        pred_connection_manager = pred_connection_cls(pred_config)
        open_pred_func = pred_connection_manager.open

        ref_config_cls = CONNECTION_CONFIG_REGISTRY[ref_protocol]
        ref_connection_cls = CONNECTION_REGISTRY[ref_protocol]
        delattr(ref_source_config, "protocol")
        ref_config = ref_config_cls(**vars(ref_source_config))

        if ref_protocol == 'cmems':
            if hasattr(ref_config.params, 'fs'):
                try:
                    if hasattr(ref_config.params.fs, '_session'):
                        if hasattr(ref_config.params.fs._session, 'close'):
                            ref_config.params.fs._session.close()
                except:
                    pass
                delattr(ref_config.params, "fs")


        if ref_protocol == "argo":
            ref_connection_manager = ref_connection_cls(ref_config, argo_index=argo_index)
        else:
            ref_connection_manager = ref_connection_cls(ref_config)
        open_ref_func = ref_connection_manager.open


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
                ref_data = ref_source
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
        if ref_data and ref_transform:
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

        gc.collect()
        return res
    except Exception as exc:
        logger.error(f"Error computing metrics for date {forecast_reference_time}: {repr(exc)}")
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
        self.results = []
        self.ref_aliases = ref_aliases

        self.ref_managers, self.ref_catalogs, self.ref_connection_params = dataset_manager.get_config()

    def evaluate(self) -> List[Dict[str, Any]]:
        """
        Évalue les métriques sur les données du dataloader pour chaque référence.

        Returns:
            List[Dict[str, Any]]: Résultats des métriques pour chaque lot et chaque référence.
        """
        try:
            for batch in self.dataloader:
                batch_results = self._evaluate_batch(
                    batch, self.dataloader,
                )
                if batch_results is None:
                    raise TypeError(
                        "Empty result batch. Make sure all datasets needed for "\
                         "the evaluation loaded properly."
                        )
                serial_results = [make_fully_serializable(res) for res in batch_results if res is not None]
                self.results.extend(serial_results) 
            return self.results

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
        dataloader: EvaluationDataloader,
    ) -> List[Dict[str, Any]]:
        delayed_tasks = []

        pred_connection_params = deep_copy_object(
            dataloader.pred_connection_params, skip_list=['dataset_processor', 'fs']
        )
        pred_connection_params = clean_for_serialization(pred_connection_params)
        pred_connection_params = self.clean_namespace(pred_connection_params)

        pred_transform=dataloader.pred_transform
        ref_transforms = None
        if dataloader.ref_transforms is not None:
            ref_transforms = dataloader.ref_transforms
        if hasattr(pred_transform, 'dataset_processor'):
            delattr(pred_transform, 'dataset_processor')

        ref_alias = batch[0].get("ref_alias")

        ref_connection_params = deep_copy_object(
            dataloader.ref_connection_params.get(ref_alias, None), skip_list=['dataset_processor', 'fs']
        )
        ref_connection_params = clean_for_serialization(ref_connection_params)
        ref_connection_params = self.clean_namespace(ref_connection_params)


        argo_index = None
        if hasattr(dataloader.ref_managers[ref_alias], 'argo_index'):
            argo_index = dataloader.ref_managers[ref_alias].get_argo_index()
        if argo_index is not None:
            scattered_argo_index = self.dataset_processor.scatter_data(
                argo_index,
                broadcast_item = False,
            )
        else:
            scattered_argo_index = None

        try:
            for entry in batch:
                delayed_tasks.append(dask.delayed(compute_metric)(
                    entry,
                    pred_connection_params,
                    ref_connection_params,
                    dataloader.pred_alias,
                    self.metrics[ref_alias],
                    pred_transform=pred_transform,
                    ref_transforms=ref_transforms,
                    argo_index=scattered_argo_index,
                ))

            batch_results = self.dataset_processor.compute_delayed_tasks(delayed_tasks)
            valid_results = [meta for meta in batch_results if meta is not None]
            return valid_results
        except Exception as exc:
            logger.error(f"Error processing entry {entry}: {repr(exc)}")
            traceback.print_exc()
            return None
