from abc import ABC
from argparse import Namespace
import gc
from os import stat
from tracemalloc import start
from typing import Any, Callable, Dict, List, Optional

import dask
from dask.distributed import Client, as_completed, get_worker

from distributed import get_worker, performance_report
from loguru import logger
from memory_profiler import profile
import numpy as np
import pandas as pd
import shapely.geometry.base as shapely_base
from shapely.geometry import mapping
import traceback
import xarray as xr


# from dctools.data.datasets.dataloader import DatasetLoader
from dctools.data.connection.config import (
    ARGOConnectionConfig, GlonetConnectionConfig,
    WasabiS3ConnectionConfig, S3ConnectionConfig, 
    FTPConnectionConfig, CMEMSConnectionConfig,
    LocalConnectionConfig
)
from dctools.data.connection.connection_manager import (
    ArgoManager, GlonetManager,
    LocalConnectionManager, S3WasabiManager,
    S3Manager, FTPManager, CMEMSManager,
)
from dctools.data.datasets.dataloader import EvaluationDataloader
from dctools.data.coordinates import CoordinateSystem
from dctools.metrics.oceanbench_metrics import DCMetric
from dctools.metrics.metrics import MetricComputer


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

'''def log_memory(stage):
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / 1e6
    print(f"[{stage}] Memory usage: {mem_mb:.2f} MB")'''


def make_fully_serializable(obj):
    # Types de base
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    # Numpy types
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    # Pandas Timestamp/Timedelta
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, pd.Timedelta):
        return str(obj)
    # Pandas DataFrame/Series
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, pd.Series):
        return obj.tolist()
    # xarray Dataset/DataArray
    if isinstance(obj, (xr.Dataset, xr.DataArray)):
        return obj.to_dict()
    # Shapely geometry
    if isinstance(obj, shapely_base.BaseGeometry):
        return mapping(obj)
    # Dataclasses
    if hasattr(obj, "__dataclass_fields__"):
        return {k: make_fully_serializable(v) for k, v in obj.__dict__.items()}
    # Classes avec __dict__
    if hasattr(obj, "__dict__"):
        return {k: make_fully_serializable(v) for k, v in obj.__dict__.items() if not k.startswith("_")}
    # Mapping (dict-like)
    if isinstance(obj, dict):
        return {make_fully_serializable(k): make_fully_serializable(v) for k, v in obj.items()}
    # Iterable (list, tuple, set)
    if isinstance(obj, (list, tuple, set)):
        return [make_fully_serializable(v) for v in obj]
    # Fallback: string


"""def log_memory_and_objects(tag=""):
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1024**2  # en MiB
    print(f"[{tag}] Mémoire utilisée : {mem:.2f} MiB")
    
    # Afficher les objets les plus nombreux
    gc.collect()
    print("Top objets vivants :")
    for obj_type, count in objgraph.most_common_types(limit=10):
        print(f"  {obj_type}: {count}")"""


"""def debug_worker_memory():
    import objgraph
    from pympler import muppy, summary
    print("==== OBJECT TYPES ====")
    objgraph.show_most_common_types(limit=10)
    print("==== MEMORY SUMMARY ====")
    all_objects = muppy.get_objects()
    summary.print_(summary.summarize(all_objects), limit=10)

    # Optionnel : dump graphique
    objgraph.show_backrefs(
        objgraph.by_type('dict')[0],
        max_depth=3,
        filename='/tmp/debug_backref.png'
    )

    gc.collect()"""

class Evaluator:
    def __init__(
        self,
        dask_client: object,
        metrics: Dict[str, List[MetricComputer]],
        dataloader: Dict[str, EvaluationDataloader],
        ref_aliases: List[str],
    ):
        """
        Initialise l'évaluateur.

        Args:
            dask_client (object): Client Dask pour la parallélisation.
            metrics (Dict[str, List[MetricComputer]]): Dictionnaire {ref_alias: [MetricComputer, ...]}.
            dataloader (Dict[str, EvaluationDataloader]): Dictionnaire {ref_alias: EvaluationDataloader}.
        """
        self.dask_client = dask_client
        self.metrics = metrics
        self.dataloader = dataloader
        self.results = []
        self.ref_aliases = ref_aliases


    def evaluate(self) -> List[Dict[str, Any]]:
        """
        Évalue les métriques sur les données du dataloader pour chaque référence.

        Returns:
            List[Dict[str, Any]]: Résultats des métriques pour chaque lot et chaque référence.
        """
        try:
            for batch in self.dataloader:
                batch_results = self._evaluate_batch(
                    batch, self.dataloader, self.dask_client,
                )
                serial_results = [make_fully_serializable(res) for res in batch_results]
                self.results.extend(serial_results) 
                # Vide le cache si besoin
                #self.dask_client.run(gc.collect)
            return self.results
        except Exception as exc:
            logger.error(f"Evaluation failed: {traceback.format_exc()}")
            raise


    def clean_namespace(self, namespace: Namespace) -> Namespace:
        try:
            # Crée une copie pour éviter les effets de bord
            ns = Namespace(**vars(namespace))
            # remove these from config to avoid serialization issues with Dask
            for key in ['dask_cluster', 'fs']:
                if hasattr(ns, key):
                    delattr(ns, key)
            return ns
        except Exception as exc:
            logger.error(f"Error cleaning namespace: {exc}")
            # Retourne la version originale si erreur
            return namespace

    def _evaluate_batch(
        self, batch: List[Dict[str, Any]],
        dataloader: EvaluationDataloader,
        dask_client: Client,
    ) -> List[Dict[str, Any]]:
        tasks = []
        for entry in batch:
            try:
                forecast_reference_time = entry.get("forecast_reference_time")
                lead_time = entry.get("lead_time")
                valid_time = entry.get("valid_time")
                pred_coords = entry.get("pred_coords")
                ref_coords = entry.get("ref_coords")
                ref_alias = entry.get("ref_alias")
                ref_is_observation = entry.get("ref_is_observation")
                logger.info(f"Process forecast: {forecast_reference_time}, lead time: {lead_time}")

                ref_transform = None
                if dataloader.ref_transforms and ref_alias in dataloader.ref_transforms:
                    ref_transform = dataloader.ref_transforms[ref_alias]

                pred_source_config = self.clean_namespace(dataloader.pred_connection_params)
                ref_source_config = self.clean_namespace(dataloader.ref_connection_params.get(ref_alias, None))

                task = dask.delayed(self._compute_metric)(
                    pred_source_config,
                    ref_source_config,
                    dataloader.pred_alias,
                    self.metrics[ref_alias],
                    pred_path=entry["pred_data"],
                    ref_source=entry["ref_data"],
                    pred_transform=dataloader.pred_transform,
                    ref_transform=ref_transform,
                    ref_alias=ref_alias,
                    ref_is_observation=ref_is_observation,
                    forecast_reference_time=forecast_reference_time,
                    lead_time=lead_time,
                    valid_time=valid_time,
                    pred_coords=pred_coords,
                    ref_coords=ref_coords,
                )
                tasks.append(task)
            except Exception as exc:
                logger.error(f"Error processing entry {entry}: {repr(exc)}")
                continue

        futures = dask_client.compute(tasks)
        results = dask_client.gather(futures)
        return results


    @staticmethod
    def _compute_metric(
        pred_source_config: Namespace,
        ref_source_config: Namespace,
        model: str,
        list_metrics: list[MetricComputer],
        pred_path: str,
        ref_source,
        pred_transform: Callable,
        ref_transform: Callable,
        ref_alias: str,
        ref_is_observation: bool,
        forecast_reference_time,
        lead_time,
        valid_time,
        pred_coords: CoordinateSystem = None,
        ref_coords: CoordinateSystem = None,
    ) -> Dict[str, Any]:
        try:
            # worker = get_worker()
            # print(f"Running on worker: {worker.address}")

            # Recrée l’objet de lecture dans le worker
            pred_config_cls = CONNECTION_CONFIG_REGISTRY[pred_source_config.protocol]
            pred_connection_cls = CONNECTION_REGISTRY[pred_source_config.protocol]
            delattr(pred_source_config, "protocol")
            pred_config = pred_config_cls(**vars(pred_source_config))
            pred_connection_manager = pred_connection_cls(pred_config)
            open_pred_func = pred_connection_manager.open

            ref_config_cls = CONNECTION_CONFIG_REGISTRY[ref_source_config.protocol]
            ref_connection_cls = CONNECTION_REGISTRY[ref_source_config.protocol]
            delattr(ref_source_config, "protocol")
            ref_config = ref_config_cls(**vars(ref_source_config))
            ref_connection_manager = ref_connection_cls(ref_config)
            open_ref_func = ref_connection_manager.open

            with open_pred_func(pred_path) as pred_data:
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
                        ref_data = open_ref_func(ref_source, ref_alias)
                else:
                    ref_data = None

                if pred_transform:
                    pred_data = pred_transform(pred_data)
                if ref_data and ref_transform:
                    ref_data = ref_transform(ref_data)

                if ref_is_observation:
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
                        
                        # Convertir chaque ligne du DataFrame en dictionnaire
                        for var_depth_label in return_res.index:
                            # Nettoyer le nom de la variable/profondeur pour en faire une clé valide
                            clean_key = var_depth_label.lower().replace(" ", "_")
                            
                            # Extraire les valeurs RMSD pour tous les lead days
                            rmsd_values = return_res.loc[var_depth_label].to_dict()
                            
                            # Structure : {variable: {lead_day: rmsd_value}}
                            res_dict[clean_key] = rmsd_values

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

            # 3. Libérer les objets volumineux
            if ref_data is not None:
                ref_data.close()
                del ref_data

            gc.collect()
            return res
        except Exception as exc:
            logger.error(f"Error computing metrics for date {forecast_reference_time}: {repr(exc)}")
            return {
                "model": model,
                "ref_alias": ref_alias,
                "result": None,
            }
