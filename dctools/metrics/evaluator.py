from abc import ABC
from argparse import Namespace
from typing import Any, Dict, List, Optional

import dask
from dask.distributed import Client

from dctools.data.dataloader import DatasetLoader
from dctools.data.dataset import DCDataset
from dctools.metrics.oceanbench_metrics import DCMetric



class Evaluator(ABC):
    def __init__(
            self,
            conf_args: Namespace,
            dask_cluster: object,
            metrics: Optional[List[DCMetric]] = None,
            data_container: Optional[Dict[str, DatasetLoader]] = None,
        ):
        """
        :param reference_loader: Instance de DatasetLoader pour charger la référence.
        :param prediction_loaders: Dictionnaire {nom_model: DatasetLoader}.
        :param metrics: Liste d'instances de métriques.
        """
        self.args = conf_args
        self.dask_cluster = dask_cluster
        self._data_container = data_container
        self._model_name = list(self._data_container.keys())[0] if data_container else None
        self._metrics = metrics

        self._plot = False
        self._tasks: Dict[str, Dict[str, Any]] = {}
        self.results: Dict[str, Dict[str, Any]] = {}

    def reset_tasks(self):
        self._tasks = {}
        self.results = {}

    def set_metrics(self, metrics: List[DCMetric]):
        """
        :param metrics: Liste d'instances de métriques.
        """
        self._metrics = metrics
        self.args.dclogger.info(f"Metrics set: {self._metrics}")

    def evaluate(self):
        if self._data_container and self._metrics:
            try:
                with Client(self.dask_cluster) as dask_client:
                    self.reset_tasks()
                    #print('before')
                    for model_name, data_loader in self._data_container.items():
                        for batch in data_loader.load(): #self._data_container.items():
                            #print('Model name:', model_name)
                            #print('Data loader:', data_loader)
                            self._tasks[model_name] = {}
                            self.results[model_name] = {}
                            date: str = str()
                            for date, pred_dataset, ref_dataset in batch:
                                #self._model_name = model_name
                                # Chargement des datasets de prédiction
                                #print('Sample:', sample)
                                self.args.dclogger.info(f"Load data for model: {model_name} date: {date}")
                                #print('Sample:', sample)

                                """for date, pred_dataset, ref_dataset in zip(
                                    sample.load_date(),
                                    sample.load_pred(),
                                    sample.load_ref(),
                                ):"""
                                #print('Date:', date)
                                self.results[model_name][date] = {}
                                self._tasks[model_name][date] = {}

                                pred_dataset_future = dask_client.scatter(pred_dataset, broadcast=True)
                                ref_dataset_future = dask_client.scatter(ref_dataset, broadcast=True)

                                for metric in self._metrics:
                                    self.args.dclogger.info(f"Compute metric: {metric.get_metric_name()} for model: {model_name} date: {date}")
                                    #print('Metric:', metric)
                                    metric_name = metric.get_metric_name()
                                    self._tasks[self._model_name][date][metric_name] = dask.delayed(metric.compute)(
                                        pred_dataset_future, ref_dataset_future
                                    )
                            self.args.dclogger.info(f"Run set of tasks: {self._tasks[model_name]}")

                            # Convertir tout le dictionnaire en liste de tâches
                            tasks_list = [
                                task for model in self._tasks.values() for date in model.values() for task in date.values()
                            ]

                            # Envoyer les tâches aux workers et récupérer les résultats
                            futures = dask_client.compute(tasks_list)  # Envoi au cluster
                            results = dask_client.gather(futures)  # Récupération des résultats
                            #print('Results:', results)
                            i = 0
                            for model_name in self._tasks:
                                #dclogger = self.args.dclogger
                                #dclogger.info(f"Show results for Model : {model_name}")
                                for date in self._tasks[model_name]:
                                    #dclogger.info(f"    Show results for Date : {date}")
                                    for metric_name in self._tasks[model_name][date]:
                                        #dclogger.info(f"        Show results for Metric : {metric_name}")
                                        #dclogger.info(f"            i: {i} Result : {results[i]}")
                                        self.results[model_name][date][metric_name] = results[i]
                                        i += 1
                                #self.args.json_logger.info(self.results)

                            # self.args.dclogger.info(f"WORKERS RESULTS: {self.results}")
                return self.results
            except Exception as exc:
                self.args.exception_handler.handle_exception(
                    exc, "Evaluator failed."
                )
        else:
            self.args.dclogger.error("No metrics or data container set.")
            return None


