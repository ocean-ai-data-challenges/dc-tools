from abc import ABC
from argparse import Namespace
from typing import Dict, List, Optional

import dask

from dctools.data.dataloader import DatasetLoader
from dctools.metrics.oceanbench_metrics import DCMetric



class Evaluator(ABC):
    def __init__(
            self,
            conf_args: Namespace,
            dask_client: object,
            metrics: Optional[List[DCMetric]] = None,
            data_container: Optional[Dict[str, DatasetLoader]] = None,
        ):
        """
        :param reference_loader: Instance de DatasetLoader pour charger la référence.
        :param prediction_loaders: Dictionnaire {nom_model: DatasetLoader}.
        :param metrics: Liste d'instances de métriques.
        """
        self.args = conf_args
        self.dask_client = dask_client
        self._data_container = data_container
        self._model_name = list(self._data_container.keys())[0] if data_container else None
        self._metrics = metrics

        self._plot = False
        self._tasks = {}
        self.results = {}
        if metrics and data_container:
            self.set_tasks()

    def reset_tasks(self):
        self._tasks = {}
        self.results = {}

    def set_data_container(self, data_container: Dict[str, DatasetLoader]):
        self._data_container = data_container
        self._model_name = list(self._data_container.keys())[0]

    def reset_data_container(self, model_name: str):
        """Reset le dataset de référence et de prédiction"""
        self._data_container[model_name].reset()

    def set_metrics(self, metrics: List[DCMetric]):
        self._metrics = metrics

    def evaluate(self):

        if self._data_container and self._metrics:
            self.reset_tasks()
            for model_name, data_loader in self._data_container.items():
                self._model_name = model_name
                # Chargement des datasets de prédiction
                self._tasks[model_name] = {}
                self.results[model_name] = {}

                for date, pred_dataset, ref_dataset in zip(
                    data_loader.load_date(),
                    data_loader.load_pred(),
                    data_loader.load_ref(),
                ):
                    self.results[model_name][date] = {}
                    self._tasks[model_name][date] = {}

                    pred_dataset_future = self.dask_client.scatter(pred_dataset, broadcast=True)
                    ref_dataset_future = self.dask_client.scatter(ref_dataset, broadcast=True)

                    for metric in self._metrics:
                        self._tasks[model_name][date][metric.get_metric_name()] = dask.delayed(metric.compute)(
                            pred_dataset_future, ref_dataset_future,
                        )
                    self.args.dclogger.info(f"Run set of tasks: {self._tasks[model_name]}")

                    # Convertir tout le dictionnaire en liste de tâches
                    tasks_list = [task for model in self._tasks.values() for date in model.values() for task in date.values()]

                    # Envoyer les tâches aux workers et récupérer les résultats
                    futures = self.dask_client.compute(tasks_list)  # Envoi au cluster
                    results = self.dask_client.gather(futures)  # Récupération des résultats

                    i = 0
                    for model_name in self._tasks:
                        for date in self._tasks[model_name]:
                            for metric_name in self._tasks[model_name][date]:
                                self.results[model_name][date][metric_name] = results[i]
                                i += 1
                    self.args.json_logger.info(self.results)

                    self.args.dclogger.info(f"WORKERS RESULTS: {self.results}")

