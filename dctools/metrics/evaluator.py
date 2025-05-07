from abc import ABC
from argparse import Namespace
from typing import Any, Dict, List, Optional

import dask
from dask.distributed import Client
from loguru import logger

# from dctools.data.datasets.dataloader import DatasetLoader
from dctools.data.datasets.dataloader import EvaluationDataloader
from dctools.metrics.oceanbench_metrics import DCMetric
from dctools.metrics.metrics import MetricComputer


class Evaluator:
    def __init__(
        self,
        dask_cluster: object,
        metrics: List[MetricComputer],
        dataloader: EvaluationDataloader,
    ):
        """
        Initialise l'évaluateur.

        Args:
            dask_cluster (object): Cluster Dask pour la parallélisation.
            metrics (List[MetricComputer]): Liste des métriques à calculer.
            dataloader (EvaluationDataloader): Dataloader pour charger les données.
        """
        self.dask_cluster = dask_cluster
        self.metrics = metrics
        self.dataloader = dataloader
        self.results = []

    def evaluate(self) -> List[Dict[str, Any]]:
        """
        Évalue les métriques sur les données du dataloader.

        Returns:
            List[Dict[str, Any]]: Résultats des métriques pour chaque lot.
        """
        try:
            with Client(self.dask_cluster) as dask_client:
                for batch in self.dataloader:
                    batch_results = self._evaluate_batch(batch, self.dataloader, dask_client)
                    self.results.extend(batch_results)
            return self.results
        except Exception as exc:
            logger.error(f"Evaluation failed: {repr(exc)}")
            return []

    def _evaluate_batch(
        self, batch: List[Dict[str, Any]],
        dataloader: EvaluationDataloader,
        dask_client: Client
    ) -> List[Dict[str, Any]]:
        """
        Évalue un lot de données.

        Args:
            batch (List[Dict[str, Any]]): Lot de données.
            dask_client (Client): Client Dask.

        Returns:
            List[Dict[str, Any]]: Résultats des métriques pour le lot.
        """
        tasks = []
        for entry in batch:
            try:
                date = entry["date"]
                pred_data = dataloader.open_pred(entry["pred_data"])
                ref_data = dataloader.open_ref(entry["ref_data"]) if entry["ref_data"] else None

                if dataloader.pred_transform:
                    pred_data = dataloader.pred_transform(pred_data)
                if dataloader.ref_transform:
                    ref_data = dataloader.ref_transform(pred_data)

                # Partager les données entre les workers
                pred_future = dask_client.scatter(pred_data, broadcast=True)
                ref_future = dask_client.scatter(ref_data, broadcast=True)

                # Construire les tâches pour chaque métrique
                for metric in self.metrics:
                    task = dask.delayed(self._compute_metric)(
                        metric, pred_future, ref_future, date
                    )
                    tasks.append(task)
            except Exception as exc:
                logger.error(f"Error processing entry {entry}: {repr(exc)}")
                continue

        # Exécuter les tâches et récupérer les résultats
        futures = dask_client.compute(tasks)
        return dask_client.gather(futures)

    @staticmethod
    def _compute_metric(metric: MetricComputer, pred_data, ref_data, date: str) -> Dict[str, Any]:
        """
        Calcule une métrique pour un lot de données.

        Args:
            metric (MetricComputer): Instance de la métrique.
            pred_data: Données de prédiction.
            ref_data: Données de référence.
            date (str): Date associée au lot.

        Returns:
            Dict[str, Any]: Résultat de la métrique.
        """
        try:
            result = metric.compute(pred_data, ref_data)
            return {"date": date, "metric": metric.get_metric_name(), "result": result}
        except Exception as exc:
            logger.error(f"Error computing metric {metric.get_metric_name()} for date {date}: {repr(exc)}")
            return {"date": date, "metric": metric.get_metric_name(), "result": None}