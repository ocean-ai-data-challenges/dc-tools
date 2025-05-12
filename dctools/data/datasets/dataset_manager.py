
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from datetime import datetime
import geopandas as gpd
import json
from loguru import logger
import pandas as pd
from shapely.geometry import box
import xarray as xr

from dctools.data.connection.config import BaseConnectionConfig
from dctools.data.datasets.dataset import BaseDataset, DatasetConfig
from dctools.data.transforms import CustomTransforms
from dctools.data.datasets.dc_catalog import DatasetCatalog
from dctools.data.datasets.dataloader import EvaluationDataloader

class MultiSourceDatasetManager:

    def __init__(self):
        """
        Initialise le gestionnaire multi-sources.
        """
        self.datasets = {}
        # self.catalog = DatasetCatalog([])  # Initialiser avec un catalogue vide

    def add_dataset(self, alias: str, dataset: BaseDataset):
        """
        Ajoute un dataset au gestionnaire avec un alias.

        Args:
            alias (str): Alias unique pour le dataset.
            dataset (BaseDataset): Instance du dataset.
        """
        if alias in self.datasets:
            raise ValueError(f"Alias '{alias}' already exists.")
        self.datasets[alias] = dataset

    def build_catalogs(self):
        """
        Construit les catalogues pour tous les datasets.
        """
        for alias, dataset in self.datasets.items():
            dataset.build_catalog()

    def get_catalog(self, alias: str) -> DatasetCatalog:
        """
        Retourne le catalogue d'un dataset.

        Args:
            alias (str): Alias du dataset.

        Returns:
            DatasetCatalog: Catalogue du dataset.
        """
        if alias not in self.datasets:
            raise ValueError(f"Alias '{alias}' not found in the manager.")
        return self.datasets[alias].get_catalog()

    def get_data(self, alias: str, path: str) -> xr.Dataset:
        """
        Charge les données d'un dataset.

        Args:
            alias (str): Alias du dataset.
            path (str): Chemin du fichier.

        Returns:
            xr.Dataset: Dataset chargé.
        """
        return self.datasets[alias].load_data(path)

    def filter_attrs(
        self, filters: dict[str, Union[Callable[[Any], bool], gpd.GeoSeries]]
    ) -> None:
        """
        Filtre les datasets en fonction des attributs.

        Args:
            filters (dict[str, Callable[[Any], bool]]): Dictionnaire de filtres.
        """
        for alias, dataset in self.datasets.items():
            dataset.filter_attrs(filters)

    def filter_by_date(self, alias: str,start: datetime, end: datetime):
        """
        Filtre le catalogue par plage temporelle.

        Args:
            start (datetime): Date de début.
            end (datetime): Date de fin.
        """
        self.datasets[alias].filter_catalog_by_date(start, end)

    def filter_by_region(self, alias: str, region: gpd.GeoSeries):
        """
        Filtre le catalogue par boîte englobante.

        Args:
            bbox (Tuple[float, float, float, float]): (lon_min, lat_min, lon_max, lat_max).
        """
        self.datasets[alias].filter_catalog_by_region(region)

    def filter_by_variable(self, alias: str, variables: List[str]):
        """
        Filtre le catalogue par variable.

        Args:
            variables (List[str]): Liste des variables à filtrer.
        """
        self.datasets[alias].filter_catalog_by_variable(variables)

    def filter_all_by_date(self, start: datetime, end: datetime):
        """
        Filtre tous les datasets gérés par cette classe par plage temporelle.

        Args:
            start (datetime): Date de début.
            end (datetime): Date de fin.
        """
        for alias, _ in self.datasets.items():
            logger.info(f"Filtrage du dataset '{alias}' par date : {start} -> {end}")
            self.filter_by_date(alias, start, end)

    def filter_all_by_region(self, region: gpd.GeoSeries):
        """
        Filtre tous les datasets gérés par cette classe par par boîte englobante.

        Args:
            bbox (Tuple[float, float, float, float]): (lon_min, lat_min, lon_max, lat_max).
        """
        for alias, _ in self.datasets.items():
            logger.info(f"Filtrage du dataset '{alias}' par bbox : {region}")
            self.filter_by_region(alias, region)

    def filter_all_by_variable(self, variables: List[str]):
        """
        Filtre tous les datasets gérés par cette classe par plage temporelle.

        Args:
            variables (List[str]): Liste des variables à filtrer.
        """
        for alias, _ in self.datasets.items():
            logger.info(f"Filtrage du dataset '{alias}' par variables : {variables}")
            self.filter_by_variable(alias, variables)

    def to_file(self, alias: str, path: Optional[str] = None) -> str:
        """
        Exporte les informations d'un dataset au format JSON.

        Args:
            alias (str): Alias du dataset à exporter.
            path (Optional[str]): Chemin pour sauvegarder le fichier JSON.

        Returns:
            str: Représentation JSON des informations du dataset.
        """
        if alias not in self.datasets:
            raise ValueError(f"Alias '{alias}' not found in the manager.")

        dataset = self.datasets[alias]
        dataset.get_catalog().to_file(path)
        '''# Construire le dictionnaire des informations du dataset
        dataset_info = {
            "alias": alias,
            #"connection_config": dataset.connection_manager.params,  # Configuration de connexion
            "catalog": dataset.get_catalog().to_json(),  # Catalogue
            #"catalog": dataset.get_catalog().get_dataframe().to_dict(orient="records"),  # Catalogue
        }'''

        '''# Convertir en JSON
        json_str = json.dumps(dataset_info, indent=4)

        # Sauvegarder dans un fichier si un chemin est fourni
        if path:
            with open(path, "w") as f:
                f.write(json_str)

        return json_str'''

    def all_to_file(self, output_dir: str):
        """
        Exporte les informations de tous les datasets au format JSON.

        Args:
            output_dir (str): Répertoire où sauvegarder les fichiers JSON.

        Raises:
            ValueError: Si le répertoire spécifié n'existe pas ou n'est pas accessible.
        """

        # Vérifier si le répertoire existe
        if not os.path.exists(output_dir):
            raise ValueError(f"Le répertoire spécifié '{output_dir}' n'existe pas.")
        if not os.path.isdir(output_dir):
            raise ValueError(f"Le chemin spécifié '{output_dir}' n'est pas un répertoire.")

        # Boucler sur tous les datasets
        for alias, _ in self.datasets.items():
            # Générer le chemin du fichier JSON
            json_filename = f"{alias}.json"
            json_path = os.path.join(output_dir, json_filename)

            # Appeler la méthode to_file() pour chaque dataset
            self.to_file(alias, path=json_path)

            logger.info(f"Dataset '{alias}' exporté au format JSON dans '{json_path}'.")

    '''@staticmethod
    def add_from_json(json_paths: List[str]) -> 'MultiSourceDatasetManager':
        """
        Charge un gestionnaire multi-sources à partir d'une liste de fichiers JSON.

        Args:
            json_paths (List[str]): Liste des chemins vers les fichiers JSON, chaque fichier
                                    contenant les informations d'un dataset.

        Returns:
            MultiSourceDatasetManager: Instance du gestionnaire multi-sources.
        """
        manager = MultiSourceDatasetManager()

        try:
            for json_path in json_paths:
                # Charger le fichier JSON
                with open(json_path, "r") as f:
                    dataset_info = json.load(f)

                # Extraire l'alias et vérifier sa présence
                alias = dataset_info.get("alias")
                if not alias:
                    raise ValueError(f"Le fichier JSON '{json_path}' ne contient pas de champ 'alias'.")

                # Charger la configuration du dataset
                config = DatasetConfig(
                    alias=dataset_info["alias"],
                    connection_config=BaseConnectionConfig(**dataset_info["connection_config"]),
                    # catalog_options=dataset_info.get("catalog_options", {}),
                )

                # Créer une instance de BaseDataset (ou une classe dérivée)
                dataset = BaseDataset(config)

                # Charger le catalogue du dataset
                if "catalog" in dataset_info:
                    catalog = DatasetCatalog(dataset_info["catalog"])
                    dataset.catalog = catalog

                # Ajouter le dataset au gestionnaire
                manager.add_dataset(alias, dataset)

        except Exception as exc:
            logger.error(f"Erreur lors du chargement du gestionnaire depuis les fichiers JSON : {repr(exc)}")
            raise

        return manager'''

    def get_dataloader(
        self,
        pred_alias: str,
        ref_alias: Optional[str] = None,
        batch_size: Optional[int] = 8,
        pred_transform: Optional[CustomTransforms] = None,
        ref_transform: Optional[CustomTransforms] = None,
    ) -> EvaluationDataloader:
        """
        Crée un EvaluationDataloader à partir des alias des datasets.

        Args:
            pred_alias (str): Alias du dataset de prédiction.
            ref_alias (str): Alias du dataset de référence.
            batch_size (int): Taille des lots.
            pred_transform (Optional[CustomTransforms]): Transformation pour les prédictions.
            ref_transform (Optional[CustomTransforms]): Transformation pour les références.

        Returns:
            EvaluationDataloader: Instance du dataloader.
        """

        pred_dataset = self.datasets[pred_alias]
        ref_dataset = self.datasets[ref_alias] if ref_alias else None

        # Récupérer les ConnectionManager associés
        pred_manager = pred_dataset.get_connection_manager()
        ref_manager = ref_dataset.get_connection_manager() if ref_alias else None

        # Filtrer le catalogue pour les alias spécifiés
        pred_catalog = self.datasets[pred_alias].get_catalog()
        ref_catalog = self.datasets[ref_alias].get_catalog() if ref_alias else None
        # logger.debug(f"Filtered catalogs: {pred_catalog}, {ref_catalog}")
        # logger.debug(f"Pred catalog: {pred_catalog.get_dataframe()}")
        # logger.debug(f"Ref catalog: {ref_catalog.get_dataframe()}")


        if pred_catalog.get_dataframe().empty:
            raise ValueError(f"Catalog entries for alias '{pred_alias}' are empty.")
        if ref_alias and ref_catalog.get_dataframe().empty:
            raise ValueError(f"Catalog entries for alias '{ref_alias}' are empty.")

        # Créer un dataloader avec les catalogues filtrés
        return EvaluationDataloader(
            pred_catalog=pred_catalog,
            ref_catalog=ref_catalog,
            pred_manager=pred_manager,
            ref_manager=ref_manager,
            batch_size=batch_size,
            pred_transform=pred_transform,
            ref_transform=ref_transform,
        )

    def _filter_catalog_by_alias(self, alias: str) -> DatasetCatalog:
        """
        Filtre le catalogue global par alias.

        Args:
            alias (str): Alias du dataset à filtrer.

        Returns:
            DatasetCatalog: Catalogue filtré pour l'alias spécifié.
        """
        filtered_df = self.catalog.get_dataframe()[self.catalog.get_dataframe()["alias"] == alias]
        return DatasetCatalog(entries=filtered_df.to_dict(orient="records"))
