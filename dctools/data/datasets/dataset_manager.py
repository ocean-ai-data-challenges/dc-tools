
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
from dctools.data.coordinates import (
    CoordinateSystem,
    LIST_VARS_GLONET,
    GLONET_DEPTH_VALS,
    RANGES_GLONET,
)
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
            # logger.debug(f"Filtrage du dataset '{alias}' par variables : {variables}")
            self.filter_by_variable(alias, variables)

    def to_json(self, alias: str, path: Optional[str] = None) -> str:
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
        dataset.get_catalog().to_json(path)

    def all_to_json(self, output_dir: str):
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

            # Appeler la méthode to_json() pour chaque dataset
            self.to_json(alias, path=json_path)

            logger.info(f"Dataset '{alias}' exporté au format JSON dans '{json_path}'.")

    def standardize_names(
        self, alias: str,
        coords_rename_dict: Dict[str, str],
        vars_rename_dict: Dict[str, str],
    ) -> None:
        """
        Standardise les noms des variables d'un dataset en fonction d'un dictionnaire de correspondance.

        Args:
            alias (str): Alias du dataset à standardiser.
            standard_names (Dict[str, str]): Dictionnaire de correspondance des noms.
        """
        if alias not in self.datasets:
            raise ValueError(f"Alias '{alias}' not found in the manager.")
        # logger.debug(f"Standardizing names for dataset '{alias}' with coords: {coords_rename_dict} and vars: {vars_rename_dict}")
        self.datasets[alias].standardize_names(coords_rename_dict, vars_rename_dict)
        # logger.debug(f"Dataset {alias} eval_variables after standardization: {self.datasets[alias].eval_variables}")

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
        if not ref_alias:
            ref_dataset = None
        else:
            ref_dataset = self.datasets[ref_alias]

        # Récupérer les ConnectionManager associés
        pred_manager = pred_dataset.get_connection_manager()
        ref_manager = ref_dataset.get_connection_manager() if ref_alias else None

        # Filtrer le catalogue pour les alias spécifiés
        pred_catalog = self.datasets[pred_alias].get_catalog()
        ref_catalog = self.datasets[ref_alias].get_catalog() if ref_alias else None
        eval_variables = pred_dataset.eval_variables

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
            pred_alias = pred_alias,
            batch_size=batch_size,
            pred_transform=pred_transform,
            ref_transform=ref_transform,
            eval_variables=eval_variables,
        )

    def _filter_catalog_by_alias(self, alias: str) -> DatasetCatalog:
        """
        Filtre le catalogue global par alias.

        Args:
            alias (str): Alias du dataset à filtrer.

        Returns:
            DatasetCatalog: Catalogue filtré pour l'alias spécifié.
        """
        filtered_df = self.datasets["alias"].get_dataframe()
        global_metadata = self.catalog.get_global_metadata()
        filtered_df = self.catalog.get_dataframe()[self.catalog.get_dataframe()["alias"] == alias]
        return DatasetCatalog(entries=filtered_df.to_dict(orient="records"), global_metadata=global_metadata)


    def get_transform(
        self,
        transform_name: str,
        dataset_alias: str,
        **kwargs,
    ) -> Any:
        """
        Factory function to create a transform based on the given name and parameters.
        """
        logger.debug(f"Creating transform {transform_name} with kwargs: {kwargs}")

        # catalog = self.get_catalog(dataset_alias)
        global_metadata = self.datasets[dataset_alias].get_global_metadata()
        coords_rename_dict = global_metadata.get("dimensions")
        vars_rename_dict= global_metadata.get("variables_rename_dict")
        keep_vars = global_metadata.get("keep_variables")


        # Configurer les transformations
        match transform_name:
            case "standardize":
                transform = CustomTransforms(
                    transform_name="standardize_dataset",
                    list_vars=keep_vars,
                    coords_rename_dict=coords_rename_dict,
                    vars_rename_dict=vars_rename_dict,
                )
                self.standardize_names(
                    dataset_alias,
                    coords_rename_dict,
                    vars_rename_dict,
                )
            case "glorys_to_glonet":
                regridder_weights = kwargs.get("regridder_weights", None)
                assert(regridder_weights is not None), "Regridder weights path must be provided for GLONET transformation"
                transform = CustomTransforms(
                    transform_name="glorys_to_glonet",
                    weights_path=regridder_weights,
                    depth_coord_vals=GLONET_DEPTH_VALS,
                    interp_ranges=RANGES_GLONET,
                )
            case _:
                transform = None

        return transform