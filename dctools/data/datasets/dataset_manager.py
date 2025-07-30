
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from datetime import datetime
import geopandas as gpd
import json
from loguru import logger
import pandas as pd
from shapely.geometry import box
from torchvision import transforms
import xarray as xr

from dctools.data.connection.config import BaseConnectionConfig
from dctools.data.coordinates import (
    LIST_VARS_GLONET,
    GLONET_DEPTH_VALS,
    RANGES_GLONET,
    CoordinateSystem,
)
from dctools.data.datasets.dataset import BaseDataset, DatasetConfig
from dctools.data.datasets.forecast import build_forecast_index_from_catalog
from dctools.data.transforms import CustomTransforms, WrapLongitudeTransform
from dctools.data.datasets.dc_catalog import DatasetCatalog
from dctools.data.datasets.dataloader import EvaluationDataloader
from dctools.utilities.file_utils import FileCacheManager

class MultiSourceDatasetManager:

    def __init__(self, time_tolerance, max_cache_files):
        """
        Initialise le gestionnaire multi-sources.
        """
        self.datasets = {}
        self.forecast_indexes = {}
        self.time_tolerance = time_tolerance
        self.file_cache = FileCacheManager(max_cache_files)

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

    def filter_all_by_date(
            self,
            start: datetime | list[datetime],
            end: datetime | list[datetime],
        ):
        """
        Filtre tous les datasets gérés par cette classe par plage temporelle.

        Args:
            start (datetime): Date(s) de début.
            end (datetime): Date(s) de fin.
        """
        for alias, _ in self.datasets.items():
            # TODO: Fix logger output when using lists as start and end
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


    def add_to_file_cache(self, filepath: str):
        self.file_cache.add(filepath)


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
        self.datasets[alias].standardize_names(coords_rename_dict, vars_rename_dict)

    def build_forecast_index(
        self, alias: str,
        init_date: str,
        end_date: str,
        n_days_forecast: int,
        n_days_interval: int,
    ):

        dataset = self.datasets[alias]
        catalog = dataset.get_catalog()
        catalog_df = catalog.get_dataframe()
        forecast_index = build_forecast_index_from_catalog(
            catalog_df,
            init_date=init_date,
            end_date=end_date,
            forecast_time_col="date_start",
            valid_time_col="date_end",
            file_col="path",
            n_days_forecast=n_days_forecast,
            n_days_interval=n_days_interval,
            lead_time_unit="days",
        )

        self.forecast_indexes[alias] = forecast_index

    def get_dataloader(
        self,
        pred_alias: str,
        ref_aliases: Optional[List[str]] = None,
        batch_size: Optional[int] = 8,
        pred_transform: Optional[CustomTransforms] = None,
        ref_transforms: Optional[List[CustomTransforms]] = None,
        forecast_mode: bool = False,
        n_days_forecast: int = 0,
        lead_time_unit: str = "days",
    ) -> EvaluationDataloader:
        """
        Crée un EvaluationDataloader à partir des alias des datasets.

        Args:
            pred_alias (str): Alias du dataset de prédiction.
            ref_aliases (Optional[List[str]]): Alias des datasets de référence.
            batch_size (int): Taille des lots.
            pred_transform (Optional[CustomTransforms]): Transformation pour les prédictions.
            ref_transforms (Optional[List[CustomTransforms]]): Transformations pour les références.
            forecast_mode (bool): Active le mode forecast.
            n_days_forecast (int): Nombre de jours de forecast à considérer.
            lead_time_unit (str): Unité du lead time ("days" ou "hours").

        Returns:
            EvaluationDataloader: Instance du dataloader.
        """

        pred_dataset = self.datasets[pred_alias]
        ref_datasets = {}
        for ref_alias in ref_aliases:
            ref_datasets[ref_alias] = self.datasets[ref_alias]

        pred_manager = pred_dataset.get_connection_manager()
        ref_managers = {}
        ref_catalogs = {}
        ref_connection_params= {}
        for ref_alias in ref_aliases:
            ref_managers[ref_alias] = ref_datasets[ref_alias].get_connection_manager()
            ref_catalogs[ref_alias] = ref_datasets[ref_alias].get_catalog()
            ref_connection_params[ref_alias] = ref_datasets[ref_alias].get_connection_config()

        pred_catalog = pred_dataset.get_catalog()
        pred_connection_params = pred_dataset.get_connection_config()

        # --- Ajout du mode forecast ---
        forecast_index = None
        if forecast_mode:
            # catalog_df = pred_catalog.get_dataframe()
            forecast_index = self.forecast_indexes[pred_alias]

        return EvaluationDataloader(
            pred_connection_params=pred_connection_params,
            ref_connection_params=ref_connection_params,
            pred_catalog=pred_catalog,
            ref_catalogs=ref_catalogs,
            pred_manager=pred_manager,
            ref_managers=ref_managers,
            pred_alias=pred_alias,
            ref_aliases=ref_aliases,
            batch_size=batch_size,
            pred_transform=pred_transform,
            ref_transforms=ref_transforms,
            forecast_mode=forecast_mode,
            forecast_index=forecast_index,
            n_days_forecast=n_days_forecast,
            time_tolerance=self.time_tolerance,
        )


    def get_transform(
        self,
        transform_name: str,
        dataset_alias: str,
        **kwargs,
    ) -> Any:
        """
        Factory function to create a transform based on the given name and parameters.
        """
        # logger.debug(f"Creating transform {transform_name} with kwargs: {kwargs}")

        # catalog = self.get_catalog(dataset_alias)
        global_metadata = self.datasets[dataset_alias].get_global_metadata()
        coord_sys = global_metadata.get("coord_system", None)
        if global_metadata is None or coord_sys is None:
            raise("Cannot import dataset metadata.")
        if isinstance(coord_sys, dict):
            # Si le metadata global est un dictionnaire, on peut l'utiliser directement
            coord_dict = coord_sys["coordinates"]
        elif isinstance(coord_sys, CoordinateSystem):
            coord_dict = coord_sys.coordinates
        coords_rename_dict = {v: k for k, v in coord_dict.items()}
        vars_rename_dict= global_metadata.get("variables_rename_dict")
        keep_vars = global_metadata.get("keep_variables")


        # Configurer les transformations
        match transform_name:
            case "standardize_lons_interpolate":
                regridder_weights = kwargs.get("regridder_weights", None)
                interp_ranges = kwargs.get("interp_ranges", None)
                transform_std = CustomTransforms(
                    transform_name="standardize_dataset",
                    list_vars=keep_vars,
                    coords_rename_dict=coords_rename_dict,
                    vars_rename_dict=vars_rename_dict,
                )
                transform_lon_vars = WrapLongitudeTransform()

                transform_interp = CustomTransforms(
                    transform_name="interpolate",
                    interp_ranges=interp_ranges,
                    weights_path=regridder_weights,
                )
                transform = transforms.Compose(
                    [
                        transform_lon_vars,
                        transform_std,
                        transform_interp,
                    ]
                )
                self.standardize_names(
                    dataset_alias,
                    coords_rename_dict,
                    vars_rename_dict,
                )
            case "standardize_interpolate":
                regridder_weights = kwargs.get("regridder_weights", None)
                interp_ranges = kwargs.get("interp_ranges", None)
                transform_std = CustomTransforms(
                    transform_name="standardize_dataset",
                    list_vars=keep_vars,
                    coords_rename_dict=coords_rename_dict,
                    vars_rename_dict=vars_rename_dict,
                )
                transform_interp = CustomTransforms(
                    transform_name="interpolate",
                    interp_ranges=interp_ranges,
                    weights_path=regridder_weights,
                )
                transform = transforms.Compose(
                    [
                        transform_std,
                        transform_interp,
                    ]
                )
                self.standardize_names(
                    dataset_alias,
                    coords_rename_dict,
                    vars_rename_dict,
                )
            case "standardize":
                transform_std = CustomTransforms(
                    transform_name="standardize_dataset",
                    list_vars=keep_vars,
                    coords_rename_dict=coords_rename_dict,
                    vars_rename_dict=vars_rename_dict,
                )
                '''transform_time = CustomTransforms(
                    transform_name="to_timestamp",
                    time_names=["date_start", "date_end"],
                )'''
                transform = transform_std
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
            case "standardize_add_coords":
                transform_standardize = CustomTransforms(
                    transform_name="standardize_dataset",
                    list_vars=keep_vars,
                    coords_rename_dict=coords_rename_dict,
                    vars_rename_dict=vars_rename_dict,
                )
                transform_add_coords = CustomTransforms(
                    transform_name="add_spatial_coords",
                    list_vars=keep_vars,
                    coords_rename_dict=coords_rename_dict,
                )
                transform = transforms.Compose(
                    [
                        transform_add_coords,
                        transform_standardize,
                    ]
                )
            case _:
                transform = None

        return transform
