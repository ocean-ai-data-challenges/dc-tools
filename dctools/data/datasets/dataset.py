

from abc import ABC, abstractmethod
import os
from typing import (
    Any, Callable, Dict, List,
    Iterator, Optional, Tuple, Type,
    Union,
)

import datetime
import geopandas as gpd
import json
from loguru import logger
import pandas as pd
from pathlib import Path
from shapely.geometry import box
import xarray as xr

from dctools.dcio.loader import FileLoader

from dctools.data.datasets.dc_catalog import DatasetCatalog

from dctools.data.connection.config import (
    BaseConnectionConfig,
    LocalConnectionConfig,
    CMEMSConnectionConfig,
    S3ConnectionConfig,
    WasabiS3ConnectionConfig,
    FTPConnectionConfig,
    GlonetConnectionConfig,
)

from dctools.data.connection.connection_manager import (
    BaseConnectionManager,
    LocalConnectionManager,
    CMEMSManager,
    S3Manager,
    S3WasabiManager,
    FTPManager,
    GlonetManager,
)


class DatasetConfig:
    """DATASET_ALIAS_MAP: Dict[str, Type[BaseConnectionConfig]] = {
        "glorys": CMEMSConnectionConfig,
        "glonet": GlonetConnectionConfig,
        "glonet_wasabi": WasabiS3ConnectionConfig,
    }"""
    CONNECTION_MANAGER_MAP: Dict[Type[BaseConnectionConfig], Type[BaseConnectionManager]] = {
        LocalConnectionConfig: LocalConnectionManager,
        CMEMSConnectionConfig: CMEMSManager,
        S3ConnectionConfig: S3Manager,
        WasabiS3ConnectionConfig: S3WasabiManager,
        FTPConnectionConfig: FTPManager,
        GlonetConnectionConfig: GlonetManager,
    }

    def __init__(
        self,
        alias: str,
        connection_config: BaseConnectionConfig,
        catalog_options: Optional[Dict[str, Any]] = None,
    ):
        """
        Configuration pour un dataset.

        Args:
            alias (str): Nom du dataset.
            connection_config (BaseConnectionConfig): Configuration de connexion.
            catalog_options (Optional[Dict[str, Any]]): Options pour le catalogue (e.g., filtres par défaut).
        """
        self.alias = alias
        self.connection_config = connection_config
        self.catalog_options = catalog_options or {}
        self.connection_manager = self._create_connection_manager()

    def _create_connection_manager(self) -> BaseConnectionManager:
        """
        Instancie le ConnectionManager approprié en fonction du type de configuration.

        Returns:
            BaseConnectionManager: Instance du ConnectionManager correspondant.
        """
        config_type = type(self.connection_config)
        if config_type not in self.CONNECTION_MANAGER_MAP:
            raise ValueError(f"Unsupported connection configuration type: {config_type}")

        manager_class = self.CONNECTION_MANAGER_MAP[config_type]
        return manager_class(self.connection_config)

class BaseDataset(ABC):
    def __init__(self, config: DatasetConfig):
        """
        Initialise un dataset avec une configuration.

        Args:
            config (DatasetConfig): Configuration du dataset.
        """
        # Utiliser CONNECTION_MANAGER_MAP pour instancier le bon ConnectionManager
        connection_manager_class = DatasetConfig.CONNECTION_MANAGER_MAP.get(type(config.connection_config))
        if not connection_manager_class:
            raise ValueError(f"Unsupported connection configuration type: {type(config.connection_config)}")

        self.connection_manager = connection_manager_class(config.connection_config)
        self.alias = config.alias
        self.catalog_type = ""
        # Vérifier si un fichier de catalogue JSON est spécifié dans catalog_options
        catalog_path = config.catalog_options.get("catalog_path") if config.catalog_options else None
        if catalog_path and Path(catalog_path).exists():
            logger.info(f"Loading catalog from JSON file: {catalog_path}")
            self.catalog = DatasetCatalog.from_json(catalog_path)
            #self._metadata = self.catalog.entries
            self._paths = self.catalog.list_paths()
            self.catalog_type = "from_catalog_file"
        else:
            logger.info("No catalog JSON file found. Generating metadata from the dataset.")
        
            self._metadata = self.connection_manager.list_files_with_metadata()  # Récupérer les métadonnées
            self._paths = [entry.path for entry in self._metadata]
            self.build_catalog()  # Construire le catalogue à partir des métadonnées
            self.catalog_type = "from_data"

    def list_paths(self) -> List[str]:
        """
        Retourne la liste des chemins des fichiers dans le dataset.

        Returns:
            List[str]: Liste des chemins des fichiers.
        """
        return self._paths

    def get_connection_manager(self) -> BaseConnectionManager:
        """
        Retourne l'instance de ConnectionManager associée au dataset.

        Returns:
            BaseConnectionManager: Instance du ConnectionManager.
        """
        return self.connection_manager

    def get_catalog(self) -> DatasetCatalog:
        """
        Retourne le catalogue du dataset.

        Returns:
            DatasetCatalog: Catalogue du dataset.
        """
        if self.catalog_is_empty():
            logger.warning("No entries in the catalog.")
            return None
        return self.catalog

    def get_metadata(self) -> List[Dict[str, Any]]:
        """
        Retourne les métadonnées des fichiers du dataset.

        Returns:
            List[Dict[str, Any]]: Liste des métadonnées.
        """
        return self._metadata

    def get_path(self, index: int) -> str:
        """
        Retourne le chemin d'un fichier à un index donné.

        Args:
            index (int): Index du fichier.

        Returns:
            str: Chemin du fichier.
        """
        return self._paths[index]

    def download(self, index: int, local_path: str):
        """
        Télécharge un fichier à partir de son index.

        Args:
            index (int): Index du fichier.
            local_path (str): Chemin local où sauvegarder le fichier.
        """
        remote_path = self.get_path(index)
        with self.connection_manager.open(remote_path, 'rb') as remote_file:
            with open(local_path, 'wb') as local_file:
                local_file.write(remote_file.read())

    '''def load_item(self, index: int) -> xr.Dataset:
        """
        Charge un fichier en tant que dataset Xarray.

        Args:
            index (int): Index du fichier.

        Returns:
            xr.Dataset: Dataset chargé.
        """
        path = self.get_path(index)
        return FileLoader.open_dataset_auto(path, self.connection_manager)'''

    def iter_data(self) -> Iterator[xr.Dataset]:
        """
        Itère sur les fichiers du dataset et les charge en tant que datasets Xarray.

        Yields:
            xr.Dataset: Dataset chargé.
        """
        for idx in range(len(self._paths)):
            yield self.load_data(idx)


    def build_catalog(self) -> None:
        """
        Construit un catalogue pour ce dataset.

        Returns:
        """
        if self.catalog_type == "from_catalog_file":
            logger.info("Le catalogue existe déjà (chargé à partir d'un fichier)")
            return
        self.catalog = DatasetCatalog(entries=self._metadata)
        #return gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")


    '''def filter_attrs(
        self, filters: dict[str, Union[Callable[[Any], bool], gpd.GeoSeries]]
    ) -> None:
        self.catalog.filter_attrs(filters)'''


    def filter_catalog_by_date(self, start: datetime, end: datetime):
        """
        Filtre le catalogue par plage temporelle.

        Args:
            start (datetime): Date de début.
            end (datetime): Date de fin.
        """
        self.catalog.filter_by_date(start, end)

    def filter_catalog_by_region(self, region: gpd.GeoSeries):
        """
        Filtre le catalogue par boîte englobante.

        Args:
            bbox (Tuple[float, float, float, float]): (lon_min, lat_min, lon_max, lat_max).
        """
        self.catalog.filter_by_region(region)

    def filter_catalog_by_variable(self, variables: List[str]):
        """
        Filtre le catalogue par les variables spécifiées.

        Args:
            variables (List[str]): Liste des noms de variables à filtrer.
        """
        if self.catalog_is_empty():
            logger.warning("Le catalogue est vide. Aucun filtrage ne sera appliqué.")
            return

        # Appeler la méthode filter_by_variables de DatasetCatalog
        self.catalog.filter_by_variables(variables)
        logger.info(f"Catalogue filtré avec succès pour les variables : {variables}")

    def load_data(self, index: int) -> xr.Dataset:
        """
        Charge un dataset à partir d'un chemin.

        Args:
            path (str): Chemin du fichier.

        Returns:
            xr.Dataset: Dataset chargé.
        """
        path = self.get_path(index)
        return self.connection_manager.open(path)

    def catalog_is_empty(self) -> bool:
        """
        Vérifie si le catalogue est vide.

        Returns:
            bool: True si le catalogue est vide, sinon False.
        """
        return self.catalog.get_dataframe().empty

    def to_json(self, path: str) -> None:
        """
        Exporte l'intégralité du contenu de BaseDataset au format JSON.

        Args:
            path (str): Chemin pour sauvegarder le fichier JSON.
        """
        try:
            logger.info(f"Exportation de BaseDataset en JSON dans {path}")
            logger.info(f"Catalogue : {self.catalog}")
            # Sauvegarder le catalogue en JSON
            self.catalog.to_json(str(path))
            # Construire un dictionnaire pour les attributs de BaseDataset
            logger.info(f"BaseDataset sauvegardé avec succès dans {path}")
        except Exception as exc:
            logger.error(f"Erreur lors de l'exportation de BaseDataset en JSON : {repr(exc)}")
            raise


class RemoteDataset(BaseDataset):
    """Generic dataset for remote sources."""
    #def __init__(self, config: DatasetConfig):
    #    super().__init__(config)

    def download(self, index: int, local_path: str):
        """
        Download a file from the remote source.

        Args:
            index (int): Index of the file to download.
            local_path (str): Path to save the downloaded file locally.
        """
        remote_path = self.get_path(index)
        with self.connection_manager.open(remote_path, 'rb') as remote_file:
            with open(local_path, 'wb') as local_file:
                local_file.write(remote_file.read())




class LocalDataset(BaseDataset):
    """Dataset pour les fichiers locaux (NetCDF ou autres)."""
    #def __init__(self, config: DatasetConfig):
    #    super().__init__(config)
    def empty_fct(self):
        """
        Fonction vide.
        """
        pass



def get_dataset_from_config(
    source: dict,
    root_data_folder: str,
    root_catalog_folder: str,
    max_samples: Optional[int] = 0,
    use_catalog: bool = False,
) -> RemoteDataset:
    """Get dataset from config."""
    # Load config
    dataset_name = source['dataset']
    config_name = source['config']
    connection_type = source['connection_type']

    data_root = os.path.join(
        root_data_folder,
        dataset_name,
    )

    catalog_path = os.path.join(
        root_catalog_folder,
        dataset_name + ".json",
    )

    if not os.path.exists(data_root):
        os.mkdir(data_root)
    if not os.path.exists(root_catalog_folder):
        os.mkdir(root_catalog_folder)

    match config_name:
        case "cmems":
            cmems_connection_config = CMEMSConnectionConfig(
                local_root=data_root,
                dataset_id=source['cmems_product_name'],
                max_samples=max_samples,
            )
            if os.path.exists(catalog_path) and use_catalog:
                # Load dataset metadata from catalog
                cmems_config = DatasetConfig(
                    alias=dataset_name,
                    connection_config=cmems_connection_config,
                    catalog_options={"catalog_path": catalog_path}
                )
            else:
                # create dataset
                cmems_config = DatasetConfig(
                    alias=dataset_name,
                    connection_config=cmems_connection_config,
                )
            # Création du dataset
            dataset = RemoteDataset(cmems_config)

        case "s3":
            if "wasabi" in dataset_name:
                s3_connection_config = WasabiS3ConnectionConfig(
                    local_root=data_root,
                    bucket=source['s3_bucket'],
                    bucket_folder=source['s3_folder'],
                    key=source['s3_key'],
                    secret_key=source['s3_secret_key'],
                    endpoint_url=source['url'],
                    max_samples=max_samples,
                )
            elif dataset_name == "glonet":
                s3_connection_config = GlonetConnectionConfig(
                    local_root=data_root,
                    endpoint_url=source['url'],
                    glonet_s3_bucket=source['s3_bucket'],
                    s3_glonet_folder=source['s3_folder'],
                    max_samples=max_samples,
                )

            if os.path.exists(catalog_path) and use_catalog:
                # Load dataset metadata from catalog
                s3_config = DatasetConfig(
                    alias=dataset_name,
                    connection_config=s3_connection_config,
                    catalog_options={"catalog_path": catalog_path}
                )
            else:
                # create dataset
                s3_config = DatasetConfig(
                    alias=dataset_name,
                    connection_config=s3_connection_config,
                )
            # Création du dataset
            dataset = RemoteDataset(s3_config)


    return dataset