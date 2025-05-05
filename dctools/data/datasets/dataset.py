

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Iterator, Optional, Tuple, Type

import datetime
import geopandas as gpd
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
        name: str,
        connection_config: BaseConnectionConfig,
        catalog_options: Optional[Dict[str, Any]] = None,
    ):
        """
        Configuration pour un dataset.

        Args:
            name (str): Nom du dataset.
            connection_config (BaseConnectionConfig): Configuration de connexion.
            catalog_options (Optional[Dict[str, Any]]): Options pour le catalogue (e.g., filtres par défaut).
        """
        self.name = name
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
        self._metadata = self.connection_manager.list_files_with_metadata()  # Récupérer les métadonnées
        self._paths = [entry.path for entry in self._metadata]
        self.name = config.name
        self.catalog = DatasetCatalog([])  # Initialiser un catalogue vide

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


    def build_catalog(self) -> gpd.GeoDataFrame:
        """
        Construit un catalogue pour ce dataset.

        Returns:
            gpd.GeoDataFrame: Catalogue sous forme de GeoDataFrame.
        """
        '''if not self._metadata:
            logger.warning("No metadata available for this dataset.")
            return gpd.GeoDataFrame()

        df = pd.DataFrame(self._metadata)
        # Vérifier les colonnes requises
        required_columns = ["date_start", "date_end", "lat_min", "lat_max", "lon_min", "lon_max"]
        for col in required_columns:
            if col not in df.columns:
                logger.warning(f"Missing required column '{col}' in metadata.")
                df[col] = None  # Ajouter des valeurs par défaut si nécessaire

        df["geometry"] = df.apply(
            lambda row: box(row["lon_min"], row["lat_min"], row["lon_max"], row["lat_max"]), axis=1
        )

        logger.debug(f"Metadata DataFrame: {df.head()}")
        logger.debug(f"Metadata DataFrame columns: {df.columns.tolist()}")'''
        self.catalog = DatasetCatalog(self._metadata)
        #return gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")


    def filter_catalog_by_date(self, start: datetime, end: datetime):
        """
        Filtre le catalogue par plage temporelle.

        Args:
            start (datetime): Date de début.
            end (datetime): Date de fin.
        """
        self.catalog.filter_by_date(start, end)

    def filter_catalog_by_bbox(self, bbox: Tuple[float, float, float, float]):
        """
        Filtre le catalogue par boîte englobante.

        Args:
            bbox (Tuple[float, float, float, float]): (lon_min, lat_min, lon_max, lat_max).
        """
        self.catalog.filter_by_bbox(bbox)

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

    def to_json(self, path: Optional[str] = None) -> str:
        """
        Exporte le catalogue au format GeoJSON.

        Args:
            path (Optional[str]): Chemin pour sauvegarder le fichier GeoJSON.

        Returns:
            str: Représentation GeoJSON du catalogue.
        """
        logger.info(f"Exporting catalog to GeoJSON at {path}")
        return self.catalog.to_json(path)


class RemoteDataset(BaseDataset):
    """Generic dataset for remote sources."""
    def __init__(self, manager: BaseConnectionManager):
        super().__init__(manager)

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
    def __init__(self, manager: BaseConnectionManager):
        super().__init__(manager)

