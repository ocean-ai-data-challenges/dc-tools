

from abc import ABC
import os
from typing import (
    Any, Dict, List,
    Iterator, Optional, Type,
)

import ast
from datetime import datetime
import geopandas as gpd
from loguru import logger
from oceanbench.core.distributed import DatasetProcessor
from pathlib import Path
import xarray as xr


from dctools.data.datasets.dc_catalog import DatasetCatalog
from dctools.data.connection.config import (
    ARGOConnectionConfig,
    BaseConnectionConfig,
    CMEMSConnectionConfig,
    FTPConnectionConfig,
    GlonetConnectionConfig,
    LocalConnectionConfig,
    S3ConnectionConfig,
    WasabiS3ConnectionConfig,
)
from dctools.data.connection.connection_manager import (
    BaseConnectionManager,
    LocalConnectionManager,
    CMEMSManager,
    S3Manager,
    S3WasabiManager,
    FTPManager,
    GlonetManager,
    ArgoManager,
)
from dctools.utilities.file_utils import FileCacheManager


class DatasetConfig:
    CONNECTION_MANAGER_MAP: Dict[Type[BaseConnectionConfig], Type[BaseConnectionManager]] = {
        LocalConnectionConfig: LocalConnectionManager,
        CMEMSConnectionConfig: CMEMSManager,
        S3ConnectionConfig: S3Manager,
        WasabiS3ConnectionConfig: S3WasabiManager,
        FTPConnectionConfig: FTPManager,
        GlonetConnectionConfig: GlonetManager,
        ARGOConnectionConfig: ArgoManager,
    }

    def __init__(
        self,
        alias: str,
        connection_config: BaseConnectionConfig,
        catalog_options: Optional[Dict[str, Any]] = None,
        keep_variables: Optional[str] = None,
        eval_variables: Optional[str] = None,
        observation_dataset: Optional[bool] = False,
        use_catalog: Optional[bool] = True,
    ):
        """
        Configuration pour un dataset.

        Args:
            alias (str): Nom du dataset.
            connection_config (BaseConnectionConfig): Configuration de connexion.
            catalog_options (Optional[Dict[str, Any]]): Options pour le catalogue (e.g., filtres par défaut).
        """
        # logger.info(f"DatasetConfig __init__: {alias}")
        self.alias = alias
        self.connection_config = connection_config
        self.catalog_options = catalog_options or {}
        self.keep_variables = keep_variables
        self.eval_variables = eval_variables
        self.observation_dataset = observation_dataset
        # self.connection_manager = self._create_connection_manager()
        self.use_catalog = use_catalog


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
        self.keep_variables = config.keep_variables
        self.eval_variables = config.eval_variables
        self.std_eval_variables = []

        # Vérifier si un fichier de catalogue JSON est spécifié dans catalog_options
        catalog_path = config.catalog_options.get("catalog_path") if config.catalog_options else None
        if config.use_catalog and catalog_path and Path(catalog_path).exists():
            self.catalog = DatasetCatalog.from_json(catalog_path, config.alias)
            self._paths = self.catalog.list_paths()
            self.catalog_type = "from_catalog_file"
            self._global_metadata = self.catalog.get_global_metadata()
        else:
            logger.info("No catalog JSON file found. Generating metadata from the dataset.")
            self._metadata = self.connection_manager.list_files_with_metadata()  # Récupérer les métadonnées
            self._global_metadata = self.get_global_metadata()
            if config.observation_dataset:
                self.observation_dataset = config.observation_dataset
            else:
                self.observation_dataset = self.get_coord_system().is_observation_dataset()
            self._global_metadata["is_observation"] = self.observation_dataset
            self.connection_manager.set_global_metadata(self._global_metadata)  # Mettre à jour les métadonnées globales

            self._paths = [entry.path for entry in self._metadata]
            self.build_catalog()  # Construire le catalogue à partir des métadonnées
            self.catalog_type = "from_data"

        # save catalog to json
        if self.catalog_type == "from_data":
            self.get_catalog().to_json(catalog_path)

        if self._global_metadata is not None:
            vars_rename_dict= self._global_metadata.get("variables_rename_dict")
            if vars_rename_dict:
                self.std_eval_variables = [
                    vars_rename_dict[var] if var in vars_rename_dict else None for var in self.eval_variables
                ]
        logger.debug(f"self.std_eval_variables: {self.std_eval_variables}")

    def list_paths(self) -> List[str]:
        """
        Retourne la liste des chemins des fichiers dans le dataset.

        Returns:
            List[str]: Liste des chemins des fichiers.
        """
        return self._paths

    def get_global_metadata(self) -> Dict[str, Any]:
        """
        Retourne les métadonnées globales du dataset.

        Returns:
            Dict[str, Any]: Métadonnées globales.
        """
        if hasattr(self, "_global_metadata"):
            return self._global_metadata
        else:
            return self.connection_manager.get_global_metadata()

    def get_connection_config(self):
        return self.connection_manager.params

    def standardize_names(
        self,
        coord_rename_dict: Dict[str, str],
        variable_rename_dict: Dict[str, str],
    ) -> None:
        logger.info(f"Standardizing names for dataset {self.alias}")
        if isinstance(coord_rename_dict, str):
            coord_rename_dict = ast.literal_eval(coord_rename_dict)
        if isinstance(variable_rename_dict, str):
            variable_rename_dict = ast.literal_eval(variable_rename_dict)
        self.eval_variables = [variable_rename_dict.get(x, x) for x in self.eval_variables]
        self.keep_variables = [variable_rename_dict.get(x, x) for x in self.keep_variables]

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

    def get_coord_system(self):
        """
        Retourne le système de coordonnées du dataset.

        Returns:
            Dict[str, Any]: Système de coordonnées.
        """
        if hasattr(self, "_global_metadata"):
            return self._global_metadata.get("coord_system", {})
        else:
            return self.connection_manager.get_global_metadata().get("coord_system", {})

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
        self.catalog = DatasetCatalog(
            self.alias, global_metadata=self._global_metadata, entries=self._metadata
        )

    '''def filter_attrs(
        self, filters: dict[str, Union[Callable[[Any], bool], gpd.GeoSeries]]
    ) -> None:
        self.catalog.filter_attrs(filters)'''

    def filter_catalog_by_date(
            self,
            start: datetime | list[datetime], 
            end: datetime | list[datetime],
            ):
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
            # Sauvegarder le catalogue en JSON
            self.catalog.to_json(str(path))
            # Construire un dictionnaire pour les attributs de BaseDataset
            logger.info(f"BaseDataset sauvegardé avec succès dans {path}")
        except Exception as exc:
            logger.error(f"Erreur lors de l'exportation de BaseDataset en JSON : {repr(exc)}")
            raise

    def get_eval_variables(self):
        return self.std_eval_variables


class RemoteDataset(BaseDataset):
    """Generic dataset for remote sources."""

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
    def empty_fct(self):
        """
        Fonction vide.
        """
        pass


def get_dataset_from_config(
    source: dict,
    root_data_folder: str,
    root_catalog_folder: str,
    dataset_processor: DatasetProcessor,
    max_samples: Optional[int] = 0,
    use_catalog: bool = True,
    file_cache: FileCacheManager=None,
    filter_values: Optional[dict] = None,
) -> RemoteDataset:
    """Get dataset from config."""
    # Load config
    dataset_name = source.get('dataset', None)
    config_name = source.get('config', None)
    keep_variables = source.get('keep_variables', None)
    eval_variables = source.get('eval_variables', None)
    file_pattern = source.get('file_pattern', None)
    observation_dataset = source.get('observation_dataset', None)
    full_day_data = source.get('full_day_data', False)

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

    init_type = "from_data"
    if use_catalog and catalog_path and Path(catalog_path).exists():
        init_type = "from_json"

    match config_name:
        case "cmems":
            cmems_connection_config = CMEMSConnectionConfig(
                dataset_processor=dataset_processor,
                init_type=init_type,
                local_root=data_root,
                dataset_id=source['cmems_product_name'],
                max_samples=max_samples,
                file_pattern=file_pattern,
                keep_variables=keep_variables,
                file_cache=file_cache,
                filter_values=filter_values,
                full_day_data=full_day_data,
            )
            # Load dataset metadata from catalog
            cmems_config = DatasetConfig(
                alias=dataset_name,
                connection_config=cmems_connection_config,
                catalog_options={"catalog_path": catalog_path},
                keep_variables=keep_variables,
                eval_variables=eval_variables,
                observation_dataset=observation_dataset,
                use_catalog=use_catalog,
            )
            # Création du dataset
            dataset = RemoteDataset(cmems_config)

        case "argopy":
            argo_connection_config = ARGOConnectionConfig(
                dataset_processor=dataset_processor,
                init_type=init_type,
                local_root=data_root,
                max_samples=max_samples,
                file_pattern=file_pattern,
                keep_variables=keep_variables,
                file_cache=file_cache,
                filter_values=filter_values,
                full_day_data=full_day_data,
            )
            argo_config = DatasetConfig(
                alias=dataset_name,
                connection_config=argo_connection_config,
                catalog_options={"catalog_path": catalog_path},
                keep_variables=keep_variables,
                eval_variables=eval_variables,
                observation_dataset=observation_dataset,
                use_catalog=use_catalog,
            )
            # Création du dataset
            dataset = RemoteDataset(argo_config)
        case "s3":
            # logger.debug(f"Creating S3 dataset with config: {source}")
            if source["connection_type"] == "wasabi":
                s3_connection_config = WasabiS3ConnectionConfig(
                    dataset_processor=dataset_processor,
                    init_type=init_type,
                    local_root=data_root,
                    bucket=source['s3_bucket'],
                    bucket_folder=source['s3_folder'],
                    key=source['s3_key'],
                    secret_key=source['s3_secret_key'],
                    endpoint_url=source['url'],
                    max_samples=max_samples,
                    file_pattern=file_pattern,
                    groups=source['groups'] if 'groups' in source else None,
                    keep_variables=keep_variables,
                    file_cache=file_cache,
                    filter_values=filter_values,
                    full_day_data=full_day_data,
                )
            elif dataset_name == "glonet":
                s3_connection_config = GlonetConnectionConfig(
                    dataset_processor=dataset_processor,
                    init_type=init_type,
                    local_root=data_root,
                    endpoint_url=source['url'],
                    glonet_s3_bucket=source['s3_bucket'],
                    s3_glonet_folder=source['s3_folder'],
                    max_samples=max_samples,
                    file_pattern=file_pattern,
                    keep_variables=keep_variables,
                    file_cache=file_cache,
                    filter_values=filter_values,
                    full_day_data=full_day_data,
                )
            s3_config = DatasetConfig(
                alias=dataset_name,
                connection_config=s3_connection_config,
                catalog_options={"catalog_path": catalog_path},
                keep_variables=keep_variables,
                eval_variables=eval_variables,
                observation_dataset=observation_dataset,
                use_catalog=use_catalog,
            )
            # Création du dataset
            dataset = RemoteDataset(s3_config)
        case "_":
            raise ValueError(f"Unknown dataset config name: {config_name}")

    return dataset
