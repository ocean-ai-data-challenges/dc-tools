"""Base dataset classes and interfaces."""

import os
from typing import (
    Any, Dict, Iterator, List,
    Optional, Type
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
    """Configuration class for Datasets."""

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
        catalog_options: Optional[Optional[Dict[str, Any]]] = None,
        keep_variables: Optional[Optional[List[str]]] = None,
        eval_variables: Optional[Optional[List[str]]] = None,
        observation_dataset: Optional[bool] = False,
        use_catalog: Optional[bool] = True,
        ignore_geometry: Optional[bool] = False,
    ):
        """
        Configuration for a dataset.

        Args:
            alias (str): Dataset name.
            connection_config (BaseConnectionConfig): Connection configuration.
            catalog_options (Optional[Dict[str, Any]]): Options for the catalog
                (e.g., default filters).
        """
        self.alias = alias
        self.connection_config = connection_config
        self.catalog_options = catalog_options or {}
        self.keep_variables = keep_variables
        self.eval_variables = eval_variables
        self.observation_dataset = observation_dataset
        self.use_catalog = use_catalog
        self.ignore_geometry = ignore_geometry


class BaseDataset:
    """Base class for all datasets."""

    def __init__(self, config: DatasetConfig):
        """
        Initializes a dataset with a configuration.

        Args:
            config (DatasetConfig): Dataset configuration.
        """
        # Use CONNECTION_MANAGER_MAP to instantiate the correct ConnectionManager
        connection_manager_class = DatasetConfig.CONNECTION_MANAGER_MAP.get(
            type(config.connection_config)
        )
        if not connection_manager_class:
            raise ValueError(
                f"Unsupported connection configuration type: {type(config.connection_config)}"
            )
        self.connection_manager = connection_manager_class(config.connection_config)
        self.alias = config.alias
        self.catalog_type = ""
        self.keep_variables = config.keep_variables
        self.eval_variables = config.eval_variables
        self.std_eval_variables = []

        # Check if a JSON catalog file is specified in catalog_options
        catalog_path = config.catalog_options.get(
            "catalog_path"
        ) if config.catalog_options else None
        if config.use_catalog and catalog_path and Path(catalog_path).exists():
            self.catalog = DatasetCatalog.from_json(
                catalog_path, config.alias,
                limit=config.connection_config.params.max_samples,
                ignore_geometry=config.ignore_geometry if hasattr(
                    config, 'ignore_geometry'
                ) else False,
            )
            self._paths = self.catalog.list_paths()
            self.catalog_type = "from_catalog_file"
            self._global_metadata = self.catalog.get_global_metadata()
        else:
            logger.info("No catalog JSON file found. Generating metadata from the dataset.")
            self._metadata = self.connection_manager.list_files_with_metadata()  # Retrieve metadata
            self._global_metadata = self.get_global_metadata()
            if config.observation_dataset:
                self.observation_dataset = config.observation_dataset
            else:
                self.observation_dataset = self.get_coord_system().is_observation_dataset()
            self._global_metadata["is_observation"] = self.observation_dataset
            # Update global metadata
            self.connection_manager.set_global_metadata(self._global_metadata)

            self._paths = [entry.path for entry in self._metadata]
            self.build_catalog()  # Build catalog from metadata
            self.catalog_type = "from_data"

        # save catalog to json
        if self.catalog_type == "from_data":
            catalog = self.get_catalog()
            if catalog is not None:
                catalog.to_json(catalog_path)

        if self._global_metadata is not None:
            vars_rename_dict= self._global_metadata.get("variables_rename_dict")
            if vars_rename_dict and self.eval_variables:
                self.std_eval_variables = [
                    vars_rename_dict[var] if var in vars_rename_dict else None
                    for var in self.eval_variables
                ]

    def list_paths(self) -> List[str]:
        """
        Returns the list of file paths in the dataset.

        Returns:
            List[str]: List of file paths.
        """
        result: List[str] = self._paths
        return result

    def get_global_metadata(self) -> Dict[str, Any]:
        """
        Returns the global metadata of the dataset.

        Returns:
            Dict[str, Any]: Global metadata.
        """
        if hasattr(self, "_global_metadata"):
            result: Dict[str, Any] = self._global_metadata
            return result
        else:
            result = self.connection_manager.get_global_metadata()
            return result

    def get_connection_config(self):
        """Get the connection configuration parameters."""
        return self.connection_manager.params

    def standardize_names(
        self,
        coord_rename_dict: Dict[str, str],
        variable_rename_dict: Dict[str, str],
    ) -> None:
        """Standardize coordinate and variable names using rename dictionaries."""
        # logger.info(f"Standardizing names for dataset {self.alias}")
        if isinstance(coord_rename_dict, str):
            coord_rename_dict = ast.literal_eval(coord_rename_dict)
        if isinstance(variable_rename_dict, str):
            variable_rename_dict = ast.literal_eval(variable_rename_dict)
        if self.eval_variables is not None:
            self.eval_variables = [variable_rename_dict.get(x, x) for x in self.eval_variables]
        if self.keep_variables is not None:
            self.keep_variables = [variable_rename_dict.get(x, x) for x in self.keep_variables]

    def get_connection_manager(self) -> BaseConnectionManager:
        """
        Returns the ConnectionManager instance associated with the dataset.

        Returns:
            BaseConnectionManager: ConnectionManager instance.
        """
        return self.connection_manager

    def get_catalog(self) -> Optional[DatasetCatalog]:
        """
        Returns the dataset catalog.

        Returns:
            DatasetCatalog: Dataset catalog.
        """
        if self.catalog_is_empty():
            logger.warning("No entries in the catalog.")
            return None
        return self.catalog

    def get_metadata(self) -> List[Any]:
        """
        Returns the metadata of the dataset files.

        Returns:
            List[Any]: List of metadata (CatalogEntry objects).
        """
        return self._metadata

    def get_path(self, index: int) -> str:
        """
        Returns the path of a file at a given index.

        Args:
            index (int): File index.

        Returns:
            str: File path.
        """
        return str(self._paths[index])

    def get_coord_system(self):
        """
        Returns the coordinate system of the dataset.

        Returns:
            Dict[str, Any]: Coordinate system.
        """
        if hasattr(self, "_global_metadata"):
            return self._global_metadata.get("coord_system", {})
        else:
            return self.connection_manager.get_global_metadata().get("coord_system", {})

    def download(self, index: int, local_path: str):
        """
        Downloads a file based on its index.

        Args:
            index (int): File index.
            local_path (str): Local path where to save the file.
        """
        remote_path = self.get_path(index)
        remote_file = self.connection_manager.open(remote_path, 'rb')
        if remote_file is None:
            raise ValueError(f"Could not open remote file: {remote_path}")
        with remote_file as rf:
            with open(local_path, 'wb') as local_file:
                local_file.write(rf.read())

    def iter_data(self) -> Iterator[xr.Dataset]:
        """
        Iterates over the dataset files and loads them as Xarray datasets.

        Yields:
            xr.Dataset: Loaded dataset.
        """
        for idx in range(len(self._paths)):
            data = self.load_data(idx)
            if data is not None:
                yield data

    def build_catalog(self) -> None:
        """Builds a catalog for this dataset."""
        if self.catalog_type == "from_catalog_file":
            logger.info("Dataset catalog file already exists. Loading ...")
            return
        self.catalog = DatasetCatalog(
            self.alias, global_metadata=self._global_metadata, entries=self._metadata
        )
        return

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
        Filters the catalog by time range.

        Args:
            start (datetime): Start date.
            end (datetime): End date.
        """
        self.catalog.filter_by_date(start, end)

    def filter_catalog_by_region(self, region: gpd.GeoSeries):
        """
        Filters the catalog by bounding box.

        Args:
            bbox (Tuple[float, float, float, float]): (lon_min, lat_min, lon_max, lat_max).
        """
        self.catalog.filter_by_region(region)

    def filter_catalog_by_variable(self, variables: List[str]):
        """
        Filters the catalog by specified variables.

        Args:
            variables (List[str]): List of variable names to filter.
        """
        if self.catalog_is_empty():
            logger.warning("Empty catalog. Skipping variable filter.")
            return

        # Call filter_by_variables method of DatasetCatalog
        self.catalog.filter_by_variables(variables)
        logger.info(f"Successfully filtered catalog. Kept variables: {variables}")

    def load_data(self, index: int) -> Optional[xr.Dataset]:
        """
        Loads a dataset from a path.

        Args:
            path (str): File path.

        Returns:
            xr.Dataset: Loaded dataset.
        """
        path = self.get_path(index)
        return self.connection_manager.open(path)

    def catalog_is_empty(self) -> bool:
        """
        Checks if the catalog is empty.

        Returns:
            bool: True if the catalog is empty, otherwise False.
        """
        return bool(self.catalog.get_dataframe().empty)

    def to_json(self, path: str) -> None:
        """
        Exports the entire BaseDataset content to JSON format.

        Args:
            path (str): Path to save the JSON file.
        """
        try:
            logger.info(f"Exporting BaseDataset to JSON in {path}")
            # Save catalog to JSON
            self.catalog.to_json(str(path))
            # Build a dictionary for BaseDataset attributes
            logger.info(f"BaseDataset saved successfully in {path}")
        except Exception as exc:
            logger.error(f"Error while exporting BaseDataset to JSON file: {repr(exc)}")
            raise

    def get_eval_variables(self):
        """Return the list of standard evaluation variables."""
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
        remote_file = self.connection_manager.open(remote_path, 'rb')
        if remote_file is None:
            raise ValueError(f"Could not open remote file: {remote_path}")
        with remote_file as rf:
            with open(local_path, 'wb') as local_file:
                local_file.write(rf.read())


class LocalDataset(BaseDataset):
    """Dataset for local files (NetCDF or others)."""

    def empty_fct(self):
        """Empty function."""
        pass


def get_dataset_from_config(
    source: dict,
    root_data_folder: str,
    root_catalog_folder: str,
    dataset_processor: DatasetProcessor,
    max_samples: Optional[int] = 0,
    use_catalog: bool = True,
    file_cache: Optional[FileCacheManager] = None,
    filter_values: Optional[Optional[dict]] = None,
) -> RemoteDataset:
    """Get dataset from config."""
    # Load config
    dataset_name: str = source.get('dataset', '')
    config_name: Optional[str] = source.get('config', None)
    keep_variables = source.get('keep_variables', None)
    eval_variables = source.get('eval_variables', None)
    file_pattern = source.get('file_pattern', None)
    observation_dataset = source.get('observation_dataset', None)
    full_day_data = source.get('full_day_data', False)
    ignore_geometry = source.get('ignore_geometry', False)

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
            connect_config_params = {
                "dataset_processor": dataset_processor,
                "init_type": init_type,
                "local_root": data_root,
                "dataset_id": source['cmems_product_name'],
                "max_samples": max_samples,
                "file_pattern": file_pattern,
                "keep_variables": keep_variables,
                "file_cache": file_cache,
                "filter_values": filter_values,
                "full_day_data": full_day_data,
            }
            cmems_connection_config = CMEMSConnectionConfig(
                connect_config_params
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
                ignore_geometry=ignore_geometry,
            )
            # Dataset creation
            dataset = RemoteDataset(cmems_config)

        case "argopy":
            connect_config_params = {
                "dataset_processor": dataset_processor,
                "init_type": init_type,
                "local_root": data_root,
                "max_samples": max_samples,
                "file_pattern": file_pattern,
                "keep_variables": keep_variables,
                "file_cache": file_cache,
                "filter_values": filter_values,
                "full_day_data": full_day_data,
            }
            argo_connection_config = ARGOConnectionConfig(
                connect_config_params
            )
            argo_config = DatasetConfig(
                alias=dataset_name,
                connection_config=argo_connection_config,
                catalog_options={"catalog_path": catalog_path},
                keep_variables=keep_variables,
                eval_variables=eval_variables,
                observation_dataset=observation_dataset,
                use_catalog=use_catalog,
                ignore_geometry=ignore_geometry,
            )
            # Dataset creation
            dataset = RemoteDataset(argo_config)
        case "s3":
            connection_type = source.get("connection_type")
            connect_config_params = {
                "dataset_processor": dataset_processor,
                "init_type": init_type,
                "local_root": data_root,
                "s3_bucket": source['s3_bucket'],
                "s3_folder": source['s3_folder'],
                "endpoint_url": source.get('url'),
                "max_samples": max_samples,
                "file_pattern": file_pattern,
                "keep_variables": keep_variables,
                "file_cache": file_cache,
                "filter_values": filter_values,
                "full_day_data": full_day_data,
                "groups": source.get('groups'),
            }

            s3_connection_config: BaseConnectionConfig
            if connection_type == "wasabi":
                connect_config_params.update({
                    "key": source['s3_key'],
                    "secret_key": source['s3_secret_key'],
                })
                s3_connection_config = WasabiS3ConnectionConfig(
                    connect_config_params
                )
            elif connection_type == "glonet":
                 s3_connection_config = GlonetConnectionConfig(
                    connect_config_params
                )
            elif connection_type == "private":
                connect_config_params.update({
                    "key": source['s3_key'],
                    "secret_key": source['s3_secret_key'],
                })
                s3_connection_config = S3ConnectionConfig(
                    connect_config_params
                )
            elif connection_type == "public":
                 s3_connection_config = S3ConnectionConfig(
                    connect_config_params
                )
            else:
                raise ValueError(
                    f"Unknown or missing connection_type '{connection_type}' "
                    f"for s3 config in dataset '{dataset_name}'. "
                    "Expected 'wasabi', 'glonet', 'private' or 'public'."
                )

            s3_config = DatasetConfig(
                alias=dataset_name,
                connection_config=s3_connection_config,
                catalog_options={"catalog_path": catalog_path},
                keep_variables=keep_variables,
                eval_variables=eval_variables,
                observation_dataset=observation_dataset,
                use_catalog=use_catalog,
                ignore_geometry=ignore_geometry,
            )
            # Dataset creation
            dataset = RemoteDataset(s3_config)
        case "_":
            raise ValueError(f"Unknown dataset config name: {config_name}")

    return dataset
