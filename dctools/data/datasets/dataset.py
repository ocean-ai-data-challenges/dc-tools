"""Base dataset classes and interfaces."""

import os
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Optional,
    Type,
    TYPE_CHECKING,
)

import ast
from datetime import datetime
from loguru import logger
from pathlib import Path
import xarray as xr

if TYPE_CHECKING:
    from oceanbench.core.distributed import DatasetProcessor  # pragma: no cover
else:
    try:
        from oceanbench.core.distributed import DatasetProcessor  # type: ignore
    except Exception:
        DatasetProcessor = Any  # type: ignore


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
from dctools.data.coordinates import get_standardized_var_name
from dctools.utilities.file_utils import FileCacheManager


def _is_legacy_argo_catalog_paths(paths: List[str]) -> bool:
    """Return True when ARGO catalog paths use legacy profile ids (wmo:cycle)."""
    for path in paths:
        if isinstance(path, str) and ":" in path:
            return True
    return False


def _has_unknown_argo_catalog_paths(paths: List[str], valid_keys: List[str]) -> bool:
    """Return True when ARGO catalog contains keys unknown to current master index."""
    valid_set = set(valid_keys)
    if not valid_set:
        return False
    for path in paths:
        if path not in valid_set:
            return True
    return False


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
        catalog_path = (
            config.catalog_options.get("catalog_path") if config.catalog_options else None
        )
        loaded_from_catalog = False

        # Detect ARGO catalog directory structure (master_index.json inside folder)
        is_argo_directory_catalog = False
        if isinstance(self.connection_manager, ArgoManager) and catalog_path:
            p_cat = Path(catalog_path)
            if p_cat.is_dir() and (p_cat / "master_index.json").exists():
                is_argo_directory_catalog = True
                logger.info(f"Detected ARGO directory catalog at {catalog_path}")
            elif not p_cat.exists() and p_cat.suffix == ".json":
                # Handle case where catalog_path is constructed as '.../argo_profiles.json'
                # but the actual catalog is in a directory next to it or with similar name
                # E.g. '.../argo_index' or '.../argo_profiles/argo_index'
                possible_dirs = [
                    p_cat.with_suffix(""),  # argo_profiles/
                    p_cat.parent / "argo_index",
                    p_cat.with_name("argo_index"),
                ]
                for d in possible_dirs:
                    if d.is_dir() and (d / "master_index.json").exists():
                        is_argo_directory_catalog = True
                        logger.info(
                            f"Detected ARGO directory catalog at {d} (instead of {catalog_path})"
                        )
                        # Update catalog_path to point to the actual directory if needed by manager?
                        # Or just rely on manager having its own config.
                        break

        # Try to load existing catalog (File or Directory for ARGO)
        exists = Path(catalog_path).exists() if catalog_path else False
        if config.use_catalog and catalog_path:
            if is_argo_directory_catalog:
                # Load ARGO catalog from directory by leveraging ArgoManager's internal indexing
                try:
                    logger.info(
                        f"Loading ARGO catalog internal index from directory: {catalog_path}"
                    )
                    # Force loading of metadata from master_index.json
                    # (ArgoManager handles the index internally via list_files_with_metadata)
                    self._metadata = self.connection_manager.list_files_with_metadata()

                    # Also populate global metadata if available
                    self._global_metadata = self.connection_manager.get_global_metadata()
                    if self._global_metadata is None:
                        self._global_metadata = {}
                    # Ensure is_observation flag is set
                    if config.observation_dataset:
                        self.observation_dataset = config.observation_dataset
                    else:
                        coord_sys = self._global_metadata.get("coord_system")
                        if coord_sys and hasattr(coord_sys, "is_observation_dataset"):
                            self.observation_dataset = coord_sys.is_observation_dataset()
                        else:
                            self.observation_dataset = False
                    self._global_metadata["is_observation"] = self.observation_dataset

                    # Create a DatasetCatalog instance wrapper to satisfy dataset interface
                    self.catalog = DatasetCatalog(
                        alias=config.alias,
                        global_metadata=self._global_metadata,
                        entries=self._metadata,
                    )

                    self._paths = self.catalog.list_paths()
                    self.catalog_type = "from_catalog_file"  # Treat as file-loaded catalog
                    loaded_from_catalog = True
                except Exception as exc:
                    logger.warning(
                        f"Failed to load ARGO catalog from '{catalog_path}': {exc}. "
                        "Falling back to metadata rebuild."
                    )
                    # Reset if failed
                    self.catalog = None
                    loaded_from_catalog = False

            elif exists:
                try:
                    self.catalog = DatasetCatalog.from_json(
                        catalog_path,
                        config.alias,
                        limit=config.connection_config.params.max_samples,
                        ignore_geometry=config.ignore_geometry
                        if hasattr(config, "ignore_geometry")
                        else False,
                    )
                    self._paths = self.catalog.list_paths()
                    self.catalog_type = "from_catalog_file"
                    self._global_metadata = self.catalog.get_global_metadata()
                    loaded_from_catalog = True
                except Exception as exc:
                    logger.warning(
                        f"Failed to load catalog '{catalog_path}' for '{config.alias}': {exc}. "
                        "Falling back to metadata rebuild from source."
                    )

        if loaded_from_catalog and isinstance(self.connection_manager, ArgoManager):
            catalog_paths = list(self._paths)
            must_rebuild = False

            if _is_legacy_argo_catalog_paths(catalog_paths):
                logger.warning(
                    "Legacy ARGO catalog detected (wmo:cycle paths). "
                    "Rebuilding catalog from Kerchunk master index."
                )
                must_rebuild = True
            else:
                try:
                    valid_keys = self.connection_manager.list_files()
                except Exception as exc:
                    logger.warning(
                        f"Could not validate ARGO catalog keys against master index: {exc}. "
                        "Rebuilding catalog from source."
                    )
                    must_rebuild = True
                    valid_keys = []

                if not must_rebuild and _has_unknown_argo_catalog_paths(catalog_paths, valid_keys):
                    logger.warning(
                        "ARGO catalog contains keys unknown to current master index. "
                        "Rebuilding catalog from source."
                    )
                    must_rebuild = True

            if must_rebuild:
                loaded_from_catalog = False
                self.catalog = None
                self._paths = []
                self._global_metadata = None
                self.catalog_type = ""

        if not loaded_from_catalog:
            logger.info("No catalog JSON file found. Generating metadata from the dataset.")
            self._metadata = self.connection_manager.list_files_with_metadata()  # Retrieve metadata
            self._global_metadata = self.get_global_metadata()
            if self._global_metadata is None:
                self._global_metadata = {}
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
            vars_rename_dict = self._global_metadata.get("variables_rename_dict")
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
        if hasattr(self, "_global_metadata") and self._global_metadata is not None:
            result: Dict[str, Any] = self._global_metadata
            return result

        result = self.connection_manager.get_global_metadata()
        if result is None:
            result = {}
        self._global_metadata = result
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

        def _standardize_var_name(name: str) -> str:
            mapped = variable_rename_dict.get(name, name)
            standard_name = get_standardized_var_name(mapped)
            if standard_name is not None:
                return standard_name
            return mapped

        if self.eval_variables is not None:
            self.eval_variables = [_standardize_var_name(x) for x in self.eval_variables]
            self.std_eval_variables = list(self.eval_variables)
        if self.keep_variables is not None:
            self.keep_variables = [_standardize_var_name(x) for x in self.keep_variables]

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
        if self.catalog is None:
            logger.warning("Catalog is not initialized.")
            return None
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
        remote_file = self.connection_manager.open(remote_path, "rb")
        if remote_file is None:
            raise ValueError(f"Could not open remote file: {remote_path}")
        with remote_file as rf:
            with open(local_path, "wb") as local_file:
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
        if self.catalog_type == "from_catalog_file" and self.catalog is not None:
            logger.info("Dataset catalog file already exists. Loading ...")
            return
        self.catalog = DatasetCatalog(
            self.alias, global_metadata=self._global_metadata, entries=self._metadata
        )
        return

    """def filter_attrs(
        self, filters: dict[str, Union[Callable[[Any], bool], gpd.GeoSeries]]
    ) -> None:
        self.catalog.filter_attrs(filters)"""

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

    def filter_catalog_by_region(self, region: Any):
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
        if self.catalog is None:
            return True
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
        if self.std_eval_variables:
            return self.std_eval_variables
        if self.eval_variables:
            return self.eval_variables
        return []


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
        remote_file = self.connection_manager.open(remote_path, "rb")
        if remote_file is None:
            raise ValueError(f"Could not open remote file: {remote_path}")
        with remote_file as rf:
            with open(local_path, "wb") as local_file:
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
    target_depth_values: Optional[List[float]] = None,
) -> RemoteDataset:
    """Get dataset from config."""
    # Load config
    dataset_name: str = source.get("dataset", "")
    config_name: Optional[str] = source.get("config", None)
    keep_variables = source.get("keep_variables", None)
    eval_variables = source.get("eval_variables", None)
    file_pattern = source.get("file_pattern", None)
    observation_dataset = source.get("observation_dataset", None)
    full_day_data = source.get("full_day_data", False)
    ignore_geometry = source.get("ignore_geometry", False)

    data_root = os.path.join(
        root_data_folder,
        dataset_name,
    )

    catalog_path = os.path.join(
        root_catalog_folder,
        dataset_name + ".json",
    )

    # Special case for ARGO catalog which can be a directory (argo_index)
    if dataset_name == "argo_profiles":
        argo_index_path = os.path.join(root_catalog_folder, "argo_index")
        if os.path.isdir(argo_index_path) and os.path.exists(
            os.path.join(argo_index_path, "master_index.json")
        ):
            logger.info(f"Using ARGO directory catalog at {argo_index_path}")
            catalog_path = argo_index_path
        elif os.path.isdir(os.path.join(root_catalog_folder, "argo_profiles", "argo_index")):
            # Also check if it's inside argo_profiles/
            path = os.path.join(root_catalog_folder, "argo_profiles", "argo_index")
            if os.path.exists(os.path.join(path, "master_index.json")):
                logger.info(f"Using ARGO directory catalog at {path}")
                catalog_path = path

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
                "dataset_id": source["cmems_product_name"],
                "max_samples": max_samples,
                "file_pattern": file_pattern,
                "keep_variables": keep_variables,
                "file_cache": file_cache,
                "filter_values": filter_values,
                "full_day_data": full_day_data,
            }
            cmems_connection_config = CMEMSConnectionConfig(connect_config_params)
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
                "local_catalog_path": catalog_path,
                "max_samples": max_samples,
                "file_pattern": file_pattern,
                "keep_variables": keep_variables,
                "file_cache": file_cache,
                "filter_values": filter_values,
                "full_day_data": full_day_data,
                # S3/Wasabi parameters for ARGO index storage
                "s3_bucket": source.get("s3_bucket"),
                "s3_folder": source.get("s3_folder"),
                "s3_key": source.get("s3_key"),
                "s3_secret_key": source.get("s3_secret_key"),
                "endpoint_url": source.get("url"),
                # ARGO-specific parameters
                "base_path": source.get("base_path"),
                "variables": source.get("variables"),
                "depth_values": target_depth_values,
                "chunks": source.get("chunks", {"N_PROF": 2000}),
            }
            argo_connection_config = ARGOConnectionConfig(connect_config_params)
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
                "s3_bucket": source["s3_bucket"],
                "s3_folder": source["s3_folder"],
                "endpoint_url": source.get("url"),
                "max_samples": max_samples,
                "file_pattern": file_pattern,
                "keep_variables": keep_variables,
                "file_cache": file_cache,
                "filter_values": filter_values,
                "full_day_data": full_day_data,
                "groups": source.get("groups"),
            }

            s3_connection_config: BaseConnectionConfig
            if connection_type == "wasabi":
                connect_config_params.update(
                    {
                        "key": source["s3_key"],
                        "secret_key": source["s3_secret_key"],
                    }
                )
                s3_connection_config = WasabiS3ConnectionConfig(connect_config_params)
            elif connection_type == "glonet":
                s3_connection_config = GlonetConnectionConfig(connect_config_params)
            elif connection_type == "private":
                connect_config_params.update(
                    {
                        "key": source["s3_key"],
                        "secret_key": source["s3_secret_key"],
                    }
                )
                s3_connection_config = S3ConnectionConfig(connect_config_params)
            elif connection_type == "public":
                s3_connection_config = S3ConnectionConfig(connect_config_params)
            elif connection_type == "anonymous":
                # Backward-compatible alias for unauthenticated/public access.
                s3_connection_config = S3ConnectionConfig(connect_config_params)
            else:
                raise ValueError(
                    f"Unknown or missing connection_type '{connection_type}' "
                    f"for s3 config in dataset '{dataset_name}'. "
                    "Expected 'wasabi', 'glonet', 'private', 'public' or 'anonymous'."
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
        case _:
            raise ValueError(f"Unknown dataset config name: {config_name}")

    return dataset
