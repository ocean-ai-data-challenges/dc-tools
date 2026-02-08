"""Dataset management and orchestration."""

import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from datetime import datetime
import geopandas as gpd
from loguru import logger
from oceanbench.core.distributed import DatasetProcessor
import pandas as pd
import xarray as xr

from dctools.data.coordinates import (
    CoordinateSystem,
)
from dctools.data.datasets.dataset import BaseDataset
from dctools.data.datasets.forecast import build_forecast_index_from_catalog
from dctools.data.transforms import CustomTransforms, get_dataset_transform
from dctools.data.datasets.dc_catalog import DatasetCatalog
from dctools.data.datasets.dataloader import EvaluationDataloader
from dctools.utilities.file_utils import FileCacheManager

class MultiSourceDatasetManager:
    """Manager for handling multiple data sources with common processing operations."""

    def __init__(
        self,
        dataset_processor: DatasetProcessor,
        target_dimensions: Dict[str, Tuple[float, float]],
        time_tolerance: Optional[pd.Timedelta] = None,
        list_references: Optional[list[str]] = None,
        max_cache_files: int = 100,
    ):
        """Initializes the multi-source manager."""
        self.dataset_processor = dataset_processor
        self.target_dimensions = target_dimensions
        self.datasets: Dict[str, BaseDataset] = {}
        self.forecast_indexes: Dict[str, pd.DataFrame] = {}
        self.time_tolerance = time_tolerance if time_tolerance is not None else pd.Timedelta("12h")
        self.list_references = list_references if list_references is not None else []
        self.file_cache = FileCacheManager(max_cache_files)

    def get_keep_variables_dict(self):
        """Get dictionary of variables to keep for each dataset."""
        keep_vars: Dict[Any, Any] = {}
        for dataset_alias in self.datasets.keys():
            keep_vars[dataset_alias] = self.datasets[dataset_alias].keep_variables
        return keep_vars

    def get_metadata_dict(self):
        """Get global metadata dictionary for all datasets."""
        metadata: Dict[Any, Any] = {}
        for dataset_alias in self.datasets.keys():
            metadata[dataset_alias] = self.datasets[dataset_alias].get_global_metadata()
        return metadata

    def add_dataset(self, alias: str, dataset: BaseDataset):
        """
        Adds a dataset to the manager with an alias.

        Args:
            alias (str): Unique alias for the dataset.
            dataset (BaseDataset): Dataset instance.
        """
        if alias in self.datasets:
            raise ValueError(f"Alias '{alias}' already exists.")
        self.datasets[alias] = dataset

    def build_catalogs(self):
        """Builds catalogs for all datasets."""
        for _alias, dataset in self.datasets.items():
            dataset.build_catalog()

    def get_catalog(self, alias: str) -> Optional[DatasetCatalog]:
        """
        Returns the catalog of a dataset.

        Args:
            alias (str): Dataset alias.

        Returns:
            DatasetCatalog: The dataset catalog.
        """
        if alias not in self.datasets:
            raise ValueError(f"Alias '{alias}' not found in the manager.")
        return self.datasets[alias].get_catalog()

    def get_data(self, alias: str, path: str) -> Optional[xr.Dataset]:
        """
        Loads data from a dataset.

        Args:
            alias (str): Dataset alias.
            path (str): File path.

        Returns:
            xr.Dataset: Loaded dataset.
        """
        return self.datasets[alias].connection_manager.open(path)

    def filter_attrs(
        self, filters: dict[str, Union[Callable[[Any], bool], gpd.GeoSeries]]
    ) -> None:
        """
        Filters datasets based on attributes.

        Args:
            filters (dict[str, Callable[[Any], bool]]): Dictionary of filters.
        """
        for _alias, dataset in self.datasets.items():
            if hasattr(dataset, 'filter_attrs'):
                dataset.filter_attrs(filters)

    def filter_by_date(self, alias: str, start: Any, end: Any):
        """
        Filters the catalog by time range.

        Args:
            start (datetime): Start date.
            end (datetime): End date.
        """
        self.datasets[alias].filter_catalog_by_date(start, end)
        catalog = self.datasets[alias].get_catalog()
        if catalog is not None:
            logger.debug(
                f"Filtered GeoDataFrame length: {len(catalog.gdf)}"
            )
        # check if catalog is not empty after filtering
        if catalog is None or len(catalog.gdf) == 0:
            logger.warning(f"Dataset '{alias}' is empty after filtering.")
            # remove dataset from manager
            return alias
        return None

    def filter_by_region(self, alias: str, region: gpd.GeoSeries):
        """
        Filters the catalog by bounding box.

        Args:
            bbox (Tuple[float, float, float, float]): (lon_min, lat_min, lon_max, lat_max).
        """
        self.datasets[alias].filter_catalog_by_region(region)
        catalog = self.datasets[alias].get_catalog()
        if catalog is not None:
            logger.debug(
                f"Filtered GeoDataFrame length: {len(catalog.gdf)}"
            )
        if catalog is None or len(catalog.gdf) == 0:
            logger.warning(f"Dataset '{alias}' is empty after filtering.")
            # remove dataset from manager
            return alias
        return None

    def filter_by_variable(self, alias: str, variables: List[str]):
        """
        Filters the catalog by variable.

        Args:
            variables (List[str]): List of variables to filter.
        """
        self.datasets[alias].filter_catalog_by_variable(variables)
        catalog = self.datasets[alias].get_catalog()
        if catalog is not None:
            logger.debug(
                f"Filtered GeoDataFrame length: {len(catalog.gdf)}"
            )
        if catalog is None or len(catalog.gdf) == 0:
            logger.warning(f"Dataset '{alias}' is empty after filtering.")
            # remove dataset from manager
            return alias
        return None

    def filter_all_by_date(
            self,
            start: datetime | list[datetime],
            end: datetime | list[datetime],
        ):
        """
        Filters all datasets managed by this class by time range.

        Args:
            start (datetime): Start date(s).
            end (datetime): End date(s).
        """
        aliases_to_remove: List[Any] = []
        for alias, _ in self.datasets.items():
            # TODO: Fix logger output when using lists as start and end
            logger.info(f"Filtering dataset '{alias}' by date: {start} -> {end}")
            res = self.filter_by_date(alias, start, end)
            aliases_to_remove.append(res)
        for alias in aliases_to_remove:
            if alias is not None :
                del self.datasets[alias]


    def filter_all_by_region(self, region: gpd.GeoSeries):
        """
        Filters all datasets managed by this class by bounding box.

        Args:
            bbox (Tuple[float, float, float, float]): (lon_min, lat_min, lon_max, lat_max).
        """
        aliases_to_remove: List[Any] = []
        for alias, _ in self.datasets.items():
            logger.info(f"Filtering dataset '{alias}' by bbox : {region}")
            res = self.filter_by_region(alias, region)
            aliases_to_remove.append(res)
        for alias in aliases_to_remove:
            if alias is not None :
                del self.datasets[alias]

    def filter_all_by_variable(self, variables: List[str]):
        """
        Filters all datasets managed by this class by time range.

        Args:
            variables (List[str]): List of variables to filter.
        """
        aliases_to_remove: List[Any] = []
        for alias, _ in self.datasets.items():
            res = self.filter_by_variable(alias, variables)
            aliases_to_remove.append(res)
        for alias in aliases_to_remove:
            if alias is not None :
                del self.datasets[alias]


    def add_to_file_cache(self, filepath: str):
        """Add a file to the file cache."""
        self.file_cache.add(filepath)


    def to_json(self, alias: str, path: Optional[str] = None) -> Optional[str]:
        """
        Exports dataset information in JSON format.

        Args:
            alias (str): Alias of the dataset to export.
            path (Optional[str]): Path to save the JSON file.

        Returns:
            str: JSON representation of the dataset information.
        """
        if alias not in self.datasets:
            raise ValueError(f"Alias '{alias}' not found in the manager.")

        dataset = self.datasets[alias]
        catalog = dataset.get_catalog()
        if catalog is None:
             logger.warning(f"Catalog for {alias} is empty or None.")
             return None
        return catalog.to_json(path)

    def all_to_json(self, output_dir: str):
        """
        Exports information of all datasets in JSON format.

        Args:
            output_dir (str): Directory where to save the JSON files.

        Raises:
            ValueError: If the specified directory does not exist or is not accessible.
        """
        # Check if directories exists
        if not os.path.exists(output_dir):
            raise ValueError(f"The specified directory '{output_dir}' does not exist.")
        if not os.path.isdir(output_dir):
            raise ValueError(f"The specified path '{output_dir}' is not a directory.")

        # Loop over all datasets
        for alias, _ in self.datasets.items():
            # Generate JSON filename
            json_filename = f"{alias}.json"
            json_path = os.path.join(output_dir, json_filename)

            # Call to_json() method for each dataset
            self.to_json(alias, path=json_path)

            logger.info(f"Dataset '{alias}' exported to JSON in '{json_path}'.")

    def standardize_names(
        self, alias: str,
        coords_rename_dict: Dict[str, str],
        vars_rename_dict: Dict[str, str],
    ) -> None:
        """
        Standardizes variable names of a dataset based on a mapping dictionary.

        Args:
            alias (str): Alias of the dataset to standardize.
            standard_names (Dict[str, str]): Dictionary mapping old names to new names.
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
        """Build forecast index for a dataset."""
        dataset = self.datasets[alias]
        catalog = dataset.get_catalog()
        if catalog is None:
            raise ValueError(f"No catalog found for alias '{alias}'.")
        catalog_df = catalog.get_dataframe()
        forecast_index = build_forecast_index_from_catalog(
            catalog_df,
            init_date=init_date,
            end_date=end_date,
            start_time_col="date_start",
            end_time_col="date_end",
            file_col="path",
            n_days_forecast=n_days_forecast,
            n_days_interval=n_days_interval,
            lead_time_unit="days",
        )

        self.forecast_indexes[alias] = forecast_index

    def get_config(self):
        """Get configuration for all reference datasets."""
        ref_managers: Dict[Any, Any] = {}
        ref_catalogs: Dict[Any, Any] = {}
        ref_connection_params: Dict[Any, Any]= {}
        ref_aliases = [alias for alias in self.list_references if alias in self.datasets]
        ref_datasets: Dict[Any, Any] = {}
        for alias in ref_aliases:
            ref_datasets[alias] = self.datasets[alias]
        for ref_alias in ref_aliases:
            if ref_alias not in self.datasets:
                logger.warning(
                    f"Reference dataset '{ref_alias}' not found in dataset manager. Skipping."
                )
                continue
            ref_managers[ref_alias] = ref_datasets[ref_alias].get_connection_manager()
            ref_catalogs[ref_alias] = ref_datasets[ref_alias].get_catalog()
            ref_connection_params[ref_alias] = ref_datasets[ref_alias].get_connection_config()
        return ref_managers, ref_catalogs, ref_connection_params

    def get_dataloader(
        self,
        pred_alias: str,
        ref_aliases: Optional[Optional[List[str]]] = None,
        batch_size: Optional[int] = 8,
        pred_transform: Optional[Optional[CustomTransforms]] = None,
        ref_transforms: Optional[Optional[List[CustomTransforms]]] = None,
        forecast_mode: bool = False,
        n_days_forecast: int = 0,
        lead_time_unit: str = "days",
    ) -> EvaluationDataloader:
        """
        Creates an EvaluationDataloader from dataset aliases.

        Args:
            pred_alias (str): Alias of the prediction dataset.
            ref_aliases (Optional[List[str]]): Aliases of the reference datasets.
            batch_size (int): Batch size.
            pred_transform (Optional[CustomTransforms]): Transform for predictions.
            ref_transforms (Optional[List[CustomTransforms]]): Transforms for references.
            forecast_mode (bool): Enable forecast mode.
            n_days_forecast (int): Number of forecast days to consider.
            lead_time_unit (str): Lead time unit ("days" or "hours").

        Returns:
            EvaluationDataloader: Dataloader instance.
        """
        pred_dataset = self.datasets[pred_alias]
        ref_datasets: Dict[Any, Any] = {}
        for ref_alias in (ref_aliases or []):
            if ref_alias not in self.datasets:
                logger.warning(
                    f"Reference dataset '{ref_alias}' not found in dataset manager. Skipping."
                )
                continue
            ref_datasets[ref_alias] = self.datasets[ref_alias]

        pred_manager = pred_dataset.get_connection_manager()

        ref_managers: Dict[Any, Any] = {}
        ref_catalogs: Dict[Any, Any] = {}
        ref_connection_params: Dict[Any, Any] = {}
        for alias, ds in ref_datasets.items():
            ref_managers[alias] = ds.get_connection_manager()
            ref_catalogs[alias] = ds.get_catalog()
            ref_connection_params[alias] = ds.get_connection_config()

        pred_catalog = pred_dataset.get_catalog()
        pred_connection_params = pred_dataset.get_connection_config()

        # --- Enable forecast mode ---
        forecast_index = None
        if forecast_mode:
            # catalog_df = pred_catalog.get_dataframe()
            forecast_index = self.forecast_indexes[pred_alias]
        dataloader_params: Dict[Any, Any] = {}
        keep_vars = self.get_keep_variables_dict()

        dataloader_params = {
            "dataset_processor": self.dataset_processor,
            "pred_connection_params": pred_connection_params,
            "ref_connection_params": ref_connection_params,
            "pred_catalog": pred_catalog,
            "ref_catalogs": ref_catalogs,
            "pred_manager": pred_manager,
            "ref_managers": ref_managers,
            "pred_alias": pred_alias,
            "ref_aliases": ref_aliases,
            "batch_size": batch_size,
            "pred_transform": pred_transform,
            "ref_transforms": ref_transforms,
            "forecast_mode": forecast_mode,
            "forecast_index": forecast_index,
            "n_days_forecast": n_days_forecast,
            "time_tolerance": self.time_tolerance,
            "target_dimensions" : self.target_dimensions,
            "keep_variables" : keep_vars,
            "metadata" : self.get_metadata_dict(),
        }
        return EvaluationDataloader(dataloader_params)


    def get_transform(
        self,
        dataset_alias: str,
        transform_name: Optional[Optional[str]] = None,
        **kwargs,
    ) -> Any:
        """Factory function to create a transform based on the given name and parameters."""
        global_metadata = self.datasets[dataset_alias].get_global_metadata()
        coord_sys = global_metadata.get("coord_system", None)
        if global_metadata is None or coord_sys is None:
            raise ValueError("Cannot import dataset metadata.")
        if isinstance(coord_sys, dict):
            # If global metadata is a dict, we can use it directly
            coord_dict = coord_sys["coordinates"]
        elif isinstance(coord_sys, CoordinateSystem):
            coord_dict = coord_sys.coordinates
        coords_rename_dict = {v: k for k, v in coord_dict.items()}
        vars_rename_dict= global_metadata.get("variables_rename_dict")
        # global_metadata.get("keep_variables")
        keep_vars = self.datasets[dataset_alias].keep_variables

        metadata = {
            "keep_vars": keep_vars,
            "coords_rename_dict": coords_rename_dict,
            "vars_rename_dict": vars_rename_dict
        }

        # Call centralized transform builder
        transform_obj = get_dataset_transform(
            alias=dataset_alias,
            metadata=metadata,
            dataset_processor=self.dataset_processor,
            transform_name=transform_name,
            config=kwargs
        )


        self.standardize_names(
            dataset_alias,
            coords_rename_dict,
            vars_rename_dict or {},
        )

        return transform_obj
