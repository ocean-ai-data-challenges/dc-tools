
from abc import ABC, abstractmethod
import logging
import re
import traceback
from typing import (
    Any, Dict, List, Optional, Type, Union
)

from argopy import DataFetcher, IndexFetcher
from argopy import set_options as argo_set_options
from argparse import Namespace
import copernicusmarine
import datetime

import dask
from loguru import logger
import os
import numpy as np
import pandas as pd
from pathlib import Path
import xarray as xr

from dctools.data.connection.config import BaseConnectionConfig
from dctools.data.datasets.dc_catalog import CatalogEntry
from dctools.data.coordinates import (
    get_dataset_geometry,
    get_dataset_geometry_light,
    CoordinateSystem,
)
from dctools.data.coordinates import (
    VARIABLES_ALIASES,
    TARGET_DEPTH_VALS,
)

from dctools.data.datasets.dc_catalog import GLOBAL_METADATA
from dctools.dcio.loader import FileLoader
from dctools.utilities.misc_utils import (
    ensure_timestamp,
    deep_copy_object,
)
# from dctools.dcio.saver import DataSaver
# from dctools.utilities.init_dask import setup_dask


def get_time_bound_values(ds: xr.Dataset) -> tuple:
    """
    Obtient les bornes min/max temporelles d'un dataset xarray de manière robuste.
    
    Explore différentes configurations où 'time' peut être :
    - Une dimension principale
    - Une coordonnée 
    - Une variable de données
    - Absente du dataset
    
    Args:
        ds: Dataset xarray
        
    Returns:
        tuple: (min_time, max_time) ou (None, None) si aucune donnée temporelle trouvée
    """
    # Liste des noms possibles pour la dimension temporelle
    time_names = ['time', 'Time', 'TIME', 'date', 'datetime', 'valid_time', 
                  'forecast_time', 'time_counter', 'profile_date']
    
    time_vals = None
    
    try:
        # Chercher dans les dimensions principales
        for time_name in time_names:
            if time_name in ds.dims:
                try:
                    time_vals = ds.coords[time_name]
                    break
                except KeyError:
                    continue
        
        # Si pas trouvé, chercher dans les coordonnées
        if time_vals is None:
            for time_name in time_names:
                if time_name in ds.coords:
                    try:
                        time_vals = ds.coords[time_name]
                        break
                    except KeyError:
                        continue
        
        # Si pas trouvé, chercher dans les variables de données
        if time_vals is None:
            for time_name in time_names:
                if time_name in ds.data_vars:
                    try:
                        time_vals = ds[time_name]
                        break
                    except KeyError:
                        continue
        
        # Si toujours pas trouvé, chercher des variables avec attributs temporels
        if time_vals is None:
            for var_name, var in ds.data_vars.items():
                if hasattr(var, 'attrs'):
                    attrs = var.attrs
                    # Chercher des indices d'attributs temporels
                    temporal_indicators = ['time', 'date', 'temporal', 'calendar']
                    if any(indicator in str(attrs).lower() for indicator in temporal_indicators):
                        try:
                            # Vérifier si ça ressemble à des données temporelles
                            if np.issubdtype(var.dtype, np.datetime64) or 'time' in var_name.lower():
                                time_vals = var
                                break
                        except Exception:
                            continue

        if time_vals is not None:
            # Vérifier le type de données
            if np.issubdtype(time_vals.dtype, np.datetime64):
                # Pour les données temporelles
                # Utiliser les méthodes xarray
                if time_vals.size == 1:   # nombre total d’éléments
                    min_val = time_vals.isel(N_POINTS=0).load().item()
                    max_val = min_val
                else:
                    min_val = pd.to_datetime(time_vals.values.min())
                    max_val = pd.to_datetime(time_vals.values.max())
                # Conversion seulement si nécessaire
                if isinstance(min_val, np.datetime64):
                    min_val = pd.Timestamp(min_val)
                    max_val = pd.Timestamp(max_val)
                else:
                    min_val = pd.to_datetime(min_val)
                    max_val = pd.to_datetime(max_val)
                return (min_val, max_val)
            elif np.issubdtype(time_vals.dtype, np.floating) or np.issubdtype(time_vals.dtype, np.integer):
                # Pour les données numériques
                if time_vals.size == 1:   # nombre total d’éléments
                    min_val = time_vals.isel(N_POINTS=0).load().item()
                    max_val = min_val
                else:
                    min_val = float(time_vals.min().values)
                    max_val = float(time_vals.max().values)

                # Vérifier que les valeurs sont valides
                if np.isnan(min_val) or np.isnan(max_val):
                    return (None, None)
                return (min_val, max_val)
            else:
                logger.warning(f"Unsupported time data type: {time_vals.dtype}")
                return (None, None)
        else:
            logger.debug("No temporal data found in dataset")
            return (None, None)
    except Exception as exc:
        logger.warning(f"Failed to get time bounds for DS: {ds} : {repr(exc)}")
        traceback.print_exc()
        return (None, None)

def clean_for_serialization(obj):
    """Nettoie les objets non-sérialisables avant pickle."""
    # Fermer/nettoyer les objets argopy
    #if hasattr(obj, '_argo_index'):
    #    obj._argo_index = None
    #if hasattr(obj, '_argopy_fetcher'):
    #    obj._argopy_fetcher = None
        
    # Nettoyer fsspec
    if hasattr(obj.params, 'fs') and hasattr(obj.params.fs, '_session'):
        try:
            if hasattr(obj.params.fs._session, 'close'):
                obj.params.fs._session.close()
        except:
            pass
        obj.params.fs = None

    # Nettoyer dataset_processor
    if hasattr(obj.params, 'dataset_processor'):
        #try:
        #    obj.params.dataset_processor.close()
        #except:
        #    pass
        obj.params.dataset_processor = None
    return obj

class BaseConnectionManager(ABC):
    def __init__(
        self, connect_config: BaseConnectionConfig | Namespace,
        call_list_files: bool = True,
        batch_size: Optional[int] = 64,
    ):
        self.connect_config = connect_config
        self.batch_size = batch_size
        if isinstance(connect_config, BaseConnectionConfig):
            self.params = connect_config.to_dict()
        elif isinstance(connect_config, Namespace):
            self.params = connect_config
        else:
            raise TypeError("Unknown type of connection config.")

        self.start_time = self.params.filter_values.get("start_time")
        self.end_time = self.params.filter_values.get("end_time")

        self.init_type = self.params.init_type
        if self.init_type != "from_json" and call_list_files:
            self._list_files = self.list_files()
        if not self.params.file_pattern:
            self.params.file_pattern = "**/*.nc"
        if not self.params.groups:
            self.params.groups = None
        self.file_cache = self.params.file_cache
        self.dataset_processor = self.params.dataset_processor


    def adjust_full_day(
        self,
        date_start: pd.Timestamp, date_end: pd.Timestamp
    ) -> tuple[pd.Timestamp, pd.Timestamp]:
        """
        Si date_start == date_end et date_start est à minuit,
        ajuste date_end pour couvrir toute la journée.
        """
        if pd.isnull(date_start) or pd.isnull(date_end):
            return date_start, date_end
        if date_start == date_end and date_start.hour == 0 and date_start.minute == 0 and date_start.second == 0:
            # Ajuste date_end à la fin de la journée
            date_end = date_start + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        return date_start, date_end

    def open(
        self, path: str,
        mode: str = "rb",
    ) -> xr.Dataset:
        """
        Open a file, prioritizing local then remote access.
        If the file is not available,
        attempt to download it locally and open it.

        Args:
            path (str): Remote path of the file.
            mode (str): Mode to open the file (default is "rb").

        Returns:
            xr.Dataset: Opened dataset.
        """
        # Tenter d'ouvrir le fichier en local
        if LocalConnectionManager.supports(path):
            dataset = self.open_local(path)
            if dataset:
                return dataset
        # Tenter d'ouvrir le fichier en ligne
        elif self.supports(path):
            dataset = self.open_remote(path, mode)
            if dataset:
                return dataset

        # Télécharger le fichier en local, puis l'ouvrir
        try:
            local_path = self._get_local_path(path)
            if not os. path. isfile(local_path):
                self.download_file(path, local_path)

            return self.open_local(local_path)
        except Exception as exc:
            logger.warning(f"Failed to open file: {path}. Error: {repr(exc)}")
            raise

    def open_local(
        self,
        local_path: str,
    ) -> Optional[xr.Dataset]:
        """
        Open a file locally if it exists.

        Args:
            local_path (str): Path to the local file.

        Returns:
            Optional[xr.Dataset]: Opened dataset, or None if the file does not exist.
        """
        if Path(local_path).exists():
            # logger.debug(f"Opening local file: {local_path}")

            return FileLoader.open_dataset_auto(
                local_path, self,
                groups=self.params.groups,
            )
        return None

    def open_remote(
        self, path: str,
        mode: str = "rb",
    ) -> Optional[xr.Dataset]:
        """
        Open a file remotely if the source supports it.

        Args:
            path (str): Remote path of the file.
            mode (str): Mode to open the file (default is "rb").

        Returns:
            Optional[xr.Dataset]: Opened dataset, or None if remote opening is not supported.
        """
        try:
            extension = Path(path).suffix
            if extension != '.zarr':
                return None
            return FileLoader.open_dataset_auto(
                path, self,
                groups=self.params.groups,
            )
        except Exception as exc:
            logger.warning(
                f"Failed to open remote file: {path}. Error: {repr(exc)}"
            )
            return None

    def download_file(self, remote_path: str, local_path: str):
        """
        Download a file from the remote source to the local path.

        Args:
            remote_path (str): Remote path of the file.
            local_path (str): Local path to save the file.
        """
        with self.params.fs.open(remote_path, "rb") as remote_file:
            with open(local_path, "wb") as local_file:
                local_file.write(remote_file.read())
                if self.file_cache is not None:
                    self.file_cache.add(local_path)

    def _get_local_path(self, remote_path: str) -> str:
        """
        Generate the local path for a given remote path.

        Args:
            remote_path (str): Remote path of the file.

        Returns:
            str: Local path of the file.
        """
        # CMEMS case (date)
        if isinstance(remote_path, datetime.datetime):
            return None
        filename = Path(remote_path).name
        return os.path.join(self.params.local_root, filename)

    def get_global_metadata(self) -> Dict[str, Any]:
        """
        Get global metadata for all files in the connection manager.

        Returns:
            Dict[str, Any]: Global metadata including spatial bounds and variable names.
        """
        if self._list_files is None:
            raise FileNotFoundError("No files found to extract global metadata.")

        return self._global_metadata if hasattr(
            self, "_global_metadata"
        ) else self.extract_global_metadata()


    def set_global_metadata(self, global_metadata: Dict[str, Any]) -> None:
        """
        Définit les métadonnées globales pour le gestionnaire de connexion,
        en ne conservant que les clés listées dans la variable de classe global_metadata.

        Args:
            global_metadata (Dict[str, Any]): Dictionnaire de métadonnées globales.
        """
        # Ne garder que les clés pertinentes
        filtered_metadata = {k: v for k, v in global_metadata.items() if k in GLOBAL_METADATA}
        self._global_metadata = filtered_metadata


    def extract_global_metadata(self) -> Dict[str, Any]:
        """
        Extract global metadata (common to all files) from a single file.

        Returns:
            Dict[str, Any]: Global metadata including spatial bounds and variable names.
        """
        files = self._list_files
        if files is None:
            raise FileNotFoundError("Empty file list! No files to extract metadata from.")

        # Boucler sur les fichiers jusqu'à trouver un fichier valide (non vide, non None)
        first_file = None
        for file_path in files:
            # Vérifier que le fichier n'est pas None, vide ou invalide
            if file_path and file_path != "" and file_path is not None:
                try:
                    # Tester si le fichier peut être ouvert
                    test_ds = self.open(file_path, "rb")
                    if test_ds is not None:
                        first_file = file_path
                        break
                except Exception as exc:
                    logger.warning(f"Could not open file {file_path}, trying next: {exc}")
                    continue
        
        if first_file is None:
            raise FileNotFoundError("No valid files found in the list to extract metadata from.")


        with self.open(first_file, "rb") as ds:
            # Extraire les métadonnées globales

            coord_sys = CoordinateSystem.get_coordinate_system(ds)

            # Inférer la résolution spatiale et temporelle
            dict_resolution = self.estimate_resolution(
                ds, coord_sys
            )

            # Associer les variables à leurs dimensions
            variables = {}
            for var_name, var in ds.variables.items():

                if var_name in self.params.keep_variables:
                    variables[var_name] = {
                        "dims": list(var.dims),
                        "std_name": var.attrs.get("standard_name", ""),
                    }

            variables_dict = CoordinateSystem.detect_oceanographic_variables(variables)
            variables_rename_dict = {v: k for k, v in variables_dict.items() if v is not None}

            global_metadata = {
                "variables": variables,
                "variables_dict": variables_dict,
                "variables_rename_dict": variables_rename_dict,
                "resolution": dict_resolution,
                "coord_system": coord_sys,
                "keep_variables": self.params.keep_variables,
            }
        return global_metadata


    def extract_metadata(
        self, path: str,
    ) -> CatalogEntry:
        """
        Extract metadata for a specific file, combining global metadata with file-specific information.

        Args:
            path (str): Path to the file.
            global_metadata (Dict[str, Any]): Global metadata to apply to all files.

        Returns:
            CatalogEntry: Metadata for the specific file as a CatalogEntry.
        """
        try:
            with self.open(path, "rb") as ds:
                time_bounds = get_time_bound_values(ds)
                date_start, date_end = time_bounds

                if self.params.full_day_data:
                    date_start, date_end = self.adjust_full_day(date_start, date_end)

                ds_region = get_dataset_geometry(ds, self._global_metadata.get('coord_system'))

                # Créer une instance de CatalogEntry
                return CatalogEntry(
                    path=path,
                    date_start=date_start,
                    date_end=date_end,
                    variables=self._global_metadata.get("variables"),
                    geometry=ds_region,
                )
        except Exception as exc:
            logger.error(
                f"Failed to extract metadata for file {path}: {traceback.format_exc()}"
            )
            raise

    @staticmethod
    def extract_metadata_worker(
        path: str,
        global_metadata: dict,
        connection_params: dict,
        class_name: Any,
    ):
        """
        Extract metadata for a specific file, combining global metadata with file-specific information.
        Version thread-safe pour éviter les conflits NetCDF.

        Args:
            path (str): Path to the file.
            global_metadata (Dict[str, Any]): Global metadata to apply to all files.

        Returns:
            CatalogEntry: Metadata for the specific file as a CatalogEntry.
        """
        
        # Dictionnaire de mapping des noms vers les classes
        CLASS_REGISTRY: Dict[Type[BaseConnectionConfig], Type[BaseConnectionManager]] = {
            "S3WasabiManager": S3WasabiManager,
            "FTPManager": FTPManager,
            "GlonetManager": GlonetManager,
            "ArgoManager": ArgoManager,
            "CMEMSManager": CMEMSManager,
            "S3Manager": S3Manager,
            "LocalConnectionManager": LocalConnectionManager,
        }
        try:            
            # Récupérer la classe depuis le registre
            if class_name not in CLASS_REGISTRY:
                raise ValueError(f"Unknown class name: {class_name}")
            
            manager_class = CLASS_REGISTRY[class_name]
            if class_name == "CMEMSManager":
                manager = manager_class(
                    connection_params, call_list_files=False,
                    do_logging=False,
                )
            else:
                manager = manager_class(
                    connection_params, call_list_files=False
                )
            ds = manager.open(path, "rb")
            if ds is None:
                logger.warning(f"Could not open {path}")
                return None
                
            time_bounds = get_time_bound_values(ds)
            date_start = time_bounds[0]
            date_end = time_bounds[1]

            ds_region = get_dataset_geometry_light(
                ds, global_metadata.get('coord_system')
            )
            
            # Fermer explicitement le dataset
            if hasattr(ds, 'close'):
                ds.close()

            if global_metadata.get("full_day_data", False):
                if date_start and date_end:
                    date_start = date_start.replace(hour=0, minute=0, second=0, microsecond=0)
                    date_end = date_end.replace(hour=23, minute=59, second=59, microsecond=999999)

            # Créer une instance de CatalogEntry
            return CatalogEntry(
                path=path,
                date_start=date_start,
                date_end=date_end,
                variables=global_metadata.get("variables"),
                geometry=ds_region,
            )
        except Exception as exc:
            logger.warning(
                f"Failed to extract metadata for file {path}: {traceback.format_exc()}"
            )
            return None

    def estimate_resolution(
        self,
        ds: xr.Dataset,
        coord_system: CoordinateSystem,
    ) -> Dict[str, Union[float, str]]:
        """
        Estimate spatial and temporal resolution from an xarray Dataset based on coordinate type and names.

        Args:
            ds: xarray.Dataset
            coord_type: 'geographic' or 'polar'
            dict_coord: dict mapping standardized keys ('lat', 'lon', 'x', 'y', 'time') to actual dataset coord names

        Returns:
            Dictionary of estimated resolutions (degrees, meters, seconds, etc.)
        """
        res = {}

        # Helper function for 1D resolution
        def compute_1d_resolution(coord):
            values = ds.coords[coord].values
            if values.ndim != 1:
                return None
            diffs = np.diff(values)
            return float(np.round(np.median(np.abs(diffs)), 6))

        # Spatial resolution
        if coord_system.coord_type == "geographic":
            lat_name = coord_system.coordinates.get("lat")
            lon_name = coord_system.coordinates.get("lon")
            if lat_name in ds.coords:
                res["latitude"] = compute_1d_resolution(lat_name)
            if lon_name in ds.coords:
                res["longitude"] = compute_1d_resolution(lon_name)

        elif coord_system.coord_type == "polar":
            x_name = coord_system.coordinates.get("x")
            y_name = coord_system.coordinates.get("y")
            if x_name in ds.coords:
                res["x"] = compute_1d_resolution(x_name)
            if y_name in ds.coords:
                res["y"] = compute_1d_resolution(y_name)

        # Temporal resolution (common to both)
        time_name = coord_system.coordinates.get("time")
        if time_name in ds.coords:
            time_values = ds.coords[time_name].values
            if time_values.ndim == 1 and len(time_values) > 1:
                diffs = np.diff(time_values)
                diffs = pd.to_timedelta(diffs).astype("timedelta64[s]").astype(int)
                res["time"] = f"{int(np.median(diffs))}s"
        return res

    def get_config_clean_copy(self):
        connection_conf = deep_copy_object(self.connect_config)
        connection_conf = clean_for_serialization(connection_conf)
        return connection_conf

    def list_files_with_metadata(self) -> List[CatalogEntry]:
        """
        Version avec client Dask intégré et configuration optimisée.
        """

        # Récupérer les métadonnées globales
        global_metadata = self.extract_global_metadata()
        self._global_metadata = global_metadata

        limit = self.params.max_samples if self.params.max_samples else len(self._list_files)
        file_list = self._list_files[-limit:]

        logger.info(f"Processing {len(file_list)} files with integrated Dask client")
        metadata_list = []

        try:

            connection_conf = self.get_config_clean_copy()

            futures = [
                self.dataset_processor.client.submit(
                    self.extract_metadata_worker,
                    path, self._global_metadata,
                    connection_conf,
                    self.__class__.__name__,
                )
                for path in file_list
            ]
            batch_results = self.dataset_processor.client.gather(futures)
            valid_results = [meta for meta in batch_results if meta is not None]
            metadata_list.extend(valid_results)

            logger.info(f"Finished processing ARGO data: {len(valid_results)}/{len(file_list)} items processed")

            self.dataset_processor.cleanup_worker_memory()

        except Exception as exc:
            logger.error(f"Dask metadata extraction failed: {exc}")
            raise

        if not metadata_list:
            logger.error("No valid metadata entries were generated.")
            raise ValueError("No valid metadata entries were generated.")

        return metadata_list


    @classmethod
    @abstractmethod
    def supports(cls, path: str) -> bool:
        pass

    @classmethod
    @abstractmethod
    def list_files(cls, path: str) -> bool:
        pass


class LocalConnectionManager(BaseConnectionManager):
    def list_files(self) -> List[str]:
        """
        List files in the local filesystem matching pattern.

        Args:

        Returns:
            List[str]: List of file paths on local disk".
        """
        root = self.params.local_root
        files = [p for p in sorted(self.params.fs.glob(f"{root}/{self.params.file_pattern}"))]
        return [f"{file}" for file in files]

    @classmethod
    def supports(cls, path: str) -> bool:
        return str(path).startswith("/") or str(path).startswith("file://")


class CMEMSManager(BaseConnectionManager):
    """Class to manage Copernicus Marine downloads."""

    def __init__(
        self, connect_config: BaseConnectionConfig,
        call_list_files: Optional[bool] = True,
        do_logging: Optional[bool] = True,
    ):
        """
        Initialise le gestionnaire CMEMS et effectue la connexion.

        Args:
            connect_config (BaseConnectionConfig): Configuration de connexion.
        """
        super().__init__(connect_config, call_list_files=False)  # Appeler l'initialisation de la classe parente

        if do_logging:
            self.cmems_login()

        if self.init_type != "from_json" and call_list_files:
            self._list_files = self.list_files()


    def get_credentials(self):
        """Get CMEMS credentials.

        Return:
            (dict): CMEMS credentials
        """
        with open(self.params.cmems_credentials_path, "rb") as f:
            lines = f.readlines()
        credentials = {}
        for line in lines:
            key, value = line.strip().split("=")
            credentials[key] = value
        return credentials

    def get_username(self):
        """Get CMEMS username.

        Return:
            (str): CMEMS username
        """
        return self.get_credentials()["cmems_username"]

    def get_password(self):
        """Get CMEMS password.

        Return:
            (str): CMEMS password
        """
        return self.get_credentials()["cmems_password"]

    def get_api_key(self):
        """Get CMEMS API key.

        Return:
            (str): CMEMS API key
        """
        return self.get_credentials()["cmems_api_key"]

    def get_url(self):
        """Get CMEMS URL.

        Return:
            (str): CMEMS URL
        """
        return self.get_credentials()["cmems_url"]

    def get_credentials_dict(self):
        """Get CMEMS credentials as a dictionary.

        Return:
            (dict): CMEMS credentials
        """
        return self.get_credentials()

    def cmems_login(self) -> str:
        """Login to Copernicus Marine."""
        logger.info("Logging to Copernicus Marine.")
        try:
            if not (Path(self.params.cmems_credentials_path).is_file()):
                logger.warning(f"Credentials file not found at {self.params.cmems_credentials_path}.")
                copernicusmarine.login(credentials_file=self.params.cmems_credentials_path)
        except Exception as exc:
            logger.error(f"login to CMEMS failed: {repr(exc)}")

    def cmems_logout(self) -> None:
        """Logout from Copernicus Marine."""
        logger.info("Logging out from Copernicus Marine.")
        try:
            copernicusmarine.logout()
        except Exception as exc:
            logger.error(f"logout from CMEMS failed: {repr(exc)}")
        return None


    def remote_file_exists(self, dt: datetime.datetime) -> bool:
        """
        Teste si un fichier CMEMS existe pour une date donnée sans l'ouvrir.
        
        Args:
            dt (datetime.datetime): Date à tester
            
        Returns:
            bool: True si le fichier existe, False sinon
        """
        try:
            if not isinstance(dt, datetime.datetime):
                dt = datetime.datetime.strptime(dt, "%Y-%m-%dT%H:%M:%S")

            start_datetime = datetime.datetime.combine(dt.date(), datetime.time.min)  # 00:00:00
            end_datetime = datetime.datetime.combine(dt.date(), datetime.time.max)    # 23:59:59.999999
            
            # Utiliser une requête minimale pour tester l'existence
            test_ds = copernicusmarine.open_dataset(
                dataset_id=self.params.dataset_id,
                start_datetime=start_datetime,
                end_datetime=end_datetime,
                # Paramètres pour minimiser la charge
                minimum_longitude=0.0,  # Zone minimale
                maximum_longitude=1.0,
                minimum_latitude=0.0,
                maximum_latitude=1.0,
                #variables=["zos"],
                credentials_file=self.params.cmems_credentials_path,
            )
            
            # Si on arrive ici sans exception, le fichier existe
            if test_ds is not None and len(test_ds.dims) > 0:
                test_ds.close()
                return True
            else:
                return False
                
        except Exception as e:
            logger.debug(f"File does not exist for date {dt}: {e}")
            return False
    
    def list_files(self) -> List[str]:
        """List files in the Copernicus Marine directory."""
        logger.info("Listing files in Copernicus Marine directory.")
        try:
            start_dt = pd.to_datetime(self.start_time)
            start_year = start_dt.year
            start_month = start_dt.month
            start_day = start_dt.day
            end_dt = pd.to_datetime(self.end_time)
            end_year = end_dt.year
            end_month = end_dt.month
            end_day = end_dt.day

            start_date = datetime.datetime(start_year, start_month, start_day)
            end_date = datetime.datetime(end_year, end_month, end_day)
            list_dates = self.list_all_days(
                start_date,
                end_date,
            )
            list_dates = list_dates [:self.params.max_samples]
            valid_dates = []
            for date in list_dates:
                valid_dates.append(date)
            return valid_dates
        except Exception as exc:
            logger.error(f"Failed to list files from CMEMS: {repr(exc)}")
            return []

    def open_remote(
            self, dt: Union[str, datetime.datetime], mode: str = "rb"
        ) -> Optional[xr.Dataset]:
        """
        Open a file remotely from CMEMS using a date.

        Args:
            dt (str): Date string in ISO format (YYYY-MM-DDTHH:MM:SS).
            mode (str): Mode to open the file (default is "rb").

        Returns:
            Optional[xr.Dataset]: Opened dataset, or None if remote opening fails.
        """
        try:
            if not isinstance(dt, datetime.datetime):
                dt = datetime.datetime.strptime(dt, "%Y-%m-%dT%H:%M:%S")
            start_datetime = datetime.datetime.combine(dt.date(), datetime.time.min)  # 00:00:00
            end_datetime = datetime.datetime.combine(dt.date(), datetime.time.max)    # 23:59:59.999999
            ds = copernicusmarine.open_dataset(
                dataset_id=self.params.dataset_id,
                start_datetime=start_datetime,
                end_datetime=end_datetime,
                vertical_axis='depth',
                #minimum_longitude=self.params.filter_values.get("min_lon"),
                #maximum_longitude=self.params.filter_values.get("max_lon"),
                #minimum_latitude=self.params.filter_values.get("min_lat"),
                #maximum_latitude=self.params.filter_values.get("max_lat"),
                # variables=[]
                credentials_file=self.params.cmems_credentials_path,
            ) 
            return ds
            # TODO : get back the remote opening after solving (?) the pickling error
            # with botocore features

        except Exception as e:
            logger.warning(f"Failed to open CMEMS dataset for date : {dt}")
            # traceback.print_exc()
            return None

    @classmethod
    def supports(cls, path: str) -> bool:
        """
        Check if the given path is supported by CMEMS.

        Args:
            path (str): Path to check.

        Returns:
            bool: True if the path is supported, False otherwise.
        """
        # CMEMS ne supporte pas un protocole spécifique comme cmems://
        # On utilise la date (format datetime) comme identifiant des fichiers individuels
        if not isinstance(path, datetime.datetime):
            path = datetime.datetime.strptime(path, "%Y-%m-%dT%H:%M:%S")
        return isinstance(path, datetime.datetime)

    def list_all_days(
            self,
            start_date: datetime.datetime,
            end_date: datetime.datetime
        ) -> list[datetime.datetime]:
        """
        Return a list of datetime.datetime objects for each day between start_date and end_date (inclusive).

        Parameters:
            start_date (datetime): The start of the range.
            end_date (datetime): The end of the range.

        Returns:
            List[datetime]: List of dates at 00:00:00 for each day in the range.
        """
        if start_date > end_date:
            raise ValueError("start_date must be before or equal to end_date.")

        start = datetime.datetime.combine(start_date.date(), datetime.datetime.min.time())
        end = datetime.datetime.combine(end_date.date(), datetime.datetime.min.time())

        n_days = (end - start).days + 1
        return [start + datetime.timedelta(days=i) for i in range(n_days)]

class FTPManager(BaseConnectionManager):
    @classmethod
    def supports(cls, path: str) -> bool:
        # FTP does not support remote opening
        return path.startswith("ftp://")

    def open_remote(self, path, mode = "rb"):
        # cannot open files remotely
        # FTP does not support remote opening
        return None

    def list_files(self) -> List[str]:
        """
        Liste les fichiers disponibles sur le serveur FTP correspondant au motif donné.

        Args:

        Returns:
            List[str]: Liste des chemins des fichiers correspondant au motif.
        """
        try:
            # Accéder au système de fichiers FTP via fsspec
            fs = self.params.fs
            remote_path = f"ftp://{self.params.host}/{self.params.ftp_folder}{self.params.file_pattern}"

            # Lister les fichiers correspondant au motif
            files = sorted(fs.glob(remote_path))
            # logger.info(f"Files found in {remote_path} : {files}")

            if not files:
                logger.warning(f"Aucun fichier trouvé sur le serveur FTP avec le motif : {self.params.file_pattern}")
            return [ f"ftp://{self.params.host}{file}" for file in files ]
        except Exception as exc:
            logger.error(f"Erreur lors de la liste des fichiers sur le serveur FTP : {repr(exc)}")
            return []


class S3Manager(BaseConnectionManager):
    @classmethod
    def supports(cls, path: str) -> bool:
        return path.startswith("s3://")

    def list_files(self) -> List[str]:
        """
        List files matching pattern.

        Args:

        Returns:
            List[str]: List of file paths.
        """
        try:
            if hasattr(self.params, "bucket"):
                logger.info(f"Accessing bucket: {self.params.bucket}")

            # Construire le chemin distant
            remote_path = f"s3://{self.params.bucket}/{self.params.bucket_folder}/{self.params.file_pattern}"

            # Utiliser fsspec pour accéder aux fichiers
            files = sorted(self.params.fs.glob(remote_path))
            files_urls = [
                f"s3://{file}"
                for file in files
            ]

            if not files_urls:
                logger.warning(f"No files found in bucket: {self.params.bucket}")
            return files_urls
        except PermissionError as exc:
            logger.error(f"Permission error while accessing bucket: {repr(exc)}")
            logger.info("List files using object-level access...")

            # Contourner le problème en listant les objets directement
            try:
                files = [
                    f"s3://{self.params.endpoint_url}/{self.params.bucket}/{obj['Key']}"
                    for obj in self.params.fs.ls(remote_path, detail=True)
                    if obj["Key"].endswith(self.params.file_pattern.split("*")[-1])
                ]
                return files
            except Exception as exc:
                logger.error(f"Failed to list files using object-level access: {repr(exc)}")
                raise

    def open_remote(
        self, path: str,
        mode: str = "rb",
    ) -> Optional[xr.Dataset]:
        """
        Open a file remotely from an S3 bucket.

        Args:
            path (str): Remote path of the file.
            mode (str): Mode to open the file (default is "rb").

        Returns:
            Optional[xr.Dataset]: Opened dataset, or None if remote opening is not supported.
        """
        try:
            # logger.debug(f"Open S3 file: {path}")
            extension = Path(path).suffix
            if extension != '.zarr':
                return None
            return(
                FileLoader.open_dataset_auto(
                    path, self, groups=self.params.groups
                )
            )
        except Exception as exc:
            logger.warning(f"Failed to open S3 file: {path}. Error: {repr(exc)}")
            return None


class S3WasabiManager(S3Manager):

    @classmethod
    def supports(cls, path: str) -> bool:
        try:
            extension = Path(path).suffix
            return (path.startswith("s3://")) and (extension == ".zarr" or extension == ".nc")
        except Exception as exc:
            logger.warning(f"Error in supports check for S3WasabiManager: {repr(exc)}")
            traceback.print_exc()
            return False

    def open_remote(
        self,
        path: str, mode: str = "rb",
    ) -> Optional[xr.Dataset]:
        """
        Open a file remotely from an S3 bucket.

        Args:
            path (str): Remote path of the file.
            mode (str): Mode to open the file (default is "rb").

        Returns:
            Optional[xr.Dataset]: Opened dataset, or None if remote opening is not supported.
        """
        try:
            extension = Path(path).suffix
            if extension != '.zarr':
                return None
            return(
                FileLoader.open_dataset_auto(
                    path, self, groups=self.params.groups
                )
            )
        except Exception as exc:
            logger.warning(f"Failed to open Wasabi S3 file: {path}. Error: {repr(exc)}")
            return None


class GlonetManager(BaseConnectionManager):
    @classmethod
    def supports(cls, path: str) -> bool:
        #return False  # do not open : download (otherwise : often get the "too many requests" error)
        return path.startswith("https://")

    def list_files(self) -> List[str]:
        """
        List files matching pattern.

        Args:

        Returns:
            List[str]: List of file paths.
        """
        start_date = "20240103"
        date = datetime.datetime.strptime(start_date, "%Y%m%d")
        list_f = []
        while True:
            if date.year < 2025:
                date_str = date.strftime("%Y%m%d")
                list_f.append(
                    f"{self.params.endpoint_url}/{self.params.glonet_s3_bucket}/{self.params.s3_glonet_folder}/{date_str}.zarr"
                    #f"https://minio.dive.edito.eu/project-glonet/public/glonet_reforecast_2024/{date_str}.zarr"
                    #f"s3://project-glonet/public/glonet_reforecast_2024/{date_str}.zarr"
                )
                date = date + datetime.timedelta(days=7)
            else:
                break
        return list_f


    def open(
        self,
        path: str,
        mode: str = "rb",
    ) -> xr.Dataset:
        return self.open_remote(path)

    def open_remote(
        self, path: str
    ) -> Optional[xr.Dataset]:
        """
        Open a file remotely from an S3 bucket.

        Args:
            path (str): Remote path of the file.
            mode (str): Mode to open the file (default is "rb").

        Returns:
            Optional[xr.Dataset]: Opened dataset, or None if remote opening is not supported.
        """
        try:
            glonet_ds = xr.open_zarr(path)
            # glonet_ds = FileLoader.open_dataset_auto(path, self)
            # logger.debug(f"Opened Glonet file: {path}")
            return glonet_ds

        except Exception as exc:
            logger.warning(f"Failed to open Glonet file: {path}. Error: {repr(exc)}")
            return None


class ArgoManager(BaseConnectionManager):

    def __init__(self, connect_config: BaseConnectionConfig | Namespace,
        lon_range: Optional[tuple[float, float]] = (-180, 180),
        lat_range: Optional[tuple[float, float]] = (-90, 90),
        lon_step: Optional[float] = 1.0,
        lat_step: Optional[float] = 1.0,
        depth_values: Optional[List[float]] = TARGET_DEPTH_VALS,
        time_step_days: Optional[int] = 1,
        custom_cache: Optional[str] = None,
        batch_size: Optional[int] = 10,
        call_list_files: Optional[bool] = True,
        argo_index: Optional[Any] = None,
    ):
        self.batch_size = batch_size

        self.idx_fetcher = IndexFetcher(src="gdac", mode="research")
        self.backup_idx_fetcher = IndexFetcher(src="erddap", mode="research")
        if custom_cache is not None:
            argo_set_options(cachedir=custom_cache)  # Cache local

        self.argo_loader = DataFetcher(src="gdac", mode="research")
        self.backup_argo_loader = DataFetcher(src="erddap", mode="research")
        self.lon_range = lon_range
        self.lat_range = lat_range
        self.lon_step = lon_step
        self.lat_step = lat_step
        self.depth_values = depth_values
        self.time_step_days = time_step_days

        super().__init__(connect_config, call_list_files=False)
        # Charger l'index
        self.argo_index = argo_index
        if argo_index is None:
            self._load_index_once()

        if self.init_type != "from_json" and call_list_files:
            self._list_files = self.list_files()

    def _load_index_once(self):
        """Load ARGO index using argopy API with proper initialization."""
        # Chargement de l'index ARGO

        # Période basée sur les filtres si disponibles
        if not hasattr(self, 'start_time') or not hasattr(self, 'end_time'):
            self.start_time = self.params.filter_values.get("start_time")
            self.end_time = self.params.filter_values.get("end_time")
        start_date = pd.to_datetime(self.start_time)
        end_date = pd.to_datetime(self.end_time)

        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        
        logger.debug(
            f"Fetching ARGO index for region lat{self.lat_range}, lon{self.lon_range}, {start_date_str} to {end_date_str}"
        )
        try:
            region_box = [
                self.lon_range[0], self.lon_range[1],  # longitude min/max
                self.lat_range[0], self.lat_range[1],  # latitude min/max
                start_date_str, end_date_str           # dates en format string
            ]
            self.argo_index = self.idx_fetcher.region(
                region_box,
            ).to_dataframe()
            # Normalisation des timestamps
            self.argo_index['date'] = pd.to_datetime(self.argo_index['date']).dt.floor("min")
        
            logger.info(f"Loaded ARGO index region with {len(self.argo_index)} profiles")
            return
            
        except Exception as argo_error:
            logger.warning(f"ARGO index loading failed: {argo_error}")
            traceback.print_exc()
            logger.warning(f"trying backup argo server")
            try:
                self.argo_index =  self.idx_fetcher.region(
                    region_box,
                ).to_dataframe()
                # Normalisation des timestamps
                self.argo_index['date'] = pd.to_datetime(self.argo_index['date']).dt.floor("min")
            
                logger.info(f"Loaded ARGO index region with {len(self.argo_index)} profiles")
                return
            except Exception as backup_error:
                logger.warning(f"Backup ARGO index loading also failed: {backup_error}")
                traceback.print_exc()
                self.argo_index = pd.DataFrame()  # Empty DataFrame
                return

    def get_argo_index(self):
        return self.argo_index

    @classmethod
    def supports(cls, path: str) -> bool:
        return True


    @staticmethod
    def filter_argo_profile_by_time(
        ds: xr.Dataset,
        tmin: pd.Timestamp,
        tmax: pd.Timestamp,
        time_var_candidates=("TIME", "JULD", "time"),
        point_dim_candidates=("N_POINTS", "n_points", "points", "obs"),
    ) -> xr.Dataset:
        """
        Filter an ARGO profile dataset by a time interval [tmin, tmax].

        Works for both:
        - Standard profiles (single timestamp for the whole cycle)
        - Trajectory-type profiles (time attached to each observation point)

        Parameters
        ----------
        ds : xr.Dataset
            ARGO profile dataset opened with argopy or xarray.
        tmin, tmax : pandas.Timestamp
            Time window for filtering.
        time_var_candidates : tuple[str], optional
            Possible names of the time variable.
        point_dim_candidates : tuple[str], optional
            Possible names of the point dimension.

        Returns
        -------
        xr.Dataset
            Filtered dataset containing only points within the time window.
        """
        # Find the time variable
        time_var = None
        for cand in time_var_candidates:
            if cand in ds:
                time_var = cand
                break
            if cand in ds.coords:
                time_var = cand
                break
        if time_var is None:
            raise ValueError("No valid time variable found in dataset")

        times = pd.to_datetime(ds[time_var].values)

        # Case 1: trajectory-like dataset with N_POINTS dimension
        point_dim = None
        for cand in point_dim_candidates:
            if cand in ds.dims:
                point_dim = cand
                break

        if point_dim is not None and times.shape[0] == ds.sizes[point_dim]:
            mask = (times >= tmin) & (times <= tmax)
            ds_filtered = ds.isel({point_dim: mask})

        # Case 2: profile dataset with scalar or 1D time
        else:
            if times.ndim == 0 or len(times) == 1:
                if tmin <= times[0] <= tmax:
                    ds_filtered = ds
                else:
                    ds_filtered = ds.isel({list(ds.dims)[0]: slice(0, 0)})  # empty slice
            else:
                # fallback: filter on "time" dimension
                ds_filtered = ds.sel(time=slice(tmin, tmax))

        return ds_filtered


    @staticmethod
    def _extract_argo_metadata(
        path: pd.Timestamp,
        connection_params: dict,
        global_metadata: dict,
        argo_index: Any,
    ) -> Optional[CatalogEntry]:
        """Version pour worker Dask - ARGO."""
        try:
            logger.debug(f"Process ARGO item: {path}")
            # Recréer le manager
            manager = ArgoManager(
                connection_params, call_list_files=False, argo_index=argo_index
            )
            
            # Traitement
            ds = manager.open(path, add_depth=False)
            if ds is None:
                return None
        
            time_bounds = get_time_bound_values(ds)
            date_start, date_end = time_bounds
            # Extraction des coordonnées ARGO
            coord_sys = global_metadata.get('coord_system', {})
            
            # Variables
            variables = global_metadata.get("variables", {})

            geometry = get_dataset_geometry(ds, coord_sys)
            
            # Créer l'entrée
            metadata = CatalogEntry(
                path=str(path),
                date_start=ensure_timestamp(date_start),
                date_end=ensure_timestamp(date_end) + pd.Timedelta(minutes=1),
                variables=variables,
                geometry=geometry,
            )
            return metadata
            
        except Exception as exc:
            logger.error(f"ARGO worker error for {path}: {exc}")
            traceback.print_exc()
            return None


    def list_files_with_metadata(self) -> List[CatalogEntry]:
        """Liste les fichiers avec leurs métadonnées."""
        global_metadata = self.extract_global_metadata()
        self._global_metadata = global_metadata

        list_dates = self._list_files
        logger.info(f"Processing {len(list_dates)} ARGO dates with Dask")

        metadata_list = []
        
        try:
            # Scatter une seule fois les objets volumineux
            connection_conf = deep_copy_object(self.connect_config)
            connection_conf = clean_for_serialization(self.connect_config)
            scattered_config = self.dataset_processor.scatter_data(
                connection_conf, broadcast_item=True)
            scattered_metadata = self.dataset_processor.scatter_data(
                self._global_metadata, broadcast_item=True)
            scattered_argo_index = self.dataset_processor.scatter_data(
                self.argo_index, broadcast_item=True)
            
            delayed_tasks = [
                dask.delayed(self._extract_argo_metadata)(
                    start_date, scattered_config, scattered_metadata, scattered_argo_index
                )
                for start_date in list_dates
            ]

            batch_results = self.dataset_processor.compute_delayed_tasks(delayed_tasks)
            valid_results = [meta for meta in batch_results if meta is not None]
            metadata_list.extend(valid_results)

            logger.info(f"Finished processing ARGO data: {len(valid_results)}/{len(list_dates)} items processed")

            self.dataset_processor.cleanup_worker_memory()

        except Exception as exc:
            logger.error(f"Dask ARGO metadata extraction failed: {exc}")
            traceback.print_exc()

        if not metadata_list:
            logger.error("No valid ARGO metadata entries were generated.")
            traceback.print_exc()
            raise ValueError("No valid ARGO metadata entries were generated.")

        return metadata_list

    def extract_cycle(self, filename):
        # Cherche un pattern du type "_<digits><optional_letter>.nc" à la fin
        match = re.search(r"_([0-9]{3,4}[A-Z]?)\.nc$", filename)
        if match:
            return match.group(1)
        return None

    def list_files(self) -> list[str]:
        """Liste les couples (wmo, cycle) sous forme de string 'wmo:cycle'."""
        if self.argo_index is None:
            return []
        else:
            if self.argo_index.empty:
                logger.warning("ARGO index is empty")
                return []
        
        start_dt = pd.to_datetime(self.start_time)
        end_dt = pd.to_datetime(self.end_time)

        # Filtrage temporel et spatial
        filtered = self.argo_index[
            (self.argo_index['date'] >= start_dt) &
            (self.argo_index['date'] <= end_dt) &
            (self.argo_index['latitude'] >= self.lat_range[0]) &
            (self.argo_index['latitude'] <= self.lat_range[1]) &
            (self.argo_index['longitude'] >= self.lon_range[0]) &
            (self.argo_index['longitude'] <= self.lon_range[1])
        ]

        if filtered.empty:
            logger.warning("No ARGO profiles found in requested range")
            return []

        # Extraire le cycle à partir du nom de fichier (3 à 4 caractèress avant ".nc")
        filtered["cycle"] = filtered["file"].apply(self.extract_cycle)
        # Pour garder la colonne cycle au format string (pour les cas alphanumériques)
        cycle_na_count = filtered["cycle"].isna().sum()
        wmo_na_count = filtered["wmo"].isna().sum()
        
        logger.debug(f"NA values in cycle column: {cycle_na_count}")
        logger.debug(f"NA values in wmo column: {wmo_na_count}")
        
        if cycle_na_count > 0:
            logger.warning(f"{cycle_na_count} profiles have invalid cycle numbers")
            # Montrer quelques exemples de fichiers problématiques
            problematic_files = filtered[filtered["cycle"].isna()]["file"].head(5)
            logger.debug(f"Examples of problematic file names: {problematic_files.tolist()}")

        filtered_clean = filtered.dropna(subset=['wmo', 'cycle'])
        
        logger.debug(f"Dataset shape after removing NA values: {filtered_clean.shape}")
        
        if filtered_clean.empty:
            logger.warning("No valid ARGO profiles after removing NA values")
            return []

        try:
            couples = sorted(set(zip(filtered_clean['wmo'], filtered_clean['cycle'])))
            logger.debug(f"Successfully created {len(couples)} unique WMO:cycle couples")
        except Exception as e:
            logger.error(f"Error creating couples even after NA removal: {e}")
            return []

        # Limite si besoin
        limit = self.params.max_samples if self.params.max_samples else len(couples)
        couples = couples[:limit]

        # Formatter en string "wmo:cycle" / supprimer les cycles "spéciaux"
        couples_str = [f"{int(wmo)}:{cycle}" for wmo, cycle in couples if len(cycle) == 3]

        logger.info(f"Found {len(couples_str)} valid ARGO profiles (formatted as 'wmo:cycle')")
        return couples_str

    def open(self, wmo_cycle_str: str, mode: str = "rb", add_depth: bool = True) -> Optional[xr.Dataset]:
        """Ouvre un profil ARGO à partir d'un string 'wmo:cycle'."""
        try:
            # Parse le string
            wmo_str, cycle_str = wmo_cycle_str.split(":")
            wmo, cycle = int(wmo_str), int(cycle_str)

            # Charger le profil (renvoie un wrapper type ArgoProfile)
            profile = self.argo_loader.profile(wmo, cycle)
            if profile is None:
                logger.warning(f"Profile WMO={wmo}, cycle={cycle} not found on server")
                return None

            # Charger les données
            try:
                profile_ds = profile.load().data
            except Exception as e:
                # logger.warning(f"Profile WMO={wmo}, cycle={cycle} failed to load: {e}")
                return None

            # Vérifier si dataset est valide
            if profile_ds is None:
                # logger.warning(f"Profile WMO={wmo}, cycle={cycle} returned None")
                return None
            if not profile_ds.dims:
                # logger.warning(f"Profile WMO={wmo}, cycle={cycle} has no dimensions")
                return None
            if "N_POINTS" not in profile_ds.sizes or profile_ds.sizes["N_POINTS"] == 0:
                # logger.warning(f"Profile WMO={wmo}, cycle={cycle} has zero points")
                return None

            # Vérifier si toutes les variables sont vides
            n_nonempty_vars = sum(
                (v.size > 0) for v in profile_ds.data_vars.values()
            )
            if n_nonempty_vars == 0:
                logger.warning(f"Profile WMO={wmo}, cycle={cycle} has no non-empty variables")
                return None

            # Ajouter dimension profondeur si besoin
            if add_depth:
                profile_ds = self._add_depth_dimension(profile_ds)
            
            # chunking
            profile_ds = profile_ds.chunk({"N_POINTS": 10})

            # logger.debug(f"Opened ARGO profile WMO={wmo}, cycle={cycle} with {profile_ds.dims['N_POINTS']} points")
            return profile_ds

        except Exception as e:
            logger.error(f"Failed to load ARGO profile {wmo_cycle_str}: {e}")
            return None


    def _add_depth_dimension(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Ajoute la profondeur comme dimension au dataset ARGO.
        
        Args:
            ds: Dataset ARGO avec dimension N_POINTS
            
        Returns:
            Dataset avec dimension 'depth' ajoutée basée sur PRES ou PRES_ADJUSTED
        """
        # Utiliser PRES_ADJUSTED comme profondeur (1 décibar ≈ 1 mètre)
        if 'PRES_ADJUSTED' in ds.data_vars:
            depth_values = ds['PRES_ADJUSTED'].values
        elif 'PRES' in ds.data_vars:
            depth_values = ds['PRES'].values
        else:
            logger.warning("No pressure variable found in ARGO data")
            return ds
        
        # Créer une nouvelle dimension 'depth' basée sur les valeurs de pression
        ds_copy = ds.copy()
        
        # Ajouter 'depth' comme coordonnée
        ds_copy = ds_copy.assign_coords(depth=('N_POINTS', depth_values))
        
        # ajouter des attributs pour la dimension depth
        ds_copy['depth'].attrs = {
            'standard_name': 'depth',
            'long_name': 'Depth',
            'units': 'meters',
            'positive': 'down',
            'comment': 'Approximated from pressure (1 dbar ≈ 1 meter)'
        }
        
        return ds_copy
