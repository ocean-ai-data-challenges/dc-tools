
from abc import ABC, abstractmethod
import traceback
from typing import (
    Any, Callable, Dict, List, Optional, Union
)

from argopy import ArgoIndex, DataFetcher, IndexFetcher
from argparse import Namespace
import copernicusmarine
from dask.distributed import Client
from dataclasses import dataclass, asdict
import datetime
import fsspec
import geopandas as gpd
from loguru import logger
import os
import numpy as np
import pandas as pd
from pathlib import Path

# import random
# from shapely.geometry import box, Point
from shapely.geometry import box
# import string
import xarray as xr

from dctools.data.connection.config import BaseConnectionConfig
from dctools.data.datasets.dc_catalog import CatalogEntry
from dctools.data.coordinates import (
    get_dataset_geometry,
    CoordinateSystem,
)
# from dctools.dcio.saver import DataSaver
from dctools.dcio.loader import FileLoader
from dctools.utilities.file_utils import read_file_tolist, FileCacheManager
# from dctools.processing.cmems_data import extract_dates_from_filename
from dctools.data.coordinates import (
    VARIABLES_ALIASES,
    GLONET_DEPTH_VALS,
)
from dctools.utilities.init_dask import setup_dask



'''def extract_metadata(
    path: str,
    coord_system: Dict[str, Any],
    variables: Dict[str, Any],
    open_func: Callable,
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
        with open_func(path, "rb") as ds:
            date_start = pd.to_datetime(
                ds.time.min().values
            ) if "time" in ds.coords else None
            date_end = pd.to_datetime(
                ds.time.max().values
            ) if "time" in ds.coords else None

        ds_region = get_dataset_geometry(ds, coord_system)

        # Créer une instance de CatalogEntry
        #date_start="1"
        #date_end="2"
        return CatalogEntry(
            path=path,
            date_start="1",  #date_start,
            date_end="2",   #date_end,
            variables=variables,
            geometry=None, #ds_region,
        )
    except Exception as exc:
        logger.error(
            f"Failed to extract metadata for file {path}: {traceback.format_exc()}"
        )
        raise'''

'''def extract_metadata_worker(manager_class, manager_params, path):
    # Recrée une instance du manager côté worker
    manager = manager_class(manager_params)
    return manager.extract_metadata(path)'''

class BaseConnectionManager(ABC):
    def __init__(self, connect_config: BaseConnectionConfig | Namespace):
        if isinstance(connect_config, BaseConnectionConfig):
            self.params = connect_config.to_dict()
        elif isinstance(connect_config, Namespace):
            self.params = connect_config
        else:
            raise TypeError("Unknown type of connection config.")
        init_type = self.params.init_type
        if init_type != "from_json":
            self._list_files = self.list_files()
        if not self.params.file_pattern:
            self.params.file_pattern = "**/*.nc"
        if not self.params.groups:
            self.params.groups = None
        self.file_cache = self.params.file_cache
        # self.dask_cluster = self.params.dask_cluster

    def open(
        self, path: str,
        mode: str = "rb",
    ) -> xr.Dataset:
        """
        Open a file, prioritizing remote access via S3. If the file is not available remotely,
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
            # logger.debug(f"Open remote file: {path}")
            dataset = self.open_remote(path, mode)
            if dataset:
                return dataset

        # Télécharger le fichier en local, puis l'ouvrir
        try:
            local_path = self._get_local_path(path)
            if not os. path. isfile(local_path):
                # logger.debug(f"Downloading file to local path: {local_path}")
                self.download_file(path, local_path)

            return self.open_local(local_path)
        except Exception as exc:
            logger.error(f"Failed to open file: {path}. Error: {repr(exc)}")
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
            #logger.info(f"Open remote file: {path}")
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

        remote_path = remote_path.replace("ftp://ftp.ifremer.fr", "")
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
        if not self._list_files:
            raise FileNotFoundError("No files found to extract global metadata.")

        return self._global_metadata if hasattr(self, "_global_metadata") else self.extract_global_metadata()


    def set_global_metadata(self, global_metadata: Dict[str, Any]) -> None:
        """
        Définit les métadonnées globales pour le gestionnaire de connexion,
        en ne conservant que les clés listées dans la variable de classe global_metadata.

        Args:
            global_metadata (Dict[str, Any]): Dictionnaire de métadonnées globales.
        """
        from dctools.data.datasets.dc_catalog import GLOBAL_METADATA

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
        if not files:
            raise FileNotFoundError("Empty file list! No files to extract metadata from.")

        first_file = files[0]
        # logger.info(f"Extracting global metadata from {first_file}")

        # Charger le fichier avec xarray
        # logger.info(f"Opening first file: {first_file}")

        with self.open(first_file, "rb") as ds:
            # Extraire les métadonnées globales

            coord_sys = CoordinateSystem.get_coordinate_system(ds)
            # dimensions = coord_sys.coordinates
            # dimensions_rename_dict = {v: k for k, v in dimensions.items()}

            # logger.debug(f"Dimensions: {coord_sys.coordinates}")

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
            # dimensions = dict(ds.dims)

            # logger.debug(f"\nDetected oceanographic variables: {variables_rename_dict}\n")

            global_metadata = {
                "variables": variables,
                "variables_dict": variables_dict,
                "variables_rename_dict": variables_rename_dict,
                "resolution": dict_resolution,
                "coord_system": coord_sys,
                "keep_variables": self.params.keep_variables,
                # "is_observation": coord_sys.is_observation_dataset(),
                #"dimensions_rename_dict": dimensions_rename_dict,
            }
        return global_metadata

    def extract_metadata(self, path: str):  #, global_metadata: Dict[str, Any]) -> CatalogEntry:
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
                date_start = pd.to_datetime(
                    ds.time.min().values
                ) if "time" in ds.coords else None
                date_end = pd.to_datetime(
                    ds.time.max().values
                ) if "time" in ds.coords else None

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

    '''def list_files_with_metadata(self) -> List[CatalogEntry]:
        """
        List all files with their metadata by combining global metadata and file-specific information.

        Returns:
            List[CatalogEntry]: List of metadata entries for each file.
        """
        # Récupérer les métadonnées globales
        global_metadata = self.extract_global_metadata()
        self._global_metadata = global_metadata

        # Initialiser une liste pour stocker les métadonnées
        metadata_list = []

        limit = self.params.max_samples if self.params.max_samples else len(self._list_files)
        # Parcourir tous les fichiers et extraire leurs métadonnées
        for path in self._list_files:
            try:
                metadata_entry = self.extract_metadata(path, global_metadata)
                metadata_list.append(metadata_entry)
                if len(metadata_list) >= limit:
                    # logger.info(f"Reached the limit of {limit} metadata entries.")
                    break
            except Exception as exc:
                logger.warning(f"Failed to extract metadata for file {path}: {repr(exc)}")

        if not metadata_list:
            logger.error("No valid metadata entries were generated.")
            raise ValueError("No valid metadata entries were generated.")

        return metadata_list'''


    def list_files_with_metadata(self) -> List[CatalogEntry]:
        """
        List all files with their metadata by combining global metadata and file-specific information.

        Returns:
            List[CatalogEntry]: List of metadata entries for each file.
        """
        # Récupérer les métadonnées globales
        global_metadata = self.extract_global_metadata()
        self._global_metadata = global_metadata

        metadata_list = []
        limit = self.params.max_samples if self.params.max_samples else len(self._list_files)
        # self._list_files = [s for s in self._list_files if s.count("_2024") >= 2]    # TODO : REMOVE

        file_list = self._list_files[:limit]

        #manager_class = self.__class__
        #manager_params = self.params

        #must_close_dask = False
        #if self.dask_cluster is None:
        #    self.dask_cluster = setup_dask()
        #    must_close_dask = True
        coord_system = self._global_metadata.get('coord_system')
        variables = self._global_metadata.get("variables")

        '''dask_client = Client(self.dask_cluster)
        # Utilise le client Dask courant pour la soumission des tâches en parallèle
        futures = [
            # dask_client.submit(extract_and_cache, path, global_metadata)  #, self.file_cache)
            dask_client.submit(
                extract_metadata,
                path, coord_system, variables,
                self.open,
            )
            for path in file_list
        ]
        results = dask_client.gather(futures)
        metadata_list = [res for res in results if res is not None]'''

        metadata_list = [self.extract_metadata(
            path
        ) for path in file_list]

        if not metadata_list:
            logger.error("No valid metadata entries were generated.")
            raise ValueError("No valid metadata entries were generated.")

        #if must_close_dask:
        #    client.close()

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

    def __init__(self, connect_config: BaseConnectionConfig):
        """
        Initialise le gestionnaire CMEMS et effectue la connexion.

        Args:
            connect_config (BaseConnectionConfig): Configuration de connexion.
        """
        super().__init__(connect_config)  # Appeler l'initialisation de la classe parente

        # logger.debug(f"CMEMS file : {self.params.cmems_credentials}")
        self.cmems_login()

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
                copernicusmarine.login()
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
    
    def list_files(self) -> List[str]:
        """List files in the Copernicus Marine directory."""
        logger.info("Listing files in Copernicus Marine directory.")
        tmp_filepath = os.path.join(self.params.local_root, "files.txt")
        try:
            '''copernicusmarine.get(
                dataset_id=self.params.dataset_id, create_file_list=tmp_filepath
            )
            self._files = read_file_tolist(tmp_filepath)'''

            # list_dates = self.list_available_dates(self.params.dataset_id)

            #return self._files

            start_date = datetime.datetime(2024, 1, 1)
            end_date = datetime.datetime(2025, 1, 3)
            list_dates = self.list_all_days(
                start_date,
                end_date,
            )
            valid_dates = []
            for date in list_dates:
                print(date)
                ds = self.open_remote(date, mode="rb")
                if ds is not None:
                    valid_dates.append(date)
            return valid_dates
        except Exception as exc:
            logger.error(f"Failed to list files from CMEMS: {repr(exc)}")
            return []

    '''def get_product_metadata(self) -> Dict[str, Any]:
        """
        Fetch product-level metadata from the CMEMS API.

        Returns:
            Dict[str, Any]: A dictionary containing metadata for the product.
        """
        product_metadata = copernicusmarine.describe(self.params.dataset_id)

        return {
            "variables": product_metadata.get("variables", []),
            "date_start": product_metadata.get("temporal_coverage_start"),
            "date_end": product_metadata.get("temporal_coverage_end"),
            "lon_min": product_metadata.get("geospatial_lon_min"),
            "lon_max": product_metadata.get("geospatial_lon_max"),
            "lat_min": product_metadata.get("geospatial_lat_min"),
            "lat_max": product_metadata.get("geospatial_lat_max"),
        }'''

    '''def get_day_bounds(self, dt: datetime) -> tuple[datetime, datetime]:
        from datetime import time
        start_of_day = datetime.datetime.combine(dt.date(), time.min)  # 00:00:00
        end_of_day = datetime.datetime.combine(dt.date(), time.max)    # 23:59:59.999999
        return start_of_day, end_of_day'''

    def open_remote(self, dt: str, mode: str = "rb") -> Optional[xr.Dataset]:
        """
        Open a file remotely from CMEMS using S3 URLs.

        Args:
            path (str): Remote S3 path of the file.
            mode (str): Mode to open the file (default is "rb").

        Returns:
            Optional[xr.Dataset]: Opened dataset, or None if remote opening fails.
        """
        try:
            #start_datetime, end_datetime = self.get_day_bounds(datetime.datetime(date))
            # print(type(dt))
            if not isinstance(dt, datetime.datetime):
                dt = datetime.datetime.strptime(dt, "%Y-%m-%dT%H:%M:%S")
            from datetime import time
            start_datetime = datetime.datetime.combine(dt.date(), time.min)  # 00:00:00
            end_datetime = datetime.datetime.combine(dt.date(), time.max)    # 23:59:59.999999
            ds = copernicusmarine.open_dataset(
                dataset_id=self.params.dataset_id,
                start_datetime=start_datetime,
                end_datetime=end_datetime,
                vertical_axis='depth',
                #minimum_longitude=-10,
                #maximum_longitude=10,
                #minimum_latitude=45,
                #maximum_latitude=55,
                # variables=["uo","vo"]  # optionnel
            )
            return ds
        except Exception as e:
            traceback.print_exc()
            return None

        # return None

    '''def download_file(self, remote_path: str, local_path: str):
        """
        Download a specific file from CMEMS.

        Args:
            path (str): Path to the file to download.
        """
        # Extraire la date à partir du nom du fichier
        filename = Path(local_path).name
        try:
            # Télécharger le fichier via l'API CMEMS
            logger.info(f"Downloading file {filename} from CMEMS...")
            copernicusmarine.get(
                dataset_id=self.params.dataset_id,
                filter=filename,
                output_directory=self.params.local_root,
                no_directories=True,
                credentials_file=self.params.cmems_credentials,
            )
            if  self.file_cache is not None:
                self.file_cache.add(filename)
        except Exception as exc:
            logger.error(f"download from CMEMS failed: {repr(exc)}")
        return None'''


    '''def get_cmems_filter_from_date(self, date: str) -> str:
        """
        Generate a filter string to select the correct file for a given date.

        Args:
            date (str): Date in 'YYYY-MM-DD' format.

        Returns:
            str: Filter string for the CMEMS API.
        """
        dt = datetime.datetime.strptime(date, "%Y-%m-%d")
        return f"*/{dt.strftime('%Y')}/{dt.strftime('%m')}/*_{dt.strftime('%Y%m%d')}_*.nc"'''

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
        # On peut utiliser un identifiant spécifique pour les chemins CMEMS
        # return "cmems" in path.lower()
        return isinstance(path, datetime.datetime) #"cmems" in path.lower()

    from datetime import datetime, timedelta

    def list_all_days(self, start_date: datetime, end_date: datetime) -> list[datetime]:
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
            logger.info(f"Files found in {remote_path} : {files}")

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
        extension = Path(path).suffix
        return path.startswith("s3://") and extension == ".zarr"

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
        #logger.info(f"List of files: {list_files}")
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

    @classmethod
    def supports(cls, path: str) -> bool:
        return True

    def filter_by_geometry(catalog_gdf: gpd.GeoDataFrame, polygon: gpd.GeoSeries) -> gpd.GeoDataFrame:
        return catalog_gdf[catalog_gdf.geometry.centroid.within(polygon.unary_union)]

    def spatial_filter(catalog_gdf: gpd.GeoDataFrame, region_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        return gpd.sjoin(catalog_gdf, region_gdf, how="inner", predicate="intersects")

    def group_profiles_daily(self, df: pd.DataFrame, spatial_res: float = 5.0):
        df["grid_lat"] = (df["lat"] // spatial_res) * spatial_res
        df["grid_lon"] = (df["lon"] // spatial_res) * spatial_res
        df["date_bin"] = df["date"].dt.date  # résolution = 1 jour
        return df.groupby(["grid_lat", "grid_lon", "date_bin"])

    def get_argo_date_range(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        """Return the min/max dates available in the ARGO index."""
        index = IndexFetcher().to_dataframe()
        min_date = index['date'].min()
        max_date = index['date'].max()
        return pd.to_datetime(min_date), pd.to_datetime(max_date)

    def load_argo_profile_from_url(wmo: int, cycle: int) -> xr.Dataset:
        """
        Télécharge et charge un profil Argo à partir de son WMO et numéro de cycle.

        Parameters
        ----------
        wmo : int
            Numéro WMO de la flotte.
        cycle : int
            Numéro de cycle du profil.

        Returns
        -------
        xr.Dataset
            Profil Argo avec temps décodé.
        """
        # Construire l'URL standard Ifremer
        filename = f"D{wmo}_{str(cycle).zfill(3)}.nc"
        url = f"https://data-argo.ifremer.fr/dac/coriolis/{wmo}/profiles/{filename}"
        import requests
        import tempfile
        # Télécharger dans un fichier temporaire
        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
            response = requests.get(url)
            if response.status_code != 200:
                raise RuntimeError(f"Erreur {response.status_code} pour l'URL : {url}")
            tmp.write(response.content)
            tmp_path = Path(tmp.name)

        # Ouvrir le fichier avec xarray (sans décodage)
        ds = xr.open_dataset(tmp_path, decode_times=False)

        # Décodage manuel du temps à partir de JULD
        if "JULD" in ds:
            ref_time = pd.Timestamp("1950-01-01")
            ds["time"] = ref_time + pd.to_timedelta(ds["JULD"].values, unit="D")
            ds = ds.assign_coords(time=ds["time"])

        return ds

    def list_files_with_metadata(
            self,
            lon_range: Optional[tuple[float, float]] = (-180, 180),
            lat_range: Optional[tuple[float, float]] = (-90, 90),
            lon_step: Optional[float]=1.0,
            lat_step: Optional[float]=1.0,
            depth_values: Optional[List[float]] = GLONET_DEPTH_VALS,
            time_step_days: Optional[int]=1,
    ) -> List[CatalogEntry]:
        """
        List all files with their metadata by combining global metadata and file-specific information.

        Args:
            geometry_filter (Optional[gpd.GeoDataFrame]): Geometry filter to apply on the catalog.

        Returns:
            List[CatalogEntry]: List of metadata entries for each file.
        """
        list_dates = self._list_files

        metadata_list: List[CatalogEntry] = []

        first_elem = True

        for n_elem, start_date in enumerate(list_dates):
            end_date = start_date + pd.Timedelta(days=1)
            # logger.info(f"Processing element: {elem.to_markdown()}")
            # logger.info(f"Processing date: {start_date} to {end_date}")
            #wmo_number = elem["wmo"]
            #cycle_number = elem["cyc"]
            #logger.info(f"Processing WMO number: {wmo_number}")
            #if not wmo_number:
            #    continue
            #argopy.set_options(ds='phy', src='erddap', mode='research'):
            #params = 'all'  # eg: 'DOXY' or ['DOXY', 'BBP700']
            #logger.info(f"Fetching data for WMO number: {wmo_number}")
            # ds = DataFetcher().float(wmo_number).to_xarray()
            argo_loader = DataFetcher()
            # logger.info(f"Fetching ARGO data for WMO number: {wmo_number}")
            # ds = argo_loader.float(wmo_number).to_xarray(
            
            # ds = argo_loader.profile(wmo_number, cycle_number).load().data
            profile = argo_loader.region([
                min(lon_range), max(lon_range),
                min(lat_range), max(lat_range),
                min(depth_values),
                max(depth_values),
                start_date, end_date
            ])
            logger.info(f"Fetched ARGO data : {profile}")
            ds = profile.load().data
            # ds = ds.to_xarray() # decode_times=False)
            logger.info(f"Dataset : {ds}")
            # get coordinate system
            if first_elem:
                coord_sys = CoordinateSystem.get_coordinate_system(ds)

                variables = {v: list(ds[v].dims) for v in ds.data_vars if v in self.params.keep_variables}
                variables_dict = CoordinateSystem.detect_oceanographic_variables(variables)
                variables_rename_dict = {v: k for k, v in variables_dict.items()}
                # dimensions = dict(ds.dims)

                coord_sys = CoordinateSystem.get_coordinate_system(ds)
                # dimensions = coord_sys.coordinates
                resolution={"lon": lon_step, "lat": lat_step, "time": time_step_days}

                global_metadata = {
                    "variables": variables,
                    "variables_dict": variables_dict,
                    "variables_rename_dict": variables_rename_dict,
                    "resolution": resolution,
                    "coord_system": coord_sys,
                    "keep_variables": self.params.keep_variables,
                    # "is_observation": coord_sys.is_observation_dataset(),
                }
                self.global_metadata = global_metadata
                first_elem = False
            # dimensions_rename_dict = {v: k for k, v in dimensions.items()}
            # geometry = gpd.GeoSeries([tile])
            geometry = get_dataset_geometry(ds, coord_sys)
            date_start = pd.to_datetime(
                ds.time.min().values
            ) if "time" in ds.coords else None
            date_end = pd.to_datetime(
                ds.time.max().values
            ) if "time" in ds.coords else None


            def adjust_full_day_if_needed(date_start: pd.Timestamp, date_end: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
                """
                Si date_start == date_end et date_start est à minuit, ajuste date_end pour couvrir toute la journée.
                """
                if pd.isnull(date_start) or pd.isnull(date_end):
                    return date_start, date_end
                if date_start == date_end and date_start.hour == 0 and date_start.minute == 0 and date_start.second == 0:
                    # Ajuste date_end à la fin de la journée
                    date_end = date_start + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
                return date_start, date_end

            date_start, date_end = adjust_full_day_if_needed(date_start, date_end)
            entry = CatalogEntry(
                path=None,  #f"argo://{df['file']}",
                date_start=date_start,
                date_end=date_end,
                variables=variables,
                geometry=geometry,
            )
            metadata_list.append(entry)

        return metadata_list
 

    def list_files(self) -> List[str]:

        #idx = ArgoIndex(host="https://data-argo.ifremer.fr", index_file="core", cache=True)
        # idx = ArgoIndex(host="https://data-argo.ifremer.fr", index_file="bgc-b", cache=True)
        # idx = ArgoIndex(host="https://data-argo.ifremer.fr", index_file="bgc-s", cache=True)
        # idx = ArgoIndex(host="https://data-argo.ifremer.fr", index_file="meta", cache=True)
        # idx = ArgoIndex()

        # idx = ArgoIndex(host="https://data-argo.ifremer.fr")  # Default host
        idx = ArgoIndex(host="ftp://ftp.ifremer.fr/ifremer/argo", index_file="ar_index_global_prof.txt")  # Default index
        # idx = ArgoIndex(index_file="bgc-s")  # Use keywords instead of exact file names
        # idx = ArgoIndex(host="https://data-argo.ifremer.fr", index_file="bgc-b", cache=True)  # Use cache for performances
        # idx = ArgoIndex(host=".", index_file="dummy_index.txt", convention="core")  # Load your own index

        limit = self.params.max_samples if self.params.max_samples else len(self._list_files)
        logger.info(f"Loading ARGO index with limit: {limit}")

        idx = idx.load(nrows=limit)
        logger.info(f"\n\nARGO Index: {idx}")
        df = idx.to_dataframe(index=True)
        logger.info(f"\n\nsize List of files 1: {idx.uri_full_index}")

        # 3. Trier les données par date croissante
        #df_sorted = df.sort_values(by='date').reset_index(drop=True)
        df["date"] = pd.to_datetime(df["date"])

        # Générer une série de dates journalières entre min et max
        list_dates = pd.date_range(start=df['date'].min(), end=df['date'].max(), freq='D')

        return list_dates   # idx.uri_full_index


    def open(
        self, path: str,
        mode: str = "rb",
    ) -> xr.Dataset:
        """
        Open an Argo dataset from a given path.

        Args:
            path (str): Path to the Argo dataset.
            mode (str): Mode to open the file (default is "rb").

        Returns:
            xr.Dataset: Opened Argo dataset.
        """
        # On suppose que le chemin est un URI Argo
        if not path.startswith("argo://"):
            raise ValueError(f"Unsupported path format: {path}")

        # Extraire les informations du chemin
        parts = path.split("/")
        if len(parts) < 3:
            raise ValueError(f"Invalid Argo path: {path}")

        # Charger le dataset avec argopy
        from argopy import DataFetcher
        ds = DataFetcher().region(parts[1:]).to_xarray()
        return ds

