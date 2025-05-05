
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import copernicusmarine
import datetime
import fsspec
from loguru import logger
import os
import pandas as pd
from pathlib import Path
import random
from shapely.geometry import box
import string
import xarray as xr

from dctools.data.connection.config import BaseConnectionConfig
from dctools.data.datasets.dc_catalog import CatalogEntry
from dctools.dcio.saver import DataSaver
from dctools.dcio.loader import FileLoader
from dctools.utilities.file_utils import read_file_tolist #, check_valid_files
from dctools.processing.cmems_data import extract_dates_from_filename
from dctools.utilities.xarray_utils import (
    get_grid_coord_names,
    extract_spatial_bounds,
    extract_variables,
)


class BaseConnectionManager(ABC):
    def __init__(self, connect_config: BaseConnectionConfig):
        self.params = connect_config.to_dict()
        self._list_files = self.list_files()

    def open(self, path: str, mode: str = "rb") -> xr.Dataset:
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
        dataset = self.open_local(path)
        if dataset:
            return dataset
        # Tenter d'ouvrir le fichier en ligne via S3
        dataset = self.open_remote(path, mode)
        if dataset:
            return dataset

        # Télécharger le fichier en local, puis l'ouvrir
        try:
            local_path = self._get_local_path(path)
            if not os. path. isfile(local_path):
                # logger.info(f"Downloading file to local path: {local_path}")
                self.download_file(path, local_path)
            return self.open_local(local_path)
        except Exception as exc:
            logger.error(f"Failed to open file: {path}. Error: {repr(exc)}")
            raise

    def open_local(self, local_path: str) -> Optional[xr.Dataset]:
        """
        Open a file locally if it exists.

        Args:
            local_path (str): Path to the local file.

        Returns:
            Optional[xr.Dataset]: Opened dataset, or None if the file does not exist.
        """
        if Path(local_path).exists():
            logger.info(f"Opening local file: {local_path}")
            return FileLoader.load_dataset(local_path)
            # return open_dataset_auto(local_path, self)
        return None

    def open_remote(self, path: str, mode: str = "rb") -> Optional[xr.Dataset]:
        """
        Open a file remotely if the source supports it.

        Args:
            path (str): Remote path of the file.
            mode (str): Mode to open the file (default is "rb").

        Returns:
            Optional[xr.Dataset]: Opened dataset, or None if remote opening is not supported.
        """
        try:
            logger.info(f"Attempting to open remote file: {path}")
            return FileLoader.open_dataset_auto(path, self)
        except Exception as exc:
            logger.warning(f"Failed to open remote file: {path}. Error: {repr(exc)}")
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

    def _get_local_path(self, remote_path: str) -> str:
        """
        Generate the local path for a given remote path.

        Args:
            remote_path (str): Remote path of the file.

        Returns:
            str: Local path of the file.
        """
        filename = Path(remote_path).name
        return os.path.join(self.params.local_root, filename)

    def get_global_metadata(self) -> Dict[str, Any]:
        """
        Extract global metadata (common to all files) from a single file.

        Returns:
            Dict[str, Any]: Global metadata including spatial bounds and variable names.
        """
        files = self._list_files
        if not files:
            raise FileNotFoundError("Empty file list! No files to extract metadata from.")

        first_file = files[0]
        logger.info(f"Extracting global metadata from {first_file}")

        # Charger le fichier avec xarray
        with self.open(first_file, "rb") as ds:
            # Extraire les métadonnées globales

            dict_dims = get_grid_coord_names(ds)
            longitude_data = ds[dict_dims['lon']].values
            latitude_data = ds[dict_dims['lat']].values

            # Inférer la résolution spatiale et temporelle
            spatial_resolution = self._infer_spatial_resolution(ds)
            temporal_resolution = self._infer_temporal_resolution(ds)

            # Extraire les dimensions
            dimensions = get_grid_coord_names(ds)

            # Associer les variables à leurs dimensions
            variables = {var: list(ds[var].dims) for var in ds.data_vars}

            global_metadata = {
                "dimensions": dimensions,
                "variables": variables,
                "spatial_resolution": spatial_resolution,
                "temporal_resolution": temporal_resolution,
                "lon_min": longitude_data.min(),
                "lon_max": longitude_data.max(),
                "lat_min": latitude_data.min(),
                "lat_max": latitude_data.max(),
            }
            
        return global_metadata

    def extract_metadata(self, path: str, global_metadata: Dict[str, Any]) -> CatalogEntry:
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
                #ds = xr.open_dataset(f)
                date_start = pd.to_datetime(ds.time.min().values) if "time" in ds.coords else None
                date_end = pd.to_datetime(ds.time.max().values) if "time" in ds.coords else None
                #date_start = ds.time.min().values if "time" in ds.coords else None
                #date_end = ds.time.max().values if "time" in ds.coords else None

                # Créer une instance de CatalogEntry
                return CatalogEntry(
                    path=path,
                    date_start=date_start,
                    date_end=date_end,
                    lat_min=global_metadata.get("lat_min"),
                    lat_max=global_metadata.get("lat_max"),
                    lon_min=global_metadata.get("lon_min"),
                    lon_max=global_metadata.get("lon_max"),
                    variables=global_metadata.get("variables"),
                    dimensions=global_metadata.get("dimensions"),
                    spatial_resolution=global_metadata.get("spatial_resolution"),
                    temporal_resolution=global_metadata.get("temporal_resolution"),
                    geometry=box(global_metadata["lon_min"], global_metadata["lat_min"], global_metadata["lon_max"], global_metadata["lat_max"]),
                )
        except Exception as exc:
            logger.error(f"Failed to extract metadata for file {path}: {repr(exc)}")
            raise

    def _infer_spatial_resolution(self, ds: xr.Dataset) -> Optional[Tuple[float, float]]:
        """
        Infère la résolution spatiale à partir des coordonnées du dataset.

        Args:
            ds (xr.Dataset): Dataset xarray.

        Returns:
            Optional[Tuple[float, float]]: Résolution spatiale (lat, lon) en degrés.
        """
        if "lat" in ds.coords and len(ds.lat) > 1 and "lon" in ds.coords and len(ds.lon) > 1:
            lat_resolution = float(ds.lat.diff(dim="lat").mean().values)
            lon_resolution = float(ds.lon.diff(dim="lon").mean().values)
            return lat_resolution, lon_resolution
        return None

    def _infer_temporal_resolution(self, ds: xr.Dataset) -> Optional[str]:
        """
        Infère la résolution temporelle à partir des coordonnées du dataset.

        Args:
            ds (xr.Dataset): Dataset xarray.

        Returns:
            Optional[str]: Résolution temporelle en format ISO 8601.
        """
        if "time" in ds.coords and len(ds.time) > 1:
            try:
                # Extraire les deux premières valeurs de temps comme scalaires
                time_0 = pd.to_datetime(ds.time[0].values)
                time_1 = pd.to_datetime(ds.time[1].values)

                # Calculer la différence temporelle
                delta = pd.to_timedelta(time_1 - time_0)

                # Retourner la résolution au format ISO 8601
                return delta.isoformat()
            except Exception as exc:
                logger.error(f"Erreur lors de l'inférence de la résolution temporelle : {repr(exc)}")
                return None
        return None

    def list_files_with_metadata(self) -> List[CatalogEntry]:
        """
        List all files with their metadata by combining global metadata and file-specific information.

        Returns:
            List[CatalogEntry]: List of metadata entries for each file.
        """
        # Récupérer les métadonnées globales
        global_metadata = self.get_global_metadata()

        # Initialiser une liste pour stocker les métadonnées
        metadata_list = []

        limit = self.params.max_samples if self.params.max_samples else len(self._list_files)
        # Parcourir tous les fichiers et extraire leurs métadonnées
        for path in self._list_files:
            try:
                metadata_entry = self.extract_metadata(path, global_metadata)
                metadata_list.append(metadata_entry)
                if len(metadata_list) >= limit:
                    logger.info(f"Reached the limit of {limit} metadata entries.")
                    break
            except Exception as exc:
                logger.warning(f"Failed to extract metadata for file {path}: {repr(exc)}")

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
    def list_files(self, pattern="**/*.nc") -> List[str]:
        """
        List files in the local filesystem matching the given pattern.

        Args:
            pattern (str): Glob pattern to filter files.

        Returns:
            List[str]: List of file paths on local disk".
        """
        root = self.params.local_root
        files = [p for p in self.params.fs.glob(f"{root}/{pattern}")]
        return [f"{file}" for file in files]

    @classmethod
    def supports(cls, path: str) -> bool:
        return path.startswith("/") or path.startswith("file://")


class CMEMSManager(BaseConnectionManager):
    """Class to manage Copernicus Marine downloads."""

    def __init__(self, connect_config: BaseConnectionConfig):
        """
        Initialise le gestionnaire CMEMS et effectue la connexion.

        Args:
            connect_config (BaseConnectionConfig): Configuration de connexion.
        """
        super().__init__(connect_config)  # Appeler l'initialisation de la classe parente

        # logger.info(f"CMEMS file : {self.params.cmems_credentials}")
        self.cmems_login()  # Appeler la fonction cmems_login pour initialiser la connexion

    def get_credentials(self):
        """Get CMEMS credentials.

        Return:
            (dict): CMEMS credentials
        """
        with open(self.params.cmems_credentials, "r") as f:
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
            if not (Path(self.params.cmems_credentials).is_file()):
                logger.warning(f"Credentials file not found at {self.params.cmems_credentials}.")
                copernicusmarine.login()
        except Exception as exc:
            logger.error(f"login to CMEMS failed: {repr(exc)}")
        # return self.params.cmems_credentials

    def cmems_logout(self) -> None:
        """Logout from Copernicus Marine."""
        logger.info("Logging out from Copernicus Marine.")
        try:
            copernicusmarine.logout()
        except Exception as exc:
            logger.error(f"logout from CMEMS failed: {repr(exc)}")
        return None


    '''def list_files(self, pattern="*.nc") -> List[str]:
        """List files in the Copernicus Marine directory."""
        logger.info("Listing files in Copernicus Marine directory.")
        tmp_filepath = os.path.join(
            self.params.local_root, "files.txt"
        )
        try:
            copernicusmarine.get(
                dataset_id=self.params.dataset_id, create_file_list=tmp_filepath
            )
            self._files = read_file_tolist(tmp_filepath, max_lines=2)
            logger.info(f"List of files: {self._files}")
            return self._files
        except Exception as exc:
            logger.error(f"Failed to list files from CMEMS: {repr(exc)}")
            return []'''
    
    def list_files(self, pattern="*.nc") -> List[str]:
        """List files in the Copernicus Marine directory."""
        logger.info("Listing files in Copernicus Marine directory.")
        tmp_filepath = os.path.join(self.params.local_root, "files.txt")
        try:
            copernicusmarine.get(
                dataset_id=self.params.dataset_id, create_file_list=tmp_filepath
            )
            self._files = read_file_tolist(tmp_filepath)
            #logger.info(f"List of files: {self._files}")
            return self._files
        except Exception as exc:
            logger.error(f"Failed to list files from CMEMS: {repr(exc)}")
            return []

    def get_product_metadata(self) -> Dict[str, Any]:
        """
        Fetch product-level metadata from the CMEMS API.

        Returns:
            Dict[str, Any]: A dictionary containing metadata for the product.
        """
        product_metadata = copernicusmarine.describe(self.params["dataset_id"])

        return {
            "variables": product_metadata.get("variables", []),
            "date_start": product_metadata.get("temporal_coverage_start"),
            "date_end": product_metadata.get("temporal_coverage_end"),
            "lon_min": product_metadata.get("geospatial_lon_min"),
            "lon_max": product_metadata.get("geospatial_lon_max"),
            "lat_min": product_metadata.get("geospatial_lat_min"),
            "lat_max": product_metadata.get("geospatial_lat_max"),
        }


    def open_remote(self, path: str, mode: str = "rb") -> Optional[xr.Dataset]:
        """
        Open a file remotely from CMEMS using S3 URLs.

        Args:
            path (str): Remote S3 path of the file.
            mode (str): Mode to open the file (default is "rb").

        Returns:
            Optional[xr.Dataset]: Opened dataset, or None if remote opening fails.
        """
        '''try:
            logger.info(f"Attempting to open remote S3 file: {path}")
            fs = fsspec.filesystem("s3", anon=True)  # Access S3 anonymously
            with fs.open(path, mode) as f:
                return xr.open_dataset(f, engine="netcdf4")
        except Exception as exc:
            logger.warning(f"Failed to open remote S3 file: {path}. Error: {repr(exc)}")'''
        return None

    def download_file(self, path: str, local_path: str):
        """
        Download a specific file from CMEMS.

        Args:
            path (str): Path to the file to download.
        """
        # Extraire la date à partir du nom du fichier
        filename = Path(path).name
        name_filter = filename
        try:
            # Télécharger le fichier via l'API CMEMS
            logger.info(f"Downloading file {filename} from CMEMS...")
            copernicusmarine.get(
                dataset_id=self.params.dataset_id,
                filter=name_filter,
                output_directory=self.params.local_root,
                no_directories=True,
                credentials_file=self.params.cmems_credentials,
            )
        except Exception as exc:
            logger.error(f"download from CMEMS failed: {repr(exc)}")
        return None

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
        return "cmems" in path.lower()


class FTPManager(BaseConnectionManager):
    @classmethod
    def supports(cls, path: str) -> bool:
        return path.startswith("ftp://")

    def list_files(self, pattern: str = "*.nc") -> List[str]:
        """
        Liste les fichiers disponibles sur le serveur FTP correspondant au motif donné.

        Args:
            pattern (str): Motif de recherche pour filtrer les fichiers (par défaut "*.nc").

        Returns:
            List[str]: Liste des chemins des fichiers correspondant au motif.
        """
        try:
            # Accéder au système de fichiers FTP via fsspec
            fs = self.params.fs
            root = self.params.local_root

            # Lister les fichiers correspondant au motif
            files = fs.glob(f"{root}/{pattern}")

            if not files:
                logger.warning(f"Aucun fichier trouvé sur le serveur FTP avec le motif : {pattern}")
            return files
        except Exception as exc:
            logger.error(f"Erreur lors de la liste des fichiers sur le serveur FTP : {repr(exc)}")
            return []


class S3Manager(BaseConnectionManager):
    @classmethod
    def supports(cls, path: str) -> bool:
        return path.startswith("s3://")

    def list_files(self, pattern: str = "*.nc") -> List[str]:
        """
        List files matching the given pattern.

        Args:
            pattern (str): Glob pattern to filter files.

        Returns:
            List[str]: List of file paths.
        """
        try:
            if hasattr(self.params, "bucket"):
                logger.info(f"Accessing bucket: {self.params.bucket}")

            # Construire le chemin distant
            remote_path = f"{self.params.endpoint_url}/{self.params.bucket}/{self.params.bucket_folder}/"
            logger.info(f"Listing files in: {remote_path}")

            # Utiliser fsspec pour accéder aux fichiers
            fs = self.params.fs
            files = fs.glob(f"{remote_path}{pattern}")

            if not files:
                logger.warning(f"No files found in bucket: {self.params.bucket}")
            return files
        except PermissionError as exc:
            logger.error(f"Permission error while accessing bucket: {repr(exc)}")
            logger.info("Attempting to list files using object-level access...")

            # Contourner le problème en listant les objets directement
            try:
                #remote_path = f"{self.params.bucket}/{self.params.bucket_folder}/"
                files = [
                    f"s3://{self.params.endpoint_url}/{self.params.bucket}/{obj['Key']}"
                    for obj in fs.ls(remote_path, detail=True)
                    if obj["Key"].endswith(pattern.split("*")[-1])
                ]
                return files
            except Exception as exc:
                logger.error(f"Failed to list files using object-level access: {repr(exc)}")
                raise

    def open_remote(self, path: str, mode: str = "rb") -> Optional[xr.Dataset]:
        """
        Open a file remotely from an S3 bucket.

        Args:
            path (str): Remote path of the file.
            mode (str): Mode to open the file (default is "rb").

        Returns:
            Optional[xr.Dataset]: Opened dataset, or None if remote opening is not supported.
        """
        try:
            logger.info(f"Attempting to open S3 file: {path}")
            return xr.open_dataset(self.params.fs.open(path, mode))
        except Exception as exc:
            logger.warning(f"Failed to open S3 file: {path}. Error: {repr(exc)}")
            return None



class S3WasabiManager(S3Manager):
    def open_remote(self, path: str, mode: str = "rb") -> Optional[xr.Dataset]:
        """
        Open a file remotely from an S3 bucket.

        Args:
            path (str): Remote path of the file.
            mode (str): Mode to open the file (default is "rb").

        Returns:
            Optional[xr.Dataset]: Opened dataset, or None if remote opening is not supported.
        """
        try:
            logger.info(f"Attempting to open S3 file: {path}")
            # return xr.open_dataset(self.params.fs.open(path, mode))
        
            s3_mapper = fsspec.get_mapper(
                # "s3://ppr-ocean-climat/DC3/IABP/LEVEL1_2023.zarr",
                path,
                client_kwargs = {
                    "aws_access_key_id": self.params.key,
                    "aws_secret_access_key": self.params.secret_key,
                    "endpoint_url": self.params.endpoint_url,
                    }
                )
            return xr.open_dataset(
                s3_mapper,
                engine="zarr"
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
        List files matching the given pattern.

        Args:
            pattern (str): Glob pattern to filter files.

        Returns:
            List[str]: List of file paths.
        """
        start_date = "2024-01-03"
        date = datetime.datetime.strptime(start_date, "%Y-%m-%d")
        list_files = []
        while True:
            if date.year < 2025:
                date_str = date.strftime("%Y-%m-%d")
                list_files.append(
                    f"https://minio.dive.edito.eu/project-glonet/public/glonet_reforecast_2024/{date_str}.zarr"
                )
                date = date + datetime.timedelta(days=7)
            else:
                break
        return list_files


    def open(self, path: str, mode: str = "rb") -> xr.Dataset:
        return self.open_remote(path)

    def open_remote(self, path: str) -> Optional[xr.Dataset]:
        """
        Open a file remotely from an S3 bucket.

        Args:
            path (str): Remote path of the file.
            mode (str): Mode to open the file (default is "rb").

        Returns:
            Optional[xr.Dataset]: Opened dataset, or None if remote opening is not supported.
        """
        try:
            glonet_ds = xr.open_zarr(
                path,
            )
            # logger.info(f"Opened Glonet file: {path}")
            """filename = os.path.basename(path)
            filepath = os.path.join("/home/k24aitmo/IMT/software/tests/Glonet", filename)
            logger.info(f"Saving Glonet file to: {filepath}")
            DataSaver.save_dataset(
                glonet_ds,
                filepath,
                file_format="zarr",
                #file_format="netcdf",
            )"""
            return glonet_ds

        except Exception as exc:
            logger.warning(f"Failed to open Glonet file: {path}. Error: {repr(exc)}")
            return None