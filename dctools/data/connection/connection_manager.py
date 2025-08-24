
from abc import ABC, abstractmethod
import traceback
from typing import (
    Any, Callable, Dict, List, Optional, Type, Union
)

from argopy import ArgoIndex, DataFetcher, IndexFetcher, set_options
from argparse import Namespace
import copernicusmarine
import dask
from dask.distributed import Client, LocalCluster
from dataclasses import dataclass, asdict
import datetime
import fsspec
import geopandas as gpd
from loguru import logger
import os
import numpy as np
import pandas as pd
from pathlib import Path
import psutil

# import random
# from shapely.geometry import box, Point
from shapely.geometry import box
# import string
import xarray as xr

from dctools.data.connection.config import BaseConnectionConfig
from dctools.data.datasets.dc_catalog import CatalogEntry
from dctools.data.coordinates import (
    get_dataset_geometry,
    get_dataset_geometry_light,
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
# from dctools.utilities.init_dask import setup_dask



def get_time_bound_values(ds: xr.Dataset) -> tuple:
    """Obtient les bornes min/max de manière sécurisée."""
    try:
        time_vals = ds["time"]
        # Vérifier le type de données
        if np.issubdtype(time_vals.dtype, np.datetime64):
            # Pour les données temporelles
            min_val = pd.to_datetime(time_vals.values.min())
            max_val = pd.to_datetime(time_vals.values.max())
            return (min_val, max_val)
        
        elif np.issubdtype(time_vals.dtype, np.floating) or np.issubdtype(time_vals.dtype, np.integer):
            # Pour les données numériques
            min_val = float(time_vals.min().values)
            max_val = float(time_vals.max().values)
            
            # Vérifier que les valeurs sont valides
            if np.isnan(min_val) or np.isnan(max_val):
                return (None, None)
                
            return (min_val, max_val)
        
        else:
            logger.warning(f"Unsupported data type: {time_vals.dtype}")
            return (None, None)
            
    except Exception as exc:
        logger.warning(f"Failed to get bounds: {exc}")
        return (None, None)


class BaseConnectionManager(ABC):
    def __init__(
        self, connect_config: BaseConnectionConfig | Namespace,
        call_list_files: bool = True,
        batch_size: Optional[int] = 1  # Taille de batch adaptée aux métadonnées
    ):
        self.connect_config = connect_config
        self.batch_size = batch_size
        if isinstance(connect_config, BaseConnectionConfig):
            self.params = connect_config.to_dict()
        elif isinstance(connect_config, Namespace):
            self.params = connect_config
        else:
            raise TypeError("Unknown type of connection config.")
        
        self.start_time = self.params.time_interval[0]
        self.end_time = self.params.time_interval[1]

        init_type = self.params.init_type
        if init_type != "from_json" and call_list_files:
            self._list_files = self.list_files()
        if not self.params.file_pattern:
            self.params.file_pattern = "**/*.nc"
        if not self.params.groups:
            self.params.groups = None
        self.file_cache = self.params.file_cache
        # self.dask_cluster = self.params.dask_cluster

    def setup_dask(self):
        """Configuration Dask robuste pour éviter les données perdues."""

        import tempfile
        
        num_workers = 1  # max(1, psutil.cpu_count() // 3)
        # Mémoire
        total_memory = psutil.virtual_memory().total / 1e9
        memory_limit = f"{int(total_memory * 0.4)}GB"  # Plus conservateur (40% au lieu de 70%)
        
        # Répertoire temporaire
        temp_dir = tempfile.mkdtemp(prefix="dask_", dir="/tmp")
        
        # Configuration environnement
        os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
        os.environ['NETCDF4_DEACTIVATE_MPI'] = '1'
        
        # Configuration Dask
        dask.config.set({
            # Scheduler conservateur
            'scheduler': 'threads',
            'temporary-directory': temp_dir,
            
            # PARAMÈTRES CRITIQUES pour éviter les données perdues
            'distributed.scheduler.keep-small-data': True,           # ← CRITIQUE
            'distributed.scheduler.bandwidth': '10e6',            # ← LIMITE
            'distributed.scheduler.work-stealing': False,           # ← DÉSACTIVE
            
            # Gestion mémoire TRÈS conservatrice
            'distributed.worker.memory.target': 0.4,                # ← Plus bas
            'distributed.worker.memory.pause': 0.5,                 # ← Plus bas
            'distributed.worker.memory.spill': 0.6,                 # ← Plus bas
            'distributed.worker.memory.terminate': 0.7,             # ← Plus bas
            
            # Timeouts TRÈS longs
            'distributed.comm.timeouts.tcp': '600s',                # ← 10 minutes
            'distributed.comm.timeouts.connect': '300s',            # ← 5 minutes
            'distributed.comm.retry.count': 10,                     # ← Plus de retries
            
            # Désactiver les optimisations problématiques
            'distributed.scheduler.active-memory-manager.start': False,
            'distributed.worker.daemon': False,
            'distributed.worker.profile.enabled': False,
            
            # Chunking plus conservateur
            'array.chunk-size': '32MB',                             # ← Plus petit
        })
        
        cluster = LocalCluster(
            n_workers=num_workers,
            threads_per_worker=1,
            memory_limit=memory_limit,
            processes=False,          # Threads plus stable
            silence_logs=True,
            dashboard_address=None,   # Désactive le dashboard
            death_timeout=600,        # 10 minutes avant de considérer un worker mort
        )
        
        logger.info(f"Dask cluster: 1 worker, {memory_limit} memory, temp_dir: {temp_dir}")
        return  Client(cluster)


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
        if self._list_files is None:
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
                # "is_observation": coord_sys.is_observation_dataset(),
                #"dimensions_rename_dict": dimensions_rename_dict,
            }
        return global_metadata


    def extract_metadata(
        self, path: str,
    ):  #, global_metadata: Dict[str, Any]) -> CatalogEntry:
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

    @staticmethod
    def extract_metadata_worker(
        path: str,
        global_metadata: dict,
        connection_params: dict,
        class_name: Any,
    ):  #, global_metadata: Dict[str, Any]) -> CatalogEntry:
        """
        Extract metadata for a specific file, combining global metadata with file-specific information.

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
            #print(f"Process file: {path}")
            import logging
            # Supprimer les logs "INFO" des workers Dask
            logging.getLogger("distributed.worker").setLevel(logging.WARNING)
            # Récupérer la classe depuis le registre
            if class_name not in CLASS_REGISTRY:
                raise ValueError(f"Unknown class name: {class_name}")
            
            manager_class = CLASS_REGISTRY[class_name]
            manager = manager_class(
                connection_params, call_list_files=False
            )

            ds = manager.open(path, "rb")
            time_bounds = get_time_bound_values(ds)
            date_start = time_bounds[0]
            date_end = time_bounds[1]
            '''date_start = pd.to_datetime(
                time_vals.min().values
            ) if "time" in ds.coords else None
            date_end = pd.to_datetime(
                time_vals.max().values
            ) if "time" in ds.coords else None'''

            ds_region = get_dataset_geometry_light(
                ds, global_metadata.get('coord_system')
            )

            # Créer une instance de CatalogEntry
            return CatalogEntry(
                path=path,
                date_start=date_start,
                date_end=date_end,
                variables=global_metadata.get("variables"),
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

        metadata_list = []
        limit = self.params.max_samples if self.params.max_samples else len(self._list_files)
        # self._list_files = [s for s in self._list_files if s.count("_2024") >= 2]    # TODO : REMOVE

        file_list = self._list_files[:limit]

        coord_system = self._global_metadata.get('coord_system')
        variables = self._global_metadata.get("variables")

        metadata_list = [self.extract_metadata(
            path
        ) for path in file_list]

        if not metadata_list:
            logger.error("No valid metadata entries were generated.")
            raise ValueError("No valid metadata entries were generated.")

        #if must_close_dask:
        #    client.close()

        return metadata_list'''

    def list_files_with_metadata(self) -> List[CatalogEntry]:
        """
        Version avec client Dask intégré et configuration optimisée.
        """
        
        # Récupérer les métadonnées globales
        global_metadata = self.extract_global_metadata()
        self._global_metadata = global_metadata

        limit = self.params.max_samples if self.params.max_samples else len(self._list_files)
        file_list = self._list_files[:limit]

        logger.info(f"Processing {len(file_list)} files with integrated Dask client")

        metadata_list = []
        dask_client = None
        
        try:
            # Configuration et création du client Dask
            dask_client = self.setup_dask()
            logger.info(f"Dask client created: {dask_client}")

            # Scatter une seule fois les objets volumineux
            scattered_config = dask_client.scatter(self.connect_config, broadcast=True)
            scattered_metadata = dask_client.scatter(self._global_metadata, broadcast=True)
            
            # Parallélisation par batch pour contrôler la charge mémoire
            
            for i in range(0, len(file_list), self.batch_size):
                batch_files = file_list[i:i + self.batch_size]
                logger.info(
                    f"Processing metadata batch {i//self.batch_size + 1}/{(len(file_list)-1)//self.batch_size + 1}"
                )
                
                # Créer les tâches pour ce batch
                delayed_tasks = [
                    dask.delayed(self.extract_metadata_worker)(
                        path, scattered_metadata,
                        scattered_config,
                        self.__class__.__name__,
                    ) for path in batch_files
                ]
                
                try:
                    # Exécuter le batch avec le client
                    batch_futures = dask_client.compute(delayed_tasks, sync=False)
                    batch_results = dask_client.gather(batch_futures)
                    
                    # Filtrer et ajouter les résultats valides
                    valid_results = [meta for meta in batch_results if meta is not None]
                    metadata_list.extend(valid_results)
                    
                    logger.info(f"Batch completed: {len(valid_results)}/{len(batch_files)} files processed")
                    
                    # Nettoyage explicite entre les batches
                    #del batch_results, batch_futures
                    #dask_client.run(self._cleanup_worker_memory)
                    
                except Exception as exc:
                    logger.error(f"Batch {i//self.batch_size + 1} failed: {exc}")
                    # Fallback séquentiel pour ce batch
                    for path in batch_files:
                        try:
                            meta = self.extract_metadata(path)
                            if meta is not None:
                                metadata_list.append(meta)
                        except Exception as file_exc:
                            logger.warning(f"Failed to process {path}: {file_exc}")

        except Exception as exc:
            logger.error(f"Dask metadata extraction failed: {exc}")
            raise
            # Fallback complet vers traitement séquentiel
            #metadata_list = self._sequential_fallback(file_list)
            
        finally:
            # Nettoyage du client Dask
            if dask_client is not None:
                try:
                    logger.info("Closing Dask client and cluster")
                    dask_client.close()
                    if hasattr(dask_client, 'cluster'):
                        dask_client.cluster.close()
                except Exception as cleanup_exc:
                    logger.warning(f"Error during Dask cleanup: {cleanup_exc}")

        if not metadata_list:
            logger.error("No valid metadata entries were generated.")
            raise ValueError("No valid metadata entries were generated.")

        return metadata_list

    @staticmethod
    def _cleanup_worker_memory():
        """Fonction de nettoyage à exécuter sur chaque worker."""
        import gc
        gc.collect()

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
    ):
        """
        Initialise le gestionnaire CMEMS et effectue la connexion.

        Args:
            connect_config (BaseConnectionConfig): Configuration de connexion.
        """
        super().__init__(connect_config, call_list_files=False)  # Appeler l'initialisation de la classe parente

        self.batch_size = 3
        logger.debug(f"CMEMS file : {self.params.cmems_credentials_path}")
        self.cmems_login()

        if call_list_files:
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
                
            from datetime import time
            start_datetime = datetime.datetime.combine(dt.date(), time.min)  # 00:00:00
            end_datetime = datetime.datetime.combine(dt.date(), time.max)    # 23:59:59.999999
            
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
                # Fermer immédiatement pour libérer les ressources
                test_ds.close()
                return True
            else:
                return False
                
        except Exception as e:
            # Toute exception indique que le fichier n'existe pas ou n'est pas accessible
            logger.debug(f"File does not exist for date {dt}: {e}")
            return False
    
    def list_files(self) -> List[str]:
        """List files in the Copernicus Marine directory."""
        logger.info("Listing files in Copernicus Marine directory.")
        tmp_filepath = os.path.join(self.params.local_root, "files.txt")
        try:
            '''copernicusmarine.get(
                dataset_id=self.params.dataset_id, create_file_list=tmp_filepath
            )
            self._files = read_file_tolist(tmp_filepath)'''
            start_dt = pd.to_datetime(self.start_time)
            start_year = start_dt.year
            start_month = start_dt.month
            start_day = start_dt.day
            end_dt = pd.to_datetime(self.end_time)
            end_year = end_dt.year
            end_month = end_dt.month
            end_day = end_dt.day

            # start_date = datetime.datetime(2024, 1, 1)
            # end_date = datetime.datetime(2025, 1, 3)
            start_date = datetime.datetime(start_year, start_month, start_day)
            end_date = datetime.datetime(end_year, end_month, end_day)
            list_dates = self.list_all_days(
                start_date,
                end_date,
            )
            list_dates = list_dates [:self.params.max_samples]
            valid_dates = []
            for date in list_dates:
                # print(date)
                #ds = self.open_remote(date, mode="rb")
                #if ds is not None:
                if self.remote_file_exists(date):
                    valid_dates.append(date)
            return valid_dates
        except Exception as exc:
            logger.error(f"Failed to list files from CMEMS: {repr(exc)}")
            return []

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
                # variables=["uo","vo"]
                credentials_file=self.params.cmems_credentials_path,
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

    def __init__(self, connect_config: BaseConnectionConfig | Namespace,
        lon_range: Optional[tuple[float, float]] = (-180, 180),
        lat_range: Optional[tuple[float, float]] = (-90, 90),
        lon_step: Optional[float] = 1.0,
        lat_step: Optional[float] = 1.0,
        depth_values: Optional[List[float]] = GLONET_DEPTH_VALS,
        time_step_days: Optional[int] = 1,
        custom_cache: Optional[str] = None,
        batch_size: Optional[int] = 10,
        call_list_files: Optional[bool] = True,
        argo_index: Optional[Any] = None,
    ):
        self.batch_size = batch_size  # Taille de batch réduite pour ARGO
        if custom_cache is not None:
            set_options(cachedir=custom_cache)
        self.argo_loader = DataFetcher()
        self.lon_range = lon_range
        self.lat_range = lat_range
        self.lon_step = lon_step
        self.lat_step = lat_step
        self.depth_values = depth_values
        self.time_step_days = time_step_days

        # Charger l'index
        self.argo_index = argo_index
        if argo_index is None:
            self._load_index_once()
        super().__init__(connect_config, call_list_files)

    '''def _load_index_once(self):
        """Charge l'index ARGO une seule fois."""
        try:
            idx = ArgoIndex()
            idx = idx.load()  # Charger TOUT l'index
            self.argo_index = idx.to_dataframe()
            logger.info(f"Loaded ARGO index with {len(self.argo_index)} profiles")
        except Exception as e:
            logger.error(f"Failed to load ARGO index: {e}")
            self.argo_index = pd.DataFrame()'''

    def find_unpicklable_objects(self, obj, path=""):
        """
        Trouve récursivement les objets non sérialisables.
        """

        import dill
        import pickle
        import traceback
        try:
            # Test avec pickle standard
            pickle.dumps(obj)
            return []
        except Exception as e:
            print(f"❌ Non sérialisable à {path}: {type(obj)} - {str(e)[:100]}")
            
            # Si c'est un dictionnaire, tester chaque clé/valeur
            if isinstance(obj, dict):
                problematic = []
                for key, value in obj.items():
                    try:
                        pickle.dumps(value)
                    except:
                        problematic.extend(self.find_unpicklable_objects(value, f"{path}.{key}"))
                return problematic
                
            # Si c'est une liste/tuple
            elif isinstance(obj, (list, tuple)):
                problematic = []
                for i, item in enumerate(obj):
                    try:
                        pickle.dumps(item)
                    except:
                        problematic.extend(self.find_unpicklable_objects(item, f"{path}[{i}]"))
                return problematic
                
            # Si c'est un objet avec __dict__
            elif hasattr(obj, "__dict__"):
                problematic = []
                for attr_name, attr_value in obj.__dict__.items():
                    try:
                        pickle.dumps(attr_value)
                    except:
                        problematic.extend(self.find_unpicklable_objects(attr_value, f"{path}.{attr_name}"))
                return problematic
                
            # Objet problématique trouvé
            return [(path, type(obj), str(e))]

    # Usage dans votre code
    def debug_serialization(self, your_object):
        """Debug la sérialisation d'un objet."""

        import dill
        import pickle
        print(f"🔍 Testing serialization of {type(your_object)}")
        
        # Test avec different serializers
        for serializer_name, serializer in [("pickle", pickle), ("dill", dill)]:
            try:
                serializer.dumps(your_object)
                print(f"✅ {serializer_name}: OK")
            except Exception as e:
                print(f"❌ {serializer_name}: {e}")
                
                # Analyse détaillée pour pickle
                if serializer_name == "pickle":
                    problematic = self.find_unpicklable_objects(your_object)
                    for path, obj_type, error in problematic:
                        print(f"  📍 {path}: {obj_type} - {error}")

    '''def debug_argo_serialization(self, start_date: pd.Timestamp):
        """Debug spécifiquement pour l'extraction de métadonnées ARGO."""
        
        print(f"🔍 Debugging ARGO serialization for {start_date}")
        
        # Test du dataset
        try:
            ds = self.open(start_date)
            print(f"✅ Dataset ouvert: {type(ds)}")
            
            # Test serialization du dataset
            self.debug_serialization(ds)
            
            # Test des métadonnées globales
            print("\n🔍 Testing global metadata...")
            self.debug_serialization(self._global_metadata)
            
            # Test coord_sys
            coord_sys = self._global_metadata.get('coord_system')
            print(f"\n🔍 Testing coord_sys: {type(coord_sys)}")
            self.debug_serialization(coord_sys)
            
            # Test de la création d'entrée
            print(f"\n🔍 Testing catalog entry creation...")
            try:
                metadata = self._create_lightweight_argo_catalog_entry(ds, start_date)
                print(f"✅ Catalog entry created: {type(metadata)}")
                self.debug_serialization(metadata)
            except Exception as e:
                print(f"❌ Catalog entry creation failed: {e}")
                traceback.print_exc()
                
        except Exception as e:
            print(f"❌ Dataset opening failed: {e}")
            traceback.print_exc()'''

    def _load_index_once(self):
        """Load ARGO index once."""
        try:
            idx = ArgoIndex()
            idx = idx.load()
            self.argo_index = idx.to_dataframe()

            # Normalize all timestamps to the minute
            self.argo_index['date'] = self.argo_index['date'].dt.floor('min')
            
            logger.info(f"Loaded ARGO index with {len(self.argo_index)} profiles")
        except Exception as e:
            logger.error(f"Failed to load ARGO index: {e}")
            self.argo_index = pd.DataFrame()

    @classmethod
    def supports(cls, path: str) -> bool:
        return True

    def filter_by_geometry(catalog_gdf: gpd.GeoDataFrame, polygon: gpd.GeoSeries) -> gpd.GeoDataFrame:
        return catalog_gdf[catalog_gdf.geometry.centroid.within(polygon.unary_union)]

    def spatial_filter(catalog_gdf: gpd.GeoDataFrame, region_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        return gpd.sjoin(catalog_gdf, region_gdf, how="inner", predicate="intersects")


    def adjust_full_day_if_needed(
        self,
        date_start: pd.Timestamp, date_end: pd.Timestamp
    ) -> tuple[pd.Timestamp, pd.Timestamp]:
        """
        Si date_start == date_end et date_start est à minuit, ajuste date_end pour couvrir toute la journée.
        """
        if pd.isnull(date_start) or pd.isnull(date_end):
            return date_start, date_end
        if date_start == date_end and date_start.hour == 0 and date_start.minute == 0 and date_start.second == 0:
            # Ajuste date_end à la fin de la journée
            date_end = date_start + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        return date_start, date_end

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

    '''def list_files_with_metadata(
            self
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
            ds = self.open(start_date)
            if ds is None:
                continue

            # get coordinate system
            if first_elem:
                coord_sys = CoordinateSystem.get_coordinate_system(ds)

                # Inférer la résolution spatiale et temporelle
                #dict_resolution = self.estimate_resolution(
                #    ds, coord_sys
                #)
                dict_resolution  = {}
                dict_resolution["time"] = "60s"
                dict_resolution["latitude"] = None
                dict_resolution["longitude"] = None

                # Associer les variables à leurs dimensions
                variables = {}
                for var_name, var in ds.variables.items():

                    if var_name in self.params.keep_variables:
                        variables[var_name] = {
                            "dims": list(var.dims),
                            "std_name": var.attrs.get("standard_name", ""),
                        }

                # variables = {v: list(ds[v].dims) for v in ds.data_vars if v in self.params.keep_variables}
                variables_dict = CoordinateSystem.detect_oceanographic_variables(variables)
                variables_rename_dict = {v: k for k, v in variables_dict.items()}
                # dimensions = dict(ds.dims)

                # coord_sys = CoordinateSystem.get_coordinate_system(ds)
                # dimensions = coord_sys.coordinates
                #resolution={"lon": self.lon_step, "lat": self.lat_step, "time": self.time_step_days}

                global_metadata = {
                    "variables": variables,
                    "variables_dict": variables_dict,
                    "variables_rename_dict": variables_rename_dict,
                    "resolution": dict_resolution,
                    "coord_system": coord_sys,
                    "keep_variables": self.params.keep_variables,
                }
                self.global_metadata = global_metadata
                first_elem = False
            geometry = get_dataset_geometry(ds, coord_sys)

            # date_start, date_end = self.adjust_full_day_if_needed(date_start, date_end)
            entry = CatalogEntry(
                path=start_date,  #f"argo://{df['file']}",
                date_start=start_date,
                date_end= start_date + pd.Timedelta(minutes=1),
                variables=variables,
                geometry=geometry,
            )
            metadata_list.append(entry)

        return metadata_list'''



    @staticmethod
    def _extract_argo_metadata(
        start_date: pd.Timestamp,
        connection_params: dict,
        global_metadata: dict,
        argo_index: Any,
    ) -> Optional[CatalogEntry]:
        """Version statique pour worker Dask - ARGO."""
        try:
            # Imports locaux dans le worker
            #from dctools.data.connection.config import ARGOConnectionConfig
            #from dctools.data.connection.connection_manager import ArgoManager
            #import pandas as pd
            #import geopandas as gpd
            #from shapely.geometry import Point
            #from dctools.data.datasets.dc_catalog import CatalogEntry
            
            # Nettoyage des paramètres - ne garder que ce qui est nécessaire
            '''clean_params = {
                key: value for key, value in connection_params.items()
                if key in ['init_type', 'local_root', 'max_samples', 'file_pattern', 'keep_variables']
                and not key.startswith('_')
            }'''

            import logging
            # Supprimer les logs "INFO" des workers Dask
            logging.getLogger("distributed.worker").setLevel(logging.WARNING)
            
            # Recréer le manager
            # config = ARGOConnectionConfig(connection_params)  # **clean_params)
            manager = ArgoManager(
                connection_params, call_list_files=False, argo_index=argo_index
            )
            
            # Traitement
            ds = manager.open(start_date)
            if ds is None:
                return None
            
            # Extraction des coordonnées ARGO
            coord_sys = global_metadata.get('coord_system', {})
            '''if isinstance(coord_sys, dict):
                coords = coord_sys.get('coordinates', {})
            else:
                coords = coord_sys.coordinates if hasattr(coord_sys, 'coordinates') else {}'''
            
            # Variables
            variables = global_metadata.get("variables", {})

            geometry = get_dataset_geometry(ds, coord_sys)
            
            # Créer l'entrée
            metadata = CatalogEntry(
                path=str(start_date),
                date_start=start_date,
                date_end=start_date + pd.Timedelta(minutes=1),
                variables=variables,
                geometry=geometry,
            )
            
            return metadata
            
        except Exception as exc:
            # Logger pas disponible dans le worker
            print(f"ARGO worker error for {start_date}: {exc}")
            import traceback
            traceback.print_exc()
            return None

    # Mise à jour de la fonction principale
    def list_files_with_metadata(self) -> List[CatalogEntry]:
        """Version avec fonction statique corrigée."""
        
        global_metadata = self.extract_global_metadata()
        self._global_metadata = global_metadata

        list_dates = self._list_files
        logger.info(f"Processing {len(list_dates)} ARGO dates with Dask")

        # Configuration Dask simple
        dask.config.set(scheduler='threads')
        
        metadata_list = []
        dask_client = None
        
        try:
            # Configuration et création du client Dask
            dask_client = self.setup_dask()
            logger.info(f"Dask client created: {dask_client}")

            # Scatter une seule fois les objets volumineux
            scattered_config = dask_client.scatter(self.connect_config, broadcast=True)
            scattered_metadata = dask_client.scatter(self._global_metadata, broadcast=True)
            scattered_argo_index = dask_client.scatter(self.argo_index, broadcast=True)

            for i in range(0, len(list_dates), self.batch_size):
                batch_dates = list_dates[i:i + self.batch_size]
                logger.info(f"Processing ARGO batch {i//self.batch_size + 1}/{(len(list_dates)-1)//self.batch_size + 1}")

                # Préparer les paramètres comme dictionnaire
                '''connection_params = {
                    'init_type': self.params.init_type,
                    'local_root': self.params.local_root,
                    'max_samples': self.params.max_samples,
                    'file_pattern': getattr(self.params, 'file_pattern', '**/*.nc'),
                    'keep_variables': self.params.keep_variables,
                }'''
                
                # Créer les tâches avec les données pré-extraites (sérialisables)
                delayed_tasks = [
                    dask.delayed(self._extract_argo_metadata)(
                        start_date, scattered_config, scattered_metadata, scattered_argo_index
                    ) 
                    for start_date in batch_dates
                ]
                
                try:
                    # Exécuter le batch avec le client
                    batch_futures = dask_client.compute(delayed_tasks, sync=False)
                    batch_results = dask_client.gather(batch_futures)
                    
                    # Filtrer et ajouter les résultats valides
                    valid_results = [meta for meta in batch_results if meta is not None]
                    metadata_list.extend(valid_results)
                    
                    
                    logger.info(f"ARGO batch completed: {len(valid_results)}/{len(batch_dates)} dates processed")

                    # Nettoyage explicite entre les batches
                    del batch_results
                    del batch_futures
                    dask_client.run(self._cleanup_worker_memory)
                    
                except Exception as exc:
                    logger.error(f"ARGO batch {i//self.batch_size + 1} failed: {exc}")
                    # Fallback séquentiel pour ce batch
                    for start_date in batch_dates:
                        try:
                            meta = self._extract_argo_metadata(
                                start_date, self.connect_config,
                                self._global_metadata, self.argo_index,
                            )
                            if meta is not None:
                                metadata_list.append(meta)

                        except Exception as date_exc:
                            logger.warning(f"Failed to process ARGO date {start_date}: {date_exc}")

        except Exception as exc:
            logger.error(f"Dask ARGO metadata extraction failed: {exc}")
            raise
        finally:
            # Nettoyage du client Dask
            if dask_client is not None:
                try:
                    logger.info("Closing Dask client and cluster")
                    dask_client.close()
                    if hasattr(dask_client, 'cluster'):
                        dask_client.cluster.close()
                except Exception as cleanup_exc:
                    logger.warning(f"Error during Dask cleanup: {cleanup_exc}")

        if not metadata_list:
            logger.error("No valid ARGO metadata entries were generated.")
            raise ValueError("No valid ARGO metadata entries were generated.")

        return metadata_list


    '''def _extract_lightweight_argo_geometry(self, ds: xr.Dataset, coord_sys) -> Any:
        """Extraction géométrique allégée pour réduire l'usage mémoire - spécifique ARGO."""
        try:
            # Pour ARGO, utiliser un sous-échantillonnage très agressif car tous les points 
            # du profil ont la même position horizontale
            return get_dataset_geometry(ds, coord_sys, max_points=100)  # Très peu de points pour ARGO
        except Exception:
            # Fallback : géométrie par défaut
            return gpd.GeoSeries([geometry.Point(0, 0)])'''

    '''@staticmethod
    def _cleanup_worker_memory():
        """Fonction de nettoyage à exécuter sur chaque worker."""
        import gc
        gc.collect()'''

    '''def _sequential_argo_fallback(self, list_dates: List[pd.Timestamp]) -> List[CatalogEntry]:
        """Méthode de fallback séquentielle optimisée pour ARGO."""
        metadata_list = []
        logger.info("Using sequential fallback for ARGO metadata extraction")
        
        for i, start_date in enumerate(list_dates):
            try:
                if i % 5 == 0:
                    logger.info(f"Sequential ARGO processing: {i}/{len(list_dates)}")
                
                meta = self._extract_argo_metadata_safe(start_date)
                if meta is not None:
                    metadata_list.append(meta)
            except Exception as exc:
                logger.warning(f"Failed to process ARGO date {start_date}: {exc}")
        
        return metadata_list'''

    def list_files(self) -> List[str]:
        """Liste les dates où il y a effectivement des données ARGO avec découpage temporel fin."""
        if self.argo_index.empty:
            return []

        start_dt = pd.to_datetime(self.start_time)
        end_dt = pd.to_datetime(self.end_time)
        
        # Découper en chunks de 6 heures pour ARGO
        chunk_size_hours = 6  # Très fin pour éviter les surcharges
        all_dates = []
        
        current_start = start_dt
        while current_start < end_dt:
            current_end = min(current_start + pd.Timedelta(hours=chunk_size_hours), end_dt)
            
            # logger.info(f"Processing ARGO temporal chunk: {current_start} to {current_end}")
            
            # Filtrer pour ce chunk temporel de quelques heures
            chunk_filtered = self.argo_index[
                (self.argo_index['date'] >= current_start) &
                (self.argo_index['date'] <= current_end) &
                (self.argo_index['latitude'] >= self.lat_range[0]) &
                (self.argo_index['latitude'] <= self.lat_range[1]) &
                (self.argo_index['longitude'] >= self.lon_range[0]) &
                (self.argo_index['longitude'] <= self.lon_range[1])
            ]

            if not chunk_filtered.empty:
                #chunk_dates = chunk_filtered['date'].dt.floor('min').unique()
                chunk_dates = chunk_filtered['date'].unique()
                all_dates.extend([pd.Timestamp(date) for date in chunk_dates])
                # logger.info(f"Found {len(chunk_dates)} minute slots with ARGO data in chunk")

            current_start = current_end
        
        # Supprimer les doublons et trier
        unique_dates = sorted(list(set(all_dates)))
        logger.info(f"Total unique minutes timestamps with ARGO data: {len(unique_dates)}")

        return unique_dates


    '''def list_files(self) -> List[str]:

        # idx = ArgoIndex(host="https://data-argo.ifremer.fr")  # Default host
        idx = ArgoIndex(host="ftp://ftp.ifremer.fr/ifremer/argo", index_file="ar_index_global_prof.txt")  # Default index

        # limit = self.params.max_samples if self.params.max_samples else len(self._list_files)
        # limit = len(self._list_files)
        #logger.info(f"Loading ARGO index with limit: {limit}")

        idx = idx.load()  #nrows=limit)
        # logger.info(f"\n\nARGO Index: {idx}")
        df = idx.to_dataframe(index=True)
        # logger.info(f"\n\nsize List of files 1: {idx.uri_full_index}")

        # 3. Trier les données par date croissante
        df["date"] = pd.to_datetime(df["date"])

        # Générer une série de dates journalières entre min et max
        date_range = pd.date_range(start=df['date'].min(), end=df['date'].max(), freq='D')

        return date_range.tolist()'''


    '''def list_files(self) -> List[str]:
        """Liste les dates où il y a effectivement des données ARGO."""
        if self.argo_index.empty:
            return []
        
        # Filtrer l'index par région ET par dates
        start_dt = pd.to_datetime(self.start_time)
        end_dt = pd.to_datetime(self.end_time)
        
        filtered_index = self.argo_index[
            (self.argo_index['date'] >= start_dt) &
            (self.argo_index['date'] <= end_dt) &
            (self.argo_index['latitude'] >= self.lat_range[0]) &
            (self.argo_index['latitude'] <= self.lat_range[1]) &
            (self.argo_index['longitude'] >= self.lon_range[0]) &
            (self.argo_index['longitude'] <= self.lon_range[1])
        ]
        
        if filtered_index.empty:
            logger.warning("No ARGO data in specified region/time")
            return []
        
        # Retourner seulement les dates où il y a des données
        unique_dates = filtered_index['date'].dt.date.unique()
        return [pd.Timestamp(date) for date in sorted(unique_dates)]'''


    '''def open(
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
        return ds'''

    '''def open(
        self, start_date: pd.Timestamp,
        mode: str = "rb",
    ) -> xr.Dataset:
        """
        Open an Argo dataset from a given path.

        Args:

        Returns:
            xr.Dataset: Opened Argo dataset.
        """
        print(f"argo process: {start_date}")

        ref_start_date = pd.to_datetime(self.start_time)
        ref_end_date = pd.to_datetime(self.end_time)

        end_date = start_date + pd.Timedelta(days=1)

        if end_date < ref_start_date or start_date > ref_end_date:
            return None
        
        start_date, end_date = self.adjust_full_day_if_needed(start_date, end_date)

        profile = self.argo_loader.region([
            min(self.lon_range), max(self.lon_range),
            min(self.lat_range), max(self.lat_range),
            min(self.depth_values),
            max(self.depth_values),
            start_date, end_date
        ])
        # logger.info(f"Fetched ARGO data : {profile}")

        # check if data is available before calling load()
        try:
            # Tenter d'obtenir l'index pour vérifier la présence de données
            if hasattr(profile, '_index') and profile._index is not None:
                index_df = profile.to_index()
                if index_df.empty:
                    #logger.warning(f"No ARGO data found for period {start_date} to {end_date}")
                    return None
            else:
                # Fallback : essayer de charger un échantillon minimal
                test_load = profile.load()
                if test_load.data is None or len(test_load.data.dims) == 0:
                    #logger.warning(f"No ARGO data found for period {start_date} to {end_date}")
                    return None
                # ds = test_load.data
            
            ds = profile.load().data
            logger.info(f"Dataset : {ds}")
            return ds
        except Exception as fetch_error:
            #logger.warning(f"No ARGO data available for {start_date} to {end_date}: {fetch_error}")
            return None'''

    '''def open(self, start_date: pd.Timestamp, mode: str = "rb") -> Optional[xr.Dataset]:
        """Version optimisée qui évite les requêtes répétées."""
        # Vérifier d'abord dans l'index local
        start_date_only = start_date.date()
        available_profiles = self.argo_index[
            (self.argo_index['date'].dt.date == start_date_only) &
            (self.argo_index['latitude'] >= self.lat_range[0]) &
            (self.argo_index['latitude'] <= self.lat_range[1]) &
            (self.argo_index['longitude'] >= self.lon_range[0]) &
            (self.argo_index['longitude'] <= self.lon_range[1])
        ]

        if available_profiles.empty:
            logger.debug(f"No ARGO profiles found for {start_date}")
            return None

        logger.info(f"Found {len(available_profiles)} ARGO profiles for {start_date}")

        # Maintenant faire une requête ciblée
        try:
            end_date = start_date + pd.Timedelta(days=1)
            
            profile = self.argo_loader.region([
                self.lon_range[0], self.lon_range[1],
                self.lat_range[0], self.lat_range[1],
                min(self.depth_values), max(self.depth_values),
                start_date, end_date
            ])
            
            # Pas besoin de vérifier l'index - on sait qu'il y a des données
            ds = profile.load().data
            return ds
            
        except Exception as e:
            logger.error(f"Failed to load ARGO data for {start_date}: {e}")
            return None'''


    def open(self, start_date: pd.Timestamp, mode: str = "rb") -> Optional[xr.Dataset]:
        """Version optimisée qui utilise les profils individuels identifiés."""
        
        # Vérifier dans l'index local les profils disponibles
        # start_date_only = start_date.date()
            # Calculer l'intervalle temporel : tous les t tels que floor(t) = start_date
        # Si start_date = "2024-01-03 12:30:45", on veut tous les t dans [12:30:45, 12:30:46[
        #interval_start = start_date
        #interval_end = start_date + pd.Timedelta(seconds=1)
        selected_profiles = self.argo_index[
            #(self.argo_index['date'] >= interval_start) &
            #(self.argo_index['date'] < interval_end) &  # Intervalle fermé-ouvert [start, start+1s[
            (self.argo_index['date'] == start_date) &  # Correspondance exacte
            (self.argo_index['latitude'] >= self.lat_range[0]) &
            (self.argo_index['latitude'] <= self.lat_range[1]) &
            (self.argo_index['longitude'] >= self.lon_range[0]) &
            (self.argo_index['longitude'] <= self.lon_range[1])
        ]
        
        if selected_profiles.empty:
            logger.debug(f"No ARGO profiles found for {start_date}")
            return None
        
        logger.info(f"Found {len(selected_profiles)} ARGO profiles for {start_date}")
        
        # Charger les profils individuellement plutôt qu'avec une requête régionale
        try:
            datasets = []
            #max_profiles = 20  # Limite pour éviter la surcharge
            
            # Filtrer par correspondance exacte temporelle
            #selected_profiles = available_profiles[
            #    available_profiles['date'] == start_date
            #]
            
            '''if selected_profiles.empty:
                # Si aucune correspondance exacte, prendre les profils dans une fenêtre de ±30 minutes
                time_tolerance = pd.Timedelta(minutes=30)
                time_window_profiles = available_profiles[
                    (available_profiles['date'] >= start_date - time_tolerance) &
                    (available_profiles['date'] <= start_date + time_tolerance)
                ]
                
                if time_window_profiles.empty:
                    logger.debug(f"No ARGO profiles found within 30 minutes of {start_date}")
                    return None
                
                # Trier par proximité temporelle et prendre les plus proches
                time_window_profiles['time_diff'] = abs(time_window_profiles['date'] - start_date)
                selected_profiles = time_window_profiles.nsmallest(max_profiles, 'time_diff')
                logger.info(f"Using {len(selected_profiles)} profiles within ±30min of {start_date}")
                return None'''
            #else:
            #    # Utiliser les profils avec correspondance exacte
            #    selected_profiles = exact_time_profiles.head(max_profiles)
            #    logger.info(f"Using {len(selected_profiles)} profiles at exact time {start_date}")
            
            for _, profile_info in selected_profiles.iterrows():
                try:
                    # Charger chaque profil individuellement par WMO et cycle
                    wmo = int(profile_info['wmo'])
                    cycle = int(profile_info['cyc'])
                    
                    # Utiliser le DataFetcher avec des coordonnées spécifiques du profil
                    profile_ds = self.argo_loader.profile(wmo, cycle).load().data
                    
                    if profile_ds is not None and len(profile_ds.N_POINTS) > 0:
                        datasets.append(profile_ds)
                        
                except Exception as e:
                    logger.warning(f"Failed to load profile WMO={wmo}, cycle={cycle}: {e}")
                    continue
            
            if datasets:
                # Concaténer tous les profils chargés
                combined_ds = xr.concat(datasets, dim='N_POINTS')
                # logger.info(f"  Successfully loaded {len(combined_ds.N_POINTS)} ARGO N_POINTS for {start_date}")


                # Ajouter la profondeur comme dimension
                combined_ds = self._add_depth_dimension(combined_ds)
                return combined_ds
            else:
                logger.warning(f"No ARGO profiles could be loaded for {start_date}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to load ARGO data for {start_date}: {e}")
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