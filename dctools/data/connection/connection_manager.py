"""Manager for different types of data connections (Local, S3, FTP, ARGO, CMEMS)."""

from abc import ABC, abstractmethod
import gc
import math
import os
import tempfile
import traceback
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from shapely.geometry import box

# CRITICAL: Patch xr.open_dataset BEFORE argopy imports.
# argopy downloads ARGO data as raw bytes (NetCDF3/CDF format) and passes them
# to xr.open_dataset() without specifying an engine.  xarray defaults to the
# netCDF4 engine whose C library cannot handle in-memory reads on this system
# (fails with [Errno 1] Operation not permitted: '<xarray-in-memory-read>').
# Forcing engine='scipy' for in-memory objects avoids the netCDF4 C library
# entirely — scipy handles NetCDF3 natively and has no file-locking issues.
import io as _io
import xarray as _xr

if not hasattr(_xr, "_original_open_dataset"):
    _xr._original_open_dataset = _xr.open_dataset

    def _open_dataset_scipy_for_inmem(filename_or_obj, *args, **kwargs):
        """Wrapper that forces engine='scipy' for in-memory data."""
        if isinstance(filename_or_obj, (bytes, _io.BytesIO, _io.BufferedIOBase)):
            kwargs.setdefault("engine", "scipy")
        # When using SciPy's netcdf_file backend, disable mmap to avoid
        # RuntimeWarning on close/eviction when arrays still reference
        # the underlying file.
        if kwargs.get("engine") == "scipy":
            _bk = kwargs.get("backend_kwargs")
            if _bk is None:
                _bk = {}
            else:
                _bk = dict(_bk)
            _bk.setdefault("mmap", False)
            kwargs["backend_kwargs"] = _bk
        # Suppress FutureWarning about MTIME timedelta decoding
        kwargs.setdefault("decode_timedelta", False)
        return _xr._original_open_dataset(filename_or_obj, *args, **kwargs)

    _xr.open_dataset = _open_dataset_scipy_for_inmem

from argparse import Namespace

try:
    import copernicusmarine  # type: ignore
except Exception:
    copernicusmarine = None
import datetime
import dask
from loguru import logger
import numpy as np
import pandas as pd
from pathlib import Path
import json
import xarray as xr


from dctools.data.connection.config import (
    BaseConnectionConfig,
    ARGOConnectionConfig,
    GlonetConnectionConfig,
    WasabiS3ConnectionConfig,
    S3ConnectionConfig,
    FTPConnectionConfig,
    CMEMSConnectionConfig,
    LocalConnectionConfig,
)

from dctools.data.datasets.dc_catalog import CatalogEntry
from dctools.data.coordinates import (
    get_dataset_geometry,
    get_dataset_geometry_light,
    CoordinateSystem,
)
from dctools.data.coordinates import get_target_depth_values

from dctools.data.datasets.dc_catalog import GLOBAL_METADATA
from dctools.dcio.loader import FileLoader
from dctools.utilities.file_utils import empty_folder
from dctools.utilities.misc_utils import (
    deep_copy_object,
    list_all_days,
)


# List of possible names for the time dimension
TIME_NAMES = [
    "time",
    "Time",
    "TIME",
    "date",
    "datetime",
    "valid_time",
    "forecast_time",
    "time_counter",
    "profile_date",
]
# List of possible names for the n_points dimension
POINT_DIM_NAMES = ("N_POINTS", "n_points", "points", "obs")


def get_time_bound_values(ds: xr.Dataset) -> tuple:
    """
    Returns the time bounds (min, max) of an xarray dataset.

    Regardless of the structure (dimension, coordinate, variable).
    """
    time_vals = None

    try:
        # Search for the time variable in dims, coords, data_vars
        for time_name in TIME_NAMES:
            if time_name in ds.dims:
                time_vals = ds[time_name] if time_name in ds.data_vars else ds.coords.get(time_name)
                if time_vals is not None:
                    break
        if time_vals is None:
            for time_name in TIME_NAMES:
                if time_name in ds.coords:
                    time_vals = ds.coords[time_name]
                    break
        if time_vals is None:
            for time_name in TIME_NAMES:
                if time_name in ds.data_vars:
                    time_vals = ds[time_name]
                    break

        # If nothing found, search for a variable with datetime64 dtype
        if time_vals is None:
            for _, var in ds.data_vars.items():
                if np.issubdtype(var.dtype, np.datetime64):
                    time_vals = var
                    break

        if time_vals is not None:
            # If array is empty
            if time_vals.size == 0:
                return (None, None)
            # If datetime
            if np.issubdtype(time_vals.dtype, np.datetime64):
                dt_vals: Any = pd.to_datetime(time_vals.values)
                # If array, take min/max
                if hasattr(dt_vals, "min") and hasattr(dt_vals, "max"):
                    dt_min = dt_vals.min()
                    dt_max = dt_vals.max()
                else:
                    dt_min = dt_max = dt_vals
                return (pd.Timestamp(dt_min), pd.Timestamp(dt_max))
            # If numeric
            elif np.issubdtype(time_vals.dtype, np.floating) or np.issubdtype(
                time_vals.dtype, np.integer
            ):
                num_vals = np.asarray(time_vals.values)
                num_min = float(np.nanmin(num_vals))
                num_max = float(np.nanmax(num_vals))
                if np.isnan(num_min) or np.isnan(num_max):
                    return (None, None)
                return (num_min, num_max)
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
    """Cleans non-serializable objects before pickle."""
    # Close/clean argopy objects
    if hasattr(obj, "_argo_index"):
        obj._argo_index = None
    if hasattr(obj, "_argopy_fetcher"):
        obj._argopy_fetcher = None
    if isinstance(obj, SimpleNamespace):
        # Clean fsspec
        if hasattr(obj, "fs"):
            if hasattr(obj.fs, "_session"):
                try:
                    if hasattr(obj.fs._session, "close"):
                        obj.fs._session.close()
                except Exception:
                    pass
            obj.fs = None
        if hasattr(obj, "params"):
            obj_params = obj.params
            if hasattr(obj_params, "fs"):
                if hasattr(obj_params.fs, "_session"):
                    try:
                        if hasattr(obj_params.fs._session, "close"):
                            obj_params.fs._session.close()
                    except Exception:
                        pass
                obj_params.fs = None

        # Clean dataset_processor
        if hasattr(obj, "dataset_processor"):
            try:
                obj.params.dataset_processor.close()
            except Exception:
                pass
            obj.dataset_processor = None
    else:
        # Clean fsspec
        if hasattr(obj.params, "fs"):
            if hasattr(obj.params.fs, "_session"):
                try:
                    if hasattr(obj.params.fs._session, "close"):
                        obj.params.fs._session.close()
                except Exception:
                    pass
            obj.params.fs = None

        # Clean dataset_processor
        if hasattr(obj.params, "dataset_processor"):
            try:
                obj.params.dataset_processor.close()
            except Exception:
                pass
            obj.params.dataset_processor = None
    return obj


class BaseConnectionManager(ABC):
    """
    Abstract base connection manager.

    Manages opening, closing and listing files for various protocols.
    """

    def __init__(
        self,
        connect_config: BaseConnectionConfig | Namespace,
        call_list_files: bool = True,
        batch_size: Optional[int] = 64,
    ):
        self.connect_config = connect_config
        self.batch_size = batch_size
        if isinstance(connect_config, BaseConnectionConfig):
            self.params = connect_config.to_dict()
        elif isinstance(connect_config, (Namespace, SimpleNamespace)):
            # Cast to SimpleNamespace to satisfy type checker
            self.params = SimpleNamespace(**vars(connect_config))
        else:
            raise TypeError(f"Unknown type of connection config: {type(connect_config)}.")

        # Extract filter values if available
        filter_values = getattr(self.params, "filter_values", None)
        self.start_time = filter_values.get("start_time") if filter_values else None
        self.end_time = filter_values.get("end_time") if filter_values else None

        self.init_type = self.params.init_type
        self._list_files = None  # Initialize to None by default
        if self.init_type != "from_json" and call_list_files:
            self._list_files = self.list_files()
        if not self.params.file_pattern:
            self.params.file_pattern = "**/*.nc"
        if not self.params.groups:
            self.params.groups = None
        self.file_cache = self.params.file_cache
        self.dataset_processor = self.params.dataset_processor

    def adjust_full_day(
        self, date_start: pd.Timestamp, date_end: pd.Timestamp
    ) -> tuple[pd.Timestamp, pd.Timestamp]:
        """Adjust date_end to cover a full day if dates are the same at midnight."""
        if pd.isnull(date_start) or pd.isnull(date_end):
            return date_start, date_end
        if (
            date_start == date_end
            and date_start.hour == 0
            and date_start.minute == 0
            and date_start.second == 0
        ):
            # Adjust date_end to the end of the day
            date_end = date_start + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        return date_start, date_end

    def open(
        self,
        path: str,
        mode: str = "rb",
    ) -> Optional[xr.Dataset]:
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
        # Attempt to open the file locally
        if LocalConnectionManager.supports(path):
            dataset = self.open_local(path)
            if dataset:
                return dataset
        # Attempt to open the file online
        elif self.supports(path):
            dataset = self.open_remote(path, mode)
            if dataset:
                return dataset

        # Download the file locally, then open it
        try:
            local_path = self._get_local_path(path)
            if not os.path.isfile(local_path):
                self.download_file(path, local_path)
                ds = self.open_local(local_path)
                return ds
            else:
                return self.open_local(local_path)
        except Exception as exc:
            logger.warning(f"Failed to open file: {path}. Error: {repr(exc)}")
            traceback.print_exc()
            return None

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

            # file_storage=None: local files must NOT use the remote
            # filesystem handle (e.g. S3).  Passing self.params.fs here
            # would make zarr try file_storage.get_mapper(local_path)
            # which routes through S3 -> "Access Denied".
            ds = FileLoader.open_dataset_auto(
                local_path,
                adaptive_chunking=False,
                groups=self.params.groups,
                variables=self.params.keep_variables,
                file_storage=None,
            )
            return ds
        return None

    def open_remote(
        self,
        path: str,
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
            if extension != ".zarr":
                return None

            return FileLoader.open_dataset_auto(
                path,
                adaptive_chunking=False,
                groups=self.params.groups,
                variables=self.params.keep_variables,
                file_storage=self.params.fs,
            )
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

        return (
            self._global_metadata
            if hasattr(self, "_global_metadata")
            else self.extract_global_metadata()
        )

    def set_global_metadata(self, global_metadata: Dict[str, Any]) -> None:
        """
        Sets the global metadata for the connection manager.

        Keeps only the keys listed in the global_metadata class variable.

        Args:
            global_metadata (Dict[str, Any]): Global metadata dictionary.
        """
        # Keep only relevant keys
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

        # Loop over files until a valid file is found (not empty, not None)
        first_file = None
        for file_path in files:
            # Verify that the file is not None, empty or invalid
            if file_path and file_path != "" and file_path is not None:
                try:
                    # Test if the file can be opened
                    test_ds = self.open(file_path, "rb")
                    if test_ds is not None:
                        first_file = file_path
                        break
                except Exception as exc:
                    logger.warning(f"Could not open file {file_path}, trying next: {exc}")
                    continue

        if first_file is None:
            raise FileNotFoundError("No valid files found in the list to extract metadata from.")

        sample_ds = self.open(first_file, "rb")
        if sample_ds is None:
            raise FileNotFoundError(f"Failed to open first file: {first_file}")

        with sample_ds as ds:
            # Extract global metadata

            coord_sys = CoordinateSystem.get_coordinate_system(ds)

            # Infer spatial and temporal resolution
            dict_resolution = self.estimate_resolution(ds, coord_sys)

            # Associate variables with their dimensions
            variables: Dict[Any, Any] = {}
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
        self,
        path: str,
    ) -> CatalogEntry:
        """
        Extract metadata combining global/file-specific info.

        Args:
            path (str): Path to the file.
            global_metadata (Dict[str, Any]): Global metadata to apply to all files.

        Returns:
            CatalogEntry: Metadata for the specific file as a CatalogEntry.
        """
        try:
            ds = self.open(path, "rb")
            if ds is None:
                raise IOError(f"Could not open file {path}")

            with ds:
                time_bounds = get_time_bound_values(ds)
                date_start, date_end = time_bounds

                if self.params.full_day_data:
                    date_start, date_end = self.adjust_full_day(date_start, date_end)

                coord_sys = self._global_metadata.get("coord_system")
                if not coord_sys:
                    logger.warning(f"No coordinate system found in global metadata for {path}")
                    coord_sys = CoordinateSystem.get_coordinate_system(ds)

                ds_region = get_dataset_geometry(ds, coord_sys)

                # Create a CatalogEntry instance
                return CatalogEntry(
                    path=path,
                    date_start=date_start,
                    date_end=date_end,
                    geometry=ds_region,
                )
        except Exception as exc:
            logger.error(f"Failed to extract metadata for file {path}: {repr(exc)}")
            raise

    @staticmethod
    def extract_metadata_worker(
        path: str,
        global_metadata: dict,
        connection_params: dict,
        class_name: Any,
        argo_index: Optional[Any] = None,
    ):
        """
        Extract metadata combining global/file-specific info.

        Thread-safe version to avoid conflicts.

        Args:
            path (str): Path to the file.
            global_metadata (Dict[str, Any]): Global metadata.

        Returns:
            CatalogEntry: Metadata for the specific file as a CatalogEntry.
        """
        try:
            open_func = create_worker_connect_config(
                connection_params,
                argo_index,
            )

            if class_name == "CMEMSManager" or class_name == "GlonetManager":
                # cmems not compatible with Dask workers (pickling errors)
                with dask.config.set(scheduler="synchronous"):
                    ds = open_func(path, "rb")
            else:
                ds = open_func(path, "rb")

            if ds is None:
                logger.warning(f"Could not open {path}")
                return None

            time_bounds = get_time_bound_values(ds)
            date_start = time_bounds[0]
            date_end = time_bounds[1]

            coord_sys = global_metadata.get("coord_system")
            if not coord_sys:
                coord_sys = CoordinateSystem.get_coordinate_system(ds)

            ds_region = get_dataset_geometry_light(ds, coord_sys)

            # Explicitly close the dataset
            # if hasattr(ds, 'close'):
            ds.close()
            del ds
            gc.collect()

            if global_metadata.get("full_day_data", False):
                if date_start and date_end:
                    date_start = date_start.replace(hour=0, minute=0, second=0, microsecond=0)
                    date_end = date_end.replace(hour=23, minute=59, second=59, microsecond=999999)

            # Create a CatalogEntry instance
            return CatalogEntry(
                path=path,
                date_start=date_start,
                date_end=date_end,
                geometry=ds_region,
            )
        except Exception:
            logger.warning(f"Failed to extract metadata for file {path}: {traceback.format_exc()}")
            return None

    def estimate_resolution(
        self,
        ds: xr.Dataset,
        coord_system: CoordinateSystem,
    ) -> Dict[str, Union[float, str]]:
        """
        Estimate resolution from dataset based on coordinates.

        Only inspects coordinate values (small arrays).  Handles both
        in-memory and dask-backed datasets safely — ``np.asarray()`` is
        used to materialise only the coordinate arrays (typically tiny).

        Args:
            ds: xarray.Dataset
            coord_system: CoordinateSystem object.

        Returns:
            Dictionary of estimated resolutions.
        """
        res: Dict[Any, Any] = {}

        # Helper function for 1D resolution
        def compute_1d_resolution(coord):
            try:
                # np.asarray materialises only this small coord array
                values = np.asarray(ds.coords[coord])
            except Exception:
                return None
            if values.ndim != 1:
                return None
            if len(values) == 1:
                return float(values[0])  # Return the unique value
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
        if time_name and time_name in ds.coords:
            try:
                time_values = np.asarray(ds.coords[time_name])
            except Exception:
                time_values = np.array([])
            if time_values.ndim == 1 and len(time_values) > 1:
                diffs = np.diff(time_values)
                diffs = pd.to_timedelta(diffs).astype("timedelta64[s]").astype(int)
                res["time"] = f"{int(np.median(diffs))}s"
        return res

    def get_config_clean_copy(self):
        """Return a clean copy of the configuration."""
        connection_conf = deep_copy_object(self.connect_config.params)
        connection_conf = clean_for_serialization(connection_conf)
        return connection_conf

    def list_files_with_metadata(self) -> List[CatalogEntry]:
        """Version with integrated Dask client and optimized configuration."""
        # Get global metadata
        global_metadata = self.extract_global_metadata()
        self._global_metadata = global_metadata

        limit = self.params.max_samples if self.params.max_samples else len(self._list_files)
        file_list = self._list_files[-limit:]

        logger.info(f"Processing {len(file_list)} files with integrated Dask client")

        if hasattr(self, "argo_index") and self.argo_index is not None:
            scattered_argo_index = self.dataset_processor.scatter_data(
                self.argo_index,
                broadcast_item=False,
            )
        else:
            scattered_argo_index = None

        try:
            connection_conf = self.get_config_clean_copy()

            batch_size = self.batch_size if self.batch_size is not None else 32
            n_batches = math.ceil(len(file_list) / batch_size)
            temp_dir = tempfile.mkdtemp(prefix="metadata_batches_")
            empty_folder(temp_dir, extension=".json")
            temp_files: List[Any] = []

            logger.info(
                f"Processing {len(file_list)} files in {n_batches} batches "
                f"(batch_size={batch_size})"
            )

            for i in range(n_batches):
                batch_paths = file_list[i * batch_size : (i + 1) * batch_size]

                delayed_tasks = [
                    dask.delayed(self.extract_metadata_worker)(
                        path,
                        self._global_metadata,
                        connection_conf,
                        self.__class__.__name__,
                        scattered_argo_index,
                    )
                    for path in batch_paths
                ]

                batch_results = self.dataset_processor.compute_delayed_tasks(
                    delayed_tasks, sync=False
                )
                valid_results = [meta for meta in batch_results if meta is not None]

                # Save the batch into a temporary file
                batch_file = f"{temp_dir}/metadata_batch_{i:08d}.json"
                with open(batch_file, "w") as f:
                    json.dump([meta.to_dict() for meta in valid_results], f, default=str, indent=2)
                temp_files.append(batch_file)

                percent = int(100 * (i + 1) / n_batches)
                logger.info(
                    f"Batch {i + 1}/{n_batches} processed ({percent}%) : {len(valid_results)} files"
                )

                # Memory cleanup
                del batch_results, valid_results
                gc.collect()

            metadata_entries: List[Any] = []
            for batch_file in temp_files:
                with open(batch_file, "r") as f:
                    batch_data = json.load(f)
                    for meta_dict in batch_data:
                        metadata_entries.append(CatalogEntry(**meta_dict))

            logger.info(
                f"Finished processing data: {len(metadata_entries)}/{len(file_list)} "
                "items processed"
            )
            return metadata_entries
            # self.dataset_processor.cleanup_worker_memory()

        except Exception as exc:
            logger.error(f"Dask metadata extraction failed: {repr(exc)}")
            raise

    @classmethod
    @abstractmethod
    def supports(cls, path: str) -> bool:
        """Check if path is supported by this manager."""
        pass

    @abstractmethod
    def list_files(self) -> List[str]:
        """List files matching the configuration."""
        pass


class LocalConnectionManager(BaseConnectionManager):
    """Manager for local files."""

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
        """Checks if path is supported by this manager (local)."""
        return str(path).startswith("/") or str(path).startswith("file://")


class CMEMSManager(BaseConnectionManager):
    """Class to manage Copernicus Marine downloads."""

    def __init__(
        self,
        connect_config: BaseConnectionConfig,
        call_list_files: Optional[bool] = True,
        do_logging: Optional[bool] = True,
    ):
        """
        Initializes the CMEMS manager and performs connection.

        Args:
            connect_config (BaseConnectionConfig): Connection configuration.
        """
        super().__init__(connect_config, call_list_files=False)  # Call parent class initialization

        if do_logging:
            self.cmems_login()

        if self.init_type != "from_json" and call_list_files:
            self._list_files = self.list_files()

    def cmems_login(self) -> None:
        """Login to Copernicus Marine."""
        if copernicusmarine is None:
            raise ImportError(
                "copernicusmarine is required for CMEMS access but failed to import. "
                "This environment may have a broken sqlite3 build."
            )
        logger.info("Logging to Copernicus Marine.")
        try:
            if not (Path(self.params.cmems_credentials_path).is_file()):
                logger.warning(
                    f"Credentials file not found at {self.params.cmems_credentials_path}."
                )
                copernicusmarine.login(credentials_file=self.params.cmems_credentials_path)
        except Exception as exc:
            logger.error(f"Login to CMEMS failed: {repr(exc)}")

    def remote_file_exists(self, dt: datetime.datetime) -> bool:
        """
        Tests if a CMEMS file exists for a given date without opening it.

        Args:
            dt (datetime.datetime): Date to test

        Returns:
            bool: True if file exists, False otherwise
        """
        if copernicusmarine is None:
            raise ImportError("copernicusmarine is required for CMEMS access but failed to import.")
        try:
            if not isinstance(dt, datetime.datetime):
                dt = datetime.datetime.strptime(dt, "%Y-%m-%dT%H:%M:%S")

            start_datetime = datetime.datetime.combine(dt.date(), datetime.time.min)  # 00:00:00
            end_datetime = datetime.datetime.combine(
                dt.date(), datetime.time.max
            )  # 23:59:59.999999

            # Use a minimal request to test existence
            test_ds = copernicusmarine.open_dataset(
                dataset_id=self.params.dataset_id,
                start_datetime=start_datetime,
                end_datetime=end_datetime,
                # Parameters to minimize load
                minimum_longitude=0.0,  # minimal zone
                maximum_longitude=1.0,
                minimum_latitude=0.0,
                maximum_latitude=1.0,
                credentials_file=self.params.cmems_credentials_path,
            )

            # If we arrive here without exception, the file exists
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
            list_dates = list_all_days(
                start_date,
                end_date,
            )
            list_dates = list_dates[: self.params.max_samples]
            valid_dates: List[Any] = []
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
        if copernicusmarine is None:
            raise ImportError("copernicusmarine is required for CMEMS access but failed to import.")
        try:
            if not isinstance(dt, datetime.datetime):
                dt = datetime.datetime.strptime(dt, "%Y-%m-%dT%H:%M:%S")
            start_datetime = datetime.datetime.combine(dt.date(), datetime.time.min)  # 00:00:00
            end_datetime = datetime.datetime.combine(
                dt.date(), datetime.time.max
            )  # 23:59:59.999999
            # Optional spatial subset (major performance win when the
            # evaluation region is smaller than global).
            _fv = getattr(self.params, "filter_values", None) or {}
            _min_lon = _fv.get("min_lon")
            _max_lon = _fv.get("max_lon")
            _min_lat = _fv.get("min_lat")
            _max_lat = _fv.get("max_lat")

            try:
                ds = copernicusmarine.open_dataset(
                    dataset_id=self.params.dataset_id,
                    start_datetime=start_datetime,
                    end_datetime=end_datetime,
                    vertical_axis="depth",
                    minimum_longitude=_min_lon,
                    maximum_longitude=_max_lon,
                    minimum_latitude=_min_lat,
                    maximum_latitude=_max_lat,
                    credentials_file=self.params.cmems_credentials_path,
                )
            except Exception:
                # Fallback: some products/backends may not accept bbox params.
                ds = copernicusmarine.open_dataset(
                    dataset_id=self.params.dataset_id,
                    start_datetime=start_datetime,
                    end_datetime=end_datetime,
                    vertical_axis="depth",
                    credentials_file=self.params.cmems_credentials_path,
                )

            # Keep only requested variables early to limit downstream IO.
            _keep = getattr(self.params, "keep_variables", None)
            if _keep:
                try:
                    _keep_vars = [v for v in list(_keep) if v in ds.variables]
                    if _keep_vars:
                        ds = ds[_keep_vars]
                except Exception:
                    pass

            return ds
        except Exception:
            logger.warning(f"Failed to open CMEMS dataset for date : {dt}")
            traceback.print_exc()
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
        # CMEMS does not support a specific protocol like cmems://
        # We use the date (datetime format) as identifier for individual files
        dt: Any = path
        if not isinstance(dt, datetime.datetime):
            dt = datetime.datetime.strptime(dt, "%Y-%m-%dT%H:%M:%S")
        return isinstance(dt, datetime.datetime)


class FTPManager(BaseConnectionManager):
    """Manager for FTP connections."""

    @classmethod
    def supports(cls, path: str) -> bool:
        """Checks if path is an FTP URL."""
        # FTP does not support remote opening
        return path.startswith("ftp://")

    def open_remote(self, path, mode="rb"):
        """Opens a remote file (not supported by FTP in this implementation)."""
        # cannot open files remotely
        # FTP does not support remote opening
        return None

    def list_files(self) -> List[str]:
        """
        List available files on the FTP server matching the given pattern.

        Args:

        Returns:
            List[str]: List of file paths matching the pattern.
        """
        try:
            # Access FTP filesystem via fsspec
            fs = self.params.fs
            remote_path = (
                f"ftp://{self.params.host}/{self.params.ftp_folder}{self.params.file_pattern}"
            )

            # List files matching the pattern
            files = sorted(fs.glob(remote_path))

            if not files:
                logger.warning(
                    f"No file found on FTP server with pattern: {self.params.file_pattern}"
                )
            return [f"ftp://{self.params.host}{file}" for file in files]
        except Exception as exc:
            logger.error(f"Error listing files on FTP server: {repr(exc)}")
            return []


class RecursionExit(Exception):
    """Exception to exit recursion when listing files."""

    def __init__(self, value):
        self.value = value


class S3Manager(BaseConnectionManager):
    """Manager for S3 connections."""

    @classmethod
    def supports(cls, path: str) -> bool:
        """Checks if path is an S3 URL."""
        return path.startswith("s3://")

    def list_first_n_files(self, fs, remote_path, n=20, pattern="*.nc"):
        """Use fsspec filesystem to quickly list up to n files recursively."""
        import fnmatch

        out: List[Any] = []
        for root, dirs, files in fs.walk(remote_path):
            for dir in dirs:
                path = f"{remote_path}/{dir}"
                self.list_first_n_files(fs, path, n=n, pattern=pattern)
            for f in files:
                if fnmatch.fnmatch(f, pattern):
                    out.append(f"{root}/{f}")
                    if n is not None:
                        if len(out) >= n:
                            raise RecursionExit(out)
        return out

    def list_files(self) -> List[str]:
        """
        List files matching pattern.

        Args:

        Returns:
            List[str]: List of file paths.
        """
        try:
            if hasattr(self.params, "s3_bucket"):
                logger.info(f"Accessing bucket: {self.params.s3_bucket}")

            # Build remote path
            remote_base_path = f"s3://{self.params.s3_bucket}/{self.params.s3_folder}"
            remote_path = f"{remote_base_path}/{self.params.file_pattern}"

            # Use fsspec to access files

            files = sorted(self.params.fs.glob(remote_path))
            # files = files[-limit:]
            files_urls = [f"s3://{file}" for file in files]

            if not files_urls:
                logger.warning(f"No files found in bucket: {self.params.s3_bucket}")
            return files_urls
        except PermissionError as exc:
            logger.error(f"Permission error while accessing bucket: {repr(exc)}")
            logger.info("List files using object-level access...")

            # Bypass the problem by listing objects directly
            try:
                files = [
                    f"s3://{self.params.endpoint_url}/{self.params.s3_bucket}/{obj['Key']}"
                    for obj in self.params.fs.ls(remote_path, detail=True)
                    if obj["Key"].endswith(self.params.file_pattern.split("*")[-1])
                ]
                return files
            except Exception as exc:
                logger.error(f"Failed to list files using object-level access: {repr(exc)}")
                raise

    def open_remote(
        self,
        path: str,
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
            extension = Path(path).suffix
            if extension != ".zarr":
                return None
            return FileLoader.open_dataset_auto(
                path,
                adaptive_chunking=False,
                groups=self.params.groups,
                variables=self.params.keep_variables,
                file_storage=self.params.fs,
            )
        except Exception as exc:
            logger.warning(f"Failed to open S3 file: {path}. Error: {repr(exc)}")
            return None


class S3WasabiManager(S3Manager):
    """Specific S3 Manager for Wasabi (inheriting from S3Manager)."""

    @classmethod
    def supports(cls, path: str) -> bool:
        """Checks if path is supported."""
        try:
            extension = Path(path).suffix
            return (path.startswith("s3://")) and (extension == ".zarr" or extension == ".nc")
        except Exception:
            logger.warning(f"Error in supports check for S3WasabiManager path: {path}")
            traceback.print_exc()
            return False

    def open_remote(
        self,
        path: str,
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
            extension = Path(path).suffix
            if extension != ".zarr":
                return None
            return FileLoader.open_dataset_auto(
                path,
                adaptive_chunking=False,
                groups=self.params.groups,
                variables=self.params.keep_variables,
                file_storage=self.params.fs,
            )
        except Exception as exc:
            logger.warning(f"Failed to open Wasabi S3 file: {path}. Error: {repr(exc)}")
            return None


class GlonetManager(BaseConnectionManager):
    """Manager for Glonet (remote files over HTTPS)."""

    @classmethod
    def supports(cls, path: str) -> bool:
        """Check if the path is an HTTPS URL (generic)."""
        # return False  # do not open : download (avoids "too many requests")
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
        list_f: List[Any] = []
        while True:
            if date.year < 2025:
                date_str = date.strftime("%Y%m%d")
                list_f.append(
                    f"{self.params.endpoint_url}/{self.params.s3_bucket}/{self.params.s3_folder}/{date_str}.zarr"
                )
                date = date + datetime.timedelta(days=7)
            else:
                break
        return list_f

    def open(
        self,
        path: str,
        mode: str = "rb",
    ) -> Optional[xr.Dataset]:
        """Open a Glonet file, preferring local cache over remote.

        When the prediction prefetch step has downloaded the zarr to
        local disk, the path will be a local directory.  Open it
        directly to avoid any HTTP/S3 traffic on the worker.
        """
        # Fast path: local zarr (prefetched by the driver)
        if not path.startswith(("https://", "http://", "s3://")):
            try:
                return xr.open_zarr(path, consolidated=True)
            except Exception:
                try:
                    return xr.open_zarr(path, consolidated=False)
                except Exception:
                    pass
            return self.open_local(path)
        return self.open_remote(path, mode=mode)

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
            glonet_ds: xr.Dataset = xr.open_zarr(path)
            return glonet_ds

        except Exception as exc:
            logger.warning(f"Failed to open Glonet file: {path}. Error: {repr(exc)}")
            return None


class ArgoManager(BaseConnectionManager):
    """Specific manager for ARGO data using ArgoInterface for scalable indexing."""

    def __init__(
        self,
        connect_config: BaseConnectionConfig | Namespace,
        depth_values: Optional[List[float]] = None,
        argo_index: Optional[Any] = None,
        call_list_files: Optional[bool] = False,
    ):
        """
        Initialize ArgoManager with ArgoInterface.

        Args:
            connect_config: Configuration for ARGO connection
            depth_values: Depth levels for interpolation
            argo_index: Deprecated compatibility argument (unused)
            call_list_files: Whether to call list_files during initialization
        """
        # Depth levels must come from global target grid (dc2.yaml), not per-dataset config.
        cfg_depths: Optional[List[float]] = None
        try:
            if hasattr(connect_config, "depth_values"):
                cfg_depths = connect_config.depth_values
        except Exception:
            cfg_depths = None

        self.depth_values = (
            depth_values or cfg_depths or get_target_depth_values(connect_config) or []
        )
        self.argo_index = argo_index

        super().__init__(connect_config, call_list_files=False)

        # Import ArgoInterface
        from dctools.data.connection.argo_data import ArgoInterface

        # Créer l'instance ArgoInterface à partir de la configuration
        self.argo_interface = ArgoInterface.from_config(self.connect_config)

        import warnings

        warnings.filterwarnings("ignore", category=FutureWarning, module="argopy")

        # Charger la métadonnée du master index si disponible
        self._master_index = None
        self._catalog = None  # Cache pour list_files_with_metadata()
        self._global_metadata = None  # ARGO uses different metadata structure
        if self.init_type != "from_json" and call_list_files:
            try:
                self._load_master_index()
            except Exception as e:
                logger.warning(f"Could not load master index: {e}")

    def _load_master_index(self):
        """Charge le master index depuis S3/local."""
        import fsspec
        import ujson

        master_path = f"{self.argo_interface.base_path}/master_index.json"
        try:
            with fsspec.open(master_path, "r", **self.argo_interface.s3_storage_options) as f:
                raw = f.read()
            # Support both plain JSON and legacy zstd-compressed format
            try:
                self._master_index = ujson.loads(raw)
            except (ValueError, TypeError):
                import zstandard as zstd

                dctx = zstd.ZstdDecompressor()
                self._master_index = ujson.loads(
                    dctx.decompress(raw if isinstance(raw, bytes) else raw.encode())
                )
            logger.debug(f"Loaded ARGO master index with {len(self._master_index)} entries")
        except FileNotFoundError:
            logger.warning(f"Master index not found at {master_path}")
            self._master_index = None
        except Exception as e:
            logger.error(f"Error loading master index: {e}")
            self._master_index = None

    def _try_auto_build_master_index(self) -> None:
        """Try to build ARGO monthly Kerchunk index when master index is missing."""
        if self._master_index is not None:
            return

        if not self.start_time or not self.end_time:
            logger.warning(
                "Master index missing and no start_time/end_time available; "
                "automatic ARGO index build is skipped."
            )
            return

        try:
            start_ts = pd.Timestamp(self.start_time)
            end_ts = pd.Timestamp(self.end_time)
        except Exception as exc:
            logger.warning(
                "Master index missing but start/end time could not be parsed "
                f"({self.start_time}, {self.end_time}): {exc}"
            )
            return

        if start_ts > end_ts:
            start_ts, end_ts = end_ts, start_ts

        temp_dir = str(Path(self.params.local_root) / "tmp_argo_refs")
        n_workers = 8

        logger.warning(
            "ARGO master index is missing. Launching automatic Kerchunk index build "
            f"for window {start_ts.date()}..{end_ts.date()}."
        )

        try:
            if hasattr(self.argo_interface, "build_time_window_monthly"):
                self.argo_interface.build_time_window_monthly(
                    start=start_ts,
                    end=end_ts,
                    temp_dir=temp_dir,
                    n_workers=n_workers,
                )
            else:
                self.argo_interface.build_multi_year_monthly(
                    start_year=int(start_ts.year),
                    end_year=int(end_ts.year),
                    temp_dir=temp_dir,
                    n_workers=n_workers,
                )
        except Exception as exc:
            logger.error(f"Automatic ARGO index build failed: {exc}")
            return

        self._load_master_index()

    # ==========================================================
    # CATALOG
    # ==========================================================

    def list_files_with_metadata(self) -> List[CatalogEntry]:
        """List available time windows from the master index.

        Liste les fenêtres temporelles disponibles à partir du master index.
        Utilise build_multi_year_monthly() pour créer l'index si nécessaire.

        Returns:
            List[CatalogEntry]: Liste des entrées de catalogue avec métadonnées
        """
        if self._catalog is not None:
            return self._catalog

        # Ensure master index is loaded lazily (normal dataset init path)
        if self._master_index is None:
            try:
                self._load_master_index()
            except Exception as exc:
                logger.warning(f"Could not load ARGO master index: {exc}")

        # Kerchunk-only ARGO path: master index is required.
        if self._master_index is None:
            self._try_auto_build_master_index()

        if self._master_index is None:
            raise FileNotFoundError(
                "ARGO master index not found. Automatic build was attempted when possible, "
                "but no usable master_index.json is available. Build the Kerchunk monthly "
                "index first (scripts/build_argo_index.py) and ensure master_index.json "
                "is available."
            )

        metadata_entries: List[Any] = []

        try:
            # Créer des entrées de catalogue à partir du master index
            # Chaque mois devient une fenêtre temporelle
            for key, info in self._master_index.items():
                start_epoch = info["start"]
                end_epoch = info["end"]

                # Convertir epoch (ns) en Timestamp
                start = pd.Timestamp(start_epoch, unit="ns")
                end = pd.Timestamp(end_epoch, unit="ns")

                # Géométrie globale pour ARGO (données mondiales)
                geom = box(-180, -90, 180, 90)

                metadata_entries.append(
                    CatalogEntry(
                        path=key,  # Utiliser la clé du mois (e.g., "2024_01")
                        date_start=start,
                        date_end=end,
                        geometry=geom,
                    )
                )

            logger.info(f"Loaded ARGO metadata: {len(metadata_entries)} monthly windows available")

        except Exception as exc:
            logger.error(f"ARGO metadata extraction failed: {exc}")
            import traceback

            traceback.print_exc()
            raise

        if not metadata_entries:
            logger.warning("No valid ARGO metadata entries were generated.")

        self._catalog = metadata_entries
        return metadata_entries

    # ==========================================================
    # Kerchunk-only ARGO manager (legacy ERDDAP/profile paths removed)
    # ==========================================================

    # ==========================================================
    # SINGLE OPEN
    # ==========================================================

    def open(self, path: str, *args: Any, **kwargs: Any) -> xr.Dataset:
        """Open an ARGO time window.

        Ouvre une fenêtre temporelle ARGO.
        Utilise open_time_window() de ArgoInterface.

        Args:
            path: Clé du mois (e.g., "2024_01") ou tuple (start, end)
            *args: Ignored extra positional arguments.
            **kwargs: Ignored extra keyword arguments.

        Returns:
            xr.Dataset: Dataset ARGO avec interpolation sur les profondeurs
        """
        # Si path est une clé de mois, récupérer les dates depuis le master index
        if isinstance(path, str) and self._master_index and path in self._master_index:
            info = self._master_index[path]
            start = pd.Timestamp(info["start"], unit="ns")
            end = pd.Timestamp(info["end"], unit="ns")
        # Sinon, path devrait être un tuple (start, end)
        elif isinstance(path, (tuple, list)) and len(path) == 2:
            start, end = path
        else:
            raise ValueError(
                f"Invalid path for ARGO: {path}. "
                "Expected month key (e.g., '2024_01') or (start, end) tuple "
                "for Kerchunk-based ARGO interface"
            )

        # Appeler open_time_window de ArgoInterface
        # Pass the already-loaded master index to avoid re-reading from S3.
        try:
            ds = self.argo_interface.open_time_window(
                start=start,
                end=end,
                depth_levels=self.depth_values,
                variables=self.params.keep_variables or self.argo_interface.variables,
                master_index=self._master_index,
            )
            return ds
        except Exception as exc:
            logger.error(f"Failed to open ARGO window {path}: {exc}")
            import traceback

            traceback.print_exc()
            raise

    def get_argo_index(self):
        """Return ARGO index object for compatibility with legacy evaluator code."""
        if self.argo_index is not None:
            return self.argo_index
        return self._master_index

    def prefetch_batch_shared_zarr(
        self,
        time_bounds_list: List[Tuple[pd.Timestamp, pd.Timestamp]],
        cache_dir: Path,
    ) -> Optional[str]:
        """Pre-download ALL ARGO profiles for a batch into one shared Zarr.

        Instead of fetching one Zarr per time-window (the old approach),
        this method:

        1. **Merges** all per-entry time windows into one global bounding
           interval — profiles that belong to multiple overlapping windows
           are downloaded exactly once.
        2. **Downloads** every profile in that interval through a single
           ``requests.Session`` (HTTP connection pooling -> one TCP+TLS
           handshake per GDAC mirror).
        3. **Writes** a single time-sorted Zarr that is opened by every
           worker.  Each worker filters by its own ``time_bounds`` via
           ``np.searchsorted`` — reads only contiguous chunks, no full
           scan.

        Typical saving vs. the per-window approach for a 10-entry batch
        with ``time_tolerance=12 h``:

        * Downloads: 10 × ~1 day -> 1 × ~11 days (massive overlap removed)
        * Zarr writes: 10 -> 1
        * Disk space: 10 small stores -> 1 compact store

        Parameters
        ----------
        time_bounds_list : list of (start, end) pd.Timestamp tuples
            One per batch entry.  May contain duplicates.
        cache_dir : Path
            Directory for the shared Zarr file.  Created if necessary.
            Files persist across batches (same global window -> cache hit).

        Returns
        -------
        str or None
            Absolute path to the shared, time-sorted Zarr, or *None*
            on failure (empty data, download error, …).
        """
        import time as _time_mod

        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        # ── 1. Merge all per-entry windows into a single bounding interval ─
        all_t0 = [pd.Timestamp(t0) for t0, _ in time_bounds_list]
        all_t1 = [pd.Timestamp(t1) for _, t1 in time_bounds_list]
        global_t0 = min(all_t0)
        global_t1 = max(all_t1)

        # Cache key based on the global window
        cache_key = f"argo_shared_{global_t0.value}_{global_t1.value}"
        zarr_path = str(cache_dir / f"{cache_key}.zarr")

        # ── 2. Cache hit? ─────────────────────────────────────────────────
        if Path(zarr_path).exists():
            try:
                probe = xr.open_zarr(zarr_path, consolidated=True)
                n = probe.sizes.get("obs", 0)
                if n > 0:
                    logger.info(
                        f"ARGO shared batch Zarr cache hit: "
                        f"{n:,} profiles ({global_t0} -> {global_t1})"
                    )
                    probe.close()
                    return zarr_path
                probe.close()
            except Exception:
                pass  # stale — rebuild

        # ── 3. Download all profiles in the global window ─────────────────
        t_start = _time_mod.time()
        n_unique_windows = len(
            set(f"{t0.value}_{t1.value}" for t0, t1 in zip(all_t0, all_t1, strict=False))
        )
        logger.info(
            f"ARGO shared batch prefetch: merging {n_unique_windows} "
            f"unique window(s) into [{global_t0} -> {global_t1}] "
            f"(single download pass)"
        )

        try:
            ds = self.argo_interface.open_time_window(
                start=global_t0,
                end=global_t1,
                depth_levels=self.depth_values,
                variables=(self.params.keep_variables or self.argo_interface.variables),
                master_index=self._master_index,
            )
        except Exception as exc:
            logger.error(f"ARGO shared batch download failed: {exc}")
            return None

        if ds is None:
            logger.warning("ARGO shared batch download returned None")
            return None

        obs_dim = "obs"
        n_obs = ds.sizes.get(obs_dim, 0)
        if n_obs == 0:
            logger.warning("ARGO shared batch download returned 0 profiles")
            return None

        # ── 4. Ensure data is sorted by TIME for searchsorted fast path ──
        time_name = "TIME"
        for candidate in ("TIME", "time", "JULD"):
            if candidate in ds.coords or candidate in ds.data_vars:
                time_name = candidate
                break

        # TIME may be a data_var (profile_refs path) or a coord (Kerchunk)
        if time_name in ds.coords:
            t_arr = np.asarray(ds.coords[time_name].values)
        elif time_name in ds.data_vars:
            t_arr = np.asarray(ds[time_name].values)
        else:
            logger.warning(
                f"ARGO shared batch: no time variable found among "
                f"coords={list(ds.coords)} data_vars={list(ds.data_vars)}"
            )
            # Write without sorting — downstream searchsorted will still
            # work but may be slightly less efficient.
            t_arr = None
        if t_arr is not None:
            if hasattr(t_arr, "compute"):
                t_arr = t_arr.compute()
            if not np.issubdtype(t_arr.dtype, np.datetime64):
                try:
                    t_arr = pd.to_datetime(t_arr).values
                except Exception:
                    pass

            if len(t_arr) > 1 and not bool(np.all(t_arr[:-1] <= t_arr[1:])):
                logger.info(f"ARGO shared batch: sorting {n_obs:,} profiles by time…")
                sort_idx = np.argsort(t_arr, kind="mergesort")
                ds = ds.isel({obs_dim: sort_idx})

        # ── 5. Materialise and write a single compact Zarr ───────────────
        # Only call .compute() if the data is backed by dask arrays;
        # the profile_refs path already loaded everything into NumPy via
        # .load(), so .compute() would just add scheduling overhead.
        _has_dask = any(hasattr(ds[v].data, "dask") for v in ds.variables)
        if _has_dask:
            ds = ds.compute(scheduler="synchronous")
        # Clean encoding from profile-level open_dataset to avoid
        # zarr write validation errors
        for var in ds.variables:
            ds[var].encoding.clear()
        ds.to_zarr(zarr_path, mode="w", consolidated=True)

        elapsed = _time_mod.time() - t_start
        logger.info(
            f"ARGO shared batch Zarr written in {elapsed:.1f}s: {n_obs:,} profiles -> {zarr_path}"
        )
        return zarr_path

    def prefetch_batch_shared_zarr_partitioned(
        self,
        time_bounds_list: List[Tuple[pd.Timestamp, pd.Timestamp]],
        cache_dir: Path,
    ) -> List[Dict[str, Any]]:
        """Pre-download ARGO profiles for a batch into one-or-more shared Zarr stores.

        This is a safer variant of :meth:`prefetch_batch_shared_zarr` for
        workloads where a batch may contain time windows far apart in time.
        Instead of merging the *entire* batch into one giant global window,
        it partitions requests by calendar month and merges only overlapping
        windows *within each month*.

        The returned partitions are designed to be consumed by the evaluator
        fast-path: each entry receives either a single Zarr path (same-month
        window) or a list of two paths (rare month-boundary window), then the
        worker filters by its exact ``time_bounds`` using ``np.searchsorted``.

        Returns
        -------
        list[dict]
            Each element is ``{"t0": Timestamp, "t1": Timestamp, "zarr_path": str}``.
            The time interval is the one used to build the Zarr store.
        """
        import time as _time_mod

        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Ensure master index is available (Kerchunk-only path)
        if self._master_index is None:
            try:
                self._load_master_index()
            except Exception:
                pass
        if self._master_index is None:
            self._try_auto_build_master_index()

        if not time_bounds_list:
            return []

        def _month_key(ts: pd.Timestamp) -> str:
            ts = pd.Timestamp(ts)
            return f"{ts.year:04d}_{ts.month:02d}"

        def _month_start(ts: pd.Timestamp) -> pd.Timestamp:
            ts = pd.Timestamp(ts)
            return pd.Timestamp(year=int(ts.year), month=int(ts.month), day=1)

        def _next_month_start(ms: pd.Timestamp) -> pd.Timestamp:
            return (pd.Timestamp(ms) + pd.offsets.MonthBegin(1)).normalize()

        def _iter_month_starts(t0: pd.Timestamp, t1: pd.Timestamp) -> List[pd.Timestamp]:
            t0 = pd.Timestamp(t0)
            t1 = pd.Timestamp(t1)
            if t1 < t0:
                t0, t1 = t1, t0
            cur = _month_start(t0)
            end = _month_start(t1)
            out: List[pd.Timestamp] = []
            while cur <= end:
                out.append(cur)
                cur = _next_month_start(cur)
            return out

        # 1) Assign each window to the month(s) it intersects (typically 1, rarely 2)
        month_windows: Dict[str, List[Tuple[pd.Timestamp, pd.Timestamp]]] = {}
        for raw_t0, raw_t1 in time_bounds_list:
            t0 = pd.Timestamp(raw_t0)
            t1 = pd.Timestamp(raw_t1)
            if t1 < t0:
                t0, t1 = t1, t0
            for ms in _iter_month_starts(t0, t1):
                me = _next_month_start(ms) - pd.Timedelta("1ns")
                clip0 = max(t0, ms)
                clip1 = min(t1, me)
                if clip1 < clip0:
                    continue
                month_windows.setdefault(_month_key(ms), []).append((clip0, clip1))

        # 2) Within each month, merge only overlapping windows (no gap-bridging)
        def _merge_overlaps(
            wins: List[Tuple[pd.Timestamp, pd.Timestamp]],
        ) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
            if not wins:
                return []
            wins_sorted = sorted(
                ((pd.Timestamp(a), pd.Timestamp(b)) for a, b in wins), key=lambda x: x[0]
            )
            merged: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
            cur0, cur1 = wins_sorted[0]
            for a, b in wins_sorted[1:]:
                if a <= cur1:
                    if b > cur1:
                        cur1 = b
                else:
                    merged.append((cur0, cur1))
                    cur0, cur1 = a, b
            merged.append((cur0, cur1))
            return merged

        partitions: List[Dict[str, Any]] = []
        for mkey, wins in month_windows.items():
            # Cache at full-month granularity: each calendar month is
            # downloaded exactly ONCE regardless of how many batches touch it
            # or which sub-window they request.  Workers always filter by
            # their exact time_bounds via searchsorted on the shared Zarr.
            zarr_path = str(cache_dir / f"argo_full_month_{mkey}.zarr")

            # Build the full-month boundaries from the first window's month
            # (all windows in `wins` share the same month key).
            _any_ts = pd.Timestamp(wins[0][0])
            month_t0 = _month_start(_any_ts)
            month_t1 = _next_month_start(month_t0) - pd.Timedelta("1ns")

            # Broadcast the representative window for the partition entries:
            # use the tightest union of all requested windows (workers still
            # slice to their own time_bounds, so this is just metadata).
            all_t0_in_month = min(t for t, _ in wins)
            all_t1_in_month = max(t for _, t in wins)

            # Cache hit? (full month already on disk)
            if Path(zarr_path).exists():
                try:
                    probe = xr.open_zarr(zarr_path, consolidated=True)
                    n = probe.sizes.get("obs", 0)
                    probe.close()
                    if n > 0:
                        logger.debug(
                            f"ARGO month cache hit: {mkey} "
                            f"({n:,} profiles already in {zarr_path})"
                        )
                        partitions.append(
                            {"t0": all_t0_in_month, "t1": all_t1_in_month, "zarr_path": zarr_path}
                        )
                        continue
                except Exception:
                    pass  # stale — rebuild

            t_start = _time_mod.time()
            logger.info(
                f"ARGO monthly shared prefetch: full month {mkey} "
                f"[{month_t0.date()} -> {month_t1.date()}]"
                f" (batch window: {all_t0_in_month.date()} -> {all_t1_in_month.date()})"
            )

            try:
                ds = self.argo_interface.open_time_window(
                    start=month_t0,
                    end=month_t1,
                    depth_levels=self.depth_values,
                    variables=(self.params.keep_variables or self.argo_interface.variables),
                    master_index=self._master_index,
                )
            except Exception as exc:
                logger.error(f"ARGO monthly shared download failed ({mkey}): {exc}")
                continue

            if ds is None or ds.sizes.get("obs", 0) == 0:
                continue

            # Ensure time-sorted for searchsorted fast path
            time_name = "TIME"
            for candidate in ("TIME", "time", "JULD"):
                if candidate in ds.coords or candidate in ds.data_vars:
                    time_name = candidate
                    break

            if time_name in ds.coords:
                t_arr = np.asarray(ds.coords[time_name].values)
            elif time_name in ds.data_vars:
                t_arr = np.asarray(ds[time_name].values)
            else:
                t_arr = None

            if t_arr is not None:
                if hasattr(t_arr, "compute"):
                    t_arr = t_arr.compute()
                if not np.issubdtype(t_arr.dtype, np.datetime64):
                    try:
                        t_arr = pd.to_datetime(t_arr).values
                    except Exception:
                        pass

                if len(t_arr) > 1 and not bool(np.all(t_arr[:-1] <= t_arr[1:])):
                    sort_idx = np.argsort(t_arr, kind="mergesort")
                    ds = ds.isel({"obs": sort_idx})

            # Materialise only if dask-backed
            _has_dask = any(hasattr(ds[v].data, "dask") for v in ds.variables)
            if _has_dask:
                ds = ds.compute(scheduler="synchronous")

            for var in ds.variables:
                ds[var].encoding.clear()
            ds.to_zarr(zarr_path, mode="w", consolidated=True)

            elapsed = _time_mod.time() - t_start
            n_profiles = ds.sizes.get("obs", 0)
            logger.debug(
                f"ARGO full month {mkey} written: {n_profiles:,} profiles "
                f"in {elapsed:.1f}s -> {zarr_path}"
            )
            partitions.append(
                {"t0": all_t0_in_month, "t1": all_t1_in_month, "zarr_path": zarr_path}
            )

        return partitions

    # --- Legacy per-window prefetch (kept for backward compatibility) ------

    def prefetch_batch_to_zarr(
        self,
        time_bounds_list: List[Tuple[pd.Timestamp, pd.Timestamp]],
        cache_dir: Path,
        n_outer_workers: int = 4,
    ) -> Dict[str, str]:
        """Pre-download ARGO time windows to local Zarr files before dispatch.

        .. deprecated::
            Use :meth:`prefetch_batch_shared_zarr` instead — it merges all
            windows into a single download pass and a single shared Zarr.

        Downloads every distinct time-window needed by a batch on the driver
        (using ArgoInterface's batch HTTP session pooling), then materialises
        the data to local Zarr files.  Dask workers subsequently read from
        local storage instead of blocking on GDAC HTTP requests.

        Args:
            time_bounds_list: List of (start, end) pd.Timestamp tuples, one
                per batch entry.  Duplicate windows are downloaded once.
            cache_dir: Local directory for temporary Zarr files (created if
                necessary).  Files persist across batches so the same window
                is never downloaded twice.
            n_outer_workers: Number of windows fetched in parallel.
                Default 4 — safe because the batch download uses HTTP session
                pooling internally (no per-profile connection overhead).

        Returns:
            Dict mapping canonical window key ``"{t0.value}_{t1.value}"`` to
            the absolute local Zarr path (str).  Windows that failed to
            download are absent from the mapping.
        """
        import time
        from concurrent.futures import ThreadPoolExecutor as _TPE, as_completed as _ac

        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Deduplicate while preserving insertion order
        unique_windows: Dict[str, Tuple[pd.Timestamp, pd.Timestamp]] = {}
        for t0, t1 in time_bounds_list:
            ts0, ts1 = pd.Timestamp(t0), pd.Timestamp(t1)
            key = f"{ts0.value}_{ts1.value}"
            unique_windows[key] = (ts0, ts1)

        result: Dict[str, str] = {}
        # Batch statistics counters
        _stats = {"cache_hit": 0, "downloaded": 0, "empty": 0, "failed": 0}
        _failure_details: List[str] = []

        def _fetch_one(key: str, t0: pd.Timestamp, t1: pd.Timestamp) -> Tuple[str, str]:
            zarr_path = str(cache_dir / f"argo_w_{key}.zarr")
            if Path(zarr_path).exists():
                _stats["cache_hit"] += 1
                return key, zarr_path
            try:
                ds = self.argo_interface.open_time_window(
                    start=t0,
                    end=t1,
                    depth_levels=self.depth_values,
                    variables=(self.params.keep_variables or self.argo_interface.variables),
                    master_index=self._master_index,
                )
                if ds is not None:
                    n_pts = ds.sizes.get("N_POINTS", ds.sizes.get("obs", 0))
                    if n_pts > 0:
                        # Materialise lazy arrays before writing
                        ds = ds.compute(scheduler="synchronous")
                        ds.to_zarr(zarr_path, mode="w", consolidated=True)
                        _stats["downloaded"] += 1
                        return key, zarr_path
                _stats["empty"] += 1
            except Exception as exc:
                _stats["failed"] += 1
                _failure_details.append(f"[{t0} -> {t1}]: {exc}")
            return key, ""

        logger.info(f"ARGO batch prefetch: {len(unique_windows)} window(s) -> {cache_dir}")
        t_start = time.time()

        if len(unique_windows) <= 1 or n_outer_workers == 1:
            # Sequential: safe, each window already parallelised internally
            for wkey, (ts0, ts1) in unique_windows.items():
                _k, _path = _fetch_one(wkey, ts0, ts1)
                if _path:
                    result[_k] = _path
        else:
            with _TPE(max_workers=n_outer_workers) as ex:
                futs = {
                    ex.submit(_fetch_one, wkey, ts0, ts1): wkey
                    for wkey, (ts0, ts1) in unique_windows.items()
                }
                for fut in _ac(futs):
                    try:
                        _k, _path = fut.result()
                        if _path:
                            result[_k] = _path
                    except Exception as exc:
                        logger.warning(f"ARGO prefetch future error: {exc}")

        elapsed = time.time() - t_start
        logger.info(
            f"ARGO batch prefetch done in {elapsed:.1f}s — "
            f"{len(result)}/{len(unique_windows)} windows OK "
            f"(cache_hit={_stats['cache_hit']}, "
            f"downloaded={_stats['downloaded']}, "
            f"empty={_stats['empty']}, "
            f"failed={_stats['failed']})"
        )
        for detail in _failure_details:
            logger.warning(f"ARGO prefetch failure {detail}")
        return result

    @classmethod
    def supports(cls, path: str) -> bool:
        """
        Check if path is supported by ARGO manager.

        Since ARGO uses monthly indexing rather than traditional file paths,
        this accepts any path when explicitly specified.

        Returns:
            bool: Always True (manager selected via configuration).
        """
        return True

    def list_files(self) -> List[str]:
        """
        List available monthly index keys.

        Returns:
            List[str]: List of month keys (YYYY-MM) from master index,
                      or empty list if index not loaded.
        """
        if self._master_index is None:
            try:
                self._load_master_index()
            except Exception as exc:
                logger.warning(f"Could not load ARGO master index: {exc}")

        if self._master_index is None:
            logger.warning("Master index not loaded, cannot list files")
            return []

        month_keys = sorted(self._master_index.keys())
        logger.debug(f"ARGO master index contains {len(month_keys)} months")
        return month_keys

    def get_global_metadata(self) -> Dict[str, Any]:
        """
        Get global metadata for ARGO dataset.

        Harmonized with the generic connection manager path:
        - build a CoordinateSystem from a real ARGO sample when possible
        - detect semantic variable mappings (`variables_dict`)
        - expose inverse mapping (`variables_rename_dict`)

        Falls back to robust defaults when no sample can be opened.

        Returns:
            Dict[str, Any]: Global metadata for ARGO.
        """
        if self._global_metadata is not None:
            return self._global_metadata

        keep_variables = (
            self.params.keep_variables
            or getattr(self.argo_interface, "variables", None)
            or ["TEMP", "PSAL", "PRES"]
        )

        fallback_coord_sys = CoordinateSystem(
            coord_type="geographic",
            coord_level="L2",
            coordinates={
                "lat": "LATITUDE",
                "lon": "LONGITUDE",
                "time": "TIME",
                "depth": "DEPTH",
                "n_points": "N_POINTS",
            },
            crs="EPSG:4326",
        )

        # Build a narrow time window to sample only a few profiles for
        # metadata extraction.  Loading an entire month (10 000+ profiles)
        # just to discover variable names / coordinate system is extremely
        # slow and wasteful.
        sample_time_window: Optional[tuple] = None

        # Prefer an already built profile-level catalog when available.
        if self._catalog:
            first_entry = self._catalog[0]
            # Use a narrow 1-hour window from the start of the first month
            sample_start = pd.Timestamp(first_entry.date_start)
            sample_end = sample_start + pd.Timedelta(hours=1)
            sample_time_window = (sample_start, sample_end)

        # Otherwise try monthly keys from master index.
        if sample_time_window is None:
            if self._master_index is None:
                try:
                    self._load_master_index()
                except Exception as exc:
                    logger.warning(
                        f"Could not load ARGO master index for metadata extraction: {exc}"
                    )
            if self._master_index:
                first_key = sorted(self._master_index.keys())[0]
                info = self._master_index[first_key]
                sample_start = pd.Timestamp(info["start"], unit="ns")
                sample_end = sample_start + pd.Timedelta(hours=1)
                sample_time_window = (sample_start, sample_end)

        # Last option: open requested time window directly (narrow).
        if sample_time_window is None and self.start_time is not None and self.end_time is not None:
            sample_start = pd.Timestamp(self.start_time)
            sample_end = sample_start + pd.Timedelta(hours=1)
            sample_time_window = (sample_start, sample_end)

        coord_sys = fallback_coord_sys
        dict_resolution: Dict[str, Any] = {}
        variables: Dict[str, Any] = {}

        sample_ds: Optional[xr.Dataset] = None
        if sample_time_window is not None:
            try:
                logger.debug(
                    f"ARGO metadata: sampling narrow window "
                    f"{sample_time_window[0]} -> {sample_time_window[1]} "
                    f"(max 10 profiles)"
                )
                sample_ds = self.argo_interface.open_time_window(
                    start=sample_time_window[0],
                    end=sample_time_window[1],
                    depth_levels=None,  # skip depth interp for metadata
                    variables=keep_variables,
                    master_index=self._master_index,
                    max_profiles=10,
                )
            except Exception as exc:
                logger.warning(
                    f"Could not open ARGO sample for global metadata ({sample_time_window}): {exc}"
                )

        if sample_ds is not None:
            # Do NOT use ``with sample_ds as ds:`` — it closes the dataset
            # and can trigger eager materialisation of dask arrays.
            ds = sample_ds
            try:
                coord_sys = CoordinateSystem.get_coordinate_system(ds)
                # estimate_resolution calls .values on coords which is cheap
                # for in-memory data (profile_refs path loads without dask).
                dict_resolution = self.estimate_resolution(ds, coord_sys)

                for requested_var in keep_variables:
                    resolved_var = None
                    if requested_var in ds.variables:
                        resolved_var = requested_var
                    else:
                        requested_var_lower = requested_var.lower()
                        for ds_var in ds.variables:
                            if str(ds_var).lower() == requested_var_lower:
                                resolved_var = ds_var
                                break
                    if resolved_var is None:
                        continue

                    var = ds[resolved_var]
                    variables[resolved_var] = {
                        "dims": list(var.dims),
                        "std_name": var.attrs.get("standard_name", ""),
                    }
            finally:
                ds.close()

        if not variables:
            variables = {
                var_name: {"dims": ["N_POINTS"], "std_name": ""} for var_name in keep_variables
            }

        variables_dict = CoordinateSystem.detect_oceanographic_variables(variables)
        variables_rename_dict = {v: k for k, v in variables_dict.items() if v is not None}

        default_metadata = {
            "variables": variables,
            "variables_dict": variables_dict,
            "variables_rename_dict": variables_rename_dict,
            "resolution": dict_resolution,
            "coord_system": coord_sys,
            "keep_variables": keep_variables,
        }
        self._global_metadata = default_metadata
        return self._global_metadata


CONNECTION_CONFIG_REGISTRY: Dict[str, Any] = {
    "argo": ARGOConnectionConfig,
    "cmems": CMEMSConnectionConfig,
    "ftp": FTPConnectionConfig,
    "glonet": GlonetConnectionConfig,
    "local": LocalConnectionConfig,
    "s3": S3ConnectionConfig,
    "wasabi": WasabiS3ConnectionConfig,
}

CONNECTION_MANAGER_REGISTRY: Dict[str, Any] = {
    "argo": ArgoManager,
    "cmems": CMEMSManager,
    "ftp": FTPManager,
    "glonet": GlonetManager,
    "local": LocalConnectionManager,
    "s3": S3Manager,
    "wasabi": S3WasabiManager,
}


def prefetch_obs_files_to_local(
    remote_paths: List[str],
    cache_dir: str,
    fs: Any,
    ref_alias: str = "",
    show_progress_bar: bool = True,
) -> Dict[str, str]:
    """
    Pre-download observation files to local disk before worker dispatch.

    Downloads all remote files (`.zarr` directories or `.nc` single files)
    to *cache_dir* so that dask workers can open them locally instead of
    issuing concurrent S3 requests.

    A single tqdm progress bar tracks overall download progress.

    Args:
        remote_paths: List of remote S3 paths (e.g. ``s3://bucket/file.zarr``).
        cache_dir: Local directory where files will be stored.
        fs: An fsspec-compatible filesystem handle (e.g. ``s3fs.S3FileSystem``).
        ref_alias: Name of the observation dataset (for logging / bar label).

    Returns:
        Dict mapping each remote path to its local path on disk.
    """
    from tqdm import tqdm as _tqdm
    import shutil as _shutil

    os.makedirs(cache_dir, exist_ok=True)
    path_map: Dict[str, str] = {}

    # De-duplicate while preserving order
    unique_paths = list(dict.fromkeys(remote_paths))

    if not unique_paths:
        return path_map

    _stats = {"cached": 0, "downloaded": 0, "failed": 0}
    _failures: List[str] = []

    _label = (
        f"Downloading observation files — {ref_alias}"
        if ref_alias
        else "Downloading observation files"
    )
    _bar = _tqdm(
        total=len(unique_paths),
        desc=_label,
        unit="file",
        leave=True,
        dynamic_ncols=True,
        disable=True,  # Always suppress download bar unless explicitly enabled
    )

    import threading as _dl_threading

    _lock = _dl_threading.Lock()

    def _download_one(rpath: str) -> None:
        """Download a single file (thread-safe)."""
        filename = Path(rpath).name
        local_path = os.path.join(cache_dir, filename)
        extension = Path(rpath).suffix

        try:
            if extension == ".zarr":
                if os.path.isdir(local_path) and os.listdir(local_path):
                    # Consolidate metadata if missing (legacy cached files)
                    _zmetadata = os.path.join(local_path, ".zmetadata")
                    if not os.path.isfile(_zmetadata):
                        try:
                            import zarr as _zarr_consolidate

                            _zarr_consolidate.consolidate_metadata(local_path)
                        except Exception:
                            pass  # non-critical
                    with _lock:
                        path_map[rpath] = local_path
                        _stats["cached"] += 1
                        _bar.update(1)
                    return
                s3_key = rpath
                if s3_key.startswith("s3://"):
                    s3_key = s3_key[5:]
                tmp_path = local_path + f".downloading.{_dl_threading.current_thread().ident}"
                if os.path.isdir(tmp_path):
                    _shutil.rmtree(tmp_path, ignore_errors=True)
                fs.get(s3_key, tmp_path, recursive=True)
                _items = os.listdir(tmp_path)
                if (
                    len(_items) == 1
                    and os.path.isdir(os.path.join(tmp_path, _items[0]))
                    and not any(f.startswith(".z") for f in _items)
                ):
                    _nested = os.path.join(tmp_path, _items[0])
                    _unwrap = tmp_path + "_unwrap"
                    os.rename(_nested, _unwrap)
                    _shutil.rmtree(tmp_path, ignore_errors=True)
                    os.rename(_unwrap, tmp_path)
                if os.path.isdir(local_path):
                    _shutil.rmtree(local_path, ignore_errors=True)
                try:
                    os.rename(tmp_path, local_path)
                except OSError as _rename_err:
                    import errno as _errno_mod
                    if _rename_err.errno == _errno_mod.ENOTEMPTY and os.path.isdir(local_path):
                        # Race condition: another thread cached this zarr first.
                        # The existing directory is valid; discard our temp copy.
                        _shutil.rmtree(tmp_path, ignore_errors=True)
                    else:
                        raise
                # Consolidate Zarr metadata so workers can open with
                # consolidated=True (single .zmetadata read vs. hundreds
                # of small files).  This prevents I/O storms when 6+
                # workers open the same stores concurrently.
                _zmetadata = os.path.join(local_path, ".zmetadata")
                if not os.path.isfile(_zmetadata):
                    try:
                        import zarr as _zarr_consolidate

                        _zarr_consolidate.consolidate_metadata(local_path)
                    except Exception:
                        pass  # non-critical
                with _lock:
                    path_map[rpath] = local_path
                    _stats["downloaded"] += 1
            else:
                if os.path.isfile(local_path):
                    with _lock:
                        path_map[rpath] = local_path
                        _stats["cached"] += 1
                        _bar.update(1)
                    return
                tmp_path = local_path + f".tmp.{_dl_threading.current_thread().ident}"
                with fs.open(rpath, "rb") as remote_file:
                    with open(tmp_path, "wb") as local_file:
                        local_file.write(remote_file.read())
                os.rename(tmp_path, local_path)
                with _lock:
                    path_map[rpath] = local_path
                    _stats["downloaded"] += 1
        except Exception as exc:
            with _lock:
                _stats["failed"] += 1
                _failures.append(f"{filename}: {exc!r}")
            logger.debug(f"Prefetch failed for {rpath}: {exc!r}")

        with _lock:
            _bar.update(1)

    # ── Parallel downloads (4 concurrent connections) ─────────────
    from concurrent.futures import ThreadPoolExecutor as _DlPool

    _MAX_DL_WORKERS = min(8, len(unique_paths))
    with _DlPool(max_workers=_MAX_DL_WORKERS) as pool:
        list(pool.map(_download_one, unique_paths))

    _bar.close()

    # Summary log
    parts = []
    if _stats["cached"]:
        parts.append(f"{_stats['cached']} cached")
    if _stats["downloaded"]:
        parts.append(f"{_stats['downloaded']} downloaded")
    if _stats["failed"]:
        parts.append(f"{_stats['failed']} FAILED")
    logger.debug(f"Prefetch {ref_alias}: {' | '.join(parts)}")
    if _failures:
        for detail in _failures[:5]:
            logger.warning(f"  Prefetch failure: {detail}")
        if len(_failures) > 5:
            logger.warning(f"  ... and {len(_failures) - 5} more failures")

    return path_map


def create_worker_connect_config(config: Any, argo_index: Optional[Any] = None) -> Callable:
    """Creates connection configurations for predictive and reference sources."""
    protocol = config.protocol

    if protocol == "cmems":
        if hasattr(config, "fs") and hasattr(config.fs, "_session"):
            try:
                if hasattr(config.fs._session, "close"):
                    config.fs._session.close()
            except Exception:
                pass
            config.fs = None

    config.dataset_processor = None

    # Recreate reading object in the worker
    config_cls = CONNECTION_CONFIG_REGISTRY[protocol]
    connection_cls = CONNECTION_MANAGER_REGISTRY[protocol]
    delattr(config, "protocol")
    config = config_cls(vars(config))

    # remove fsspec handler 'fs' from Config, otherwise: serialization
    if protocol == "cmems":
        if hasattr(config.params, "fs") and hasattr(config.params.fs, "_session"):
            try:
                if hasattr(config.params.fs._session, "close"):
                    config.params.fs._session.close()
            except Exception:
                pass
            config.params.fs = None

    if protocol == "cmems":
        connection_manager = connection_cls(
            config,
            call_list_files=False,
            do_logging=True,
        )
    elif protocol == "argo":
        connection_manager = connection_cls(
            config,
            argo_index=argo_index,
            call_list_files=False,
        )
    else:
        connection_manager = connection_cls(config, call_list_files=False)
    open_func: Callable[..., Any] = connection_manager.open

    return open_func
