
from abc import ABC, abstractmethod
from argparse import Namespace
import os
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple, Union
import warnings

import boto3
from botocore import UNSIGNED
from botocore.config import Config
import numpy as np

# from torchgeo.datasets import Landsat
import xarray as xr

from dctools.dcio.loader import FileLoader
from dctools.dcio.saver import DataSaver
from dctools.processing.cmems_data import create_glorys_ndays_forecast
from dctools.processing.gridded_data import GriddedDataProcessor
from dctools.utilities.errors import DCExceptionHandler
from dctools.utilities.file_utils import  get_list_filter_files, remove_listof_files
from dctools.utilities.misc_utils import get_dates_from_startdate
from dctools.utilities.net_utils import download_s3_file, S3Url, CMEMSManager
from dctools.utilities.xarray_utils import rename_coordinates, rename_variables, DICT_RENAME_CMEMS

class DCDataset(ABC):
    """Data challenge custom dataset."""

    def __init__(
        self,
        conf_args: Namespace,
        root_data_dir: str,
        transform_fct: Optional[Callable[[xr.Dataset], xr.Dataset]] = None,
        save_after_preprocess: bool = False,
        lazy_load: bool = True,
        file_format: Optional[str] = 'netcdf',
    ):
        """
        Arguments:
            list_files (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
                on a sample.
        """
        self.args = conf_args
        self.root_dir = root_data_dir
        self.transform_fct = transform_fct
        self.dclogger = conf_args.dclogger
        self.exception_handler = conf_args.exception_handler
        self.save_after_preprocess = save_after_preprocess
        self.lazy_load = lazy_load
        self.file_format = file_format

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def get_data(self, index: int):
        pass

    @abstractmethod
    def get_date(self, index: int):
        pass

    #@abstractmethod
    #def get_labels(self, index: int):
    #    pass

    def preprocess_data(self, dataset):
        if self.transform_fct is not None:
            # preprocess dataset
            dataset = self.transform_fct(dataset)
        return dataset

    def __getitem__(self, index: int) -> Optional[xr.Dataset]:
        if index < self.__len__():
            dataset = self.get_data(index)
            # labels = self.get_labels(index)
            dataset = self.preprocess_data(dataset)
            if self.save_after_preprocess:
                file_extension = '.nc' if self.file_format == 'netcdf' else '.zarr'
                filename = self.get_date(index) + file_extension
                dataset_filepath = os.path.join(self.root_dir, filename)
                if not os.path.isfile(dataset_filepath):
                    self.args.dclogger.info(f"Save dataset to file: {dataset_filepath}")
                    DataSaver.save_dataset(
                        dataset,
                        dataset_filepath,
                        self.exception_handler,
                        self.dclogger,
                        file_format=self.file_format,
                    )
            return dataset
        else:
            raise(IndexError)


class DCEmptyDataset(DCDataset):
    """Empty dataset."""

    def __init__(
        self,
        conf_args: Namespace,
        root_data_dir: str,
        list_dates: List[str],
        transform_fct: Optional[Callable[[xr.Dataset], xr.Dataset]] = None,
        save_after_preprocess: bool = False,
        lazy_load: bool = True,
        file_format: Optional[str] = 'netcdf',
    ):
        super().__init__(
            conf_args, root_data_dir,
            transform_fct, save_after_preprocess,
            lazy_load, file_format,
        )
        self.list_dates = list_dates

    def __len__(self):
        return 0

    def get_data(self, index: int):
        return None

    def get_date(self, index: int):
        return(self.list_dates[index])



class CmemsDataset(DCDataset):
    """Class to manage data from Copernicus Marine service."""
    def __init__(
        self,
        conf_args: Namespace,
        root_data_dir: str,
        cmems_product_name: str,
        list_dates: List[str],
        transform_fct: Optional[Callable[[xr.Dataset], xr.Dataset]] = None,
        save_after_preprocess: bool = False,
        lazy_load: bool = True,
        file_format: Optional[str] = 'netcdf',
    ):
        super().__init__(
            conf_args, root_data_dir,
            transform_fct, save_after_preprocess,
            lazy_load, file_format,
        )
        self.cmems_manager = CMEMSManager(
            conf_args.dclogger, conf_args.exception_handler
        )
        self.root_data_dir = root_data_dir
        self.list_dates = list_dates
        self.logged: bool = False
        self.cmems_product_name = cmems_product_name

    def get_labels(self, index: int):
        labels = None
        return labels

    def get_date(self, index: int) -> str:
        return self.list_dates[index]

    def __len__(self):
        return len(self.list_dates)

    def get_cmems_sample(self, date: str) -> List[str]:
        name_filter = self.cmems_manager.get_cmems_filter_from_date(
            date
        )
        if not self.logged:
            self.cmems_manager.cmems_login()
        self.download_file(name_filter)


    def download_file(self, name_filter: str):
        self.cmems_manager.cmems_download(
            product_id=self.cmems_product_name,
            output_dir=self.root_data_dir,
            name_filter=name_filter,
            tmp_dir=self.root_data_dir,
        )

class CmemsGlorysDataset(CmemsDataset):
    """Class to manage data from Copernicus Marine service."""
    def __init__(
        self,
        conf_args: Namespace,
        root_data_dir: str,
        cmems_product_name: str,
        cmems_file_prefix: str,
        list_dates: List[str],
        transform_fct: Optional[Callable[[xr.Dataset], xr.Dataset]] = None,
        file_extension: str = '.zarr',
        save_after_preprocess: bool = False,
        lazy_load: bool = True,
        file_format: Optional[str] = 'zarr',
    ):
        super().__init__(
            conf_args, root_data_dir,
            cmems_product_name, list_dates,
            transform_fct=None,
            save_after_preprocess=save_after_preprocess,
            lazy_load=lazy_load,
            file_format=file_format,
        )
        self.args = conf_args
        self.root_data_dir = root_data_dir
        self.cmems_file_prefix = cmems_file_prefix
        self.file_format = file_format
        self.file_extension = file_extension
        self.preprocess_files = transform_fct

    def __len__(self):
        return len(self.list_dates)

    def get_labels(self, index: int):
        labels = None
        return labels
 
    def get_data(
        self, index: int,
    ) -> xr.Dataset:
            self.args.dclogger.info(
                f"Get data for date: {self.list_dates[index]}"
            )
            # get the start date of the forecast
            start_date = self.list_dates[index]
            list_dates = get_dates_from_startdate(
                start_date, self.args.glonet_n_days_forecast
            )
            first_date = list_dates[0]
            glorys_filename = first_date + self.file_extension
            glorys_filepath = os.path.join(self.root_data_dir, glorys_filename)
            self.args.dclogger.info(f"glorys_filepath0: {glorys_filepath}")
            if not (os.path.exists(glorys_filepath)):
                list_mercator_files = get_list_filter_files(
                    self.root_data_dir,
                    extension='.nc',
                    regex=self.cmems_file_prefix,
                    prefix=True,
                )
                if len(list_mercator_files) != self.args.glonet_n_days_forecast:
                    for date in list_dates:
                        self.get_cmems_sample(date)
                    list_mercator_files = get_list_filter_files(
                        self.root_data_dir,
                        extension='.nc',
                        regex=self.cmems_file_prefix,
                        prefix=True,
                    )
                self.args.dclogger.info(f"glorys_filepath1: {glorys_filepath}")
                glorys_data = create_glorys_ndays_forecast(
                    nc_path=self.root_data_dir,
                    list_nc_files=list_mercator_files,
                    start_date=first_date,
                    zarr_path=glorys_filepath,
                    transform_fct=self.preprocess_files,
                    dclogger=self.dclogger,
                    exception_handler=self.exception_handler
                )
                # remove downloaded files
                list_mercator_files = get_list_filter_files(
                    self.root_data_dir,
                    extension='.nc',
                    regex=self.cmems_file_prefix,
                    prefix=True,
                )
                """if len(list_mercator_files) > 0:
                    self.dclogger.info("Remove temporary Mercator files.")
                    remove_listof_files(
                        list_mercator_files, self.root_data_dir, self.exception_handler
                    )"""
                """list_tmp_files = get_list_filter_files(
                    self.root_data_dir,
                    extension='.txt',
                    regex="files_to_download",
                    prefix=True,
                )
                if (len(list_tmp_files) > 0):
                    remove_listof_files(
                        list_tmp_files, self.root_data_dir, self.exception_handler
                    )"""

                return glorys_data
            else:
                if self.lazy_load:
                    glorys_data = FileLoader.lazy_load_dataset(
                        glorys_filepath, self.exception_handler,
                        self.dclogger,
                    )
                else:
                    glorys_data = FileLoader.load_dataset(
                        glorys_filepath, self.exception_handler,
                        self.dclogger,
                    )
                return glorys_data


class GlonetDataset(DCDataset):
    """Class to manage forecasts from Glonet models."""
    def __init__(
        self,
        conf_args: Namespace,
        root_data_dir: str,
        list_dates: List[str],
        transform_fct: Optional[Callable[[xr.Dataset], xr.Dataset]] = None,
        save_after_preprocess: bool = False,
        lazy_load: bool = True,
        file_format: Optional[str] = 'netcdf',
    ):
        super().__init__(
            conf_args, root_data_dir,
            transform_fct, save_after_preprocess,
            lazy_load, file_format,
        )
        self.list_dates = list_dates
        self.args = conf_args
        self.s3_client = boto3.client(
            "s3",
            config=Config(signature_version=UNSIGNED),
            endpoint_url=conf_args.glonet_base_url,
        )

    def __len__(self):
        return len(self.list_dates)

    def get_labels(self, index: int):
        labels = None
        return labels

    def get_date(self, index: int) -> str:
        return self.list_dates[index]

    def get_data(self, index: int):
        """Download glonet forecast file from Edito.

        Args:
            filename (str): name of the file to download.
        """
        self.args.dclogger.info(
            f"Get data for date: {self.list_dates[index]}"
        )
        # get the start date of the forecast
        start_date = self.list_dates[index]
        glonet_filename = start_date + '.nc'
        local_file_path = os.path.join(self.args.glonet_data_dir, glonet_filename)
        """print(f"start_date: {start_date}")
        print(f"local_file_path: {local_file_path}")
        print(f"glonet_filename: {glonet_filename}")
        print(f"self.list_dates: {self.list_dates}")"""
        glonet_s3_filepath = os.path.join(
            self.args.s3_glonet_folder,
            glonet_filename
        )
        if not (Path(local_file_path).is_file()):
            download_s3_file(
                s3_client=self.s3_client,
                bucket_name=self.args.glonet_s3_bucket,
                file_name=glonet_s3_filepath,
                local_file_path=local_file_path,
                dclogger=self.args.dclogger,
                exception_handler=self.args.exception_handler,
            )
        assert(Path(local_file_path).is_file())
        if self.lazy_load:
            glonet_data = FileLoader.lazy_load_dataset(
                local_file_path, self.args.exception_handler,
                self.args.dclogger,
            )
        else:
            glonet_data = FileLoader.load_dataset(
                local_file_path, self.args.exception_handler,
                self.args.dclogger,
            )
        return glonet_data
      
class MODISLeadmapDataset(DCDataset):
    """
    Class to manage MODIS Leadmaps (for sea-ice monitoring).
    """
    def __init__(
        self,
        conf_args: Namespace,
        list_dates: List[str | np.datetime64],
        root_data_dir: str | None = None,
        transform_fct: Optional[Callable[[xr.Dataset], xr.Dataset]] = None,
        save_after_preprocess: bool = False,
        lazy_load: bool = True,
        file_format: Optional[str] = 'netcdf',
    ):
        """
        Init method for the MODISLeadmapDataset class.

        Parameters
        ----------
        conf_args : Namespace
            Configuration arguments coming from either the command line or the
            specific DC's YAML configuration file.
        list_dates : List[str  |  np.datetime64]
            List of dates to use as samples.
        root_data_dir : str, optional
            If specified, directory from which to recover the data. If not,
            retrieves data from S3 bucket (not yet implemented).
        transform_fct : Callable[[xr.Dataset], xr.Dataset], optional
            Function to call on the data during preprocessing.
        save_after_preprocess : bool, optional
            Controls whether to save the results of preprocessing to the disk,
            by default False.
        lazy_load : bool, optional
            Whether to load the data to memory instantly or to wait until it is
            necessary for computation, by default True.
        file_format : str, optional
            Format in which to save the preprocessed samples, by default 'netcdf'.
        min_latitude, max_latitude, min_longitude, max_longitude : float, optional
            Bounds of the horizontal region for which to retrieve data. 
        """

        if root_data_dir is None:
            raise NotImplementedError(
                "Not specifying root_data_dir is not yet supported."
                )
        
        super().__init__(
            conf_args, root_data_dir,
            transform_fct, save_after_preprocess,
            lazy_load, file_format,
        )
        if isinstance(list_dates[0], np.datetime64):
            self.list_dates = list_dates
        else:
            self.list_dates = [np.datetime64(date, "D") for date in list_dates]

        _date_array = np.array(self.list_dates)
        # Check that dates are valid and drop them if not
        months = _date_array.astype('datetime64[M]').astype(int) % 12 + 1
        valid_mask = (months >= 11) | (months <= 4) # Data from November to April
        self.list_dates = list(_date_array[valid_mask])

        if np.any(~valid_mask):
            warnings.warn(
                "Dates outside the data range (November to April) were dropped."
                )
            
        self.data_cache = {}

    
    def __len__(self):
        return len(self.list_dates)

    def get_data(self, index: int):
        date = self.get_date(index)
        
        # Open corresponding file (and put it in cache)
        winter_str = self.get_winter_string(date)
        try:
            winter_ds = self.data_cache[winter_str]
        except KeyError:
            data_path = Path(self.root_dir) / f"{winter_str}_ArcLeads.nc"
            try:
                winter_ds = xr.open_dataset(data_path)
            except FileNotFoundError:
                print(
                    f"File {data_path} not found. " \
                    "Check that the specified year is not in the dataset."
                    )

            self.data_cache[winter_str] = winter_ds

        return winter_ds.sel(
            time=date,
            method="nearest",
            tolerance=np.timedelta64(24, "h"),
            )
        

    def get_date(self, index: int) -> np.datetime64:
        return self.list_dates[index]
    
    @staticmethod
    def get_winter_string(date:np.datetime64) -> str:
        """
        Get a string representing the winter season the date belongs to.

        Useful since MODIS Leadmaps use things like \"202021\" to specify years.
        """
        year = date.astype('datetime64[Y]').astype(int) + 1970
        month = date.astype('datetime64[M]').astype(int) % 12 + 1

        if 11 <= month <= 12:
            winter_start = year
        elif 1 <= month <= 4:
            winter_start = year - 1
        else:
            raise ValueError(f"Date {date} is not in the Northern Hemisphere winter (Nov to Apr)")

        winter_end = winter_start + 1
        return f"{winter_start}{winter_end % 100:02d}"