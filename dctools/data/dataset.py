
from abc import ABC, abstractmethod
from argparse import Namespace
import os
from pathlib import Path
from typing import (
    Any, Callable, Dict, Iterable, List,
    Mapping, Optional, Tuple, Union
)
# import dask

# from torchgeo.datasets import Landsat
import xarray as xr

from dctools.dcio.loader import FileLoader
from dctools.dcio.saver import DataSaver
from dctools.processing.cmems_data import create_glorys_ndays_forecast
# from dctools.processing.gridded_data import GriddedDataProcessor
# from dctools.utilities.errors import DCExceptionHandler
from dctools.utilities.file_utils import  get_list_filter_files #, remove_listof_files
from dctools.utilities.misc_utils import get_dates_from_startdate
from dctools.utilities.net_utils import CMEMSManager, FTPManager, S3Manager
from dctools.utilities.xarray_utils import get_time_info #rename_coordinates, rename_variables, DICT_RENAME_CMEMS

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
            root_data_dir (string): Directory with all the images.
                on a sample.
        """
        self.args = conf_args
        self.root_data_dir = root_data_dir
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
                dataset_filepath = os.path.join(self.root_data_dir, filename)
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
            self.close_all()
            raise(IndexError)

    @abstractmethod
    def close_all(self):
        pass


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

    def close_all(self):
        pass

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

    def close_all(self):
        self.cmems_manager.cmems_logout()

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

class S3Dataset(DCDataset):
    """Class to manage datasets in S3 storage."""
    def __init__(
        self,
        conf_args: Namespace,
        root_data_dir: str,
        list_dates: List[str],
        s3_url: str,
        s3_access_key: str,
        s3_bucket: Optional[str] = None,
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
        self.s3_bucket = s3_bucket
        self.s3_manager = S3Manager(
            conf_args.dclogger,
            conf_args.exception_handler,
            s3_url=s3_url,
            s3_access_key=s3_access_key,
            bucket_name=None,
        )
    def __len__(self):
        return len(self.list_dates)

    def get_labels(self, index: int):
        labels = None
        return labels

    def get_date(self, index: int) -> str:
        return self.list_dates[index]

    @abstractmethod
    def get_data(self, index: int):
        """Download glonet forecast file from Edito.

        Args:
            index (int): index of the date to download.
        """
        pass

    def upload_data(self, filepath: str, dest_bucket: str, dest_key: str) -> None:
        """Upload."""
        self.s3_manager.upload_file(self, file_path=filepath, s3_key=dest_key, bucket_name=dest_bucket)

    def close_all(self):
        self.args.dclogger.info("Close S3 client.")
        self.s3_client.close()


"""import fsspec
import xarray as xr

s3_mapper = fsspec.get_mapper(
            "s3://ppr-ocean-climat/DC3/IABP/LEVEL1_2023.zarr",
            client_kwargs = {
                "aws_access_key_id": <ta clé>,
                "aws_secret_access_key": <ta clé secrète>,
                "endpoint_url": "https://s3.eu-west-2.wasabisys.com",
                }
            )

xr.open_dataset(
    s3_mapper,
    engine="zarr"
    )"""

class GlonetDataset(S3Dataset):
    """Class to manage forecasts from Glonet models."""
    def __init__(
        self,
        conf_args: Namespace,
        root_data_dir: str,
        list_dates: List[str],
        s3_url: str,
        s3_access_key: str,
        s3_bucket: Optional[str] = None,
        s3_folder: Optional[str] = None,
        transform_fct: Optional[Callable[[xr.Dataset], xr.Dataset]] = None,
        save_after_preprocess: bool = False,
        lazy_load: bool = True,
        file_format: Optional[str] = 'netcdf',
    ):
        super().__init__(
            conf_args, root_data_dir, list_dates,
            s3_url,s3_access_key, s3_bucket,
            transform_fct, save_after_preprocess,
            lazy_load, file_format,
        )
        self.s3_folder = s3_folder

    def get_data(self, index: int):
        """Download glonet forecast file from Edito.

        Args:
            index (int): index of the date to download.
        """
        self.args.dclogger.info(
            f"Get data for date: {self.list_dates[index]}"
        )
        # get the start date of the forecast
        start_date = self.list_dates[index]
        glonet_filename = start_date + '.nc'
        local_file_path = os.path.join(self.root_data_dir, glonet_filename)
        """print(f"start_date: {start_date}")
        print(f"local_file_path: {local_file_path}")
        print(f"glonet_filename: {glonet_filename}")
        print(f"self.list_dates: {self.list_dates}")"""
        glonet_s3_filepath = os.path.join(
            self.s3_folder,
            glonet_filename
        )
        if not (Path(local_file_path).is_file()):
            self.s3_manager.download_file(
                s3_key=glonet_s3_filepath,
                bucket_name=self.s3_bucket,
                dest_path=local_file_path,
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

    def close_all(self):
        self.args.dclogger.info("Close S3 client.")
        self.s3_client.close()

class FTPDataset(DCDataset):
    """Class to manage data from FTP servers."""
    def __init__(
        self,
        conf_args: Namespace,
        root_data_dir: str,
        ftp_hostname: str,
        ftp_dir: str,
        transform_fct: Optional[Callable[[xr.Dataset], xr.Dataset]] = None,
        save_after_preprocess: bool = False,
        lazy_load: bool = True,
        file_format: Optional[str] = 'netcdf',
        ftp_user: Optional[str] = "anonymous",
        ftp_pass: Optional[str] = "",
    ):
        super().__init__(
            conf_args, root_data_dir,
            transform_fct, save_after_preprocess,
            lazy_load, file_format,
        )
        self.ftp_manager = FTPManager(
            conf_args.dclogger, conf_args.exception_handler,
            ftp_hostname, ftp_dir, ftp_user, ftp_pass,
        )
        self.root_data_dir = root_data_dir
        self.init_server()

    def init_server(self):
        """Create a list of files to download from the FTP server."""
        self.ftp_manager.init_ftp()
        self.files_list = self.ftp_manager.get_files_list()
        #self.ftp_manager.close_ftp()

    def __len__(self):
        """Get the number of files in the FTP server.

        Returns:
            (int): number of files
        """
        return len(self.files_list)

    def get_date(self, index):
        """Get the date of the file at the given index.
        Args:
            index (int): index of the file
        Returns:
            (str): date of the file
        """
        return self.files_list[index]['date']

    def close_all(self):
        """Close the FTP server connection."""
        self.ftp_manager.close_ftp()
        self.args.dclogger.info("Close FTP connection.")
        #self.ftp_manager.close_ftp()

class IfremerFTPDataset(FTPDataset):
    """Class to manage data from Ifremer FTP servers."""
    def __init__(
        self,
        conf_args: Namespace,
        root_data_dir: str,
        ftp_hostname: str,
        ftp_dir: str,
        transform_fct: Optional[Callable[[xr.Dataset], xr.Dataset]] = None,
        save_after_preprocess: bool = False,
        lazy_load: bool = True,
        file_format: Optional[str] = 'netcdf',
        ftp_user: Optional[str] = "anonymous",
        ftp_pass: Optional[str] = "",
    ):
        super().__init__(
            conf_args, root_data_dir,
            ftp_hostname, ftp_dir,
            transform_fct, save_after_preprocess,
            lazy_load, file_format,
            ftp_user, ftp_pass,
        )

    def get_data(self, index: int) -> xr.Dataset:
        """Download the data
        Args:
            index: index of the file in the dataset
        """
        downl_name = self.ftp_manager.download_file(index, self.root_data_dir)
        if self.lazy_load:
            dataset = FileLoader.lazy_load_dataset(
                downl_name, self.exception_handler,
                self.dclogger,
            )
        else:
            dataset = FileLoader.load_dataset(
                downl_name, self.exception_handler,
                self.dclogger,
            )
        return dataset

    def get_time(self, index):
        """Get the timestamp of the data stored in a file at the given index.
        Args:
            index (int): index of the file
        Returns:
            (str): timestamp of the data
        """
        return get_time_info(self.files_list[index])