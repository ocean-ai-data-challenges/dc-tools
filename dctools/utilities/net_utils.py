#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""NET Utilities functions."""

import datetime
import os
import random
import string
from typing import List, Optional

#import boto3
#from botocore import UNSIGNED
#from botocore.config import Config
import copernicusmarine
import ftputil
from loguru import logger
#from mypy_boto3_s3.client import S3Client
from pathlib import Path
#from urllib.parse import urlparse

#from dctools.utilities.file_utils import read_file_tolist, check_valid_files



#import boto3
#from botocore.exceptions import ClientError
#from urllib.parse import urlparse
import os


'''
class CMEMSManager:
    """Class to manage Copernicus Marine downloads."""

    def __init__(
        self,
        cmems_credentials_path: Optional[str] = None,
    ) -> None:
        """Init.

        Args:
            cmems_credentials(Optional[str]): path to CMEMS credentials file
        """
        if cmems_credentials_path:
            self.cmems_credentials = cmems_credentials_path
        else:
            self.cmems_credentials = os.path.expanduser(
                "~/.copernicusmarine/.copernicusmarine-credentials"
            )

    def get_credentials(self):
        """Get CMEMS credentials.

        Return:
            (dict): CMEMS credentials
        """
        with open(self.cmems_credentials, "r") as f:
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
            if not (Path(self.cmems_credentials).is_file()):
                copernicusmarine.login()
        except Exception as exc:
            logger.error(f"login to CMEMS failed: {repr(exc)}")
        return self.cmems_credentials

    def cmems_logout(self) -> None:
        """Logout from Copernicus Marine."""
        logger.info("Logging out from Copernicus Marine.")
        try:
            copernicusmarine.logout()
        except Exception as exc:
            logger.error(f"logout from CMEMS failed: {repr(exc)}")
        return None


    def cmems_download(
        self, product_id: str, output_dir: str, name_filter: str, tmp_dir:str
    ) -> None:
        """Download a Copernicus Marine product.

        Args:
            product_id(str): product id
            output_dir(str): output directory
            name_filter(str): filter to apply to filenames
        """
        logger.info(f"Downloading product {product_id} from Copernicus Marine.")
        try:
            rnd_str = ''.join(random.choice(string.ascii_uppercase) for _ in range(8))
            #list_files_path = os.path.join(tmp_dir, f"files_to_download_{rnd_str}.txt")
            logger.info(f"product_id: {product_id}  name_filter: {name_filter}")
            """copernicusmarine.get(
                dataset_id=product_id,
                filter=name_filter,
                create_file_list=list_files_path,
            )
            copernicusmarine.get(
                dataset_id=product_id,
                output_directory=output_dir,
                no_directories = True,
                credentials_file=self.cmems_credentials,
                file_list=list_files_path,
            )"""
            copernicusmarine.get(
                dataset_id=product_id,
                filter=name_filter,
                output_directory=output_dir,
                no_directories = True,
                credentials_file=self.cmems_credentials,
            )

        except Exception as exc:
            logger.error(f"download from CMEMS failed: {repr(exc)}")
        return None

    def get_cmems_filter_from_date(self, date: str) -> str:
        """Give a filter to select correct file when downloading from CMEMS.

        Args:
            date (datetime.datetime): date of file to download.

        Returns:
            str: filter string
        """
        dt = datetime.datetime.strptime(date, '%Y-%m-%d')
        filter = f"*/{dt.strftime('%Y')}/{dt.strftime('%m')}/*_{dt.strftime('%Y')}{dt.strftime('%m')}{dt.strftime('%d')}_*.nc"
        return filter


class FTPManager:
    """Class to manage downloads from ftp servers."""

    def __init__(
        self,
        ftp_host: str,
        ftp_dir: str,
        ftp_user: Optional[str] = "anonymous",
        ftp_pass: Optional[str] = "",
    ) -> None:
        """Init.

        Args:
            cmems_credentials(Optional[str]): path to CMEMS credentials file
        """
        self.ftp_host = ftp_host
        self.ftp_dir = ftp_dir
        self.ftp_user = ftp_user
        self.ftp_pass = ftp_pass
        self.ftp = None


    # some utility functions that we gonna need
    def get_size_format(self, n, suffix="B"):
        # converts bytes to scaled format (e.g KB, MB, etc.)
        for unit in ["", "K", "M", "G", "T", "P"]:
            if n < 1024:
                return f"{n:.2f}{unit}{suffix}"
            n /= 1024

    def get_datetime_format(self, date_time):
        # convert to datetime object
        date_time = datetime.strptime(date_time, "%Y%m%d%H%M%S")
        # convert to human readable date time string
        return date_time.strftime("%Y/%m/%d %H:%M:%S")

    def init_ftp(self):
        """Init FTP session."""
        logger.info(f"Connecting to ftp server: {self.ftp_host}.")
        self.ftp = ftputil.FTPHost(
            self.ftp_host, self.ftp_user, self.ftp_pass
        )
        self._files_list = self.recursive_list_ftp(self.ftp_dir)

    def close_ftp(self):
        """Close FTP session."""
        if self.ftp:
            self.ftp.close()
            logger.info("FTP session closed.")
        else:
            logger.error("Cannot close FTP session: not initialized.")

    def get_files_list(self):
        """Get the files tree of the FTP server.
        Returns:
            (dict): files tree
        """
        return self._files_list

    def recursive_list_ftp(self, dir_path):
        """Recursively list files in ftp server.
        Args:
            dir_path (str): path to list files from
        """
        logger.info(f"Listing files in ftp server: {self.ftp_host}.")
        self.ftp.chdir(dir_path)
        ftp_path = self.ftp_dir
        files_list = []
        for entry in self.ftp.listdir(self.ftp.curdir):
            #logger.info(f"entry: {entry}")
            if self.ftp.path.isdir(entry):
                ftp_path = os.path.join(ftp_path , entry)
                self.recursive_list_ftp(ftp_path)
            else:
                #logger.info(f"entry: {entry}")
                #fdate, ftime = self.get_time_from_filename(entry)
                files_list.append({
                    #'date': fdate,
                    #'time': ftime,
                    'dir': self.ftp.curdir,
                    'ncfile': entry
                })
        return files_list

    def get_time_from_filename(self, filename):
        """Get time from filename.

        Args:
            filename (str): name of the file

        Returns:
            (str): time in format YYYYMMDDHHMMSS
        """
        # get the file time from the filename
        # split the filename by "_" and take the first part
        filename = filename.split("-")[0]
        year = filename[:4]
        month = filename[4:6]
        day = filename[6:8]
        time = filename[8:]
        date = f"{year}-{month}-{day}"
        return (date, time)


    def download_file(self, index, local_dirpath):
        """Download a file from ftp server.

        Args:self.files_list (int): index of the file to download
            local_file_path (str): path where to save the downloaded file
        """
        logger.info(f"Downloading file {index} from ftp server.")

        list_entry = self._files_list[index]
        name = os.path.join(list_entry['dir'], list_entry['ncfile'])
        local_name = os.path.join(local_dirpath, list_entry['ncfile'])
        self.ftp.download(name, local_name)
        return local_name
'''