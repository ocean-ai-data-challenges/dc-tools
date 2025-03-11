#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""NET Utilities functions."""

import datetime
import logging
import os
from typing import List, Optional

import copernicusmarine
from mypy_boto3_s3.client import S3Client
from pathlib import Path
from urllib.parse import urlparse

from dctools.utilities.errors import DCExceptionHandler


class S3Url(object):
    """Class to manipulate S3 urls."""

    def __init__(self, url):
        """Init.

        Args:
            url(str): S3 url
        """
        self._parsed = urlparse(url, allow_fragments=False)

    @property
    def bucket(self):
        """Get bucket name from url.

        Return:
            (str): bucket name
        """
        return self._parsed.netloc

    @property
    def key(self):
        """Get file key from url.

        Return:
            (str): file key
        """
        if self._parsed.query:
            return self._parsed.path.lstrip("/") + "?" + self._parsed.query
        else:
            return self._parsed.path.lstrip("/")

    @property
    def url(self):
        """Get url.

        Returns:
            (str): url
        """
        return self._parsed.geturl()

def download_s3_file(
    s3_client: S3Client,
    bucket_name: str,
    file_name: str,
    local_file_path: str,
    dclogger: logging.Logger,
    exception_handler: DCExceptionHandler,
) -> None:
    """Download a file from s3 server.

    Args:
        s3_client: (S3Client) boto3 S3 client
        bucket_name(str): name of s3 bucket
        filename(str): file to download from bucket
        outpath(str): path where to save the downloaded file
    """
    dclogger.info(f"Downloading from s3 bucket: {bucket_name}.")
    try:
        s3_client.download_file(
            Bucket=bucket_name,
            Key=file_name,
            Filename=local_file_path,
        )
    except Exception as exc:
        exception_handler.handle_exception(exc, "S3 download failed.")

def list_files_in_s3bucket_folder(
        s3_client: S3Client, bucket_name: str, s3_folder_name: str
    ) -> List[str]:
    """
    List all files in a S3 bucket folder.

    Args:
        s3_client: (S3Client) boto3 S3 client
        bucket_name(str): name of s3 bucket
        s3_folder_name(str): bucket folder to list files from
    """
    list_files = []
    response = s3_client.list_objects_v2(
        Bucket=bucket_name, Prefix=s3_folder_name
    )
    files = response.get("Contents")
    for file in files:
        list_files.append(file)
    return sorted(list_files)


class CMEMSManager:
    """Class to manage Copernicus Marine downloads."""

    def __init__(
        self,
        dclogger: logging.Logger,
        exception_handler: DCExceptionHandler,
        cmems_credentials_path: Optional[str] = None
    ) -> None:
        """Init.

        Args:
            dclogger(logging.Logger): logger instance
            exception_handler(DCExceptionHandler): exception handler instance
            cmems_credentials(Optional[str]): path to CMEMS credentials file
        """
        self.dclogger = dclogger
        self.exception_handler = exception_handler
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
        self.dclogger.info("Logging to Copernicus Marine.")
        try:
            if not (Path(self.cmems_credentials).is_file()):
                copernicusmarine.login()
        except Exception as exc:
            self.exception_handler.handle_exception(exc, "login to CMEMS failed.")
        return self.cmems_credentials

    def cmems_logout(self) -> None:
        """Logout from Copernicus Marine."""
        self.dclogger.info("Logging out from Copernicus Marine.")
        try:
            copernicusmarine.logout()
        except Exception as exc:
            self.exception_handler.handle_exception(exc, "logout from CMEMS failed.")
        return None

    def cmems_download(self, product_id: str, output_dir: str, filter: str) -> None:
        """Download a Copernicus Marine product.

        Args:
            product_id(str): product id
            output_dir(str): output directory
            filter(str): filter to apply to filenames
        """
        self.dclogger.info(f"Downloading product {product_id} from Copernicus Marine.")
        try:
            copernicusmarine.get(
                dataset_id=product_id,
                filter=filter,
                output_directory=output_dir,
                no_directories = True,
                credentials_file=self.cmems_credentials,
            )
        except Exception as exc:
            self.exception_handler.handle_exception(exc, "download from CMEMS failed.")
        return None

    def get_cmems_filter_from_date(self, date:datetime.datetime) -> str:
        """Give a filter to select correct file when downloading from CMEMS.

        Args:
            date (datetime.datetime): date of file to download.

        Returns:
            str: filter string
        """

        filter = f"*/{date.strftime('%Y')}/{date.strftime('%m')}/*_{date.strftime('%Y')}{date.strftime('%m')}{date.strftime('%d')}_*.nc"
        print(f"date: {date}     filter: {filter}")
        return filter
