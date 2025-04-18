#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""NET Utilities functions."""

import datetime
import logging
import os
import random
import string
from typing import List, Optional

import boto3
from botocore import UNSIGNED
from botocore.config import Config
import copernicusmarine
import ftputil
from mypy_boto3_s3.client import S3Client
from pathlib import Path
from urllib.parse import urlparse

from dctools.utilities.errors import DCExceptionHandler
from dctools.utilities.file_utils import read_file_tolist, check_valid_files



import boto3
from botocore.exceptions import ClientError
from urllib.parse import urlparse
import os

class S3Manager:
    def __init__(
        self,
        dclogger: logging.Logger,
        exception_handler: DCExceptionHandler,
        s3_url: str,
        s3_access_key: str=None,
        bucket_name: str=None,
    ):
        """self.s3_session = boto3.Session(
            s3_id=s3_id,
            s3_access_key=s3_access_key,
            s3_session_token=s3_session_token,
        )
        self.s3_client = self.s3_session.client('s3')"""

        if s3_url and s3_access_key:
            self.s3_client = boto3.client(
                "s3",
                aws_access_key_id=s3_url,
                aws_secret_access_key=s3_access_key,
            )
        else:
            self.s3_client = boto3.client(
                "s3",
                config=Config(signature_version=UNSIGNED),
                endpoint_url=s3_url,
            )
        self.default_bucket = bucket_name
        self.dclogger = dclogger
        self.exception_handler = exception_handler

    def upload_file(self, file_path: str, s3_key: str, bucket_name: str=None):
        """Upload a file to S3"""
        bucket = bucket_name if bucket_name is not None else self.default_bucket
        try:
            # usage : s3.upload_file("local_file.txt", "bucket-name", "folder/file.txt")
            self.s3_client.upload_file(file_path, bucket, s3_key)
            self.dclogger.info(f"Uploaded {file_path} to s3://{bucket}/{s3_key}")
        except ClientError as exc:
            self.exception_handler.handle_exception(exc, "S3 upload failed.")

    def download_file(self, s3_key: str, dest_path: str, bucket_name: str=None):
        """Download a file from S3"""
        try:
            bucket = bucket_name if bucket_name is not None else self.default_bucket
            self.s3_client.download_file(
                Bucket=bucket,
                Key=s3_key,
                Filename=dest_path,
            )
            self.dclogger.info(f"Downloaded s3://{bucket_name}/{s3_key} to {dest_path}")
        except Exception as exc:
            self.exception_handler.handle_exception(exc, "S3 download failed.")

    def list_s3bucket_content(self, bucket_name, prefix=""):
        """List objects under a prefix in a bucket"""
        try:
            paginator = self.s3_client.get_paginator('list_objects_v2')
            page_iterator = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
            files = []
            for page in page_iterator:
                for obj in page.get('Contents', []):
                    files.append(obj['Key'])
            return files
        except ClientError as exc:
            self.exception_handler.handle_exception(exc, "S3 download failed.")

    '''def delete_object(self, bucket_name, s3_key):
        """Delete a single object from a bucket"""
        try:
            self.s3.delete_object(Bucket=bucket_name, Key=s3_key)
            print(f"Deleted s3://{bucket_name}/{s3_key}")
        except ClientError as e:
            print(f"Error deleting object: {e}")
            raise'''

    '''def generate_presigned_url(self, bucket_name, s3_key, expiration=3600): #604800
        """Generate a pre-signed URL for a file"""
        try:
            url = self.s3.generate_presigned_url(
                'get_object',
                Params={'Bucket': bucket_name, 'Key': s3_key},
                ExpiresIn=expiration
            )
            return url
        except ClientError as e:
            print(f"Error generating presigned URL: {e}")
            raise'''

    def parse_s3_url(self, s3_url):
        """Parse an S3 URL and return bucket and key"""
        parsed = urlparse(s3_url)
        if parsed.scheme != "s3":
            raise ValueError("URL must start with s3://")
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")
        return bucket, key

    def bucket_exists(self, bucket_name):
        """Check if a bucket exists"""
        try:
            self.s3_client.head_bucket(Bucket=bucket_name)
            return True
        except ClientError:
            return False

    def create_bucket(self, bucket_name):
        """Create a new bucket"""
        try:
            if self.bucket_exists(bucket_name):
                print(f"Bucket {bucket_name} already exists.")
                return
            self.s3_client.create_bucket(Bucket=bucket_name)
            print(f"Bucket {bucket_name} created.")
        except ClientError as exc:
            self.exception_handler.handle_exception(exc, "Error creating S3 Bucket.")


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


    def cmems_download(
        self, product_id: str, output_dir: str, name_filter: str, tmp_dir:str
    ) -> None:
        """Download a Copernicus Marine product.

        Args:
            product_id(str): product id
            output_dir(str): output directory
            name_filter(str): filter to apply to filenames
        """
        self.dclogger.info(f"Downloading product {product_id} from Copernicus Marine.")
        try:
            rnd_str = ''.join(random.choice(string.ascii_uppercase) for _ in range(8))
            #list_files_path = os.path.join(tmp_dir, f"files_to_download_{rnd_str}.txt")
            self.dclogger.info(f"product_id: {product_id}  name_filter: {name_filter}")
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
            self.exception_handler.handle_exception(exc, "download from CMEMS failed.")
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
        dclogger: logging.Logger,
        exception_handler: DCExceptionHandler,
        ftp_host: str,
        ftp_dir: str,
        ftp_user: Optional[str] = "anonymous",
        ftp_pass: Optional[str] = "",
    ) -> None:
        """Init.

        Args:
            dclogger(logging.Logger): logger instance
            exception_handler(DCExceptionHandler): exception handler instance
            cmems_credentials(Optional[str]): path to CMEMS credentials file
        """
        self.dclogger = dclogger
        self.exception_handler = exception_handler
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
        self.dclogger.info(f"Connecting to ftp server: {self.ftp_host}.")
        self.ftp = ftputil.FTPHost(
            self.ftp_host, self.ftp_user, self.ftp_pass
        )
        self._files_list = self.recursive_list_ftp(self.ftp_dir)

    def close_ftp(self):
        """Close FTP session."""
        if self.ftp:
            self.ftp.close()
            self.dclogger.info("FTP session closed.")
        else:
            self.dclogger.error("Cannot close FTP session: not initialized.")

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
        self.dclogger.info(f"Listing files in ftp server: {self.ftp_host}.")
        self.ftp.chdir(dir_path)
        ftp_path = self.ftp_dir
        files_list = []
        for entry in self.ftp.listdir(self.ftp.curdir):
            #self.dclogger.info(f"entry: {entry}")
            if self.ftp.path.isdir(entry):
                ftp_path = os.path.join(ftp_path , entry)
                self.recursive_list_ftp(ftp_path)
            else:
                #self.dclogger.info(f"entry: {entry}")
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
        self.dclogger.info(f"Downloading file {index} from ftp server.")

        list_entry = self._files_list[index]
        name = os.path.join(list_entry['dir'], list_entry['ncfile'])
        local_name = os.path.join(local_dirpath, list_entry['ncfile'])
        self.ftp.download(name, local_name)
        return local_name