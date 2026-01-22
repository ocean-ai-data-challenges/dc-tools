"""Connection configuration classes."""

from abc import ABC
import os
from types import SimpleNamespace
from typing import Optional

import fsspec
from loguru import logger
from oceanbench.core.distributed import DatasetProcessor

from dctools.utilities.file_utils import FileCacheManager
from dctools.utilities.misc_utils import get_home_path



class BaseConnectionConfig(ABC):
    def __init__(self, protocol: str, **kwargs):
        self.protocol = protocol
        self.params = SimpleNamespace(**kwargs)
        setattr(self.params, "protocol", protocol)
        assert hasattr(self.params, "local_root"), "Attribute \"local_root\" is required"
        if not os.path.exists(self.params.local_root):
            logger.error(f"Invalid path : {self.params.local_root}")
            raise FileNotFoundError()

    def to_dict(self) -> dict:
        return self.params


class LocalConnectionConfig(BaseConnectionConfig):
    def __init__(
        self,
        params: dict,
    ):
        """Init.
        
        Args:
            root (str): path to local directory
        """
        for key, value in params.items():
            setattr(self, key, value)
        fs = self.create_fs()
        super().__init__(
            "local",
            init_type=self.init_type,
            local_root=self.local_root,
            fs=fs,
            max_samples=self.max_samples or None,
            file_pattern=self.file_pattern or "**/*.nc",
            groups=self.groups or None,
            keep_variables=self.keep_variables or None,
            file_cache=self.file_cache or None,
            dataset_processor=self.dataset_processor or None,
            filter_values=self.filter_values or None,
            full_day_data=self.full_day_data or False,
        )

    def create_fs(self):
        fs = fsspec.filesystem("file")
        return fs

class CMEMSConnectionConfig(BaseConnectionConfig):
    def __init__(
        self,
        params: dict,
    ):
        """Init.

        Args:
            cmems_credentials(Optional[str]): path to CMEMS credentials file
        """
        for key, value in params.items():
            setattr(self, key, value)

        self.cache_dir = "/tmp/s3_cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        fs = self.create_fs()
        if cmems_credentials_path:
            cmems_credentials_path = cmems_credentials_path
        else:
            home_path = get_home_path()
            cmems_credentials_path = os.path.join(
                home_path, ".copernicusmarine", ".copernicusmarine-credentials"
            )
        super().__init__(
            "cmems",
            init_type=self.init_type,
            local_root=self.local_root, dataset_id=self.dataset_id,
            cmems_credentials_path=cmems_credentials_path or None,
            fs=fs,
            max_samples=self.max_samples or None,
            file_pattern=self.file_pattern or "**/*.nc",
            groups=self.groups or None,
            keep_variables=self.keep_variables or None,
            file_cache=self.file_cache or None,
            dataset_processor=self.dataset_processor or None,
            filter_values=self.filter_values or None,
            full_day_data=self.full_day_data or False,
        )

    def create_fs(self):
        fs = fsspec.filesystem(
            "file",
            cache_storage=self.cache_dir,
            cache_type='filecache',  # Cache sur disque
            cache_check=False,  # Ne pas vérifier si le fichier distant a changé
        )
        return fs

class FTPConnectionConfig(BaseConnectionConfig):
    def __init__(
        self,
        params: dict,
    ):
        for key, value in params.items():
            setattr(self, key, value)

        #self.host = host
        #self.user = user
        #self.password = password
        fs = self.create_fs()
        super().__init__(
            "ftp",
            init_type=self.init_type,
            local_root=self.local_root, host=self.host,
            user=self.user or None,
            password=self.password or None,
            fs=fs,
            ftp_folder=self.ftp_folder or None,
            max_samples=self.max_samples or None,
            file_pattern=self.file_pattern or "**/*.nc",
            groups=self.groups or None,
            keep_variables=self.keep_variables or None,
            file_cache=self.file_cache or None,
            dataset_processor=self.dataset_processor or None,
            filter_values=self.filter_values or None,
            full_day_data=self.full_day_data or False,
        )

    def create_fs(self):

        fs = fsspec.filesystem("ftp", host=self.host, username=self.user, password=self.password)
        return fs

class S3ConnectionConfig(BaseConnectionConfig):
    def __init__(
        self,
        params: dict,
    ):
        self.protocol = "s3"
        for key, value in params.items():
            setattr(self, key, value)
        fs = self.create_fs()

        super().__init__(
            self.protocol,
            init_type=self.init_type,
            local_root=self.local_root,
            bucket=self.bucket,
            bucket_folder=self.bucket_folder,
            key=self.key or None,
            secret_key=self.secret_key or None,
            endpoint_url=self.endpoint_url or None,
            fs=fs,
            max_samples=self.max_samples or None,
            file_pattern=self.file_pattern or "**/*.nc",
            groups=self.groups or None,
            keep_variables=self.keep_variables or None,
            file_cache=self.file_cache or None,
            dataset_processor=self.dataset_processor or None,
            filter_values=self.filter_values or None,
            full_day_data=self.full_day_data or False,
        )

    def create_fs(self):
        client_kwargs={'endpoint_url': self.endpoint_url} if self.endpoint_url else None
        if not self.key or not self.secret_key:
            fs = fsspec.filesystem('s3', anon=True, client_kwargs=client_kwargs)
        else:
            fs = fsspec.filesystem(
                "s3", key=self.key, secret=self.secret_key, client_kwargs=client_kwargs
            )
        return fs

class WasabiS3ConnectionConfig(S3ConnectionConfig):
    def __init__(
        self,
        params: dict,
    ):
        for key, value in params.items():
            setattr(self, key, value)
        s3_params = {
            "init_type": self.init_type,
            "local_root": self.local_root,
            "bucket": self.bucket,
            "bucket_folder": self.bucket_folder,
            "key": self.key,
            "secret_key": self.secret_key or None,
            "endpoint_url": self.endpoint_url,
            "max_samples": self.max_samples,
            "file_pattern": self.file_pattern or "**/*.nc",
            "groups": self.groups or None,
            "keep_variables": self.keep_variables or None,
            "file_cache": self.file_cache or None,
            "dataset_processor": self.dataset_processor or None,
            "filter_values": self.filter_values or None,
            "full_day_data": self.full_day_data or False,
            "protocol": "wasabi",
        }
        super().__init__(s3_params)


class GlonetConnectionConfig(BaseConnectionConfig):
    def __init__(
        self,
        params: dict,
    ):
        for key, value in params.items():
            setattr(self, key, value)
        fs = self.create_fs()

        super().__init__(
            "glonet", 
            init_type=self.init_type,
            local_root=self.local_root,
            fs=fs,
            endpoint_url=self.endpoint_url,
            glonet_s3_bucket=self.glonet_s3_bucket,
            s3_glonet_folder=self.s3_glonet_folder,
            max_samples=self.max_samples if hasattr(self, 'max_samples') else None,
            file_pattern=self.file_pattern or "**/*.nc",
            groups=self.groups if hasattr(self, 'groups') else None,
            keep_variables=self.keep_variables if hasattr(self, 'keep_variables') else None,
            file_cache=self.file_cache if hasattr(self, 'file_cache') else None,
            dataset_processor=self.dataset_processor if hasattr(self, 'dataset_processor') else None,
            filter_values=self.filter_values if hasattr(self, 'filter_values') else None,
            full_day_data=self.full_day_data if hasattr(self, 'full_day_data') else False,
        )

    def create_fs(self):
        endpoint_url = self.endpoint_url
        client_kwargs={'endpoint_url': endpoint_url} if endpoint_url else None
        fs = fsspec.filesystem(
            's3', anon=True, client_kwargs=client_kwargs,

        )
        return fs


class ARGOConnectionConfig(BaseConnectionConfig):
    def __init__(
        self,
        params: dict,
    ):
        for key, value in params.items():
            setattr(self, key, value)
        fs = self.create_fs()

        super().__init__(
            "argo",
            init_type=self.init_type,
            local_root=self.local_root,
            fs=fs,
            max_samples=self.max_samples or None,
            file_pattern=self.file_pattern or "**/*.nc",
            groups=self.groups if hasattr(self, 'groups') else None,
            keep_variables=self.keep_variables if hasattr(self, 'keep_variables') else None,
            file_cache=self.file_cache if hasattr(self, 'file_cache') else None,
            dataset_processor=self.dataset_processor if hasattr(self, 'dataset_processor') else None,
            filter_values=self.filter_values if hasattr(self, 'filter_values') else None,
            full_day_data=self.full_day_data if hasattr(self, 'full_day_data') else False,
        )

    def create_fs(self):
        fs = fsspec.filesystem("file")
        return fs
