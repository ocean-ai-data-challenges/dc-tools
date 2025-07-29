
from abc import ABC, abstractmethod
import os
from types import SimpleNamespace
from typing import Optional

import fsspec
from loguru import logger

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

    #@abstractmethod
    def to_dict(self) -> dict:
        return self.params


class LocalConnectionConfig(BaseConnectionConfig):
    def __init__(
        self,
        init_type: str,
        local_root: str,
        max_samples: Optional[int] = None,
        file_pattern: Optional[str] = "**/*.nc",
        groups: Optional[list[str]] = None,
        keep_variables: Optional[list[str]] = None,
        file_cache: Optional[FileCacheManager] = None,
        dask_cluster: Optional[object] = None,
    ):
        """Init.
        Args:
            root (str): path to local directory
        """
        fs = fsspec.filesystem("file")
        super().__init__(
            "local",
            init_type=init_type,
            local_root=local_root,
            fs=fs,
            max_samples=max_samples,
            file_pattern=file_pattern,
            groups=groups,
            keep_variables=keep_variables,
            file_cache=file_cache,
            dask_cluster=dask_cluster,
        )


class CMEMSConnectionConfig(BaseConnectionConfig):
    def __init__(
        self,
        init_type: str,
        local_root: str,
        dataset_id: str,
        cmems_credentials_path: Optional[str] = None,
        max_samples: Optional[int] = None,
        file_pattern: Optional[str] = "**/*.nc",
        groups: Optional[list[str]] = None,
        keep_variables: Optional[list[str]] = None,
        file_cache: Optional[FileCacheManager] = None,
        dask_cluster: Optional[object] = None,
    ):
        """Init.

        Args:
            cmems_credentials(Optional[str]): path to CMEMS credentials file
        """
        fs = fsspec.filesystem("file")
        if cmems_credentials_path:
            cmems_credentials_path = cmems_credentials_path
        else:
            home_path = get_home_path()
            '''cmems_credentials = os.path.expanduser(
                "~/.copernicusmarine/.copernicusmarine-credentials"
            )'''
            cmems_credentials_path = os.path.join(
                home_path, ".copernicusmarine", ".copernicusmarine-credentials"
            )
        super().__init__(
            "cmems",
            init_type=init_type,
            local_root=local_root, dataset_id=dataset_id,
            cmems_credentials_path=cmems_credentials_path, fs=fs,
            max_samples=max_samples,
            file_pattern=file_pattern,
            groups=groups,
            keep_variables=keep_variables,
            file_cache=file_cache,
            dask_cluster=dask_cluster,
        )


class FTPConnectionConfig(BaseConnectionConfig):
    def __init__(
        self, 
        init_type: str,
        local_root: str, host: str,
        ftp_folder: str,
        user: str = None, password: str = None,
        max_samples: Optional[int] = None,
        file_pattern: Optional[str] = "**/*.nc",
        groups: Optional[list[str]] = None,
        keep_variables: Optional[list[str]] = None,
        file_cache: Optional[FileCacheManager] = None,
        dask_cluster: Optional[object] = None,
    ):
        fs = fsspec.filesystem("ftp", host=host, username=user, password=password)
        super().__init__(
            "ftp",
            init_type=init_type,
            local_root=local_root, host=host,
            user=user, password=password, fs=fs,
            ftp_folder=ftp_folder,
            max_samples=max_samples,
            file_pattern=file_pattern,
            groups=groups,
            keep_variables=keep_variables,
            file_cache=file_cache,
            dask_cluster=dask_cluster,
        )


class S3ConnectionConfig(BaseConnectionConfig):
    def __init__(
        self,
        init_type: str,
        local_root: str,
        bucket: str,
        bucket_folder: str,
        key: Optional[str] = None,
        secret_key: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        max_samples: Optional[int] = None,
        file_pattern: Optional[str] = "**/*.nc",
        groups: Optional[list[str]] = None,
        keep_variables: Optional[list[str]] = None,
        file_cache: Optional[FileCacheManager] = None,
        dask_cluster: Optional[object] = None,
        protocol: str = "s3",

    ):
        client_kwargs={'endpoint_url': endpoint_url} if endpoint_url else None
        if not key or not secret_key:
            fs = fsspec.filesystem('s3', anon=True, client_kwargs=client_kwargs)
        else:
            fs = fsspec.filesystem(
                "s3", key=key, secret=secret_key, client_kwargs=client_kwargs
            )
        super().__init__(
            protocol, 
            init_type=init_type,
            local_root=local_root,
            bucket=bucket,
            bucket_folder=bucket_folder,
            key=key, secret_key=secret_key,
            endpoint_url=endpoint_url,
            fs=fs, max_samples=max_samples,
            file_pattern=file_pattern,
            groups=groups,
            keep_variables=keep_variables,
            file_cache=file_cache,
            dask_cluster=dask_cluster,
        )


class WasabiS3ConnectionConfig(S3ConnectionConfig):
    def __init__(
        self,
        init_type: str,
        local_root: str,
        bucket: str,
        bucket_folder: str,
        key: Optional[str] = None,
        secret_key: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        max_samples: Optional[int] = None,
        file_pattern: Optional[str] = "**/*.nc",
        groups: Optional[list[str]] = None,
        keep_variables: Optional[list[str]] = None,
        file_cache: Optional[FileCacheManager] = None,
        dask_cluster: Optional[object] = None,
    ):
        super().__init__(
            init_type=init_type,
            local_root=local_root,
            bucket=bucket,
            bucket_folder=bucket_folder,
            key=key, secret_key=secret_key,
            endpoint_url=endpoint_url,
            max_samples=max_samples,
            file_pattern=file_pattern,
            groups=groups,
            keep_variables=keep_variables,
            file_cache=file_cache,
            dask_cluster=dask_cluster,
            protocol="wasabi",
        )

class GlonetConnectionConfig(BaseConnectionConfig):
    def __init__(
        self,
        init_type: str,
        local_root: str,
        endpoint_url: str,
        glonet_s3_bucket: str,
        s3_glonet_folder: str,
        max_samples: Optional[int] = None,
        file_pattern: Optional[str] = "**/*.nc",
        groups: Optional[list[str]] = None,
        keep_variables: Optional[list[str]] = None,
        file_cache: Optional[FileCacheManager] = None,
        dask_cluster: Optional[object] = None,
    ):
        client_kwargs={'endpoint_url': endpoint_url} if endpoint_url else None
        fs = fsspec.filesystem('s3', anon=True, client_kwargs=client_kwargs)

        super().__init__(
            "glonet", 
            init_type=init_type,
            local_root=local_root,
            fs=fs,
            endpoint_url=endpoint_url,
            glonet_s3_bucket=glonet_s3_bucket,
            s3_glonet_folder=s3_glonet_folder,
            max_samples=max_samples,
            file_pattern=file_pattern,
            groups=groups,
            keep_variables=keep_variables,
            file_cache=file_cache,
            dask_cluster=dask_cluster,
        )


class ARGOConnectionConfig(BaseConnectionConfig):
    def __init__(
        self,
        init_type: str,
        local_root: str,
        max_samples: Optional[int] = None,
        file_pattern: Optional[str] = "**/*.nc",
        groups: Optional[list[str]] = None,
        keep_variables: Optional[list[str]] = None,
        file_cache: Optional[FileCacheManager] = None,
        dask_cluster: Optional[object] = None,
    ):
        fs = fsspec.filesystem("file")

        super().__init__(
            "argo",
            init_type=init_type,
            local_root=local_root,
            fs=fs,
            max_samples=max_samples,
            file_pattern=file_pattern,
            groups=groups,
            keep_variables=keep_variables,
            file_cache=file_cache,
            dask_cluster=dask_cluster,
        )

