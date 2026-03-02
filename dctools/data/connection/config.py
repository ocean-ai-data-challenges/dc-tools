"""Connection configuration classes."""

from abc import ABC, abstractmethod
import os
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import fsspec
from loguru import logger

from dctools.utilities.misc_utils import get_home_path


class BaseConnectionConfig(ABC):
    """Base class for connection configurations."""

    def __init__(self, protocol: str, **kwargs):
        self.protocol = protocol
        self.params = SimpleNamespace(**kwargs)
        self.params.protocol = protocol
        # For ARGO/S3, local_root may be None - only check if provided
        if hasattr(self.params, "local_root") and self.params.local_root is not None:
            if not os.path.exists(self.params.local_root):
                logger.error(f"Invalid path : {self.params.local_root}")
                # raise FileNotFoundError()

    def to_dict(self) -> SimpleNamespace:
        """Convert configuration to dictionary."""
        return self.params

    @abstractmethod
    def create_fs(self):
        """Create filesystem object."""
        pass


class LocalConnectionConfig(BaseConnectionConfig):
    """Configuration for local file connection."""

    init_type: str
    local_root: str
    max_samples: Optional[int] = None
    file_pattern: Optional[str] = None
    groups: Optional[Any] = None
    keep_variables: Optional[List[str]] = None
    file_cache: Optional[Any] = None
    dataset_processor: Optional[Any] = None
    filter_values: Optional[Dict] = None
    full_day_data: bool = False

    def __init__(
        self,
        params: dict,
    ):
        """Init.

        Args:
            params (dict): parameters.
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
        """Create local filesystem."""
        fs = fsspec.filesystem("file")
        return fs


class CMEMSConnectionConfig(BaseConnectionConfig):
    """Configuration for CMEMS connection."""

    init_type: str
    local_root: str
    dataset_id: str
    cmems_credentials_path: Optional[str] = None
    max_samples: Optional[int] = None
    file_pattern: Optional[str] = None
    groups: Optional[Any] = None
    keep_variables: Optional[List[str]] = None
    file_cache: Optional[Any] = None
    dataset_processor: Optional[Any] = None
    filter_values: Optional[Dict] = None
    full_day_data: bool = False

    def __init__(
        self,
        params: dict,
    ):
        """Init.

        Args:
            params (dict): parameters.
        """
        for key, value in params.items():
            setattr(self, key, value)

        self.cache_dir = "/tmp/s3_cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        fs = self.create_fs()

        cmems_credentials_path = getattr(self, "cmems_credentials_path", None)
        if not cmems_credentials_path:
            home_path = get_home_path()
            cmems_credentials_path = os.path.join(
                home_path, ".copernicusmarine", ".copernicusmarine-credentials"
            )

        super().__init__(
            "cmems",
            init_type=self.init_type,
            local_root=self.local_root,
            dataset_id=self.dataset_id,
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
        """Create filesystem."""
        fs = fsspec.filesystem(
            "file",
            cache_storage=self.cache_dir,
            cache_type="filecache",  # On-disk cache
            cache_check=False,  # Don't check whether the remote file changed
        )
        return fs


class FTPConnectionConfig(BaseConnectionConfig):
    """Configuration for FTP connection."""

    init_type: str
    local_root: str
    host: str
    user: Optional[str] = None
    password: Optional[str] = None
    ftp_folder: Optional[str] = None
    max_samples: Optional[int] = None
    file_pattern: Optional[str] = None
    groups: Optional[Any] = None
    keep_variables: Optional[List[str]] = None
    file_cache: Optional[Any] = None
    dataset_processor: Optional[Any] = None
    filter_values: Optional[Dict] = None
    full_day_data: bool = False

    def __init__(
        self,
        params: dict,
    ):
        """Init."""
        for key, value in params.items():
            setattr(self, key, value)

        # self.host = host
        # self.user = user
        # self.password = password
        fs = self.create_fs()
        super().__init__(
            "ftp",
            init_type=self.init_type,
            local_root=self.local_root,
            host=self.host,
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
        """Create filesystem."""
        fs = fsspec.filesystem("ftp", host=self.host, username=self.user, password=self.password)
        return fs


class S3ConnectionConfig(BaseConnectionConfig):
    """Configuration for S3 Connection."""

    init_type: str
    local_root: str
    s3_bucket: str
    s3_folder: str
    key: Optional[str] = None
    secret_key: Optional[str] = None
    endpoint_url: Optional[str] = None
    max_samples: Optional[int] = None
    file_pattern: Optional[str] = None
    groups: Optional[Any] = None
    keep_variables: Optional[List[str]] = None
    file_cache: Optional[Any] = None
    dataset_processor: Optional[Any] = None
    filter_values: Optional[Dict] = None
    full_day_data: bool = False

    def __init__(
        self,
        params: dict,
    ):
        """Init."""
        self.protocol = "s3"
        for key, value in params.items():
            setattr(self, key, value)
        fs = self.create_fs()

        super().__init__(
            self.protocol,
            init_type=self.init_type,
            local_root=self.local_root,
            s3_bucket=self.s3_bucket,
            s3_folder=self.s3_folder,
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
        """Create filesystem."""
        # Use config_kwargs for s3fs to create the Config object internaly
        # instead of passing a constructed Config object in client_kwargs which causes
        # "multiple values for keyword argument 'config'" error with recent aiobotocore/s3fs.
        config_kwargs = {"connect_timeout": 30, "read_timeout": 60}

        client_kwargs: Dict[str, Any] = {}
        if self.endpoint_url:
            client_kwargs["endpoint_url"] = self.endpoint_url

        if not self.key or not self.secret_key:
            fs = fsspec.filesystem(
                "s3", anon=True, client_kwargs=client_kwargs, config_kwargs=config_kwargs
            )
        else:
            fs = fsspec.filesystem(
                "s3",
                key=self.key,
                secret=self.secret_key,
                client_kwargs=client_kwargs,
                config_kwargs=config_kwargs,
            )
        return fs


class WasabiS3ConnectionConfig(S3ConnectionConfig):
    """Configuration for Wasabi S3 Connection."""

    init_type: str
    local_root: str
    s3_bucket: str
    s3_folder: str
    key: Optional[str] = None
    secret_key: Optional[str] = None
    endpoint_url: Optional[str] = None
    max_samples: Optional[int] = None
    file_pattern: Optional[str] = None
    groups: Optional[Any] = None
    keep_variables: Optional[List[str]] = None
    file_cache: Optional[Any] = None
    dataset_processor: Optional[Any] = None
    filter_values: Optional[Dict] = None
    full_day_data: bool = False

    def __init__(
        self,
        params: dict,
    ):
        """Init."""
        for key, value in params.items():
            setattr(self, key, value)
        s3_params = {
            "init_type": self.init_type,
            "local_root": self.local_root,
            "s3_bucket": self.s3_bucket,
            "s3_folder": self.s3_folder,
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
    """Configuration for Glonet Connection."""

    init_type: str
    local_root: str
    endpoint_url: Optional[str] = None
    s3_bucket: str
    s3_folder: str
    max_samples: Optional[int] = None
    file_pattern: Optional[str] = None
    groups: Optional[Any] = None
    keep_variables: Optional[List[str]] = None
    file_cache: Optional[Any] = None
    dataset_processor: Optional[Any] = None
    filter_values: Optional[Dict] = None
    full_day_data: bool = False

    def __init__(
        self,
        params: dict,
    ):
        """Init."""
        for key, value in params.items():
            setattr(self, key, value)
        fs = self.create_fs()

        super().__init__(
            "glonet",
            init_type=self.init_type,
            local_root=self.local_root,
            fs=fs,
            endpoint_url=self.endpoint_url,
            s3_bucket=self.s3_bucket,
            s3_folder=self.s3_folder,
            max_samples=self.max_samples if hasattr(self, "max_samples") else None,
            file_pattern=self.file_pattern or "**/*.nc",
            groups=self.groups if hasattr(self, "groups") else None,
            keep_variables=self.keep_variables if hasattr(self, "keep_variables") else None,
            file_cache=self.file_cache if hasattr(self, "file_cache") else None,
            dataset_processor=self.dataset_processor
            if hasattr(self, "dataset_processor")
            else None,
            filter_values=self.filter_values if hasattr(self, "filter_values") else None,
            full_day_data=self.full_day_data if hasattr(self, "full_day_data") else False,
        )

    def create_fs(self):
        """Create filesystem."""
        # Use config_kwargs for s3fs to create the Config object internaly
        # instead of passing a constructed Config object in client_kwargs which causes
        # "multiple values for keyword argument 'config'" error with recent aiobotocore/s3fs.
        config_kwargs = {"connect_timeout": 30, "read_timeout": 60}

        client_kwargs: Dict[str, Any] = {}
        if self.endpoint_url:
            client_kwargs["endpoint_url"] = self.endpoint_url

        fs = fsspec.filesystem(
            "s3", anon=True, client_kwargs=client_kwargs, config_kwargs=config_kwargs
        )
        return fs


class ARGOConnectionConfig(BaseConnectionConfig):
    """Configuration for Argo Connection with S3/Wasabi support."""

    init_type: str
    local_root: str
    s3_bucket: Optional[str] = None
    s3_folder: Optional[str] = None
    s3_key: Optional[str] = None
    s3_secret_key: Optional[str] = None
    endpoint_url: Optional[str] = None
    base_path: Optional[str] = None
    depth_values: Optional[List[float]] = None
    variables: Optional[List[str]] = None
    chunks: Optional[Dict[str, int]] = None
    max_fetch_retries: Optional[int] = None
    retry_backoff_seconds: Optional[float] = None
    max_samples: Optional[int] = None
    file_pattern: Optional[str] = None
    groups: Optional[Any] = None
    keep_variables: Optional[List[str]] = None
    file_cache: Optional[Any] = None
    dataset_processor: Optional[Any] = None
    filter_values: Optional[Dict] = None
    full_day_data: bool = False

    def __init__(
        self,
        params: dict,
    ):
        """Init."""
        for key, value in params.items():
            setattr(self, key, value)
        fs = self.create_fs()

        max_fetch_retries = getattr(self, "max_fetch_retries", None)
        if max_fetch_retries is None:
            max_fetch_retries = 4

        retry_backoff_seconds = getattr(self, "retry_backoff_seconds", None)
        if retry_backoff_seconds is None:
            retry_backoff_seconds = 0.8

        super().__init__(
            "argo",
            init_type=getattr(self, "init_type", "from_connection"),
            local_root=getattr(self, "local_root", None),
            local_catalog_path=getattr(self, "local_catalog_path", None),
            fs=fs,
            s3_bucket=getattr(self, "s3_bucket", None),
            s3_folder=getattr(self, "s3_folder", None),
            s3_key=getattr(self, "s3_key", None),
            s3_secret_key=getattr(self, "s3_secret_key", None),
            endpoint_url=getattr(self, "endpoint_url", None),
            base_path=getattr(self, "base_path", None),
            depth_values=getattr(self, "depth_values", None),
            variables=getattr(self, "variables", None),
            chunks=getattr(self, "chunks", {"N_PROF": 2000}),
            max_fetch_retries=max_fetch_retries,
            retry_backoff_seconds=retry_backoff_seconds,
            max_samples=self.max_samples or None,
            file_pattern=self.file_pattern or "**/*.nc",
            groups=self.groups if hasattr(self, "groups") else None,
            keep_variables=self.keep_variables if hasattr(self, "keep_variables") else None,
            file_cache=self.file_cache if hasattr(self, "file_cache") else None,
            dataset_processor=self.dataset_processor
            if hasattr(self, "dataset_processor")
            else None,
            filter_values=self.filter_values if hasattr(self, "filter_values") else None,
            full_day_data=self.full_day_data if hasattr(self, "full_day_data") else False,
        )

    def create_fs(self):
        """Create filesystem (S3 if configured, otherwise local)."""
        if self.s3_bucket and self.endpoint_url:
            # Fix for "multiple values for keyword argument 'config'" error
            config_kwargs = {"connect_timeout": 30, "read_timeout": 60}

            client_kwargs: Dict[str, Any] = {"endpoint_url": self.endpoint_url}

            if self.s3_key and self.s3_secret_key:
                fs = fsspec.filesystem(
                    "s3",
                    key=self.s3_key,
                    secret=self.s3_secret_key,
                    client_kwargs=client_kwargs,
                    config_kwargs=config_kwargs,
                )
            else:
                fs = fsspec.filesystem(
                    "s3", anon=True, client_kwargs=client_kwargs, config_kwargs=config_kwargs
                )
        else:
            # Local filesystem fallback
            fs = fsspec.filesystem("file")
        return fs

    def get_storage_options(self) -> Dict[str, Any]:
        """Get S3 storage options for ArgoInterface."""
        if self.s3_bucket and self.endpoint_url:
            storage_opts = {"client_kwargs": {"endpoint_url": self.endpoint_url}}
            if self.s3_key and self.s3_secret_key:
                storage_opts["key"] = self.s3_key
                storage_opts["secret"] = self.s3_secret_key
            else:
                storage_opts["anon"] = True
            return storage_opts
        return {}
