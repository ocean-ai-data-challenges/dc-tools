"""Configuration creation utilities for data connections."""

from typing import Any, Callable, Dict, Optional
from dctools.data.connection.config import (
    ARGOConnectionConfig, GlonetConnectionConfig,
    WasabiS3ConnectionConfig, S3ConnectionConfig,
    FTPConnectionConfig, CMEMSConnectionConfig,
    LocalConnectionConfig
)

from dctools.data.connection.connection_manager import (
    ArgoManager, GlonetManager,
    LocalConnectionManager, S3WasabiManager,
    S3Manager, FTPManager, CMEMSManager,
)


CONNECTION_MANAGER_REGISTRY: Dict[str, Any] = {
    "argo": ArgoManager,
    "cmems": CMEMSManager,
    "ftp": FTPManager,
    "glonet": GlonetManager,
    "local": LocalConnectionManager,
    "s3": S3Manager,
    "wasabi": S3WasabiManager,
}

CONNECTION_CONFIG_REGISTRY: Dict[str, Any] = {
    "argo": ARGOConnectionConfig,
    "cmems": CMEMSConnectionConfig,
    "ftp": FTPConnectionConfig,
    "glonet": GlonetConnectionConfig,
    "local": LocalConnectionConfig,
    "s3": S3ConnectionConfig,
    "wasabi": WasabiS3ConnectionConfig,
}

def create_worker_connect_config(
    # pred_source_config: Any,
    config: Any,
    argo_index: Optional[Any] = None
) -> Callable:
    """Create connection configurations for prediction and reference sources."""
    protocol = config.protocol
    # ref_protocol = ref_source_config.protocol


    if protocol == 'cmems':
        if hasattr(config, 'fs') and hasattr(config.fs, '_session'):
            try:
                if hasattr(config.fs._session, 'close'):
                    config.fs._session.close()
            except Exception:
                pass
            config.fs = None

    config.dataset_processor = None

    # Recreate the reader object in the worker
    config_cls = CONNECTION_CONFIG_REGISTRY[protocol]
    connection_cls = CONNECTION_MANAGER_REGISTRY[protocol]
    delattr(config, "protocol")
    config = config_cls(params=vars(config))

    # remove fsspec handler 'fs' from Config, otherwise: serialization error
    if protocol == 'cmems':
        if hasattr(
            config.params, 'fs') and hasattr(config.params.fs, '_session'
        ):
            try:
                if hasattr(config.params.fs._session, 'close'):
                    config.params.fs._session.close()
            except Exception:
                pass
            config.params.fs = None


    if protocol == 'cmems':
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
        connection_manager = connection_cls(
            config, call_list_files=False
        )
    open_func: Callable[..., Any] = connection_manager.open

    return open_func
