

from typing import Any
from dctools.data.connection.config import (
    ARGOConnectionConfig, GlonetConnectionConfig,
    WasabiS3ConnectionConfig, S3ConnectionConfig, 
    FTPConnectionConfig, CMEMSConnectionConfig,
    LocalConnectionConfig
)

from dctools.data.connection.connection_manager import (
    ArgoManager, GlonetManager,
    LocalConnectionManager, S3WasabiManager,
    S3Manager, FTPManager, CMEMSManager, clean_for_serialization,
)


CONNECTION_MANAGER_REGISTRY = {
    "argo": ArgoManager,
    "cmems": CMEMSManager,
    "ftp": FTPManager,
    "glonet": GlonetManager,
    "local": LocalConnectionManager,
    "s3": S3Manager,
    "wasabi": S3WasabiManager,
}

CONNECTION_CONFIG_REGISTRY = {
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
    argo_index: Any = None
) -> tuple:
    """Crée les configurations de connexion pour les sources prédictives et de référence."""
    protocol = config.protocol
    # ref_protocol = ref_source_config.protocol


    if protocol == 'cmems':
        if hasattr(config, 'fs') and hasattr(config.fs, '_session'):
            try:
                if hasattr(config.fs._session, 'close'):
                    config.fs._session.close()
            except:
                pass
            config.fs = None

    config.dataset_processor = None

    # Recrée l'objet de lecture dans le worker
    config_cls = CONNECTION_CONFIG_REGISTRY[protocol]
    connection_cls = CONNECTION_MANAGER_REGISTRY[protocol]
    delattr(config, "protocol")
    config = config_cls(**vars(config))

    # remove fsspec handler 'fs' from Config, otherwise: serialization error
    if protocol == 'cmems': 
        if hasattr(
            config.params, 'fs') and hasattr(config.params.fs, '_session'
        ):
            try:
                if hasattr(config.params.fs._session, 'close'):
                    config.params.fs._session.close()
            except:
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
    open_func = connection_manager.open

    return open_func
