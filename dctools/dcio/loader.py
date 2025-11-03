#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Classes and functions for loading xarray Datasets."""

import gc
import os
from typing import Any, List, Optional, Union

from fsspec import FSMap
from loguru import logger
import netCDF4
import traceback
import xarray as xr

# Configuration pour la compatibilité Dask
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
os.environ['NETCDF4_DEACTIVATE_MPI'] = '1'


def configure_xarray_for_dask():
    """
    Configure xarray pour une utilisation optimale avec Dask workers.
    À appeler une fois au début de l'application.
    """
    import dask
    
    # Configuration Dask pour xarray
    dask.config.set({
        'array.chunk-size': '256MB',
        'array.slicing.split_large_chunks': False,
        'distributed.worker.daemon': False,
        'distributed.comm.timeouts.tcp': 300,
        'distributed.comm.timeouts.connect': 180,
    })
    
    # Configuration xarray
    xr.set_options(
        file_cache_maxsize=1,  # Cache minimal pour éviter les conflits
        warn_for_unclosed_files=False,  # Évite les warnings dans les workers
    )


def list_all_group_paths(nc_path: str) -> List[str]:
    """
    Liste tous les chemins de groupes dans un fichier NetCDF.
    Thread-safe pour utilisation avec Dask.
    """
    def walk(grp, prefix=""):
        paths = []
        for name, subgrp in grp.groups.items():
            full = f"{prefix}/{name}" if prefix else name
            paths.append(full)
            paths.extend(walk(subgrp, full))
        return paths
    try:
        # Utiliser mode 'r' avec format='NETCDF4' pour compatibilité Dask
        with netCDF4.Dataset(nc_path, "r", format='NETCDF4') as nc:
            groups = walk(nc)
        gc.collect()
        return groups
    except Exception as e:
        logger.warning(f"Could not read groups from {nc_path}: {e}")
        traceback.print_exc()
        return []

def open_and_concat_groups(
    source: Union[FSMap, str],
    group_paths: List[str] = None,
    dask_safe: bool = True,
    **xr_kwargs
) -> xr.Dataset:
    """
    Ouvre récursivement les groupes NetCDF imbriqués et concatène leurs variables dans un seul Dataset.
    Les noms de variables sont préfixés par le chemin du groupe (avec '__' comme séparateur).
    Version optimisée pour Dask workers.

    Args:
        source (Union[FSMap, str]): Chemin du fichier NetCDF ou FSMap (fsspec).
        group_paths (List[str] or None): Liste des chemins complets des groupes à ouvrir.
        dask_safe (bool): Utiliser les paramètres Dask-safe.
        **xr_kwargs: Arguments additionnels pour xr.open_dataset.

    Returns:
        xr.Dataset: Dataset concaténé.
    """
    # Configuration par défaut pour Dask
    safe_kwargs = {
        'chunks': 'auto',
        'engine': 'netcdf4',
        **xr_kwargs
    }
    
    if dask_safe:
        safe_kwargs.update({
            'lock': False,  # Important pour les workers Dask
            'cache': False,  # Évite les conflits de cache
        })
    
    if not group_paths:
        return xr.open_dataset(source, **safe_kwargs)

    datasets = []
    for group_path in group_paths:
        try:
            ds = xr.open_dataset(source, group=group_path, **safe_kwargs)
            prefix = group_path.replace("/", "__")
            ds = ds.rename({var: f"{prefix}__{var}" for var in ds.data_vars})
            datasets.append(ds)
        except Exception as e:
            logger.warning(f"Failed to open group {group_path}: {e}")
            continue

    if not datasets:
        raise ValueError("No valid groups could be opened")

    ds_merged = xr.merge(datasets, compat="no_conflicts", join="outer")
    return ds_merged


class FileLoader:
    @staticmethod
    def open_dataset_auto(
        path: str,
        manager: Any,
        groups: Optional[list[str]] = None,
        engine: Optional[str] = "netcdf4",
        dask_safe: bool = True,
    ) -> xr.Dataset:
        """
        Open a dataset automatically, handling both NetCDF and Zarr formats.
        Optimized for Dask worker compatibility.

        Args:
            path (str): Path to the dataset (local or remote).
            manager (Any): Connection manager providing the filesystem.
            groups (Optional[list[str]]): NetCDF groups to open.
            engine (Optional[str]): Engine to use for opening files.
            dask_safe (bool): Whether to use Dask-safe configurations.

        Returns:
            xr.Dataset: Opened dataset.
        """
        try:
            # Configuration de base pour Dask
            base_kwargs = {
                "chunks": 'auto',
            }
            
            if dask_safe:
                base_kwargs.update({
                    "lock": False,  # Désactive le verrouillage pour les workers
                    "cache": False,  # Évite les conflits de cache
                })
            
            # Convertir en string si c'est un objet file-like
            path_str = str(path)
            
            # Vérifier si c'est un fichier Zarr ou NetCDF
            if path_str.endswith(".zarr"):
                logger.debug(f"Opening Zarr dataset: {path_str}")
                zarr_kwargs = {"chunks": "auto"}
                if hasattr(manager, 'params') and hasattr(manager.params, 'fs'):
                    return xr.open_zarr(manager.params.fs.get_mapper(path), **zarr_kwargs)
                else:
                    return xr.open_zarr(path, **zarr_kwargs)
            else:
                # Gérer les objets file-like avec le bon engine
                if hasattr(path, 'read'):  # C'est un objet file-like
                    logger.debug(f"Opening NetCDF from file-like object with h5netcdf engine")
                    # Utiliser h5netcdf pour les objets file-like
                    base_kwargs["engine"] = "h5netcdf"
                    return xr.open_dataset(path, **base_kwargs)
                else:
                    # C'est un chemin string, utiliser l'engine spécifié
                    base_kwargs["engine"] = engine
                    
                    # Lister les groupes pour les fichiers locaux
                    group_paths = list_all_group_paths(path_str)
                    if group_paths:
                        ds = open_and_concat_groups(
                            path,
                            group_paths=group_paths,
                            dask_safe=dask_safe,
                            **base_kwargs,
                        )
                        return ds
                    else:
                        ds = xr.open_dataset(path, **base_kwargs)
                        return ds
                        
        except Exception as exc:
            logger.error(f"Failed to open dataset {path}: {repr(exc)}")
            raise

    @staticmethod
    def load_dataset(
        file_path: str,
        adaptive_chunking: bool = False,
        groups: Optional[list[str]] = None,
        engine: Optional[str] = "netcdf4",
        variables: Optional[list[str]] = None,
        dask_safe: bool = True,
    ) -> xr.Dataset | None:
        """
        Load a dataset with Dask-safe configurations.
        
        Args:
            file_path (str): Path to the file.
            adaptive_chunking (bool): Whether to use adaptive chunking.
            groups (Optional[list[str]]): NetCDF groups to load.
            engine (Optional[str]): Engine to use.
            variables (Optional[list[str]]): Variables to keep.
            dask_safe (bool): Whether to use Dask-safe configurations.
        
        Returns:
            xr.Dataset | None: Loaded dataset or None if error.
        """
        try:
            # Configuration de base pour Dask
            open_kwargs = {"chunks": 'auto', "engine": engine}
            
            if dask_safe:
                open_kwargs.update({
                    "lock": False,  # Important pour éviter les deadlocks
                    "cache": False,  # Évite les conflits de cache entre workers
                })
            
            # NOTE: variables filtering doit être fait APRÈS ouverture du dataset
            # pour éviter les erreurs avec des variables non définies
            
            if file_path.endswith(".nc"):
                group_paths = list_all_group_paths(file_path)
                if group_paths:
                    # Ouvre chaque groupe séparément, puis concatène
                    datasets = []
                    for group_path in group_paths:
                        try:
                            ds = xr.open_dataset(file_path, group=group_path, **open_kwargs)
                            datasets.append(ds)
                        except Exception as e:
                            logger.warning(f"Failed to open group {group_path}: {e}")
                            continue
                    
                    if datasets:
                        ds = xr.merge(datasets, compat="no_conflicts", join="outer")
                    else:
                        logger.error(f"No valid groups found in {file_path}")
                        return None
                else:
                    ds = xr.open_dataset(file_path, **open_kwargs)
                
            elif file_path.endswith(".zarr"):
                zarr_kwargs = {"chunks": 'auto'}
                ds = xr.open_zarr(file_path, **zarr_kwargs)
            else:
                raise ValueError(f"Unsupported file format {file_path}.")
            
            # Filtrage des variables après ouverture
            if variables and ds is not None:
                available_vars = list(ds.variables.keys())
                vars_to_drop = [v for v in available_vars if v not in variables]
                if vars_to_drop:
                    ds = ds.drop_vars(vars_to_drop, errors='ignore')
            
            return ds
            
        except Exception as error:
            logger.error(f"Error when loading file {file_path}: {error}")
            return None
