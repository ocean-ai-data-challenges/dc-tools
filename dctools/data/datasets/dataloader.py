#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Dataloder."""

import traceback
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Type, Union

import dask
import geopandas as gpd
from loguru import logger
import numpy as np
from oceanbench.core.distributed import DatasetProcessor
import pandas as pd
import xarray as xr
import torch
from xrpatcher import XRDAPatcher

from dctools.data.connection.config import BaseConnectionConfig
from dctools.data.connection.connection_manager import (
    ArgoManager,
    BaseConnectionManager,
    CMEMSManager,
    FTPManager,
    LocalConnectionManager,
    GlonetManager,
    S3Manager,
    S3WasabiManager,
    deep_copy_object,
    clean_for_serialization,
)
from dctools.utilities.xarray_utils import (
    filter_variables, interp_single_argo_profile,
)


# Dictionnaire de mapping des noms vers les classes
CLASS_REGISTRY: Dict[Type[BaseConnectionConfig], Type[BaseConnectionManager]] = {
    "S3WasabiManager": S3WasabiManager,
    "FTPManager": FTPManager,
    "GlonetManager": GlonetManager,
    "ArgoManager": ArgoManager,
    "CMEMSManager": CMEMSManager,
    "S3Manager": S3Manager,
    "LocalConnectionManager": LocalConnectionManager,
}


def add_coords_as_dims(ds: xr.Dataset, coords=("LATITUDE", "LONGITUDE")) -> xr.Dataset:
    """
    Add given coordinates as dimensions to all data variables in the dataset,
    broadcasting them if necessary. Handles the case where coordinates exist
    only as per-point arrays (e.g., Argo profiles with N_POINTS).
    
    Parameters
    ----------
    ds : xr.Dataset
        Input dataset.
    coords : tuple of str
        Names of the coordinates to promote to dimensions (if present in ds).
    
    Returns
    -------
    xr.Dataset
        Dataset where the given coords are added as dimensions for all variables.
    """
    out = ds.copy()

    for coord in coords:
        if coord not in ds:
            continue

        coord_da = ds[coord]

        # Cas Argo : coordonnée 1D constante sur N_POINTS
        if coord_da.ndim == 1 and coord_da.dims == ("N_POINTS",):
            unique_vals = coord_da.to_series().unique()
            if len(unique_vals) == 1:
                value = unique_vals[0]

                # Supprimer l'ancienne coordonnée pour éviter le conflit
                out = out.drop_vars(coord)

                # Ajouter comme nouvelle dimension de taille 1
                out = out.expand_dims({coord: [value]})

                # Broadcast sur toutes les variables
                for v in out.data_vars:
                    out[v] = out[v].broadcast_like(out[coord])

                continue

        # Cas général (coords déjà bien définies)
        out = out.assign_coords({coord: coord_da})
        for v in out.data_vars:
            if coord not in out[v].dims:
                out[v] = out[v].broadcast_like(out[coord])

    return out


def swath_to_points(
        ds,
        vars_to_keep=None,
        drop_coords=["num_lines", "num_pixels", "num_nadir"],
        coords_to_keep=None,
    ):  # , drop_missing=True):
    """
    Convert a swath-style Dataset into a 1D point collection along 'n_points'.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset, possibly with swath-like dimensions (e.g. num_lines, num_pixels).
    vars_to_keep : list of str, optional
        Variables to retain. Default = all data variables.
    drop_missing : bool, default True
        If True, drop points where *all* variables are NaN or _FillValue.

    Returns
    -------
    xr.Dataset
        Flattened dataset with dimension 'n_points'.
        Includes time, lat, lon if present.
    """
    # Identify swath dimensions
    if "n_points" in ds.dims:
        # Already flat
        return ds
    
    swath_dims = [d for d in ds.dims if d not in ("time", "n_points")]
    if not swath_dims:
        raise ValueError("No swath dims found (already 1D or unexpected format).")

    # Select variables
    if vars_to_keep is None:
        vars_to_keep = list(ds.data_vars)

    # Stack swath dims into 'n_points'
    ds_flat = ds[vars_to_keep].stack(n_points=swath_dims)

    # Sauvegarder les coordonnées importantes avant suppression
    coords_to_reassign = {}
    for coord in coords_to_keep:
        if coord in ds.coords:
            arr = ds.coords[coord]
            # Si la coordonnée dépend des swath dims, on la réindexe sur n_points
            if set(arr.dims) <= set(swath_dims):
                coords_to_reassign[coord] = arr.stack(n_points=swath_dims).values
            elif "n_points" in arr.dims:
                coords_to_reassign[coord] = arr.values

    # Supprimer les coordonnées orphelines (non utilisées)
    for coord in drop_coords:
        if coord in ds_flat.coords:
            ds_flat = ds_flat.drop_vars(coord)

    # Ré-attacher les coordonnées importantes
    for coord, vals in coords_to_reassign.items():
        ds_flat = ds_flat.assign_coords({coord: ("n_points", vals)})

    # Reset attributes to avoid concat conflicts
    ds_flat.attrs = {}

    # Ensure time is broadcast to n_points
    if "time" in ds_flat.coords and ds_flat["time"].ndim < ds_flat["n_points"].ndim:
        # Case: time per line only → broadcast to pixels
        ds_flat = ds_flat.assign_coords(
            time=("n_points", np.repeat(ds_flat["time"].values, np.prod([ds_flat.sizes[d] for d in swath_dims[1:]])))
        )

    return ds_flat


def add_time_dim(
    ds: xr.Dataset,
    input_df: pd.DataFrame,
    points_dim: str,
    time_coord,
    idx: int,
):
    """
    Ensure that dataset has a 'time' dimension compatible with swath/n-points structure.
    Covers cases:
      - No time info available (fallback: mid_time from metadata).
      - One unique time value for all points.
      - Multiple time values (per-point time).
      - Existing time coordinate.
    """
    if time_coord is None:
        # Fallback: use metadata mid_time
        file_info = input_df.iloc[idx]
        mid_time = file_info["date_start"] + (file_info["date_end"] - file_info["date_start"]) / 2
        ds = ds.assign_coords(time=(points_dim, np.full(ds.sizes[points_dim], mid_time)))
        if "time" not in ds.dims:
            ds = ds.expand_dims(time=[mid_time])
        return ds

    # Standardize time_coord to pandas datetime
    time_values = pd.to_datetime(getattr(time_coord, "values", time_coord))

    if points_dim in getattr(time_coord, "dims", []):
        # Case: time per point
        unique_times = pd.unique(time_values)

        if len(unique_times) == 1:
            # Only one unique time → add as scalar dimension if not already there
            if "time" in ds.dims:
                return ds  # already present
            else:
                return ds.expand_dims(time=[unique_times[0]])
        else:
            # Per-point times: assign as coordinate
            ds = ds.assign_coords(time=(points_dim, time_values))
            return ds

    else:
        # Case: time scalar or broadcastable
        if np.ndim(time_values) == 0:
            time_val = pd.to_datetime(time_values)
        else:
            time_val = pd.to_datetime(time_values[0])

        if "time" in ds.dims:
            # Already has a time dimension → just overwrite if needed
            ds = ds.assign_coords(time=[time_val])
            return ds
        else:
            return ds.expand_dims(time=[time_val])


def preprocess_one_npoints(
    source, is_swath, n_points_dims, filtered_df, idx,
    connection_params, class_name, alias,
    keep_variables_list, target_dimensions,
    coordinates,
    argo_index=None, time_bounds=None,
):
    if class_name not in CLASS_REGISTRY:
        raise ValueError(f"Unknown class name: {class_name}")
  
    manager_class = CLASS_REGISTRY[class_name]
    if class_name == "CMEMSManager":
        manager = manager_class(
            connection_params, call_list_files=False,
            do_logging=False,
        )
    elif class_name == "ArgoManager":
        manager = manager_class(
            connection_params, call_list_files=False,
            argo_index=argo_index,
        )

    else:
        manager = manager_class(
            connection_params, call_list_files=False
        )
    try:
        # open dataset
        if alias is not None:
            ds = manager.open(source, alias)
        else:
            ds = manager.open(source)
        if ds is None:
            return None

        ds_float32 = ds.copy()
        for var in ds_float32.data_vars:
            if np.issubdtype(ds_float32[var].dtype, np.floating):
                ds_float32[var] = ds_float32[var].astype(np.float32)

        ds_filtered = filter_variables(ds_float32, keep_variables_list)

        if is_swath:
            coords_to_keep = [
                coordinates.get('time', None),
                coordinates.get('depth', None),
                coordinates.get('lat', None),
                coordinates.get('lon', None),
            ]
            coords_to_keep = list(filter(lambda x: x is not None, coords_to_keep))
            ds_filtered = swath_to_points(
                ds_filtered, vars_to_keep=keep_variables_list,
                coords_to_keep=list(coordinates.keys()),
            )

        # Chercher une coordonnée/variable temporelle
        time_name = coordinates['time']
        if time_name in ds_filtered.variables and time_name not in ds_filtered.coords:
            ds_filtered = ds_filtered.set_coords(time_name)
        time_coord = ds_filtered.coords[time_name]

        # Filtrer les valeurs de temps (si profil ARGO)
        if class_name == "ArgoManager" and time_bounds is not None:
            ds_filtered = manager.filter_argo_profile_by_time(
                ds_filtered,
                tmin=time_bounds[0],
                tmax=time_bounds[1],
            )

        # Identifier la dimension de points
        points_dim = None
        for dim_name in n_points_dims:
            if dim_name in ds_filtered.dims:
                points_dim = dim_name
                break
        
        if points_dim is None:
            logger.warning(f"Dataset {idx}: No points dimension found")
            return None

        ds_with_time = add_time_dim(
            ds_filtered, filtered_df, points_dim=points_dim, time_coord=time_coord, idx=idx
        )

        if coordinates.get('depth', None) in list(ds_with_time.coords):
            # sous-échantilloner depth aux valeurs de la grille cible (par interpolation)
            ds_interp = interp_single_argo_profile(ds_with_time, target_dimensions['depth'])
        else:
            ds_interp = ds_with_time

        ds_interp = ds_interp.chunk({points_dim: 50000})
        return ds_interp

    except Exception as e:
        logger.warning(f"Failed to process n_points dataset {idx}: {e}")
        traceback.print_exc()
        return None


class EvaluationDataloader:
    def __init__(
        self,
        params: dict,
    ):
        """
        Initialise le dataloader pour les ensembles de données.

        Args:
            params: dictionnaire des paramètres
        """

        for key, value in params.items():
            setattr(self, key, value)
        self.pred_coords = self.pred_catalog.get_global_metadata().get("coord_system", None)
        self.ref_coords = {ref_alias: ref_catalog.get_global_metadata().get("coord_system", None)
                           for ref_alias, ref_catalog in self.ref_catalogs.items()}

        self.optimize_for_parallel = True
        self.min_batch_size_for_parallel = 4

    def __len__(self):
        if self.forecast_mode and self.forecast_index is not None:
            return len(self.forecast_index)
        return len(self.pred_catalog.get_dataframe())

    def __iter__(self):
        return self._generate_batches()

    def _find_matching_ref(self, valid_time, ref_alias):
        """Trouve le fichier de référence couvrant valid_time pour ref_alias."""
        ref_df = self.ref_catalogs[ref_alias].get_dataframe()
        match = ref_df[(ref_df["date_start"] <= valid_time) & (ref_df["date_end"] >= valid_time)]
        if not match.empty:
            return match.iloc[0]["path"]
        return None

    def _generate_batches(self) -> Generator[List[Dict[str, Any]], None, None]:
        batch = []
        for _, row in self.forecast_index.iterrows():
            # Vérifier si on a suffisamment de données pour ce forecast
            forecast_reference_time = row["forecast_reference_time"]
            lead_time = row["lead_time"]
            valid_time = row["valid_time"]

            # Calculer la fin du forecast complet (dernier lead time)
            max_lead_time = self.n_days_forecast - 1  # 0-indexé
            if hasattr(self, 'lead_time_unit') and self.lead_time_unit == "hours":
                forecast_end_time = forecast_reference_time + pd.Timedelta(hours=max_lead_time)
            else:
                forecast_end_time = forecast_reference_time + pd.Timedelta(days=max_lead_time)
            
            # Vérifier que le forecast complet est dans la plage de données disponibles
            for ref_alias in self.ref_aliases:
                if ref_alias:
                    ref_catalog = self.ref_catalogs[ref_alias]
                    ref_df = ref_catalog.get_dataframe()
                    
                    # Trouver la date de fin maximale disponible dans les données de référence
                    max_available_date = ref_df["date_end"].max()
                    
                    # Si la fin du forecast dépasse les données disponibles, ignorer cette entrée
                    if forecast_end_time > max_available_date:
                        logger.debug(f"Skipping forecast starting at {forecast_reference_time}: "
                                f"forecast ends at {forecast_end_time} but data only available until {max_available_date}")
                        continue
                
                entry = {
                    "forecast_reference_time": forecast_reference_time,
                    "lead_time": lead_time,
                    "valid_time": valid_time,
                    "pred_data": row["file"],
                    "ref_data": None,
                    "ref_alias": ref_alias,
                    "pred_coords": self.pred_coords,
                    "ref_coords": self.ref_coords[ref_alias] if ref_alias else None,
                }
                
                if ref_alias:
                    ref_catalog = self.ref_catalogs[ref_alias]
                    coord_system = ref_catalog.get_global_metadata().get("coord_system")
                    is_observation = coord_system.is_observation_dataset() if coord_system else False
                    
                    if is_observation:
                        # Logique observation : filtrer le catalogue d'observation sur l'intervalle temporel du forecast_index
                        obs_time_interval = (valid_time, valid_time)
                        keep_vars = self.keep_variables[ref_alias]
                        rename_vars_dict = self.metadata[ref_alias]['variables_dict']
                        keep_vars = [rename_vars_dict[var] for var in keep_vars if var in rename_vars_dict]
                        target_dimensions = self.target_dimensions
                        ref_entries = ObservationDataViewer(
                            ref_catalog.get_dataframe(),
                            self.open_ref, ref_alias,
                            keep_vars, target_dimensions, self.metadata[ref_alias],
                            self.time_tolerance,
                            dataset_processor=getattr(self, 'dataset_processor', None),
                        )
                        ref_manager = self.ref_managers[ref_alias]
                        ref_manager_class_name = ref_manager.__class__.__name__
                        connection_params = ref_manager.get_config_clean_copy()
                        filtered_obs = ref_entries.preprocess_datasets(
                            ref_manager,
                            ref_manager_class_name,
                            connection_params,
                            obs_time_interval,
                        )

                        if filtered_obs is None:
                            continue
                        entry["ref_data"] = filtered_obs
                        entry["ref_is_observation"] = True
                    else:
                        # Logique gridded : associer le fichier de référence couvrant valid_time
                        ref_path = self._find_matching_ref(valid_time, ref_alias)
                        if ref_path is None:
                            logger.debug(f"No reference data found for valid_time {valid_time}")
                            continue
                        entry["ref_data"] = ref_path
                        entry["ref_is_observation"] = False
                
                batch.append(entry)
                # Adapter la taille de batch selon le mode parallélisation
                # target_batch_size = self._get_optimal_batch_size()
                if len(batch) >= self.batch_size:
                    yield batch
                    batch = []
        if batch:
            yield batch


    def open_pred(self, pred_entry: str) -> xr.Dataset:
        pred_data = self.pred_manager.open(pred_entry, self.file_cache)
        return pred_data

    def open_ref(self, ref_entry: str, ref_alias: str) -> xr.Dataset:
        ref_data = self.ref_managers[ref_alias].open(ref_entry, self.file_cache)
        return ref_data
    

class TorchCompatibleDataloader:
    def __init__(
        self,
        dataloader: EvaluationDataloader,
        patch_size: Tuple[int, int],
        stride: Tuple[int, int],
    ):
        """
        Initialise un dataloader compatible avec PyTorch.

        Args:
            dataloader (EvaluationDataloader): Le dataloader existant.
            patch_size (Tuple[int, int]): Taille des patches (hauteur, largeur).
            stride (Tuple[int, int]): Pas de glissement pour les patches.
        """
        self.dataloader = dataloader
        self.patch_size = patch_size
        self.stride = stride

    def __len__(self):
        """
        Retourne le nombre total de lots dans le dataloader.
        """
        return len(self.dataloader)

    def __iter__(self) -> Generator[Dict[str, torch.Tensor], None, None]:
        """
        Génère des lots de données compatibles avec PyTorch.

        Yields:
            Dict[str, torch.Tensor]: Un dictionnaire contenant les patches de données.
        """
        for batch in self.dataloader:
            for entry in batch:
                pred_data = entry["pred_data"]
                ref_data = entry["ref_data"]

                # Générer des patches pour les données de prédiction
                pred_patches = self._generate_patches(pred_data)

                # Générer des patches pour les données de référence (si disponibles)
                ref_patches = self._generate_patches(ref_data) if ref_data is not None else None

                # Retourner les patches sous forme de tensors PyTorch
                yield {
                    "date": entry["date"],
                    "pred_patches": pred_patches,
                    "ref_patches": ref_patches,
                }

    def _generate_patches(self, dataset: xr.Dataset) -> torch.Tensor:
        """
        Génère des patches à partir d'un dataset xarray.

        Args:
            dataset (xr.Dataset): Le dataset xarray.

        Returns:
            torch.Tensor: Les patches sous forme de tensor PyTorch.
        """
        patcher = XRDAPatcher(
            data=dataset,
            patch_size=self.patch_size,
            stride=self.stride,
        )
        patches = patcher.extract_patches()
        return torch.tensor(patches)

def concat_with_dim(
    datasets: List[xr.Dataset],
    concat_dim: str,
    sort: bool = True,
):
    datasets_with_dim = []
    for i, ds in enumerate(datasets):
        if concat_dim not in ds.dims:
                ds = ds.expand_dims({concat_dim: [i]})
        datasets_with_dim.append(ds)

    result = dask.delayed(
        xr.concat)(datasets_with_dim, dim=concat_dim,
        coords="minimal",
        compat="override", join="outer"
    )
    if sort:
        result = dask.delayed(result.sortby)(concat_dim)
    return result


class ObservationDataViewer:
    def __init__(
        self,
        source: Union[xr.Dataset, List[xr.Dataset], pd.DataFrame, gpd.GeoDataFrame],
        load_fn: Callable[[str], xr.Dataset],
        alias: str,
        keep_vars: List[str],
        target_dimensions: Dict[str, Any],
        dataset_metadata: Any,
        time_tolerance: pd.Timedelta = pd.Timedelta("12h"),
        dataset_processor: Optional[DatasetProcessor] = None,
    ):
        """
        Parameters:
            source: either
                - one or more xarray Datasets (data already loaded)
                - a DataFrame/GeoDataFrame containing metadata, including file links
            load_fn: a callable that loads a dataset given a link (required if source is a DataFrame)
            alias: optional alias to pass to load_fn if needed
            time_tolerance: time tolerance to apply when filtering by time
        """
        self.is_metadata = isinstance(source, (pd.DataFrame, gpd.GeoDataFrame))
        self.load_fn = load_fn
        self.time_tolerance = time_tolerance
        self.alias = alias
        self.keep_vars = keep_vars
        self.target_dimensions = target_dimensions
        self.dataset_processor = dataset_processor 

        if self.is_metadata:
            if self.load_fn is None:
                raise ValueError("A `load_fn(link: str)` must be provided when using metadata.")
            self.meta_df = source
        else:
            self.datasets = source if isinstance(source, list) else [source]
        self.coordinates = dataset_metadata['coord_system'].coordinates

    def preprocess_datasets(
        self,
        dataset_manager: Any,
        class_name: str,
        connect_config: BaseConnectionConfig,
        time_interval: tuple
    ) -> xr.Dataset:
        """
        Version qui évite les conflits de dimensions en supprimant les variables num_nadir
        et ne conserve que les variables océanographiques 2D principales.
        """
        connection_params = deep_copy_object(
            connect_config, skip_list=['dataset_processor', 'fs']
        )
        connection_params = clean_for_serialization(connection_params)

        if class_name not in CLASS_REGISTRY:
            raise ValueError(f"Unknown class name: {class_name}")
        
        if class_name == "ArgoManager":
            argo_index = dataset_manager.get_argo_index()
            scattered_argo_index = self.dataset_processor.scatter_data(
                argo_index, broadcast_item=False)
        else:
            scattered_argo_index = None
    
        t0, t1 = time_interval
        t0 = t0 - self.time_tolerance
        t1 = t1 + self.time_tolerance
        time_bounds = (t0, t1)
        filtered_df = self.filter_by_time(t0, t1)

        if filtered_df.empty:
            logger.warning(f"No {self.alias} Data for time interval: {time_interval}")
            return None

        # chargement des fichiers  
        dataset_paths = [row["path"] for _, row in filtered_df.iterrows()]

        first_ds = None
        while first_ds is None:
            if self.alias is not None:
                first_ds = dataset_manager.open(dataset_paths[0], self.alias)
            else:
                first_ds = dataset_manager.open(dataset_paths[0])

        swath_dims = {"num_lines", "num_pixels", "num_nadir"}
        n_points_dims = {"n_points", "N_POINTS", "points", "obs"}
    
        # Données avec dimension n_points/N_POINTS uniquement
        if any(dim in first_ds.dims for dim in n_points_dims) and not swath_dims.issubset(first_ds.dims):            
            try:
                # Nettoyer et traiter les datasets
                delayed_tasks = []
                for idx, dataset_path in enumerate(dataset_paths):
                    delayed_tasks.append(dask.delayed(preprocess_one_npoints)(
                        dataset_path, False, n_points_dims, filtered_df, idx,
                        connection_params, class_name, self.alias,
                        self.keep_vars, self.target_dimensions,
                        self.coordinates,
                        scattered_argo_index, time_bounds,
                    ))
                batch_results = self.dataset_processor.compute_delayed_tasks(
                    delayed_tasks, sync=False
                )
                cleaned_datasets = [meta for meta in batch_results if meta is not None]

                if not cleaned_datasets:
                    logger.error("No n_points datasets could be processed")
                    traceback.print_exc()
                    return None
                
                # Concaténer le long de la dimension temporelle
                if len(cleaned_datasets) == 1:
                    result = cleaned_datasets[0]
                    logger.info("Single n_points dataset - no concatenation needed")
                else:
                    try:
                        result = concat_with_dim(cleaned_datasets, dim="time")
                        logger.info(f"Successfully concatenated {len(cleaned_datasets)} n_points datasets")
                    except Exception as e:
                        logger.error(f"N_points concatenation failed: {e}")
                        result = cleaned_datasets[0]  # Fallback au premier
                
                # Ajouter métadonnées sur la stratégie utilisée
                result.attrs.update({
                    'data_processing_strategy': 'n_points_with_time_dimension',
                    'original_datasets': len(dataset_paths),
                    'processed_datasets': len(cleaned_datasets),
                    'final_variables': list(result.data_vars.keys()),
                    'time_steps': result.sizes.get('time', 1),
                })
                
                logger.info(f"Final n_points result: {result.sizes.get('time', 1)} time steps, "
                        f"{len(result.data_vars)} variables")
                return result
                
            except Exception as e:
                logger.error(f"Complete n_points processing failed: {e}")
                traceback.print_exc()
                return None

        # Données Swath
        elif swath_dims.issubset(first_ds.dims):
            logger.info("Swath data detected - reshaping and filtering to n_points")
            
            try:
                is_swath_data = True
                # nettoyer les datasets
                delayed_tasks = []
                for idx, dataset_path in enumerate(dataset_paths):
                    delayed_tasks.append(dask.delayed(preprocess_one_npoints)(
                        dataset_path, is_swath_data, n_points_dims, filtered_df, idx,
                        connection_params, class_name, self.alias,
                        self.keep_vars, self.target_dimensions,
                        self.coordinates,
                        time_bounds,
                    ))
                batch_results = self.dataset_processor.compute_delayed_tasks(delayed_tasks)
                cleaned_datasets = [meta for meta in batch_results if meta is not None]

                if not cleaned_datasets:
                    logger.error("No datasets could be cleaned and reshaped")
                    return None

                # vérifier la compatibilité
                first_ds = cleaned_datasets[0]
                target_n_points = first_ds.sizes['n_points']
                target_vars = set(first_ds.data_vars.keys())
                
                logger.info(f"Target structure: {target_n_points:,} points, variables: {target_vars}")
                
                # Filtrer les datasets compatibles
                compatible_datasets = []
                for i, ds in enumerate(cleaned_datasets):
                    if (ds.sizes['n_points'] == target_n_points and 
                        set(ds.data_vars.keys()) == target_vars):
                        compatible_datasets.append(ds)
                    else:
                        logger.warning(f"Dataset {i} incompatible: {ds.sizes['n_points']} points, "
                                    f"variables: {set(ds.data_vars.keys())}")
                
                if not compatible_datasets:
                    logger.error("No compatible datasets after filtering")
                    return cleaned_datasets[0]  # Retourner au moins le premier
                
                logger.info(f"Found {len(compatible_datasets)} compatible datasets")
                
                # concaténation
                try:
                    result = concat_with_dim(compatible_datasets, concat_dim="n_points", sort=False)

                except Exception as e:
                    logger.error(f"Concatenation failed: {e}")
                    result = compatible_datasets[0]  # Fallback au premier
                
                # Ajouter métadonnées sur la stratégie utilisée
                result.attrs.update({
                    'data_processing_strategy': 'num_nadir_filtered_2d_only',
                    'original_datasets': len(dataset_paths),
                    'processed_datasets': len(compatible_datasets),
                })
                
                return result
                
            except Exception as e:
                logger.error(f"Complete processing failed: {e}")
                traceback.print_exc()
                return first_ds

        # données non-swath, non n_points
        elif "time" in first_ds.dims or "time" in first_ds.coords:
            try:
                if self.alias is not None:
                    all_datasets = [dataset_manager.open(row["path"], self.alias) for _, row in filtered_df.iterrows()][:5]
                else:
                    all_datasets = [dataset_manager.open(row["path"]) for _, row in filtered_df.iterrows()]
                for idx, ds in enumerate(all_datasets):
                    all_datasets[idx] = filter_variables(ds, self.keep_vars)
                combined = concat_with_dim(all_datasets, concat_dim="time", sort=True)
                combined = dask.delayed(combined.sel)(time=slice(t0, t1))
                return combined
            except Exception as e:
                logger.error(f"Failed to concatenate time series data: {e}")
                traceback.print_exc()
                return first_ds

        else:
            logger.warning("Unknown data structure, returning first dataset")
            return first_ds

    def filter_by_time(self, t0: pd.Timestamp, t1: pd.Timestamp) -> List[xr.Dataset]:
        """
        Returns a list of datasets that fall within the time window.
        If source is metadata, loads only the required datasets.
        """
        
        # Convertir t0, t1 en datetime64[s] (précision seconde)
        t0_clean = np.datetime64(pd.Timestamp(t0).floor('s'))
        t1_clean = np.datetime64(pd.Timestamp(t1).ceil('s'))
        
        # Convertir les colonnes DataFrame en datetime64[s] également
        df_date_start = pd.to_datetime(self.meta_df["date_start"]).dt.floor('s')
        df_date_end = pd.to_datetime(self.meta_df["date_end"]).dt.floor('s')
        
        # Conversion explicite en datetime64[s] pour uniformiser le format
        df_date_start_clean = df_date_start.values.astype('datetime64[s]')
        df_date_end_clean = df_date_end.values.astype('datetime64[s]')
        
        # Filtrage temporel
        mask1 = df_date_start_clean <= t1_clean
        mask2 = df_date_end_clean >= t0_clean
        combined_mask = mask1 & mask2
        filtered = self.meta_df[combined_mask]

        return filtered

    def filter_by_time_and_region(
        self,
        time_range: Tuple[pd.Timestamp, pd.Timestamp],
        lon_bounds: Tuple[float, float],
        lat_bounds: Tuple[float, float]
    ) -> List[xr.Dataset]:
        """
        Filters by both time and bounding box [lon_min, lon_max], [lat_min, lat_max].
        Only applies to datasets that contain time and spatial coordinates.
        """
        t0, t1 = time_range
        lon_min, lon_max = lon_bounds
        lat_min, lat_max = lat_bounds

        if self.is_metadata:
            filtered = self.meta_df[
                (self.meta_df["date_start"] >= t0) & (self.meta_df["date_end"] <= t1) &
                (self.meta_df["lon"] >= lon_min) & (self.meta_df["lon"] <= lon_max) &
                (self.meta_df["lat"] >= lat_min) & (self.meta_df["lat"] <= lat_max)
            ]
            return [self.load_fn(row["link"]) for _, row in filtered.iterrows()]
        else:
            result = []
            for ds in self.datasets:
                if not all(k in ds.coords for k in ["lat", "lon", "time"]):
                    continue
                ds_subset = ds.sel(
                    time=slice(t0, t1),
                    lon=slice(lon_min, lon_max),
                    lat=slice(lat_min, lat_max)
                )
                result.append(ds_subset)
            return result
