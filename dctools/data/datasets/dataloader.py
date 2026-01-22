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
from concurrent.futures import ThreadPoolExecutor, as_completed
# from pathlib import Path
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
)
from dctools.utilities.xarray_utils import (
    filter_variables,
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
            # Optimize: Avoid full load .to_series().unique() which is eager and RAM heavy
            # Use min/max check instead (lazy on dask)
            cmin = coord_da.min(skipna=True)
            cmax = coord_da.max(skipna=True)
            
            # If dask, compute small scalars only
            if hasattr(cmin.data, "compute"):
                import dask
                cmin, cmax = dask.compute(cmin, cmax)
            else:
                cmin = cmin.values
                cmax = cmax.values
            
            # If min matches max, it's a constant value
            if cmin == cmax:
                value = cmin.item() if hasattr(cmin, "item") else cmin

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
        drop_coords=["num_lines", "num_pixels", "num_nadir"],
        coords_to_keep=None,
        n_points_dim = "n_points",
    ):  # , drop_missing=True):
    """
    Convert a swath-style Dataset into a 1D point collection along 'n_points'.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset, possibly with swath-like dimensions (e.g. num_lines, num_pixels).
    drop_missing : bool, default True
        If True, drop points where *all* variables are NaN or _FillValue.

    Returns
    -------
    xr.Dataset
        Flattened dataset with dimension 'n_points'.
        Includes time, lat, lon if present.
    """

    if "n_points" in ds.dims:
        # Already flat
        return ds
    
    swath_dims = [d for d in ds.dims if d not in ("time", n_points_dim)]
    if not swath_dims:
        raise ValueError("No swath dims found (already 1D or unexpected format).")

    # Stack swath dims into 'n_points'
    ds_flat = ds.stack(n_points=swath_dims)

    # Sauvegarder les coordonnées importantes avant suppression
    coords_to_reassign = {}
    for coord in coords_to_keep:
        if coord in ds.coords:
            arr = ds.coords[coord]
            # Si la coordonnée dépend des swath dims, on la réindexe sur n_points
            if set(arr.dims) <= set(swath_dims):
                # Use .data to preserve dask arrays instead of .values (which forces compute)
                coords_to_reassign[coord] = arr.stack(n_points=swath_dims).data
            elif n_points_dim in arr.dims:
                coords_to_reassign[coord] = arr.data

    # Supprimer les coordonnées orphelines (non utilisées)
    for coord in drop_coords:
        if coord in ds_flat.coords:
            ds_flat = ds_flat.drop_vars(coord)

    # Ré-attacher les coordonnées importantes
    for coord, vals in coords_to_reassign.items():
        ds_flat = ds_flat.assign_coords({coord: (n_points_dim, vals)})

    # Reset attributes to avoid concat conflicts
    ds_flat.attrs = {}

    # Ensure time is broadcast to n_points
    if "time" in ds_flat.coords and ds_flat["time"].ndim < ds_flat[n_points_dim].ndim:
        pass
        # Case: time per line only → broadcast to pixels
        # ds_flat = ds_flat.assign_coords(
        #     time=(n_points_dim, np.repeat(ds_flat["time"].values, np.prod([ds_flat.sizes[d] for d in swath_dims[1:]])))
        # )

    return ds_flat


def add_time_dim(
    ds: xr.Dataset,
    input_df: pd.DataFrame,
    n_points_dim: str,
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
        ds = ds.assign_coords(time=(n_points_dim, np.full(ds.sizes[n_points_dim], mid_time)))
        if "time" not in ds.dims:
            ds = ds.expand_dims(time=[mid_time])
        return ds

    # Check if time_coord is a dask array (lazy)
    is_lazy = hasattr(time_coord, "chunks") or (hasattr(time_coord, "data") and hasattr(time_coord.data, "chunks"))

    if is_lazy:
        # Avoid loading .values and computing unique()
        # Assume per-point coordinates (safest default for n_points dims)
        
        try:
             # Si déjà datetime, on garde tel quel
             data_to_assign = time_coord
             
             # Attention: si time_coord a des dimensions (ex: n_points), il faut assigner avec la dim
             dims = getattr(time_coord, "dims", (n_points_dim,)) # Tuple de dimensions
             
             # Si time_coord vient de .coords, il a déjà ses dimensions
             # Sinon on suppose n_points_dim
             
             ds = ds.assign_coords(time=data_to_assign)
             return ds
        except Exception as e:
             logger.warning(f"Lazy time assignment failed, falling back to eager: {e}")
             # Fallback to eager execution below
             pass

    # Standardize time_coord to pandas datetime
    # This forces loading into memory (.values)
    # Check if already datetime64 to avoid costly pd.to_datetime conversion on huge arrays
    raw_values = getattr(time_coord, "values", time_coord)
    
    if np.issubdtype(raw_values.dtype, np.datetime64):
         time_values = raw_values
    else:
         try:
             # Fast path for large arrays: pd.to_datetime can be slow on large object arrays
             time_values = pd.to_datetime(raw_values)
         except Exception:
             # Fallback if errors
             time_values = pd.to_datetime(raw_values, errors='coerce')

    # Case: time per point
    # OPTIMIZATION: Avoid pd.unique on massive arrays if not strictly necessary
    # Check shape first
    is_scalar = (np.ndim(time_values) == 0) or (time_values.size == 1)
    
    if n_points_dim in getattr(time_coord, "dims", []) and not is_scalar:
        # If huge array, assume unique times are many -> treat as coordinate
        # Only check uniqueness if relatively small (<100k) to save CPU/RAM
        if time_values.size < 100_000:
            unique_times = pd.unique(time_values)
            is_single_time = (len(unique_times) == 1)
        else:
            is_single_time = False # Assume variation to stay safe and fast

        if is_single_time:
            # Only one unique time → add as scalar dimension if not already there
            # ...existing code...
            unique_val = unique_times[0]
            if "time" in ds.dims or "time" in ds.coords:
                return ds  # already present
            else:
                return ds.expand_dims(time=[unique_val])
        else:
            try:
                # Per-point times: assign as coordinate
                ds = ds.assign_coords(time=(n_points_dim, time_values))
                return ds
            except Exception as e:
                logger.error(f"Error assigning time coordinates: {e}")
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


def filter_by_time(df: pd.DataFrame, t0: pd.Timestamp, t1: pd.Timestamp) -> pd.DataFrame:
    """
    Filtre le DataFrame pour ne garder que les entrées dont l'intervalle
    [date_start, date_end] recoupe [t0, t1].
    """
    df_copy = df.copy()
    # Conversion en datetime si besoin
    date_start = pd.to_datetime(df_copy["date_start"])
    date_end = pd.to_datetime(df_copy["date_end"])
    mask = (date_start <= t1) & (date_end >= t0)
    return df_copy[mask]


def extrapolate_to_surface(var_name, valid_depths, valid_vals):
    if len(valid_vals) == 0:
        return np.nan
    
    if var_name in ["TEMP", "temperature"]:
        # Température : gradient réduit vers la surface
        if len(valid_depths) >= 2:
            # Chercher une profondeur différente de valid_depths[0]
            depth_diff = 0
            i = 1
            while i < len(valid_depths) and abs(depth_diff) < 1e-6:
                depth_diff = valid_depths[i] - valid_depths[0]
                i += 1
            
            if abs(depth_diff) < 1e-6:  # Toutes les profondeurs sont identiques
                surface_val = valid_vals[0]
            else:
                gradient = (valid_vals[i-1] - valid_vals[0]) / depth_diff
                surface_val = valid_vals[0] - gradient * 0.5 * valid_depths[0]
        else:
            surface_val = valid_vals[0]
    
    elif var_name in ["PSAL", "salinity"]:
        # Même logique pour la salinité
        if len(valid_depths) >= 2:
            depth_diff = 0
            i = 1
            while i < len(valid_depths) and abs(depth_diff) < 1e-6:
                depth_diff = valid_depths[i] - valid_depths[0]
                i += 1
            
            if abs(depth_diff) < 1e-6:
                surface_val = valid_vals[0]
            else:
                gradient = (valid_vals[i-1] - valid_vals[0]) / depth_diff
                surface_val = valid_vals[0] - gradient * 0.3 * valid_depths[0]
        else:
            surface_val = valid_vals[0]
    
    else:
        # Variables autres : extrapolation linéaire standard
        if len(valid_depths) >= 2:
            depth_diff = 0
            i = 1
            while i < len(valid_depths) and abs(depth_diff) < 1e-6:
                depth_diff = valid_depths[i] - valid_depths[0]
                i += 1
            
            if abs(depth_diff) < 1e-6:
                surface_val = valid_vals[0]
            else:
                gradient = (valid_vals[i-1] - valid_vals[0]) / depth_diff
                surface_val = valid_vals[0] - gradient * valid_depths[0]
        else:
            surface_val = valid_vals[0]
    
    return surface_val

def preprocess_argo_profiles(
    profile_sources: List[str],
    open_func: Callable[[str, Optional[str]], xr.Dataset],
    alias: str,
    time_bounds: Tuple[pd.Timestamp, pd.Timestamp],
    depth_levels: Union[List[float], np.ndarray],
    n_points_dim: str = "N_POINTS"
):
    interp_profiles = []
    time_vals = []
    threshold_list_profiles = 20     # TODO : remove this after storing preprocessed profiles to avoid reprocessing them at each timestep

    def process_one_profile(profile_source):
        try:
            # open dataset
            ds = open_func(profile_source, alias) if alias is not None else open_func(profile_source)
            if ds is None:
                return None, None

            # ds = ds.argo.interp_std_levels(target_dimensions['depth']) # inutilisable sur un profil unique
            # ds = ds.argo.filter_qc()   # inutile dans le mode "research" d'argopy
            # ds_filtered = filter_variables(ds, keep_variables_list)

            ds = ds.rename({"PRES_ADJUSTED": "depth"})
            ds = ArgoManager.filter_argo_profile_by_time(
                ds,
                tmin=time_bounds[0],
                tmax=time_bounds[1],
            )
            if n_points_dim not in ds.dims or ds.sizes.get(n_points_dim, 0) == 0:
                logger.warning(f"Argo profile {profile_source} is empty after time filtering, skipping.")
                return None, None

            lat = ds["LATITUDE"].isel(N_POINTS=0).values.item()
            lon = ds["LONGITUDE"].isel(N_POINTS=0).values.item()
            time = pd.to_datetime(ds["TIME"].values)
            if isinstance(time, (np.ndarray, list)) and len(time) > 1:
                mean_time = pd.to_datetime(time).mean()
            else:
                mean_time = pd.to_datetime(time[0] if isinstance(time, (np.ndarray, list)) else time)

            depths = ds["depth"].values

            data_dict = {}
            for v in ds.data_vars:
                if v == "depth":
                    continue
                vals = ds[v].values
                # Filtrer les NaN
                valid_mask = ~np.isnan(vals)
                if not np.any(valid_mask):
                    # Si toutes les valeurs sont NaN, créer un array de NaN
                    interp_vals = np.full_like(depth_levels, np.nan, dtype=float)
                else:
                    # Filtrer les valeurs et les profondeurs correspondantes
                    valid_vals = vals[valid_mask]
                    valid_depths = depths[valid_mask]

                    # Extrapolation vers la surface
                    surface_val = extrapolate_to_surface(v, valid_depths, valid_vals)
        
                    interp_vals = np.interp(
                        depth_levels,
                        valid_depths,
                        valid_vals,
                        left=surface_val,
                        right=np.nan
                    )
                data_dict[v] = ("depth", interp_vals)

            for v in ds.data_vars:
                if v == "depth":
                    continue
                vals = ds[v].values
                if np.all(np.isnan(vals)):
                    interp_vals = np.full_like(depth_levels, np.nan, dtype=float)
                else:
                    interp_vals = np.interp(
                        depth_levels,
                        depths,
                        vals,
                        left=np.nan,
                        right=np.nan
                    )
                data_dict[v] = ("depth", interp_vals)
            ds.close()
            interp_ds = xr.Dataset(
                data_dict,
                coords={
                    "depth": depth_levels,
                    "lat": lat,
                    "lon": lon,
                    "time": time
                }
            )
            return interp_ds, mean_time
        except Exception as e:
            logger.warning(f"Failed to process Argo profile {profile_source}: {e}")
            traceback.print_exc()
            return None, None

    # Parallélisation avec ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=2) as executor:  # adapte max_workers à ton CPU
        futures = [executor.submit(process_one_profile, src) for src in profile_sources[0:threshold_list_profiles]] # TODO remove this shortcut after optimizing the processing: thousands of files to preprocess at each timestep !
        for future in as_completed(futures):
            interp_ds, mean_time = future.result()
            if interp_ds is not None:
                interp_profiles.append(interp_ds)
                time_vals.append(mean_time)

    if len(interp_profiles) == 0:
        return None
    # Convertit chaque élément en pd.Timestamp scalaire
    clean_time_vals = []
    for t in time_vals:
        if isinstance(t, (pd.DatetimeIndex, np.ndarray, list)):
            # Prend le premier élément si c'est un index ou une liste
            clean_time_vals.append(pd.to_datetime(t[0]))
        else:
            clean_time_vals.append(pd.to_datetime(t))
    mean_time = pd.Series(clean_time_vals).mean()
    interp_profiles = [ds.drop_vars("time") if "time" in ds.coords else ds for ds in interp_profiles]

    if len(interp_profiles) == 1:
        combined = interp_profiles[0]
    else:
        combined = xr.concat(interp_profiles, dim=n_points_dim)

    combined = combined.assign_coords(time=mean_time)
    return combined


def preprocess_one_npoints(
    source, is_swath, 
    n_points_dim,
    filtered_df, idx,
    alias, open_func,
    keep_variables_list,
    target_dimensions,
    coordinates,
    time_bounds=None,
):
    try:
        # open dataset
        if alias is not None:
            ds = open_func(source, alias)
        else:
            ds = open_func(source)
        if ds is None:
            return None

        if is_swath:
            coords_to_keep = [
                coordinates.get('time', None),
                coordinates.get('depth', None),
                coordinates.get('lat', None),
                coordinates.get('lon', None),
            ]
            coords_to_keep = list(filter(lambda x: x is not None, coords_to_keep))
            ds = swath_to_points(
                ds,
                coords_to_keep=list(coordinates.keys()),
                n_points_dim=n_points_dim,
            )

        # Chercher une coordonnée/variable temporelle
        time_name = coordinates['time']
        if time_name in ds.variables and time_name not in ds.coords:
            ds = ds.set_coords(time_name)

        time_coord = ds.coords[time_name]
        
        if n_points_dim not in ds.dims:
            logger.warning(f"Dataset {idx}: No points dimension found")
            return None

        ds_with_time = add_time_dim(
            ds, filtered_df, n_points_dim=n_points_dim, time_coord=time_coord, idx=idx
        )

        import gc
        ds_interp = ds_with_time
        # Use balanced chunks for n_points to avoid memory spiking in concatenation
        # 100k points is a moderate compromise (~8-10MB per var)
        ds_interp = ds_interp.chunk({n_points_dim: 100000})
        
        del ds
        del ds_with_time
        gc.collect()

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
        try:
            # Vérifier que le forecast complet est dans la plage de données disponibles
            for ref_alias in self.ref_aliases:
                logger.info(f"=============  PREPROCESSING REFERENCE DATASET: {ref_alias} ==============")
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
                        ref_df = ref_catalog.get_dataframe()
                        
                        # Trouver la date de fin maximale disponible dans les données de référence
                        max_available_date = ref_df["date_end"].max()
                        
                        # Si la fin du forecast dépasse les données disponibles, ignorer cette entrée
                        if forecast_end_time > max_available_date:
                            logger.debug(f"Skipping forecast starting at {forecast_reference_time}: "
                                    f"forecast ends at {forecast_end_time} but data only available until {max_available_date}")
                            if batch:
                                yield batch
                                batch = []
                            break
                        
                        # ref_catalog = self.ref_catalogs[ref_alias]
                        coord_system = ref_catalog.get_global_metadata().get("coord_system")
                        is_observation = coord_system.is_observation_dataset() if coord_system else False
                        
                        if is_observation:
                            # Logique observation : filtrer le catalogue d'observation sur l'intervalle temporel du forecast_index
                            obs_time_interval = (valid_time, valid_time)
                            keep_vars = self.keep_variables[ref_alias]
                            rename_vars_dict = self.metadata[ref_alias]['variables_dict']
                            keep_vars = [rename_vars_dict[var] for var in keep_vars if var in rename_vars_dict]

                            t0, t1 = obs_time_interval
                            t0 = t0 - self.time_tolerance
                            t1 = t1 + self.time_tolerance
                            time_bounds = (t0, t1)

                            entry["ref_data"] = {
                                "source": ref_catalog,
                                "keep_vars": keep_vars,
                                "target_dimensions": self.target_dimensions,
                                "metadata": self.metadata[ref_alias],
                                "time_bounds": time_bounds,
                            }
                            entry["ref_is_observation"] = True
                            entry["obs_time_interval"] = obs_time_interval
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
                if batch:  # dernier batch de ref_alias
                    yield batch
                    batch = []
        except Exception as e:
            logger.error(f"Error generating batches: {e}")
            traceback.print_exc()


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


def concat_with_dim_delayed(
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


def concat_with_dim(
    datasets: List[xr.Dataset],
    concat_dim: str,
    sort: bool = True,
):
    for i, ds in enumerate(datasets):
        if "time" in ds.coords:
            # Check dtype without loading data
            dtype = ds.coords["time"].dtype
            if np.issubdtype(dtype, np.integer):
                ref_date = ds.coords["time"].attrs.get("units", None)
                if ref_date:
                    # Use lazy decoding if possible
                    # relying on xarray's decode_cf_datetime if variables were decoded, 
                    # but here we seem to do manual fix.
                    # We avoid .values to keep it lazy.
                    import cftime
                    
                    def decode_time_lazy(x, units):
                        try:
                            from cftime import num2date
                            return num2date(x, units)
                        except ImportError:
                            base_date = pd.to_datetime(units.split("since")[1].strip().split(" ")[0])
                            return base_date + pd.to_timedelta(x, unit="D")

                    # If it's a dask array, use map_blocks
                    if hasattr(ds.coords["time"].data, "map_blocks"):
                         # Note: map_blocks implies we know the output chunks/dtype. 
                         # Converting int to object (cftime) or datetime64.
                         # This acts as a best effort to keep it lazy.
                         # Simpler approach: use xarray's decode_cf functionality if possible
                         # or just wrap in dask.map_blocks
                         pass 
                         # For now, let's just avoid the block that forces .values if we can't do it lazily easily
                         # But users might need this conversion.
                         # Default to xarray.coding.times.decode_cf_datetime if applicable?
                         pass

            elif dtype == "O":
                pass
            else:
                 # It's likely datetime64 or similar, ensure it
                 pass
                 # ds = ds.assign_coords(time=ds.coords["time"].astype("datetime64[ns]")) 
                 # astype on dask array is lazy.

        datasets[i] = ds


    datasets_with_dim = []
    for i, ds in enumerate(datasets):
        if concat_dim not in ds.dims:
                ds = ds.expand_dims({concat_dim: [i]})
        datasets_with_dim.append(ds)

    result = xr.concat(datasets_with_dim, dim=concat_dim,
        coords="minimal",
        compat="override", join="outer"
    )
    if sort:
        result = result.sortby(concat_dim)
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
        time_bounds: Tuple[pd.Timestamp, pd.Timestamp],
        # time_tolerance: pd.Timedelta = pd.Timedelta("12h"),
        n_points_dim: str,
        dataset_processor: Optional[DatasetProcessor] = None,
        include_geometry: bool = False,
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
        # self.time_tolerance = time_tolerance
        self.alias = alias
        self.keep_vars = keep_vars
        self.target_dimensions = target_dimensions
        self.n_points_dim = n_points_dim
        self.dataset_processor = dataset_processor 
        self.time_bounds = time_bounds

        if self.is_metadata:
            if self.load_fn is None:
                raise ValueError("A 'load_fn(link: str)' must be provided when using metadata.")
            self.meta_df = source
        else:
            self.datasets = source if isinstance(source, list) else [source]
        self.coordinates = dataset_metadata['coord_system'].coordinates
        self.include_geometry = include_geometry


    def preprocess_datasets(
        self,
        dataframe: pd.DataFrame,
        load_to_memory: bool = False,
    ) -> xr.Dataset:
        """
        Preprocess the input DataFrame and single observations files and return an xarray Dataset.
        """
        # remove "geometry" fields if needed:
        if not self.include_geometry and "geometry" in dataframe.columns:
            dataframe = dataframe.drop(columns=["geometry"])

        # chargement des fichiers  
        dataset_paths = [row["path"] for _, row in dataframe.iterrows()]

        first_ds = None
        while first_ds is None:
            if self.alias is not None:
                first_ds = self.load_fn(dataset_paths[0], self.alias)
            else:
                first_ds = self.load_fn.open(dataset_paths[0])

        # swath_dims = {"num_lines", "num_pixels", "num_nadir"}
        reduced_swath_dims = {"num_lines", "num_pixels"}

        # si profils argo, prétraitement particulier :

        if self.alias == "argo_profiles":
            # logger.info("Argo profiles detected - special preprocessing")
            try:
                result = preprocess_argo_profiles(
                    profile_sources=dataset_paths,
                    open_func=self.load_fn,
                    alias=self.alias,
                    time_bounds=self.time_bounds,
                    depth_levels=self.target_dimensions.get('depth', np.array([])),
                )
                if result is None:
                    logger.error("No Argo profiles could be processed")
                    return None
                # logger.info(f"Final Argo result: {result.sizes.get('profile', 1)} profiles, "
                #        f"{len(result.data_vars)} variables")
                if load_to_memory:
                    result = result.compute()

                '''# sauvegarde du dataset prétraité dans un fichier Zarr
                argo_dir = "..."
                time_val = result.coords["time"].values

                # Si c'est un tableau avec une seule valeur
                if isinstance(time_val, (np.ndarray, list)) and len(time_val) == 1:
                    time_str = str(pd.to_datetime(time_val[0]))
                else:
                    time_str = str(pd.to_datetime(time_val))
                argo_name = f"argo_profiles_{time_str}.zarr"
                import os
                argo_path = os.path.join(argo_dir, argo_name)
                result.to_zarr(argo_path, mode="w", consolidated=True)'''

                return result
            except Exception as e:
                logger.error(f"Argo preprocessing failed: {e}")
                traceback.print_exc()
                return None
    
        # Données avec dimension n_points/N_POINTS uniquement
        elif (self.n_points_dim in first_ds.dims) and not reduced_swath_dims.issubset(first_ds.dims):            
            try:
                # Nettoyer et traiter les datasets
                if self.dataset_processor is not None:
                    delayed_tasks = []
                    for idx, dataset_path in enumerate(dataset_paths):
                        delayed_tasks.append(dask.delayed(preprocess_one_npoints)(
                            dataset_path, False, self.n_points_dim, dataframe, idx,
                            self.alias, self.load_fn,
                            self.keep_vars, self.target_dimensions,
                            self.coordinates,
                            self.time_bounds,
                        ))
                    batch_results = self.dataset_processor.compute_delayed_tasks(
                        delayed_tasks, sync=False
                    )
                else:
                    batch_results = []
                    for idx, dataset_path in enumerate(dataset_paths):
                        result = preprocess_one_npoints(
                            dataset_path, False, self.n_points_dim, dataframe, idx,
                            self.alias, self.load_fn,
                            self.keep_vars, self.target_dimensions,
                            self.coordinates,
                            self.time_bounds,
                        )
                        batch_results.append(result)
                cleaned_datasets = [meta for meta in batch_results if meta is not None]

                if not cleaned_datasets:
                    logger.error("No n_points datasets could be processed")
                    traceback.print_exc()
                    return None
                
                # Concaténer le long de la dimension temporelle
                if len(cleaned_datasets) == 1:
                    result = cleaned_datasets[0]
                    # logger.info("Single n_points dataset - no concatenation needed")
                else:
                    try:
                        result = concat_with_dim(cleaned_datasets, dim="time")
                        # logger.info(f"Successfully concatenated {len(cleaned_datasets)} n_points datasets")
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
                
                # logger.info(f"Final n_points result: {result.sizes.get('time', 1)} time steps, "
                #        f"{len(result.data_vars)} variables")
                if load_to_memory:
                    result = result.compute()
                return result
                
            except Exception as e:
                logger.error(f"Complete n_points processing failed: {e}")
                traceback.print_exc()
                return None

        # Données Swath
        elif reduced_swath_dims.issubset(first_ds.dims):
            # logger.info("Swath data detected - reshaping and filtering to n_points")
            
            try:
                is_swath_data = True
                # nettoyer les datasets
                if self.dataset_processor is not None:
                    delayed_tasks = []
                    for idx, dataset_path in enumerate(dataset_paths):
                        delayed_tasks.append(dask.delayed(preprocess_one_npoints)(
                            dataset_path, is_swath_data, self.n_points_dim, dataframe, idx,
                            self.alias, self.load_fn,
                            self.keep_vars, self.target_dimensions,
                            self.coordinates,
                            self.time_bounds,
                        ))
                    batch_results = self.dataset_processor.compute_delayed_tasks(delayed_tasks)
                else:
                    batch_results = []
                    for idx, dataset_path in enumerate(dataset_paths):
                        result = preprocess_one_npoints(
                            dataset_path, is_swath_data, self.n_points_dim, dataframe, idx,
                            self.alias, self.load_fn,
                            self.keep_vars, self.target_dimensions,
                            self.coordinates,
                            self.time_bounds,
                        )
                        batch_results.append(result)
                cleaned_datasets = [meta for meta in batch_results if meta is not None]

                if not cleaned_datasets:
                    logger.error("No datasets could be cleaned and reshaped")
                    return None

                # vérifier la compatibilité
                first_ds = cleaned_datasets[0]
                target_n_points = first_ds.sizes['n_points']
                target_vars = set(first_ds.data_vars.keys())
                
                # logger.info(f"Target structure: {target_n_points:,} points, variables: {target_vars}")
                
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
                
                # logger.info(f"Found {len(compatible_datasets)} compatible datasets")
                
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
                
                if load_to_memory:
                    result = result.compute()
                return result
                
            except Exception as e:
                logger.error(f"Complete processing failed: {e}")
                traceback.print_exc()
                return first_ds

        # données non-swath, non n_points
        elif "time" in first_ds.dims or "time" in first_ds.coords:
            try:
                if self.alias is not None:
                    all_datasets = [self.load_fn(row["path"], self.alias) for _, row in dataframe.iterrows()]
                else:
                    all_datasets = [self.load_fn(row["path"]) for _, row in dataframe.iterrows()]
                all_datasets = [x for x in all_datasets if x is not None]
                for idx, ds in enumerate(all_datasets):
                    all_datasets[idx] = filter_variables(ds, self.keep_vars)
                combined = concat_with_dim(all_datasets, concat_dim="time", sort=True)
                # combined = dask.delayed(combined.sel)(time=slice(t0, t1))
                if load_to_memory:
                    combined = combined.compute()
                return combined
            except Exception as e:
                logger.error(f"Failed to concatenate time series data: {e}")
                traceback.print_exc()
                return first_ds

        else:
            logger.warning("Unknown data structure, returning first dataset")
            return first_ds

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
