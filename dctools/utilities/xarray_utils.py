#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Misc. functions to aid in the processing xr.Datasets and DataArrays."""

import ast
from typing import Any, Dict, List, Optional, Union

import cftime
from loguru import logger
import numpy as np
import pandas as pd
import xarray as xr



def create_empty_dataset(dimensions: dict) -> xr.Dataset:
    """
    Crée un Dataset Xarray vide à partir d'un dictionnaire de dimensions.

    Args:
        dimensions (dict): Dictionnaire où les clés sont les noms des dimensions
                           et les valeurs sont les tailles des dimensions.

    Returns:
        xr.Dataset: Dataset Xarray vide avec les dimensions spécifiées.
    """
    coords = {dim: range for dim, range in dimensions.items()}
    return xr.Dataset(coords=coords)


def rename_coordinates(ds: xr.Dataset, rename_dict: dict) -> xr.Dataset:
    """
    Renomme les coordonnées/dimensions d'un Dataset xarray en utilisant swap_dims pour garantir
    que les nouvelles coordonnées deviennent des dimensions indexées (évite les warnings xarray).

    Args:
        ds (xr.Dataset): Dataset à modifier.
        rename_dict (dict): Dictionnaire {ancien_nom: nouveau_nom}.

    Returns:
        xr.Dataset: Dataset renommé.
    """
    # Renommer les coordonnées (sans toucher aux dimensions)
    coords_to_rename = {k: v for k, v in rename_dict.items() if k in ds.coords and k != v}
    if coords_to_rename:
        ds = ds.rename(coords_to_rename)

    # S'assurer que les nouvelles coordonnées sont bien présentes
    for old, new in coords_to_rename.items():
        if new in ds.variables and new not in ds.coords:
            ds = ds.set_coords(new)

    # Utiliser swap_dims pour transformer les coordonnées en dimensions principales
    swap_dict = {}
    for old, new in coords_to_rename.items():
        if old in ds.dims and new in ds.coords and old != new:
            swap_dict[old] = new
    if swap_dict:
        ds = ds.swap_dims(swap_dict)

    # Supprimer les anciennes coordonnées si elles existent encore
    for old, new in coords_to_rename.items():
        if old in ds.coords and old != new:
            ds = ds.drop_vars(old)
    return ds


def rename_variables(ds: xr.Dataset, rename_dict: Optional[dict] = None):
    """Rename variables according to a given dictionary."""
    try:
        if not rename_dict:
            return ds
        if isinstance(rename_dict, str):
            rename_dict = ast.literal_eval(rename_dict)

        # keep only keys that are in the dataset variables
        rename_dict = {k: v for k, v in rename_dict.items() if k in list(ds.data_vars)}
        # Remove invalid values from the dictionary
        rename_dict = {k: v for k, v in rename_dict.items() if k != v and v is not None}
        if not rename_dict or len(rename_dict) == 0:
            return ds

        ds = ds.rename_vars(rename_dict)

        # Supprimer les anciennes variables si elles existent encore
        for old, new in rename_dict.items():
            if old in ds.variables and new in ds.variables and old != new:
                ds = ds.drop_vars(old)
        return ds
    except Exception as e:
        logger.error(f"Error renaming variables: {e}")
        raise ValueError(f"Failed to rename variables in dataset: {e}") from e


def rename_coords_and_vars(
    ds: xr.Dataset,
    rename_coords_dict: Optional[dict] = None,
    rename_vars_dict: Optional[dict] = None,
):
    """Rename variables and coordinates according to given dictionaries."""
    try:
        ds = rename_variables(ds, rename_vars_dict)
        ds = rename_coordinates(ds, rename_coords_dict)

        return ds
    except Exception as e:
        logger.error(f"Error renaming coordinates or variables: {e}")
        raise ValueError(f"Failed to rename coordinates or variables in dataset: {e}") from e

def subset_variables(ds: xr.Dataset, list_vars: List[str]):
    """Extract a sub-dataset containing only listed variables, preserving attributes."""

    for variable_name in ds.variables:
        var_std_name = ds[variable_name].attrs.get("standard_name",'').lower()
        if not var_std_name:
            var_std_name = ds[variable_name].attrs.get("std_name", '').lower()

    # Crée un sous-dataset avec uniquement les variables listées
    subset = ds[list_vars]

    # Détecter les coordonnées présentes dans le sous-dataset
    coords_to_set = [c for c in subset.data_vars if c in ds.coords]
    if coords_to_set:
        subset = subset.set_coords(coords_to_set)

    # Supprime les coordonnées orphelines (non utilisées)
    # subset = subset.reset_coords(drop=True)

    for variable_name in subset.variables:
        var_std_name = subset[variable_name].attrs.get("standard_name",'').lower()
        if not var_std_name:
            var_std_name = subset[variable_name].attrs.get("std_name", '').lower()

    return subset


def apply_standard_dimension_order(da: xr.DataArray) -> xr.DataArray:
    """
    Réorganise les dimensions d'un DataArray dans l'ordre standard : (time, depth, lat, lon).
    Si une dimension n'existe pas, elle est ignorée.
    """
    # Ordre standard souhaité
    standard_order = ['time', 'depth', 'lat', 'lon']
    
    # Garder seulement les dimensions qui existent dans le DataArray
    existing_dims = [dim for dim in standard_order if dim in da.dims]
    
    # Ajouter les autres dimensions non-standard à la fin (si il y en a)
    other_dims = [dim for dim in da.dims if dim not in standard_order]
    final_order = existing_dims + other_dims
    
    return da.transpose(*final_order)

def assign_coordinate(
    ds: xr.Dataset, coord_name: str, coord_vals: List[Any], coord_attrs: Dict[str, str]
):
    ds = ds.assign_coords({coord_name: (coord_name, coord_vals, coord_attrs)})
    return ds

def get_glonet_time_attrs(start_date: str):
    glonet_time_attrs = {
        'units': f"days since {start_date} 00:00:00", 'calendar': "proleptic_gregorian"
    }
    return glonet_time_attrs

def get_time_info(ds: xr.Dataset):
    """
    Analyze the main time axis of an xarray Dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset to analyze.

    Returns
    -------
    dict
        A dictionary containing:
        - start: the earliest timestamp (pandas.Timestamp or string from attributes)
        - end: the latest timestamp
        - duration: the time range duration (if computable)
        - calendar: the calendar used (if specified)
    """
    # Step 1: Try to find a valid time coordinate
    for name in ds.coords:
        if name.lower() in {"time", "t", "date", "datetime"}:
            time_coord = ds.coords[name]
            calendar = time_coord.attrs.get("calendar", "standard")

            # Case 1: Already decoded to datetime-like values
            if np.issubdtype(time_coord.dtype, np.datetime64) or isinstance(time_coord.values[0], cftime.DatetimeBase):
                times = pd.to_datetime(time_coord.values)
                break

            # Case 2: Try decoding from CF metadata
            units = time_coord.attrs.get("units")
            if units:
                try:
                    decoded_time = xr.decode_cf(xr.Dataset({name: time_coord}))[name]
                    times = pd.to_datetime(decoded_time.values)
                    break
                except Exception as exc:
                    logger.error(f"Error decoding time axis: {repr(exc)}")
                    continue
    else:
        # Step 2 fallback: Use global attributes
        start = ds.attrs.get("time_coverage_start")
        end = ds.attrs.get("time_coverage_end")
        duration = None
        if start and end:
            try:
                parsed_start = pd.to_datetime(start)
                parsed_end = pd.to_datetime(end)
                duration = parsed_end - parsed_start
            except Exception:
                pass
        return {
            "start": start,
            "end": end,
            "duration": duration,
            "calendar": None
        }

    # Step 3: Extract times and compute
    if len(times) == 0:
        return {
            "start": None,
            "end": None,
            "duration": None,
            "calendar": calendar
        }

    start = times.min()
    end = times.max()
    duration = end - start

    return {
        "start": start,
        "end": end,
        "duration": duration,
        "calendar": calendar
    }

def filter_time_interval(ds: xr.Dataset, start_time: str, end_time: str) -> xr.Dataset:
    """
    Filters an xarray Dataset to only include data within a specified time range.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset containing time coordinates.
    start_time : str
        The start time in ISO format (e.g., "2024-05-02").
    end_time : str
        The end time in ISO format (e.g., "2024-05-11").

    Returns
    -------
    xr.Dataset
        A filtered dataset containing only data within the time range.
        Returns None if no data is within the time range.
    """
    # Analyze the time axis
    time_info = get_time_info(ds)
    
    # Convert start and end times to pandas Timestamp
    start_time = pd.to_datetime(start_time)
    end_time = pd.to_datetime(end_time)

    # Check if the time axis is present and valid
    if time_info["start"] is None or time_info["end"] is None:
        return None

    # Filter the time coordinate within the given interval
    time_coord = ds.coords["time"]
    mask = (time_coord >= start_time) & (time_coord <= end_time)

    # If no data falls within the time range, return None
    if mask.sum() == 0:
        logger.warning("No data found in the specified time interval.")
        return None

    return ds.sel(time=mask)


def extract_spatial_bounds(ds: xr.Dataset) -> dict:
    """
    Extract spatial bounds from an xarray Dataset, handling various coordinate naming conventions.

    Args:
        ds (xr.Dataset): The xarray dataset.

    Returns:
        dict: Dictionary with lat/lon min/max.
    """
    lat_vals = ds["lat"].values
    lon_vals = ds["lon"].values

    return {
        "lat_min": float(lat_vals.min()),
        "lat_max": float(lat_vals.max()),
        "lon_min": float(lon_vals.min()),
        "lon_max": float(lon_vals.max()),
    }

def filter_spatial_area(
    ds: xr.Dataset,
    lat_min: float, lat_max: float,
    lon_min: float, lon_max: float
) -> xr.Dataset:
    """
    Filters an xarray Dataset to only include data within a specified spatial area.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset containing latitude and longitude coordinates.
    lat_min : float
        Minimum latitude.
    lat_max : float
        Maximum latitude.
    lon_min : float
        Minimum longitude.
    lon_max : float
        Maximum longitude.

    Returns
    -------
    xr.Dataset
        A filtered dataset containing only data within the specified spatial area.
        Returns None if no data is within the area.
    """
    # Step 1: Check if latitude and longitude coordinates exist
    if "lat" not in ds.coords or "lon" not in ds.coords:
        logger.warning("Latitude or longitude coordinates not found in the dataset.")
        return None

    # Step 2: Apply the spatial filter
    lat_coord = ds.coords["lat"]
    lon_coord = ds.coords["lon"]
    
    lat_mask = (lat_coord >= lat_min) & (lat_coord <= lat_max)
    lon_mask = (lon_coord >= lon_min) & (lon_coord <= lon_max)

    # Step 3: Filter the dataset using the masks
    if lat_mask.sum() == 0 or lon_mask.sum() == 0:
        logger.warning("No data found in the specified spatial area.")
        return None

    # Step 4: Return the filtered dataset
    return ds.sel(lat=lat_mask, lon=lon_mask)


def extract_variables(ds: xr.Dataset) -> List[str]:
    return list(ds.data_vars.keys())


def reset_time_coordinates(dataset: xr.Dataset) -> xr.Dataset:
    """
    Remplace les valeurs des coordonnées de temps par une suite débutant à 0.

    Args:
        dataset (xr.Dataset): Le Dataset Xarray à modifier.

    Returns:
        xr.Dataset: Le Dataset avec les coordonnées de temps modifiées.
    """
    if "time" not in dataset.coords:
        raise ValueError("Le dataset ne contient pas de coordonnées 'time'.")

    # Générer une suite de valeurs commençant à 0
    new_time_values = range(len(dataset.coords["time"]))

    # Remplacer les valeurs des coordonnées de temps
    dataset = dataset.assign_coords(time=new_time_values)
    return dataset

def subsample_dataset(
    ds: xr.Dataset,
    subsample_values: Dict[str, Union[List, np.ndarray, slice]] = None,
    method: str = "nearest",
    tolerance: Optional[Dict[str, float]] = None
) -> xr.Dataset:
    """
    Sous-échantillonne un dataset xarray sur une ou plusieurs dimensions.
    
    Args:
        ds (xr.Dataset): Dataset xarray d'entrée
        subsample_values (Dict[str, Union[List, np.ndarray, slice]]): 
            Dictionnaire spécifiant les valeurs à garder pour chaque dimension.
            - Clé : nom de la dimension (ex: 'time', 'lat', 'lon', 'depth')
            - Valeur : liste/array des valeurs à sélectionner, ou slice object
            Si une dimension n'est pas dans le dictionnaire, toutes ses valeurs sont gardées.
        method (str): Méthode de sélection ('nearest', 'exact', 'ffill', 'bfill')
        tolerance (Optional[Dict[str, float]]): Tolérance pour chaque dimension lors de la sélection
        
    Returns:
        xr.Dataset: Dataset sous-échantillonné
        
    Examples:
        # Sous-échantillonner temps et profondeur
        subsample_values = {
            'time': pd.date_range('2024-01-01', '2024-01-10', freq='2D'),
            'depth': [0, 10, 50, 100]
        }
        ds_sub = subsample_dataset(ds, subsample_values)
        
        # Utiliser des slices pour sous-échantillonner
        subsample_values = {
            'lat': slice(-60, 60, 2),  # Latitudes de -60 à 60 avec pas de 2
            'time': slice('2024-01-01', '2024-01-10')
        }
        ds_sub = subsample_dataset(ds, subsample_values)
    """
    
    if subsample_values is None:
        subsample_values = {}
    
    if tolerance is None:
        tolerance = {}
    
    ds_result = ds.copy()
    
    # Parcourir chaque dimension à sous-échantillonner
    for dim_name, values in subsample_values.items():
        
        # Vérifier que la dimension existe dans le dataset
        if dim_name not in ds_result.dims:
            logger.warning(f"Dimension '{dim_name}' not found in dataset. Available dimensions: {list(ds_result.dims)}")
            continue
        
        try:
            if isinstance(values, slice):
                # Cas où on utilise un slice
                ds_result = ds_result.sel({dim_name: values})
                
            else:
                # Obtenir la tolérance pour cette dimension
                dim_tolerance = tolerance.get(dim_name, None)
                
                # Construire les arguments pour sel()
                sel_kwargs = {dim_name: values, 'method': method}
                if dim_tolerance is not None:
                    sel_kwargs['tolerance'] = dim_tolerance
                
                ds_result = ds_result.sel(**sel_kwargs)
                
        except Exception as e:
            logger.error(f"Failed to subsample dimension '{dim_name}': {e}")
            # En cas d'erreur, essayer avec drop=True pour ignorer les valeurs manquantes
            try:
                if not isinstance(values, slice):
                    sel_kwargs['drop'] = True
                    ds_result = ds_result.sel(**sel_kwargs)
                    logger.info(f"Successfully subsampled '{dim_name}' with drop=True")
                else:
                    logger.error(f"Cannot subsample dimension '{dim_name}' with slice")
                    continue
            except Exception as e2:
                logger.error(f"Failed to subsample dimension '{dim_name}' even with drop=True: {e2}")
                continue
    
    # Log des informations sur le résultat
    original_shape = {dim: size for dim, size in ds.dims.items()}
    result_shape = {dim: size for dim, size in ds_result.dims.items()}
    
    logger.info(f"Dataset subsampling completed:")
    logger.info(f"  Original shape: {original_shape}")
    logger.info(f"  Result shape: {result_shape}")
    
    return ds_result


def subsample_dataset_by_indices(
    ds: xr.Dataset,
    subsample_indices: Dict[str, Union[List[int], np.ndarray, slice]] = None
) -> xr.Dataset:
    """
    Sous-échantillonne un dataset xarray en utilisant des indices plutôt que des valeurs.
    
    Args:
        ds (xr.Dataset): Dataset xarray d'entrée
        subsample_indices (Dict[str, Union[List[int], np.ndarray, slice]]): 
            Dictionnaire spécifiant les indices à garder pour chaque dimension.
            
    Returns:
        xr.Dataset: Dataset sous-échantillonné
        
    Examples:
        # Garder seulement certains indices temporels et de profondeur
        subsample_indices = {
            'time': [0, 2, 4, 6],  # Garder les indices 0, 2, 4, 6
            'depth': slice(0, 10, 2)  # Garder les indices 0, 2, 4, 6, 8
        }
        ds_sub = subsample_dataset_by_indices(ds, subsample_indices)
    """
    
    if subsample_indices is None:
        subsample_indices = {}
    
    ds_result = ds.copy()
    
    # Parcourir chaque dimension à sous-échantillonner
    for dim_name, indices in subsample_indices.items():
        
        # Vérifier que la dimension existe dans le dataset
        if dim_name not in ds_result.dims:
            logger.warning(f"Dimension '{dim_name}' not found in dataset. Available dimensions: {list(ds_result.dims)}")
            continue
        
        try:
            # logger.debug(f"Subsampling dimension '{dim_name}' with indices: {indices}")
            ds_result = ds_result.isel({dim_name: indices})
            
        except Exception as e:
            logger.error(f"Failed to subsample dimension '{dim_name}' by indices: {e}")
            continue
    
    # Log des informations sur le résultat
    original_shape = {dim: size for dim, size in ds.dims.items()}
    result_shape = {dim: size for dim, size in ds_result.dims.items()}
    
    logger.info(f"Dataset subsampling by indices completed:")
    logger.info(f"  Original shape: {original_shape}")
    logger.info(f"  Result shape: {result_shape}")
    
    return ds_result


def subsample_dataset_uniform(
    ds: xr.Dataset,
    subsample_steps: Dict[str, int] = None
) -> xr.Dataset:
    """
    Sous-échantillonne un dataset xarray de manière uniforme avec un pas donné.
    
    Args:
        ds (xr.Dataset): Dataset xarray d'entrée
        subsample_steps (Dict[str, int]): 
            Dictionnaire spécifiant le pas pour chaque dimension.
            Par exemple: {'time': 2, 'depth': 3} prendra 1 valeur sur 2 pour le temps
            et 1 valeur sur 3 pour la profondeur.
            
    Returns:
        xr.Dataset: Dataset sous-échantillonné
        
    Examples:
        # Prendre 1 valeur sur 2 pour le temps, 1 sur 3 pour la profondeur
        subsample_steps = {
            'time': 2,
            'depth': 3,
            'lat': 4,
            'lon': 4
        }
        ds_sub = subsample_dataset_uniform(ds, subsample_steps)
    """
    
    if subsample_steps is None:
        subsample_steps = {}
    
    ds_result = ds.copy()
    
    # Parcourir chaque dimension à sous-échantillonner
    for dim_name, step in subsample_steps.items():
        
        # Vérifier que la dimension existe dans le dataset
        if dim_name not in ds_result.dims:
            logger.warning(f"Dimension '{dim_name}' not found in dataset. Available dimensions: {list(ds_result.dims)}")
            continue
        
        if not isinstance(step, int) or step < 1:
            logger.warning(f"Invalid step value for dimension '{dim_name}': {step}. Must be a positive integer.")
            continue
        
        try:
            # Créer un slice avec le pas spécifié
            indices_slice = slice(None, None, step)
            # logger.debug(f"Subsampling dimension '{dim_name}' with step {step}")
            ds_result = ds_result.isel({dim_name: indices_slice})
            
        except Exception as e:
            logger.error(f"Failed to subsample dimension '{dim_name}' with step {step}: {e}")
            continue
    
    # Log des informations sur le résultat
    original_shape = {dim: size for dim, size in ds.dims.items()}
    result_shape = {dim: size for dim, size in ds_result.dims.items()}
    
    logger.info(f"Uniform dataset subsampling completed:")
    logger.info(f"  Original shape: {original_shape}")
    logger.info(f"  Result shape: {result_shape}")
    
    return ds_result


def get_dimension_info(ds: xr.Dataset, dim_name: str) -> Dict[str, any]:
    """
    Obtient des informations détaillées sur une dimension du dataset.
    
    Args:
        ds (xr.Dataset): Dataset xarray
        dim_name (str): Nom de la dimension
        
    Returns:
        Dict: Informations sur la dimension (taille, valeurs min/max, type, etc.)
    """
    
    if dim_name not in ds.dims:
        return {"exists": False}
    
    coord = ds.coords.get(dim_name)
    if coord is None:
        return {
            "exists": True,
            "size": ds.dims[dim_name],
            "has_coordinates": False
        }
    
    values = coord.values
    
    info = {
        "exists": True,
        "size": ds.dims[dim_name],
        "has_coordinates": True,
        "dtype": str(values.dtype),
        "first_value": values[0] if len(values) > 0 else None,
        "last_value": values[-1] if len(values) > 0 else None,
    }
    
    # Ajouter min/max pour les types numériques
    if np.issubdtype(values.dtype, np.number):
        info["min_value"] = float(np.min(values))
        info["max_value"] = float(np.max(values))
    
    return info


def suggest_subsample_values(
    ds: xr.Dataset, 
    target_sizes: Dict[str, int] = None
) -> Dict[str, Union[List, slice]]:
    """
    Suggère des valeurs de sous-échantillonnage pour réduire le dataset à des tailles cibles.
    
    Args:
        ds (xr.Dataset): Dataset xarray
        target_sizes (Dict[str, int]): Tailles cibles pour chaque dimension
        
    Returns:
        Dict: Valeurs suggérées pour le sous-échantillonnage
        
    Example:
        target_sizes = {'time': 10, 'depth': 5, 'lat': 100, 'lon': 100}
        suggestions = suggest_subsample_values(ds, target_sizes)
        ds_sub = subsample_dataset(ds, suggestions)
    """
    
    if target_sizes is None:
        target_sizes = {}
    
    suggestions = {}
    
    for dim_name, target_size in target_sizes.items():
        if dim_name not in ds.dims:
            logger.warning(f"Dimension '{dim_name}' not found in dataset")
            continue
        
        current_size = ds.dims[dim_name]
        
        if target_size >= current_size:
            logger.info(f"Target size for '{dim_name}' ({target_size}) >= current size ({current_size}), no subsampling needed")
            continue
        
        # Calculer le pas pour obtenir approximativement la taille cible
        step = max(1, current_size // target_size)
        
        if dim_name in ds.coords:
            # Si on a des coordonnées, prendre des valeurs spécifiques
            coord_values = ds.coords[dim_name].values
            subsampled_values = coord_values[::step][:target_size]
            suggestions[dim_name] = subsampled_values.tolist()
            logger.info(f"Dimension '{dim_name}': {current_size} -> {len(subsampled_values)} values (step={step})")
        else:
            # Sinon, utiliser un slice
            suggestions[dim_name] = slice(None, None, step)
            estimated_size = (current_size + step - 1) // step
            logger.info(f"Dimension '{dim_name}': {current_size} -> ~{estimated_size} values (step={step})")
    
    return suggestions


def preview_display_dataset(ds, variables=None, max_values=500000):
    """Affiche un dataset en gérant la mémoire."""
    print("DATASET SUMMARY")
    print("="*50)
    print(f"Nombre de dimensions: {len(ds.dims)}")
    print(f"Nombre de coordonnées: {len(ds.coords)}")
    print(f"Nombre de variables: {len(ds.data_vars)}")
    print(f"Taille totale: {ds.nbytes / 1e6:.2f} MB")
    
    print("\nDIMENSIONS:")
    for dim, size in ds.dims.items():
        print(f"  {dim}: {size}")
    
    print("\nCOORDONNÉES:")
    for coord_name, coord in ds.coords.items():
        print(f"  {coord_name}: {coord.shape} {coord.dtype}")
        if coord.size <= max_values:
            if coord.size <= 20:
                print(f"    Valeurs: {coord.values}")
            else:
                print(f"    Premières valeurs: {coord.values[:5]}...")
                print(f"    Dernières valeurs: ...{coord.values[-5:]}")
    
    print("\nVARIABLES:")
    if variables is not None:
        display_variables = variables
    else:
        display_variables = list(ds.data_vars)
    for var_name, var in ds.data_vars.items():
        if var_name not in display_variables:
            continue
        print(f"  {var_name}: {var.dims} {var.shape} {var.dtype}")

        try:
            # Afficher quelques statistiques
            if np.issubdtype(var.dtype, np.number):
                valid_data = var
                if valid_data.size > 0:
                    print(f"    Min: {float(valid_data.min()):.3f}")
                    print(f"    Max: {float(valid_data.max()):.3f}")
                    print(f"    Mean: {float(valid_data.mean()):.3f}")
        except Exception as e:
            print(f"    (Erreur calcul stats: {e})")
