#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Misc. functions to aid in the processing xr.Datasets and DataArrays."""

import ast
from datetime import datetime
import gc
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cftime
import geopandas as gpd
from loguru import logger
from matplotlib.pyplot import grid
from memory_profiler import profile
import numpy as np
import pandas as pd
from pathlib import Path
import psutil
import xarray as xr
import xesmf as xe

from dctools.data.coordinates import (
    GEO_STD_COORDS
)
HAS_PYINTERP = False


def log_memory(stage):
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / 1e6
    print(f"[{stage}] Memory usage: {mem_mb:.2f} MB")

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
    # ds = ds.copy(deep=True)
    # 1. Renommer les coordonnées (sans toucher aux dimensions)
    log_memory("START rename_coordinates")
    coords_to_rename = {k: v for k, v in rename_dict.items() if k in ds.coords and k != v}
    if coords_to_rename:
        ds = ds.rename(coords_to_rename)

    # 2. S'assurer que les nouvelles coordonnées sont bien présentes
    for old, new in coords_to_rename.items():
        if new in ds.variables and new not in ds.coords:
            ds = ds.set_coords(new)

    # 3. Utiliser swap_dims pour transformer les coordonnées en dimensions principales
    swap_dict = {}
    for old, new in coords_to_rename.items():
        if old in ds.dims and new in ds.coords and old != new:
            swap_dict[old] = new
    if swap_dict:
        ds = ds.swap_dims(swap_dict)

    # 4. Supprimer les anciennes coordonnées si elles existent encore
    for old, new in coords_to_rename.items():
        if old in ds.coords and old != new:
            ds = ds.drop_vars(old)
    log_memory("END rename_coordinates")
    return ds


def rename_variables(ds: xr.Dataset, rename_dict: Optional[dict] = None):
    """Rename variables according to a given dictionary."""
    try:
        log_memory("START rename_variables")
        if not rename_dict:
            return ds
        # ds = ds.copy(deep=True)
        if isinstance(rename_dict, str):
            rename_dict = ast.literal_eval(rename_dict)

        # keep only keys that are in the dataset variables
        rename_dict = {k: v for k, v in rename_dict.items() if k in list(ds.data_vars)}
        # Remove invalid values from the dictionary
        rename_dict = {k: v for k, v in rename_dict.items() if k != v and v is not None}
        if not rename_dict or len(rename_dict) == 0:
            return ds

        ds = ds.rename_vars(rename_dict)

        # Supprimer les anciennes variables si elles existent encore (rare mais possible)
        for old, new in rename_dict.items():
            if old in ds.variables and new in ds.variables and old != new:
                ds = ds.drop_vars(old)
        log_memory("END rename_variables")
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
        log_memory("START rename_coords_and_vars")
        #logger.debug(f"Renaming variables in dataset with dictionary: {rename_vars_dict}")
        ds = rename_variables(ds, rename_vars_dict)
        #logger.debug(f"Renaming coordinates in dataset with dictionary: {rename_coords_dict}")
        ds = rename_coordinates(ds, rename_coords_dict)

        log_memory("END rename_coords_and_vars")
        return ds
    except Exception as e:
        logger.error(f"Error renaming coordinates or variables: {e}")
        raise ValueError(f"Failed to rename coordinates or variables in dataset: {e}") from e

def subset_variables(ds: xr.Dataset, list_vars: List[str]):
    """Extract a sub-dataset containing only listed variables, preserving attributes."""
    # ds = ds.copy(deep=True)
    for variable_name in ds.variables:
        var_std_name = ds[variable_name].attrs.get("standard_name",'').lower()
        if not var_std_name:
            var_std_name = ds[variable_name].attrs.get("std_name", '').lower()

    # Crée un sous-dataset avec uniquement les variables listées
    # subset = xr.Dataset({var: ds[var].copy(deep=True) for var in list_vars if var in ds})
    '''subset = xr.Dataset(
        {var: ds[var].copy(deep=True) for var in list_vars if var in ds},
        coords={k: v for k, v in ds.coords.items() if k in ds.dims}
    )'''
    subset = ds[list_vars]   #.(deep=True)

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
    '''if hasattr(ds, "close"):
        ds.close()
    del ds
    gc.collect()'''
    return subset



import xarray as xr
import numpy as np
from typing import Literal, Optional
import pyinterp
import shutil
import tempfile
import xesmf as xe

'''
def make_pyinterp_compatible(ds: xr.Dataset) -> xr.Dataset:
    """
    Adds required 'standard_name' attributes to coordinates for pyinterp compatibility.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset with lat/lon/(depth) coordinates.

    Returns
    -------
    xr.Dataset
        Dataset with updated coordinate attributes.
    """
    ds = ds.copy()
    coord_map = {
        "lon": "longitude",
        "longitude": "longitude", 
        "x": "longitude",
        "lat": "latitude",
        "latitude": "latitude",
        "y": "latitude",
        "depth": "depth",
        "z": "depth"
    }
    for name, std in coord_map.items():
        if name in ds.coords:
            ds[name].attrs.setdefault("standard_name", std)
    rename_dims = {}
    if "lat" in ds.dims:
        rename_dims["lat"] = "latitude"
    if "lon" in ds.dims:
        rename_dims["lon"] = "longitude"
    ds = ds.rename(rename_dims)
    return ds'''


def ensure_cf_attrs(da):
    if "lon" in da.coords:
        da.coords["lon"].attrs.setdefault("standard_name", "longitude")
        da.coords["lon"].attrs.setdefault("units", "degrees_east")
        da.coords["lon"].attrs.setdefault("axis", "X")
    if "lat" in da.coords:
        da.coords["lat"].attrs.setdefault("standard_name", "latitude")
        da.coords["lat"].attrs.setdefault("units", "degrees_north")
        da.coords["lat"].attrs.setdefault("axis", "Y")
    return da


import xarray as xr
import numpy as np
import tempfile, shutil, os
import pyinterp.backends.xarray
from pyinterp.backends.xarray import Grid2D


'''def interpolate_pyinterp(ds: xr.Dataset, target_grid: dict, method: str="bilinear") -> xr.Dataset:
    # 1. Renommage dims pour pyinterp
    # ds = ds.copy(deep=True)

    log_memory("START interpolate_pyinterp")
    ren = {d: d.replace("lat", "latitude").replace("lon", "longitude") for d in ds.dims}
    ds = ds.rename(ren)

    lat_t = target_grid["lat"]
    lon_t = target_grid["lon"]

    has_time = "time" in ds.dims
    has_depth = "depth" in ds.dims and "depth" in target_grid.keys()
    time_vals = ds.time.values if has_time else [None]
    depth_vals = ds.depth.values if has_depth else [None]

    # Zarr temp store
    tmp = tempfile.mkdtemp(prefix="interp_zarr_")
    zpath = os.path.join(tmp, "out.zarr")
    first = True

    # Itérations time / depth
    try:
        for t in time_vals:
            for d in depth_vals:
                sel = {}
                if t is not None: sel["time"] = t
                if d is not None and "depth" in ds.dims: sel["depth"] = d
                ds_slice = ds.sel(sel, method="nearest").squeeze(drop=True)

                var_out = {}
                for vn, da in ds_slice.data_vars.items():
                    if not {"latitude", "longitude"}.issubset(da.dims): continue
                    da2 = da.squeeze()
                    if set(da2.dims) != {"latitude", "longitude"}: continue

                    grid = Grid2D(da2)
                    arr1d = grid.interp(lon_t.ravel(), lat_t.ravel(), bounds_error=False)
                    arr2 = arr1d.reshape(len(lat_t), len(lon_t))
                    da_interp = xr.DataArray(arr2, dims=("latitude","longitude"),
                                            coords={"latitude": lat_t, "longitude": lon_t},
                                            attrs=da.attrs)
                    # reinject time/depth
                    if t is not None: da_interp = da_interp.expand_dims(time=[t])
                    if d is not None: da_interp = da_interp.expand_dims(depth=[d])
                    var_out[vn] = da_interp

                if not var_out: continue
                ds_out = xr.Dataset(var_out)
                if first:
                    ds_out.attrs.update(ds.attrs)
                    ds_out.to_zarr(zpath, mode="w")
                    first = False
                else:
                    dim_app = "time" if has_time else "depth"
                    ds_out.to_zarr(zpath, append_dim=dim_app)
                del ds_out
                gc.collect()
    except Exception as axc:
        traceback.print_exc()

    # lecture finale
    out = xr.open_zarr(zpath, chunks={})
    shutil.rmtree(tmp)
    log_memory("END rename_dimensions")
    return out'''


#@profile
'''def interpolate_xesmf(
    ds: xr.Dataset,
    target_grid: xr.Dataset,
    reuse_weights: bool,
    weights_file: Optional[str],
    method: str = "bilinear",
) -> xr.Dataset:
    """
    Interpolate a gridded oceanographic dataset to a target grid using xESMF,
    by looping over time, depth, and variables to reduce memory usage.

    Parameters
    ----------
    ds : xr.Dataset
        The source dataset to interpolate.
    target_grid : xr.Dataset
        The target grid as an xarray Dataset (must include lat/lon).
    reuse_weights : bool
        Whether to reuse weights (if weights_file is provided).
    weights_file : Optional[str]
        Path to a weights file to reuse or store. Required if reuse_weights is True.
    method : str
        Interpolation method, e.g., 'bilinear', 'nearest_s2d', etc.

    Returns
    -------
    xr.Dataset
        The interpolated dataset on the target grid.
    """
    # ds = ds.copy(deep=True)
    log_memory("START interpolate_xesmf")
    regridder = xe.Regridder(
        ds,
        target_grid,
        method=method,
        reuse_weights=reuse_weights,
        filename=weights_file if reuse_weights else None
    )

    interpolated_vars = []

    time_coords = ds.time.values if "time" in ds.dims else [None]
    depth_coords = ds.depth.values if "depth" in ds.dims else [None]

    # Create temporary Zarr store
    temp_dir = tempfile.mkdtemp(prefix="interpol_zarr_")
    zarr_path = Path(temp_dir) / "interpolated.zarr"

    for var_name in ds.data_vars:
        var = ds[var_name]
        dims = var.dims

        interpolated_slices = []

        for t in time_coords:
            for z in depth_coords:
                # Subset
                subset = var
                if t is not None:
                    subset = subset.sel(time=t)
                if z is not None:
                    subset = subset.sel(depth=z)

                # Ensure 2D input for xesmf
                subset_2d = subset if {'lat', 'lon'} <= set(subset.dims) else subset.squeeze()
                # del subset
                # Interpolation
                try:
                    interp_result = regridder(subset_2d)
                    # del subset_2d
                except Exception as e:
                    print(f"Failed interpolation for var={var_name}, time={t}, depth={z}: {e}")
                    continue

                # Expand dims back
                if z is not None:
                    interp_result = interp_result.expand_dims("depth")
                    interp_result["depth"] = [z]
                if t is not None:
                    interp_result = interp_result.expand_dims("time")
                    interp_result["time"] = [np.datetime64(t)]

                interp_result.name = var_name
                interpolated_slices.append(interp_result)
                # del interp_result

        # Concatenate all slices
        if interpolated_slices:
            interpolated_var = xr.concat(interpolated_slices, dim=[dim for dim in ['time', 'depth'] if dim in interpolated_slices[0].dims])
            interpolated_vars.append(interpolated_var)
            # del interpolated_slices

    # Combine into one dataset
    interpolated_ds = xr.merge(interpolated_vars)
    #del interpolated_vars

    # Save to Zarr (lazy, no .load())
    #interpolated_ds.to_zarr(str(zarr_path), mode="w", compute=True)

    # Close the in-memory dataset to free memory
    # del interpolated_ds

    # Open the Zarr file in lazy mode and return
    # lazy_ds = xr.open_zarr(str(zarr_path), chunks="auto")
    log_memory("END interpolate_xesmf")
    return interpolated_ds'''


'''
import dask
def interpolate_xesmf_lazy(
    ds: xr.Dataset,
    #variable_names: List[str],
    target_grid: xr.Dataset,
    reuse_weights: bool = True,
    weights_file: str = None,
    method: str = "bilinear",
) :
    """
    Lazily interpolate multiple variables in a Dataset using xESMF, with automatic regridder creation or loading.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset containing the variables to interpolate.
    variable_names : list of str
        List of variable names to interpolate.
    target_grid : xr.Dataset
        Grid onto which to regrid the variables. Must contain 'lat' and 'lon'.
    weights_dir : str
        Directory where to save/load regridder weights (as .nc files).
    method : str, optional
        Regridding method to use (e.g., 'bilinear', 'nearest_s2d'), by default "bilinear".
    reuse_weights : bool, optional
        Whether to reuse weights from files in `weights_dir` if they exist, by default True.

    Returns
    -------
    dask.delayed.Delayed
        Delayed computation returning the interpolated Dataset when computed.
    """
    #os.makedirs(weights_dir, exist_ok=True)

    log_memory("START interpolate_xesmf")
    delayed_datasets = []

    variable_names = ds.data_vars
    for var_name in variable_names:
        if var_name not in ds:
            raise ValueError(f"Variable '{var_name}' not found in the input dataset.")

        # Extraire uniquement la variable
        var_ds = ds[[var_name]]

        # Chemin du fichier de poids pour cette variable
        # weights_file = os.path.join(weights_dir, f"weights_{var_name}_{method}.nc")

        # Création ou chargement du regridder
        regridder = xe.Regridder(
            var_ds,
            target_grid,
            method=method,
            filename=weights_file,
            reuse_weights=reuse_weights
        )

        # Lazy interpolation via dask.delayed
        @dask.delayed
        def interpolate_variable(ds_block, regridder, var_name):
            interpolated = regridder(ds_block[var_name])
            return interpolated.to_dataset(name=var_name)

        delayed_result = interpolate_variable(var_ds, regridder, var_name)
        delayed_datasets.append(delayed_result)

    # Merge lazy datasets
    combined_delayed = xr.merge(delayed_datasets)

    log_memory("END interpolate_xesmf")
    return combined_delayed

'''



def interpolate_xarray_dataset(ds: xr.Dataset, 
                              varname: str,
                              coordinates: Dict[str, np.ndarray],
                              interpolator: str = "bilinear",
                              **kwargs) -> np.ndarray:
    """
    Interpolate an xarray dataset using pyinterp, automatically handling 2D or 3D cases
    based on available dimensions.
    
    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset containing the variable to interpolate
    varname : str
        Name of the variable to interpolate
    coordinates : dict
        Dictionary containing interpolation coordinates:
        - 'longitude': array of longitude values
        - 'latitude': array of latitude values  
        - 'time': array of time values (optional, for 3D interpolation)
        - 'depth': array of depth values (optional, for 4D interpolation)
    interpolator : str, optional
        Interpolation method, by default "bilinear"
        Options: "bilinear", "inverse_distance_weighting", "nearest"
    **kwargs
        Additional arguments passed to the interpolation function
        
    Returns
    -------
    numpy.ndarray
        Interpolated values at the specified coordinates
        
    Raises
    ------
    ValueError
        If required coordinates are missing or dimensions are not supported
    """
    
    # Check if the variable exists in the dataset
    if varname not in ds.data_vars:
        raise ValueError(f"Variable '{varname}' not found in dataset")
    
    data_array = ds[varname]
    dims = list(data_array.dims)
    
    # Determine interpolation type based on dimensions
    has_depth = any(dim in dims for dim in ['depth', 'z', 'level'])
    has_time = 'time' in dims
    
    # Validate required coordinates
    required_coords = ['longitude', 'latitude']
    if has_time:
        required_coords.append('time')
    if has_depth:
        required_coords.append('depth')
        
    missing_coords = [coord for coord in required_coords if coord not in coordinates]
    if missing_coords:
        raise ValueError(f"Missing required coordinates: {missing_coords}")
    
    # Create interpolation coordinates dictionary
    interp_coords = {
        'longitude': coordinates['longitude'],
        'latitude': coordinates['latitude']
    }
    
    # Handle different interpolation cases
    try:
        if has_time and not has_depth:
            # 3D interpolation (longitude, latitude, time)
            print("Performing 3D interpolation (lon, lat, time)")
            grid = pyinterp.backends.xarray.Grid3D(data_array)
            interp_coords['time'] = coordinates['time']
            result = grid.trivariate(interp_coords, interpolator=interpolator, **kwargs)
            
        elif has_depth and not has_time:
            # 3D interpolation (longitude, latitude, depth)
            print("Performing 3D interpolation (lon, lat, depth)")
            grid = pyinterp.backends.xarray.Grid3D(data_array)
            interp_coords['depth'] = coordinates['depth']
            result = grid.trivariate(interp_coords, interpolator=interpolator, **kwargs)
            
        elif has_depth and has_time:
            # Process time step by time step for 4D data
            print("Performing 4D interpolation (lon, lat, time, depth) - processing by time steps")
            time_coords = coordinates['time']
            n_points = len(coordinates['longitude'])
            n_times = len(time_coords)
            
            # Initialize result array
            result = np.full((n_times, n_points), np.nan)
            
            # Process each time step
            for t_idx, time_val in enumerate(time_coords):
                try:
                    # Select data for this time step
                    data_at_time = data_array.sel(time=time_val, method='nearest')
                    
                    # Create 3D grid for this time step (lon, lat, depth)
                    grid_3d = pyinterp.backends.xarray.Grid3D(data_at_time)
                    
                    # Interpolate for this time step
                    time_result = grid_3d.trivariate({
                        'longitude': coordinates['longitude'],
                        'latitude': coordinates['latitude'],
                        'depth': coordinates['depth']
                    }, interpolator=interpolator, **kwargs)
                    
                    result[t_idx, :] = time_result
                    
                except Exception as e:
                    print(f"Warning: Failed to interpolate time step {t_idx}: {e}")
                    continue
            
            result = result.flatten() if n_times == 1 else result
            
        else:
            # 2D interpolation (longitude, latitude)
            print("Performing 2D interpolation (lon, lat)")
            grid = pyinterp.backends.xarray.Grid2D(data_array)
            result = grid.bivariate(interp_coords, interpolator=interpolator, **kwargs)
        
        return result
        
    except Exception as e:
        print(f"Error during interpolation: {e}")
        print(f"Data array dimensions: {dims}")
        print(f"Data array shape: {data_array.shape}")
        print(f"Available coordinates: {list(coordinates.keys())}")
        raise



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


def interpolate_xesmf(ds: xr.Dataset,
                                 # varnames: list,
                                 target_grid: Dict[str, np.ndarray],
                                 reuse_weights: bool = False,
                                 weights_file: Optional[str] = None,
                                 interpolator: str = "bilinear",
                                 **kwargs) -> xr.Dataset:
    """
    Interpolate multiple variables from an xarray dataset and return a new Dataset
    with the same variables and dimensions as the original, but interpolated onto
    the new coordinate grid.
    """
    log_memory("START interpolate_pyinterp")
    varnames = ds.data_vars
    # Create a working copy of the dataset
    ds_work = ds.copy()

    # Detect coordinate names in the original dataset
    lon_coord_orig = None
    lat_coord_orig = None
    
    for coord_name in ['longitude', 'lon']:
        if coord_name in ds_work.coords:
            lon_coord_orig = coord_name
            break
    
    for coord_name in ['latitude', 'lat']:
        if coord_name in ds_work.coords:
            lat_coord_orig = coord_name
            break
    
    if lon_coord_orig is None or lat_coord_orig is None:
        raise ValueError("Could not find longitude/latitude coordinates in dataset")
    
    # Rename coordinates to pyinterp standard if needed
    rename_dict = {}
    if lon_coord_orig != 'longitude':
        rename_dict[lon_coord_orig] = 'longitude'
    if lat_coord_orig != 'latitude':
        rename_dict[lat_coord_orig] = 'latitude'
    
    if rename_dict:
        ds_work = ds_work.rename(rename_dict)
        print(f"Renamed coordinates: {rename_dict}")
    
    # Extract coordinate arrays
    lon_interp = np.asarray(target_grid['lon'])
    lat_interp = np.asarray(target_grid['lat'])
    
    # Handle optional coordinates
    time_interp = None
    depth_interp = None
    if 'time' in target_grid:
        time_interp = np.asarray(target_grid['time'])
    if 'depth' in target_grid:
        depth_interp = np.asarray(target_grid['depth'])
    
    # Create new coordinate system using original coordinate names
    new_coords = {}
    new_coords[lon_coord_orig] = lon_interp
    new_coords[lat_coord_orig] = lat_interp
    
    if time_interp is not None and 'time' in ds_work.coords:
        new_coords['time'] = time_interp
    
    if depth_interp is not None:
        depth_coord = next((k for k in ['depth', 'z', 'level'] if k in ds_work.coords), 'depth')
        new_coords[depth_coord] = depth_interp
  
    # Initialize result dataset with new coordinates
    data_vars = {}

    # Interpolate each requested variable
    for varname in varnames:
        if varname not in ds_work.data_vars:
            print(f"Warning: Variable '{varname}' not found in dataset, skipping...")
            continue
            
        try:
            print(f"Interpolating variable: {varname}")
            
            # Get original variable
            original_var = ds_work[varname]
            original_dims = list(original_var.dims)
            
            # Determine interpolation type based on dimensions
            has_depth = any(dim in original_dims for dim in ['depth', 'z', 'level'])
            has_time = 'time' in original_dims
            
            # Case 1: 2D variables (lon, lat only)
            if not has_time and not has_depth:
                print(f"  -> 2D interpolation for {varname}")
                grid_2d = pyinterp.backends.xarray.Grid2D(original_var)
                
                # Create meshgrid for interpolation points
                lon_mesh, lat_mesh = np.meshgrid(lon_interp, lat_interp, indexing='ij')
                
                # Interpolate
                result = grid_2d.bivariate({
                    'longitude': lon_mesh.ravel(),
                    'latitude': lat_mesh.ravel()
                }, interpolator=interpolator, **kwargs)
                
                # Reshape to grid
                interpolated_values = result.reshape(len(lon_interp), len(lat_interp))
                
                # Create dimensions using original names
                new_dims = [lon_coord_orig, lat_coord_orig]
                
            # Case 2: 3D variables with depth but no time (lon, lat, depth)
            elif has_depth and not has_time:
                print(f"  -> 3D interpolation (lon, lat, depth) for {varname}")
                grid_3d = pyinterp.backends.xarray.Grid3D(original_var)
                
                # Create meshgrid for interpolation points
                lon_mesh, lat_mesh, depth_mesh = np.meshgrid(
                    lon_interp, lat_interp, depth_interp, indexing='ij')
                
                # Interpolate
                result = grid_3d.trivariate({
                    'longitude': lon_mesh.ravel(),
                    'latitude': lat_mesh.ravel(),
                    'depth': depth_mesh.ravel()
                }, interpolator=interpolator, **kwargs)
                
                # Reshape to grid
                interpolated_values = result.reshape(len(lon_interp), len(lat_interp), len(depth_interp))
                
                # Determine depth coordinate name
                depth_coord_name = next((k for k in ['depth', 'z', 'level'] if k in original_dims), 'depth')
                new_dims = [lon_coord_orig, lat_coord_orig, depth_coord_name]
                
            # Case 3: 3D variables with time but no depth (time, lon, lat)
            elif has_time and not has_depth:
                print(f"  -> 3D interpolation (time, lon, lat) for {varname}")
                n_times = len(time_interp) if time_interp is not None else len(ds_work.time)
                n_lon = len(lon_interp)
                n_lat = len(lat_interp)
                
                # Initialize result array
                interpolated_values = np.full((n_times, n_lon, n_lat), np.nan)
                
                # Create meshgrid for spatial coordinates
                lon_mesh, lat_mesh = np.meshgrid(lon_interp, lat_interp, indexing='ij')
                flat_lon = lon_mesh.ravel()
                flat_lat = lat_mesh.ravel()
                
                # Process each time step
                times_to_process = time_interp if time_interp is not None else ds_work.time.values
                
                for t_idx, time_val in enumerate(times_to_process):
                    try:
                        # Select data for this time step
                        data_at_time = original_var.sel(time=time_val, method='nearest')
                        
                        # Create 2D grid for this time step
                        grid_2d = pyinterp.backends.xarray.Grid2D(data_at_time)
                        
                        # Interpolate for this time step
                        time_result = grid_2d.bivariate({
                            'longitude': flat_lon,
                            'latitude': flat_lat
                        }, interpolator=interpolator)  #, **kwargs)
                        
                        # Reshape and store
                        interpolated_values[t_idx, :, :] = time_result.reshape(n_lon, n_lat)
                        
                    except Exception as e:
                        print(f"    Warning: Failed to interpolate {varname} at time step {t_idx}: {e}")
                        continue
                
                new_dims = ['time', lon_coord_orig, lat_coord_orig]
                
            # Case 4: 4D variables with time and depth (time, lon, lat, depth)
            elif has_time and has_depth:
                print(f"  -> 4D interpolation (time, lon, lat, depth) for {varname}")
                n_times = len(time_interp) if time_interp is not None else len(ds_work.time)
                n_lon = len(lon_interp)
                n_lat = len(lat_interp)
                n_depth = len(depth_interp) if depth_interp is not None else len(ds_work.depth)
                
                # Initialize result array
                interpolated_values = np.full((n_times, n_lon, n_lat, n_depth), np.nan)
                
                # Create meshgrid for spatial coordinates
                lon_mesh, lat_mesh, depth_mesh = np.meshgrid(
                    lon_interp, lat_interp, depth_interp, indexing='ij')
                flat_lon = lon_mesh.ravel()
                flat_lat = lat_mesh.ravel()
                flat_depth = depth_mesh.ravel()
                
                # Process each time step
                times_to_process = time_interp if time_interp is not None else ds_work.time.values
                
                for t_idx, time_val in enumerate(times_to_process):
                    try:
                        # Select data for this time step
                        data_at_time = original_var.sel(time=time_val, method='nearest')
                        
                        # Create 3D grid for this time step
                        grid_3d = pyinterp.backends.xarray.Grid3D(data_at_time)
                        
                        # Interpolate for this time step
                        time_result = grid_3d.trivariate({
                            'longitude': flat_lon,
                            'latitude': flat_lat,
                            'depth': flat_depth
                        }, interpolator=interpolator) #, **kwargs)
                        
                        # Reshape and store
                        interpolated_values[t_idx, :, :, :] = time_result.reshape(n_lon, n_lat, n_depth)
                        
                    except Exception as e:
                        print(f"    Warning: Failed to interpolate {varname} at time step {t_idx}: {e}")
                        continue
                
                # Determine depth coordinate name
                depth_coord_name = next((k for k in ['depth', 'z', 'level'] if k in original_dims), 'depth')
                new_dims = ['time', lon_coord_orig, lat_coord_orig, depth_coord_name]
            
            # Create new variable with interpolated values using original coordinate names
            data_vars[varname] = (new_dims, interpolated_values, original_var.attrs)
            print(f"  -> Successfully interpolated variable: {varname}")
            
        except Exception as e:
            print(f"Error interpolating variable {varname}: {e}")
            continue
    

    # Create result dataset with original coordinate names
    ds_result = xr.Dataset(
        data_vars=data_vars,
        coords=new_coords,
        attrs=ds.attrs
    )

    # Réorganiser toutes les variables selon l'ordre standard
    standard_order = ['time', 'depth', 'lat', 'lon']
    for var_name in ds_result.data_vars:
        da = ds_result[var_name]
        # Garder seulement les dimensions qui existent
        existing_dims = [dim for dim in standard_order if dim in da.dims]
        # Ajouter les autres dimensions à la fin
        other_dims = [dim for dim in da.dims if dim not in standard_order]
        final_order = existing_dims + other_dims
        
        if final_order != list(da.dims):  # Seulement si l'ordre change
            ds_result[var_name] = da.transpose(*final_order)

    log_memory("END interpolate_pyinterp")
    return ds_result



def create_interpolation_coordinates(lon_points: np.ndarray,
                                   lat_points: np.ndarray,
                                   time_points: np.ndarray = None,
                                   depth_points: np.ndarray = None) -> Dict[str, np.ndarray]:
    """
    Helper function to create interpolation coordinates dictionary.
    
    Parameters
    ----------
    lon_points : numpy.ndarray
        Longitude coordinates for interpolation
    lat_points : numpy.ndarray
        Latitude coordinates for interpolation
    time_points : numpy.ndarray, optional
        Time coordinates for interpolation
    depth_points : numpy.ndarray, optional
        Depth coordinates for interpolation
        
    Returns
    -------
    dict
        Dictionary of interpolation coordinates
    """
    
    coords = {
        'lon': np.asarray(lon_points),
        'lat': np.asarray(lat_points)
    }
    
    if time_points is not None:
        coords['time'] = np.asarray(time_points)
    
    if depth_points is not None:
        coords['depth'] = np.asarray(depth_points)
    
    return coords





'''
 
import xarray as xr
import dask
import pyinterp
import numpy as np

def interpolate_xesmf(
    ds: xr.Dataset,
    target_grid: dict,
    reuse_weights: bool = True,
    weights_file: Optional[str] = None,
    time_dim: str = "time",
    vertical_dim: str = "depth",
    lon_dim: str = "lon",
    lat_dim: str = "lat",
    method: str = "linear",
    drop_missing: bool = True,
) -> xr.Dataset:
    """
    Interpolate a gridded dataset onto a target grid using pyinterp, in a lazy (Dask) fashion.
    """
    log_memory("START interpolate_pyinterp")
    out_vars = []

    # Ensure Dask chunking for lazy evaluation
    if not ds.chunks:
        chunk_dims = {d: 1 for d in ds.dims if d in [time_dim, vertical_dim]}
        chunk_dims.update({lat_dim: 128, lon_dim: 128})
        ds = ds.chunk(chunk_dims)

    lat_t = target_grid[lat_dim]
    lon_t = target_grid[lon_dim]

    time_vals = ds[time_dim].values if time_dim in ds.dims else [None]
    depth_vals = ds[vertical_dim].values if vertical_dim in ds.dims else [None]

    for var_name, da in ds.data_vars.items():
        dims = set(da.dims)
        if not {lat_dim, lon_dim}.issubset(dims):
            if drop_missing:
                continue
            else:
                out_vars.append(da)
                continue

        slices = []
        for t in time_vals:
            for z in depth_vals:
                # Sélectionne la tranche à interpoler
                da_sel = da
                if t is not None and time_dim in da.dims:
                    da_sel = da_sel.sel({time_dim: t})
                if z is not None and vertical_dim in da.dims:
                    da_sel = da_sel.sel({vertical_dim: z})

                # Interpolation 2D avec pyinterp
                lon = da_sel[lon_dim].values
                lat = da_sel[lat_dim].values
                data = da_sel.values


                # Vérifier que la variable est bien 2D (lat, lon)
                if set(da_sel.dims) != {lat_dim, lon_dim}:
                    continue

                # Interpolation 2D avec pyinterp
                grid = pyinterp.backends.xarray.Grid2D(da_sel)
                if method == "bilinear":
                    lon_grid, lat_grid = np.meshgrid(lon_t, lat_t)
                    interp = grid.bivariate(
                        {lon_dim: lon_t, lat_dim: lat_t},
                        bounds_error=False
                    )
                elif method == "nearest":
                    interp = grid.bivariate(
                        {lon_dim: lon_t, lat_dim: lat_t},
                        method="nearest",
                        bounds_error=False
                    )
                elif method == "bicubic":
                    interp = grid.bicubic(
                        {lon_dim: lon_t, lat_dim: lat_t},
                        bounds_error=False
                    )
                else:
                    raise ValueError(f"Unknown interpolation method: {method}")
                interp_da = xr.DataArray(
                    interp,
                    dims=(lat_dim, lon_dim),
                    coords={lat_dim: lat_t, lon_dim: lon_t},
                    name=var_name,
                )
                if z is not None and vertical_dim in da.dims:
                    interp_da = interp_da.expand_dims({vertical_dim: [z]})
                if t is not None and time_dim in da.dims:
                    interp_da = interp_da.expand_dims({time_dim: [t]})
                slices.append(interp_da)

        # Concatène sur les axes appropriés
        if slices:
            concat_dims = [dim for dim in [time_dim, vertical_dim] if dim in da.dims]
            interp_var = xr.concat(slices, dim=concat_dims)
            out_vars.append(interp_var)

    interpolated_ds = xr.merge(out_vars)
    log_memory("END interpolate_pyinterp")
    return interpolated_ds
'''



'''
def interpolate_xesmf(
    ds: xr.Dataset,
    target_grid: dict,
    reuse_weights: bool = True,
    weights_file: Optional[str] = None,
    time_dim: str = "time",
    vertical_dim: str = "depth",
    lon_dim: str = "lon",
    lat_dim: str = "lat",
    method: str = "linear",
    drop_missing: bool = True,
) -> xr.Dataset:
    """
    Interpolate a gridded dataset onto a target grid using pyinterp, variable by variable, time/depth slice by slice.
    Returns a Dask-backed Dataset if input is chunked.
    """
    log_memory("START interpolate_pyinterp")
    out_vars = []

    # Prépare les valeurs cibles
    lat_t = target_grid[lat_dim]
    lon_t = target_grid[lon_dim]
    has_time = time_dim in ds.dims
    has_depth = vertical_dim in ds.dims and vertical_dim in target_grid
    time_vals = ds[time_dim].values if has_time else [None]
    depth_vals = ds[vertical_dim].values if has_depth else [None]

    for var_name, da in ds.data_vars.items():
        dims = set(da.dims)
        if not {lat_dim, lon_dim}.issubset(dims):
            if drop_missing:
                continue
            else:
                out_vars.append(da)
                continue

        slices = []
        for t in time_vals:
            for z in depth_vals:
                # Sélectionne la tranche à interpoler
                da_sel = da
                if t is not None and time_dim in da.dims:
                    da_sel = da_sel.sel({time_dim: t})
                if z is not None and vertical_dim in da.dims:
                    da_sel = da_sel.sel({vertical_dim: z})

                # Interpolation 2D avec pyinterp
                lon = da_sel[lon_dim].values
                lat = da_sel[lat_dim].values
                data = da_sel.values
                grid = pyinterp.backends.xarray.Grid2D(lon, lat, data)
                interp = grid.interp(lon_t, lat_t, method=method, bounds_error=False)
                interp_da = xr.DataArray(
                    interp,
                    dims=(lat_dim, lon_dim),
                    coords={lat_dim: lat_t, lon_dim: lon_t},
                    name=var_name,
                )
                if z is not None and vertical_dim in da.dims:
                    interp_da = interp_da.expand_dims({vertical_dim: [z]})
                if t is not None and time_dim in da.dims:
                    interp_da = interp_da.expand_dims({time_dim: [t]})
                slices.append(interp_da)

        # Concatène sur les axes appropriés
        if slices:
            concat_dims = [dim for dim in [time_dim, vertical_dim] if dim in da.dims]
            interp_var = xr.concat(slices, dim=concat_dims)
            out_vars.append(interp_var)

    interpolated_ds = xr.merge(out_vars)
    log_memory("END interpolate_pyinterp")
    return interpolated_ds
'''

#@profile
def interpolate_dataset(
        ds: xr.Dataset, ranges: Dict[str, np.ndarray],
        weights_filepath: Optional[str] = None,
        interpolation_lib: Optional[str] = "pyinterp",
    ) -> xr.Dataset:

    # ds = ds.copy(deep=True)
    for variable_name in ds.variables:
        var_std_name = ds[variable_name].attrs.get("standard_name",'').lower()
        if not var_std_name:
            var_std_name = ds[variable_name].attrs.get("std_name", '').lower()

    # 1. Sauvegarder les attributs des coordonnées AVANT interpolation
    coords_attrs = {}
    for coord in ds.coords:
        coords_attrs[coord] = ds.coords[coord].attrs.copy()

    # (optionnel) Sauvegarder aussi les attrs des variables si besoin
    vars_attrs = {}
    for var in ds.data_vars:
        vars_attrs[var] = ds[var].attrs.copy()

    out_dict = {}
    for key in ranges.keys():
        out_dict[key] = ranges[key]
    for dim in GEO_STD_COORDS.keys():
        if dim not in out_dict.keys():
            out_dict[dim] = ds.coords[dim].values
    # ds_out = create_empty_dataset(out_dict)

    #for key in ranges.keys():
    #    assert(key in list(ds.dims))
    ranges = {k: v for k, v in ranges.items() if k in ds.dims}

    # TODO : adapt chunking depending on the dataset type
    # ds_out = ds_out.chunk(chunks={"lat": 10, "lon": 10, "time": 1})

    if interpolation_lib == "pyinterp":
        ds = interpolate_pyinterp(
            ds,
            target_grid=out_dict,
        )
    elif interpolation_lib == "xesmf":
        if weights_filepath and Path(weights_filepath).is_file():
            # Use precomputed weights
            logger.debug(f"Using interpolation precomputed weights from {weights_filepath}")
            '''regridder = xe.Regridder(
                ds, ds_out, "bilinear", reuse_weights=True, filename=weights_filepath
            )'''
            ds = interpolate_xesmf(
                ds,
                target_grid=out_dict,
                reuse_weights=True,
                weights_file=weights_filepath,
                method="bilinear",
            )
        else:
            ds = interpolate_xesmf(
                ds,
                target_grid=out_dict,
                reuse_weights=False,
                weights_file=weights_filepath,
                method="bilinear",
            )
    else:
        raise("Unknown interpolation lib")

    # Compute weights
    '''regridder = xe.Regridder(
        ds, ds_out, "bilinear",
    )
    # Save the weights to a file
    regridder.to_netcdf(weights_filepath)'''
    # Regrid the dataset

    # ds_out = regridder(ds)



    # 2. Réaffecter les attributs des variable
    for var in ds.data_vars:
        if var in vars_attrs:
            ds[var].attrs = vars_attrs[var].copy()

    # 3. Réaffecter les attributs des coordonnées
    for coord in ds.coords:
        if coord in coords_attrs:
            # Crée un nouveau DataArray avec les attrs sauvegardés
            new_coord = xr.DataArray(
                ds.coords[coord].values,
                dims=ds.coords[coord].dims,
                attrs=coords_attrs[coord].copy()
            )
            ds = ds.assign_coords({coord: new_coord})

    for variable_name in ds.variables:
        var_std_name = ds[variable_name].attrs.get("standard_name",'').lower()
        if not var_std_name:
            var_std_name = ds[variable_name].attrs.get("std_name", '').lower()

    return ds

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
    # Step 1: Analyze the time axis
    time_info = get_time_info(ds)
    
    # Convert start and end times to pandas Timestamp
    start_time = pd.to_datetime(start_time)
    end_time = pd.to_datetime(end_time)

    # Step 2: Check if the time axis is present and valid
    if time_info["start"] is None or time_info["end"] is None:
        return None

    # Step 3: Filter the time coordinate within the given interval
    time_coord = ds.coords["time"]
    mask = (time_coord >= start_time) & (time_coord <= end_time)

    # If no data falls within the time range, return None
    if mask.sum() == 0:
        logger.warning("No data found in the specified time interval.")
        return None

    # Step 4: Return the filtered dataset
    return ds.sel(time=mask)


def extract_spatial_bounds(ds: xr.Dataset) -> dict:
    """
    Extract spatial bounds from an xarray Dataset, handling various coordinate naming conventions.

    Args:
        ds (xr.Dataset): The xarray dataset.

    Returns:
        dict: Dictionary with lat/lon min/max.
    """
    # Tentatives courantes pour les noms de coordonnées
    #lat_names = ["lat", "latitude"]
    #lon_names = ["lon", "longitude"]

    #lat_var = next((name for name in lat_names if name in ds.coords), None)
    #lon_var = next((name for name in lon_names if name in ds.coords), None)

    #if lat_var is None or lon_var is None:
    #    raise ValueError("Could not identify latitude or longitude coordinates in dataset.")

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



class UnifiedObservationView:
    def __init__(
        self,
        source: Union[xr.Dataset, List[xr.Dataset], pd.DataFrame, gpd.GeoDataFrame],
        load_fn: Callable[[str], xr.Dataset],
        alias: Optional[str] = None,
        time_tolerance: pd.Timedelta = pd.Timedelta("12h"),
    ):
        """
        Parameters:
            source: either
                - one or more xarray Datasets (data already loaded)
                - a DataFrame/GeoDataFrame containing metadata, including file links
            load_fn: a callable that loads a dataset given a link (required if source is a DataFrame)
        """
        self.is_metadata = isinstance(source, (pd.DataFrame, gpd.GeoDataFrame))
        self.load_fn = load_fn
        self.time_tolerance = time_tolerance
        self.alias = alias

        if self.is_metadata:
            if self.load_fn is None:
                raise ValueError("A `load_fn(link: str)` must be provided when using metadata.")
            self.meta_df = source
        else:
            self.datasets = source if isinstance(source, list) else [source]


    #@profile
    def open_concat_in_time(self, time_interval: tuple) -> xr.Dataset:
        """
        Filtre les métadonnées selon l'intervalle de temps, ouvre les fichiers correspondants,
        puis concatène les datasets le long de la dimension 'time'.

        Parameters
        ----------
        time_interval : tuple
            (start_time, end_time) sous forme de pd.Timestamp ou de string compatible pandas.

        Returns
        -------
        xr.Dataset
            Dataset concaténé sur la dimension 'time'.
        """
        log_memory("START open_concat_in_time")
        t0, t1 = time_interval
        t0 = t0 - self.time_tolerance
        t1 = t1 + self.time_tolerance
        # Filtrage des métadonnées
        filtered = self.meta_df[
            (self.meta_df["date_start"] <= t1) & (self.meta_df["date_end"] >= t0)
        ]
        if filtered.empty:
            logger.warning("Aucune donnée dans l'intervalle de temps demandé.")
            return None

        # Ouverture des fichiers NetCDF/Zarr
        if self.alias is not None:
            datasets = [self.load_fn(row["path"], self.alias) for _, row in filtered.iterrows()]
        else:
            datasets = [self.load_fn(row["path"]) for _, row in filtered.iterrows()]

        # Concaténation sur la dimension 'time'
        combined = xr.concat(datasets, dim="time")
        # Optionnel : trier et supprimer les doublons temporels
        combined = combined.sortby("time")
        combined = combined.sel(time=slice(t0, t1))
        for dataset in datasets:
            if hasattr(dataset, "close"):
                dataset.close()
            del dataset
        gc.collect()
        log_memory("END open_concat_in_time")

        return combined

    def filter_by_time(self, time_range: Tuple[pd.Timestamp, pd.Timestamp]) -> List[xr.Dataset]:
        """
        Returns a list of datasets that fall within the time window.
        If source is metadata, loads only the required datasets.
        """
        t0, t1 = time_range

        if self.is_metadata:
            filtered = self.meta_df[
                (self.meta_df["date_start"] >= t0) & (self.meta_df["date_end"] <= t1)
            ]
            if filtered.empty:
                return []

            return [self.load_fn(row["link"]) for _, row in filtered.iterrows()]
        else:
            return [
                ds.sel(time=slice(t0, t1)) for ds in self.datasets
                if "time" in ds.dims or "time" in ds.coords
            ]

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
