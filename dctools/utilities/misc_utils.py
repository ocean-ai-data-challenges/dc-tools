#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Miscellaneous utils functions."""

import os
from typing import Dict, List


from cartopy import crs as ccrs
from cartopy.feature import NaturalEarthFeature
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import mapping, Polygon, base as shapely_base
import xarray as xr


def get_dates_from_startdate(start_date: str, ndays: int) -> List[str]:
    """Get dates of n days after start_date.

    Args:
        date (str): start date
        ndays (int): number of days after start_date

    Returns:
        List[str]: list of n dates.
    """
    list_days = []
    for nday in range(0, ndays):
        time_stamp = pd.to_datetime(start_date) + pd.DateOffset(days=nday)
        list_days.append(time_stamp.strftime('%Y-%m-%d'))
    return list_days

def get_home_path():
    if 'HOME' in os.environ:
        #logger.info(f"HOME: {os.environ['HOME']}")
        home_path = os.environ['HOME']
    elif 'USERPROFILE' in os.environ:
        #logger.info(f"USER: {os.environ['USERPROFILE']}")
        home_path = os.environ['USERPROFILE']
    elif 'HOMEPATH' in os.environ:
        #logger.info(f"HOME: {os.environ['HOMEPATH']}")
        home_path = os.environ['HOMEPATH']
    return home_path



def visualize_netcdf_with_geometry(
    ds: xr.Dataset, geometry: gpd.GeoSeries, coordinates: Dict[str, str]
):
    # Charger les données NetCDF

    # Extraire les coordonnées et la variable à visualiser
    lon = ds[coordinates['lon']]
    lat = ds[coordinates['lat']]
    variable = ds['zos']  # variable à visualiser


    # Créer une GeoDataFrame pour la géométrie
    gdf = gpd.GeoDataFrame({'geometry': [geometry]}, crs="EPSG:4326")

    # Configurer la projection
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], crs=ccrs.PlateCarree())

    # Tracer les données NetCDF
    variable.plot(ax=ax, transform=ccrs.PlateCarree(), cmap='viridis')

    # Tracer la géométrie
    gdf.plot(ax=ax, edgecolor='red', facecolor='none', linewidth=2, transform=ccrs.PlateCarree())


    # Afficher la carte
    plt.show()

def walk_obj(obj):
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield from walk_obj(value)
    elif isinstance(obj, (list, tuple, set)):
        for item in obj:
            yield from walk_obj(item)
    else:
        yield obj

def transform_in_place(obj, func):
    if isinstance(obj, dict):
        for k, v in obj.items():
            obj[k] = transform_in_place(v, func)
        return obj
    elif isinstance(obj, list):
        for i in range(len(obj)):
            obj[i] = transform_in_place(obj[i], func)
        return obj
    else:
        # Pour les types immuables : appliquer la fonction directement
        return func(obj)

def make_serializable(obj):
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, pd.DataFrame) or isinstance(obj, gpd.GeoDataFrame):
        return obj.to_json()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def make_timestamps_serializable(gdf: pd.DataFrame) -> pd.DataFrame:
    gdf = gdf.copy()
    for col in gdf.columns:
        if pd.api.types.is_datetime64_any_dtype(gdf[col]):
            gdf[col] = gdf[col].apply(lambda x: x.isoformat() if pd.notnull(x) else None)
    return gdf

def make_fully_serializable(obj):
    # Types de base
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    # Numpy types
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    # Pandas Timestamp/Timedelta
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, pd.Timedelta):
        return str(obj)
    # Pandas DataFrame/Series
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, pd.Series):
        return obj.tolist()
    # xarray Dataset/DataArray
    if isinstance(obj, (xr.Dataset, xr.DataArray)):
        return obj.to_dict()
    # Shapely geometry
    if isinstance(obj, shapely_base.BaseGeometry):
        return mapping(obj)
    # Dataclasses
    if hasattr(obj, "__dataclass_fields__"):
        return {k: make_fully_serializable(v) for k, v in obj.__dict__.items()}
    # Classes avec __dict__
    if hasattr(obj, "__dict__"):
        return {k: make_fully_serializable(v) for k, v in obj.__dict__.items() if not k.startswith("_")}
    # Mapping (dict-like)
    if isinstance(obj, dict):
        return {make_fully_serializable(k): make_fully_serializable(v) for k, v in obj.items()}
    # Iterable (list, tuple, set)
    if isinstance(obj, (list, tuple, set)):
        return [make_fully_serializable(v) for v in obj]
    # Fallback: string representation
    return str(obj)

def add_noise_with_snr(signal: np.ndarray, snr_db: float, seed: int = None) -> np.ndarray:
    """
    Add Gaussian noise to a NumPy array to achieve a desired SNR (in decibels).

    Parameters
    ----------
    signal : np.ndarray
        Input signal array.
    snr_db : float
        Desired Signal-to-Noise Ratio in decibels (dB).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    noisy_signal : np.ndarray
        The signal with added Gaussian noise.
    """
    if seed is not None:
        np.random.seed(seed)

    signal_power = np.mean(signal ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear

    noise = np.random.normal(loc=0.0, scale=np.sqrt(noise_power), size=signal.shape)
    noisy_signal = signal + noise
    return noisy_signal


def nan_to_none(obj):
    if isinstance(obj, float) and np.isnan(obj):
        return None
    if isinstance(obj, pd.Interval):
        # Convertit en string ou tuple
        return str(obj)  # ou (obj.left, obj.right)
    if isinstance(obj, dict):
        return {k: nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [nan_to_none(v) for v in obj]
    return obj


def to_float32(ds: xr.Dataset) -> xr.Dataset:
    """Convertit toutes les variables numériques en float32."""
    for var in ds.data_vars:
        if np.issubdtype(ds[var].dtype, np.floating):
            ds[var] = ds[var].astype("float32")
    return ds