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
from shapely.geometry import mapping, Polygon
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

    # Ajouter des caractéristiques naturelles (par exemple, côtes)
    #ax.add_feature(NaturalEarthFeature('physical', 'coastline', '10m', edgecolor='black'))

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
    return obj



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

'''import folium
import geopandas as gpd

m = folium.Map(zoom_start=2)
for _, row in catalog.iterrows():
    geojson = gpd.GeoSeries([row.geometry]).to_json()
    folium.GeoJson(geojson, tooltip=row.path).add_to(m)
'''


'''
# Exemple d'instance CatalogEntry
catalog_entry = CatalogEntry(
    path="path/to/netcdf_file.nc",
    date_start="2023-01-01",
    date_end="2023-01-31",
    variables={"temperature": ["time", "lat", "lon"]},
    dimensions={"lat": "latitude", "lon": "longitude"},
    coord_type="geographic",
    crs="EPSG:4326",
    geometry=Polygon([(-10, 40), (-10, 50), (0, 50), (0, 40), (-10, 40)])
)

visualize_netcdf_with_geometry(catalog_entry)'''
