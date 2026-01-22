# -*- coding: UTF-8 -*-

"""Miscellaneous utility functions."""

import os
import pickle
import traceback
import psutil
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Union

import copy
from cartopy import crs as ccrs
from dask.distributed import get_worker, get_client
import datetime
import dill
import geopandas as gpd
import json
from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import mapping, base as shapely_base
import xarray as xr

from collections import deque
from datetime import datetime, date, time, timedelta
from dask.distributed import get_worker
import os


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
        home_path = os.environ['HOME']
    elif 'USERPROFILE' in os.environ:
        home_path = os.environ['USERPROFILE']
    elif 'HOMEPATH' in os.environ:
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
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, gpd.GeoDataFrame):
        return obj.to_json()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.generic,)):
        return obj.item()
    if isinstance(obj, xr.Dataset) or isinstance(obj, xr.DataArray):
        return obj.to_dict()
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [make_serializable(v) for v in obj]
    return obj

def make_timestamps_serializable(gdf: pd.DataFrame) -> pd.DataFrame:
    gdf = gdf.copy()
    for col in gdf.columns:
        if pd.api.types.is_datetime64_any_dtype(gdf[col]):
            gdf[col] = gdf[col].apply(lambda x: x.isoformat() if pd.notnull(x) else None)
    return gdf

def _replace_nan_in_nested_list(obj):
    """Fonction helper pour remplacer les NaN dans les listes imbriquées."""
    if isinstance(obj, list):
        return [_replace_nan_in_nested_list(item) for item in obj]
    elif isinstance(obj, float):
        if np.isnan(obj):
            return None
        elif np.isposinf(obj):
            return "Infinity"
        elif np.isneginf(obj):
            return "-Infinity"
        return obj
    else:
        return obj



def serialize_optimized(obj):
    """
    Convert object to JSON-serializable form efficiently for large structures.

    Handles:
        - dict
        - list, tuple, set
        - np.ndarray
        - pandas.Timestamp
        - pandas.Interval
        - pandas.DataFrame (converted to records)
        - pandas.Series
        - basic scalars
    """
    # Basic types
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj

    # numpy scalar
    if isinstance(obj, np.generic):
        return obj.item()

    # numpy array
    if isinstance(obj, np.ndarray):
        if obj.shape == ():  # scalar
            return obj.item()
        else:
            return obj.tolist()  # avoid recursion for performance

    # pandas Timestamp
    if isinstance(obj, pd.Timestamp):
        if pd.isna(obj):
            return None
        return obj.isoformat()

    # pandas Interval
    if isinstance(obj, pd.Interval):
        return {"left": obj.left, "right": obj.right, "closed": obj.closed}

    # pandas DataFrame
    if isinstance(obj, pd.DataFrame):
        # convert each row to dict but process columns recursively
        records = []
        for row in obj.itertuples(index=False, name=None):
            row_dict = {col: serialize_optimized(val) for col, val in zip(obj.columns, row)}
            records.append(row_dict)
        return records

    # pandas Series
    if isinstance(obj, pd.Series):
        return [serialize_optimized(x) for x in obj.tolist()]

    # dict
    if isinstance(obj, dict):
        return {k: serialize_optimized(v) for k, v in obj.items()}

    # list / tuple / set
    if isinstance(obj, (list, tuple, set)):
        return [serialize_optimized(x) for x in obj]

    # fallback: convert unknown object to string
    return str(obj)


def serialize_structure(obj):
    """
    Serialize a complex object into JSON and save to file.
    """
    serializable_obj = serialize_optimized(obj)
    return serializable_obj


def _safe_repr(x, maxlen=120):
    try:
        s = repr(x)
    except Exception:
        s = str(type(x))
    if len(s) > maxlen:
        return s[:maxlen-3] + "..."
    return s

def to_float32(obj: Any) -> Any:
    """Recursively converts all float64 data to float32 in xarray objects or dicts."""
    if isinstance(obj, (xr.Dataset, xr.DataArray)):
        return obj.astype("float32")
    elif isinstance(obj, dict):
        return {k: to_float32(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_float32(v) for v in obj]
    return obj


def print_structure_types(
    obj,
    indent: int = 0,
    max_depth: int = 6,
    max_items: int = 10,
    show_values: bool = False,
    follow_attrs: bool = False,
    visited: set = None,
):
    """
    Print hierarchically the types contained in `obj` with some metadata.

    Parameters
    ----------
    obj:
        Any Python object to inspect.
    indent:
        Current indentation level (used internally for recursion).
    max_depth:
        Maximum recursion depth.
    max_items:
        Max items to display per container (dict/list/array).
    show_values:
        If True, show small values for scalars/strings.
    follow_attrs:
        If True, explore object.__dict__ for arbitrary objects (dangerous for pandas).
    visited:
        Internal: set of visited object ids to avoid infinite recursion.
    """
    prefix = "  " * indent
    if visited is None:
        visited = set()

    oid = id(obj)
    if oid in visited:
        print(f"{prefix}- <CYCLE detected: {type(obj)}> (id={oid})")
        return
    visited.add(oid)

    t = type(obj)
    # Simple scalars
    if obj is None or isinstance(obj, (bool, int, float, str)):
        val = _safe_repr(obj) if show_values else ""
        print(f"{prefix}- {t} {val}")
        return

    # Numpy generic scalar
    if isinstance(obj, np.generic):
        try:
            val = obj.item()
        except Exception:
            val = str(obj)
        val_str = f": {_safe_repr(val)}" if show_values else ""
        print(f"{prefix}- numpy scalar {obj.dtype}{val_str}")
        return

    # Numpy array (including object arrays)
    if isinstance(obj, np.ndarray):
        dtype = getattr(obj, "dtype", None)
        shape = getattr(obj, "shape", None)
        print(f"{prefix}- numpy.ndarray dtype={dtype} shape={shape}")
        if obj.shape == ():  # scalar ndarray
            try:
                sval = obj.item()
            except Exception:
                sval = None
            if show_values:
                print(f"{prefix}  scalar value: {_safe_repr(sval)}")
            return
        # For non-scalar arrays, show a few items (flattened)
        flat = obj.flat
        n = min(max_items, obj.size)
        for i, x in enumerate(flat):
            if i >= n:
                print(f"{prefix}  ... ({obj.size - n} more elements)")
                break
            print(f"{prefix}  index {i}:")
            print_structure_types(x, indent + 2, max_depth, max_items, show_values, follow_attrs, visited)
        return

    # pandas Timestamp / Timedelta / Interval
    if isinstance(obj, (pd.Timestamp, np.datetime64)):
        print(f"{prefix}- pandas.Timestamp: {str(pd.to_datetime(obj))}")
        return
    if isinstance(obj, pd.Timedelta):
        print(f"{prefix}- pandas.Timedelta: {str(obj)}")
        return
    if isinstance(obj, pd.Interval):
        print(f"{prefix}- pandas.Interval left={_safe_repr(obj.left)} right={_safe_repr(obj.right)} closed={obj.closed}")
        return

    # Pandas DataFrame / Series
    if isinstance(obj, pd.DataFrame):
        print(f"{prefix}- pandas.DataFrame shape={obj.shape}")
        # show columns and dtypes
        for col in obj.columns[:max_items]:
            print(f"{prefix}  column: {col!r} dtype={obj[col].dtype}")
        if obj.shape[0] > 0 and max_items > 0:
            sample = obj.head(3).to_dict(orient="records")
            print(f"{prefix}  sample (up to 3 rows):")
            for row in sample:
                print(f"{prefix}    {_safe_repr(row, 200)}")
        return
    if isinstance(obj, pd.Series):
        print(f"{prefix}- pandas.Series dtype={obj.dtype} length={len(obj)}")
        if show_values:
            print(f"{prefix}  head: {_safe_repr(list(obj.head(5)), 200)}")
        return

    # xarray
    if isinstance(obj, xr.Dataset):
        print(f"{prefix}- xarray.Dataset dims={dict(obj.dims)}")
        # list variables
        vars_list = list(obj.data_vars)[:max_items]
        print(f"{prefix}  data_vars: {vars_list} (+{max(0, len(obj.data_vars)-len(vars_list))} more)")
        return
    if isinstance(obj, xr.DataArray):
        print(f"{prefix}- xarray.DataArray dims={obj.dims} shape={obj.shape}")
        return

    # dict
    if isinstance(obj, dict):
        n = len(obj)
        print(f"{prefix}- dict len={n}")
        for i, (k, v) in enumerate(obj.items()):
            if i >= max_items:
                print(f"{prefix}  ... ({n - max_items} more keys)")
                break
            print(f"{prefix}  key: {k!r} (type={type(k)})")
            if indent + 1 >= max_depth:
                print(f"{prefix}    - max depth reached")
            else:
                print_structure_types(v, indent + 2, max_depth, max_items, show_values, follow_attrs, visited)
        return

    # list/tuple/set/deque
    if isinstance(obj, (list, tuple, set, frozenset, deque)):
        n = len(obj)
        print(f"{prefix}- {t.__name__} len={n}")
        for i, item in enumerate(list(obj)[:max_items]):
            print(f"{prefix}  index {i}:")
            if indent + 1 >= max_depth:
                print(f"{prefix}    - max depth reached")
            else:
                print_structure_types(item, indent + 2, max_depth, max_items, show_values, follow_attrs, visited)
        if n > max_items:
            print(f"{prefix}  ... ({n - max_items} more elements)")
        return

    # pathlib.Path, datetime.date/time
    if isinstance(obj, (datetime, date, time)):
        print(f"{prefix}- {type(obj).__name__} {str(obj)}")
        return

    # bytes / bytearray
    if isinstance(obj, (bytes, bytearray, memoryview)):
        print(f"{prefix}- {type(obj).__name__} len={len(obj)} repr={_safe_repr(obj, 80)}")
        return

    # Generic object: optionally follow __dict__ if requested
    print(f"{prefix}- {t} (generic object)")
    if follow_attrs and hasattr(obj, "__dict__"):
        attrs = {k: v for k, v in vars(obj).items() if not k.startswith("_")}
        if not attrs:
            print(f"{prefix}  (no public attrs or all private - skip)")
            return
        print(f"{prefix}  public attrs: {list(attrs.keys())}")
        for k, v in attrs.items():
            print(f"{prefix}    attr: {k}")
            if indent + 1 >= max_depth:
                print(f"{prefix}      - max depth reached")
            else:
                print_structure_types(v, indent + 3, max_depth, max_items, show_values, follow_attrs, visited)
    else:
        # try to print small repr
        print(f"{prefix}  repr: {_safe_repr(obj, 200)}")


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
    if isinstance(obj, float) and (pd.isna(obj) or obj != obj):
        return None
    if isinstance(obj, pd.Interval):
        # Convertit en string ou tuple
        return str(obj)  # ou (obj.left, obj.right)
    if isinstance(obj, pd.Timestamp) and pd.isna(obj):
        return None
    if isinstance(obj, pd.NaT.__class__):  # Pour NaTType
        return None
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


def find_unpicklable_objects(obj, path=""):
    """
    Trouve récursivement les objets non sérialisables.
    """
    try:
        # Test avec pickle standard
        pickle.dumps(obj)
        return []
    except Exception as e:
        print(f"Not serializable at {path}: {type(obj)} - {str(e)[:100]}")
        
        # Si c'est un dictionnaire, tester chaque clé/valeur
        if isinstance(obj, dict):
            problematic = []
            for key, value in obj.items():
                try:
                    pickle.dumps(value)
                except:
                    problematic.extend(find_unpicklable_objects(value, f"{path}.{key}"))
            return problematic
            
        # Si c'est une liste/tuple
        elif isinstance(obj, (list, tuple)):
            problematic = []
            for i, item in enumerate(obj):
                try:
                    pickle.dumps(item)
                except:
                    problematic.extend(find_unpicklable_objects(item, f"{path}[{i}]"))
            return problematic
            
        # Si c'est un objet avec __dict__
        elif hasattr(obj, "__dict__"):
            problematic = []
            for attr_name, attr_value in obj.__dict__.items():
                try:
                    pickle.dumps(attr_value)
                except:
                    problematic.extend(find_unpicklable_objects(attr_value, f"{path}.{attr_name}"))
            return problematic
            
        # Objet problématique trouvé
        return [(path, type(obj), str(e))]


def log_memory(stage):
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / 1e6
    print(f"[{stage}] Memory usage: {mem_mb:.2f} MB")


def is_dask_worker():
    """Vérifie via les variables d'environnement."""
    return any([
        'DASK_WORKER_NAME' in os.environ,
        'DASK_SCHEDULER_ADDRESS' in os.environ,
        os.environ.get('DASK_WORKER', False)
    ])

def ensure_timestamp(date_input):
    """Convertit en Timestamp seulement si ce n'est pas déjà un Timestamp."""
    if isinstance(date_input, pd.Timestamp):
        return date_input
    else:
        return pd.to_datetime(date_input)


def deep_copy_object(obj: Any, skip_list: Optional[List[Union[str, type]]] = None) -> Any:
    """Version simplifiée qui gère mieux les types spéciaux."""
    if skip_list is None:
        skip_list = []
    
    # Types primitifs
    if obj is None or isinstance(obj, (str, int, float, bool, bytes, type)):
        return obj
    
    # Vérifier skip_list
    obj_type = type(obj)
    if obj_type in skip_list or obj_type.__name__ in skip_list:
        return obj
    
    # Types non-copiables
    non_copyable_types = ['Client', 'LocalCluster', 'DatasetProcessor', 'ArgoIndex', 'DataFetcher']
    if any(name in obj_type.__name__ for name in non_copyable_types):
        return obj
    
    # Gestion spéciale en premier pour éviter les erreurs
    if isinstance(obj, SimpleNamespace):
        copied_vars = {k: deep_copy_object(v, skip_list) if k not in skip_list else v 
                      for k, v in vars(obj).items()}
        return SimpleNamespace(**copied_vars)
    
    if hasattr(obj, '__class__') and obj.__class__.__name__ == 'Namespace':
        from argparse import Namespace
        copied_vars = {k: deep_copy_object(v, skip_list) if k not in skip_list else v 
                      for k, v in vars(obj).items()}
        return Namespace(**copied_vars)
    
    # Essayer pickle d'abord
    try:
        pickle.dumps(obj)
        return copy.deepcopy(obj)
    except:
        pass
    
    # Collections standards
    if isinstance(obj, list):
        return [deep_copy_object(item, skip_list) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(deep_copy_object(item, skip_list) for item in obj)
    elif isinstance(obj, dict):
        return {k: deep_copy_object(v, skip_list) if k not in skip_list else v 
                for k, v in obj.items()}
    elif isinstance(obj, set):
        return {deep_copy_object(item, skip_list) for item in obj}
    
    # Fallback : retourner l'original
    return obj

def get_active_workers_count():
    """Retourne le nombre de workers Dask actifs."""
    try:
        # Méthode 1 : Via le client Dask
        client = get_client()
        workers_info = client.scheduler_info()['workers']
        active_workers = len(workers_info)
        return active_workers
    except:
        # Pas de client Dask actif
        return 0

def get_dask_config_workers():
    """Retourne la configuration des workers Dask."""
    try:
        client = get_client()
        return {
            'n_workers': len(client.scheduler_info()['workers']),
            'threads_per_worker': client.scheduler_info().get('threads_per_worker', 1),
            'total_cores': sum(w['nthreads'] for w in client.scheduler_info()['workers'].values())
        }
    except:
        return {'n_workers': 0, 'threads_per_worker': 0, 'total_cores': 0}


def get_current_worker_id():
    """Obtient l'ID du worker actuel depuis l'intérieur d'une tâche."""
    try:
        worker = get_worker()
        return {
            'worker_id': worker.id,
            'worker_address': worker.address,
            'worker_name': getattr(worker, 'name', 'unknown'),
            'worker_threads': getattr(worker.state, 'nthreads', 'unknown'),
            'worker_memory_limit': getattr(worker.memory_manager, 'memory_limit', 'unknown'),
        }
    except Exception as e:
        return {'error': f"Not running in worker context: {e}"}


def show_worker_info():
    """Fonction exécutée sur un worker Dask."""
    try:
        # Récupérer le worker actuel
        worker = get_worker()
        
        # Informations sur le worker
        worker_id = worker.id
        worker_address = worker.address
        worker_name = getattr(worker, 'name', 'unknown')
        process_id = os.getpid()
        
        print(f"Worker ID: {worker_id}")
        print(f"Worker Name: {worker_name}")
        print(f"Worker Address: {worker_address}")
        print(f"Process ID: {process_id}")
        
    except ValueError as e:
        # Pas dans un contexte de worker Dask
        print(f"Not running in worker context: {e}")
        return f"Not in worker (PID: {os.getpid()})"


def find_unpicklable_objects(obj, path="root", max_depth=15, visited=None):
    """
    Explore récursivement un objet et affiche les sous-objets non picklables avec leur chemin.
    """
    if visited is None:
        visited = set()
    obj_id = id(obj)
    if obj_id in visited or max_depth < 0:
        return
    visited.add(obj_id)
    try:
        pickle.dumps(obj)
    except Exception as e:
        print(f"Unpicklable at {path}: {type(obj)} - {e}")
        # Explorer les attributs __dict__ si possible
        if hasattr(obj, "__dict__"):
            for k, v in obj.__dict__.items():
                find_unpicklable_objects(v, f"{path}.{k}", max_depth-1, visited)
        # Explorer les items si dict
        elif isinstance(obj, dict):
            for k, v in obj.items():
                find_unpicklable_objects(v, f"{path}[{repr(k)}]", max_depth-1, visited)
        # Explorer les éléments si list/tuple/set
        elif isinstance(obj, (list, tuple, set)):
            for i, v in enumerate(obj):
                find_unpicklable_objects(v, f"{path}[{i}]", max_depth-1, visited)
        # Explorer les slots
        elif hasattr(obj, "__slots__"):
            for k in obj.__slots__:
                v = getattr(obj, k, None)
                find_unpicklable_objects(v, f"{path}.{k}", max_depth-1, visited)


def list_all_days(
        start_date: datetime,
        end_date: datetime
    ) -> list[datetime]:
    """
    Return a list of datetime.datetime objects for each day between start_date and end_date (inclusive).

    Parameters:
        start_date (datetime): The start of the range.
        end_date (datetime): The end of the range.

    Returns:
        List[datetime]: List of dates at 00:00:00 for each day in the range.
    """
    if start_date > end_date:
        raise ValueError("start_date must be before or equal to end_date.")

    start = datetime.combine(start_date.date(), datetime.min.time())
    end = datetime.combine(end_date.date(), datetime.min.time())

    n_days = (end - start).days + 1
    return [start + timedelta(days=i) for i in range(n_days)]
