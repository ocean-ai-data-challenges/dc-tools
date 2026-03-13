# -*- coding: UTF-8 -*-

"""Miscellaneous utility functions."""

import copy
import os
import pickle
from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import Any, List, Optional, Union

import numpy as np
import pandas as pd
import xarray as xr

try:
    import geopandas as gpd
except ImportError:
    gpd = None


def get_home_path() -> str:
    """Get the user home directory path in a cross-platform way.

    Returns:
        str: The path to the user's home directory.
    """
    if "HOME" in os.environ:
        home_path = os.environ["HOME"]
    elif "USERPROFILE" in os.environ:
        home_path = os.environ["USERPROFILE"]
    elif "HOMEPATH" in os.environ:
        home_path = os.environ["HOMEPATH"]
    return home_path


def transform_in_place(obj, func):
    """Recursively apply a function to all elements of a nested structure in place.

    Args:
        obj: The object to transform (dict, list, or scalar).
        func (Callable): The function to apply to leaf elements.

    Returns:
        Any: The transformed object.
    """
    if isinstance(obj, dict):
        for k, v in obj.items():
            obj[k] = transform_in_place(v, func)
        return obj
    elif isinstance(obj, list):
        for i in range(len(obj)):
            obj[i] = transform_in_place(obj[i], func)
        return obj
    else:
        # For immutable types: apply function directly
        return func(obj)


def make_serializable(obj):
    """Make an object JSON serializable.

    Handles timestamps, DataFrames, GeoDataFrames, ndarrays, xarray objects,
    dictionaries, and lists.

    Args:
        obj (Any): The object to convert.

    Returns:
        Any: The JSON-serializable object.
    """
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if gpd is not None and isinstance(obj, gpd.GeoDataFrame):
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
    """Convert datetime columns in a DataFrame to ISO format strings.

    Args:
        gdf (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with serializable timestamps.
    """
    gdf = gdf.copy()
    for col in gdf.columns:
        if pd.api.types.is_datetime64_any_dtype(gdf[col]):
            gdf[col] = gdf[col].apply(lambda x: x.isoformat() if pd.notnull(x) else None)
    return gdf


def _replace_nan_in_nested_list(obj):
    """Helper function to replace NaNs in nested lists."""
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
    """Convert object to JSON-serializable form efficiently for large structures.

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
        records: List[Any] = []
        for row in obj.itertuples(index=False, name=None):
            row_dict = {
                col: serialize_optimized(val)
                for col, val in zip(obj.columns, row, strict=False)
            }
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
    """Serialize a complex object into JSON and save to file."""
    serializable_obj = serialize_optimized(obj)
    return serializable_obj


def to_float32(obj: Any) -> Any:
    """Recursively converts all float64 data to float32 in xarray objects or dicts."""
    if isinstance(obj, xr.Dataset):
        # Specific implementation for Dataset to be safe with types
        ds = obj.copy()
        for var in ds.data_vars:
            if np.issubdtype(ds[var].dtype, np.floating):
                ds[var] = ds[var].astype("float32")
        return ds
    elif isinstance(obj, xr.DataArray):
        if np.issubdtype(obj.dtype, np.floating):
            return obj.astype("float32")
        return obj
    elif isinstance(obj, dict):
        return {k: to_float32(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_float32(v) for v in obj]
    return obj


def add_noise_with_snr(
    signal: np.ndarray, snr_db: float, seed: Optional[int] = None,
) -> np.ndarray:
    """Add Gaussian noise to a NumPy array to achieve a desired SNR (in decibels).

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

    signal_power = np.mean(signal**2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear

    noise = np.random.normal(loc=0.0, scale=np.sqrt(noise_power), size=signal.shape)
    noisy_signal: np.ndarray = signal + noise
    return noisy_signal


def nan_to_none(obj):
    """Recursively replace NaN values with None in an object (dict, list, or scalar).

    Args:
        obj (Any): The input object.

    Returns:
        Any: The object with NaNs replaced by None.
    """
    if isinstance(obj, float) and np.isnan(obj):
        return None
    if isinstance(obj, float) and (pd.isna(obj) or obj != obj):
        return None
    if isinstance(obj, pd.Interval):
        # Converts to string or tuple
        return str(obj)  # or (obj.left, obj.right)
    if isinstance(obj, pd.Timestamp) and pd.isna(obj):
        return None
    if isinstance(obj, pd.NaT.__class__):  # For NaTType
        return None
    if isinstance(obj, dict):
        return {k: nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [nan_to_none(v) for v in obj]
    return obj


def deep_copy_object(
    obj: Any, skip_list: Optional[List[Union[str, type]]] = None,
) -> Any:
    """Simplified version that handles special types better."""
    if skip_list is None:
        safe_skip_list: List[Any] = []
    else:
        safe_skip_list = list(skip_list)  # copy to be safe

    # Primitive types
    if obj is None or isinstance(obj, (str, int, float, bool, bytes, type)):
        return obj

    # Check skip_list
    obj_type = type(obj)
    if obj_type in safe_skip_list or obj_type.__name__ in safe_skip_list:
        return obj

    # Non-copyable types
    non_copyable_types = [
        "Client", "LocalCluster", "DatasetProcessor", "ArgoIndex", "DataFetcher",
    ]
    if any(name in obj_type.__name__ for name in non_copyable_types):
        return obj

    # Special handling first to avoid errors
    if isinstance(obj, SimpleNamespace):
        copied_vars = {
            k: deep_copy_object(v, safe_skip_list) if k not in safe_skip_list else v
            for k, v in vars(obj).items()
        }
        return SimpleNamespace(**copied_vars)

    if hasattr(obj, "__class__") and obj.__class__.__name__ == "Namespace":
        from argparse import Namespace

        copied_vars = {
            k: deep_copy_object(v, safe_skip_list) if k not in safe_skip_list else v
            for k, v in vars(obj).items()
        }
        return Namespace(**copied_vars)

    # Try pickle first
    try:
        pickle.dumps(obj)
        return copy.deepcopy(obj)
    except Exception:
        pass

    # Standard collections
    if isinstance(obj, list):
        return [deep_copy_object(item, safe_skip_list) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(deep_copy_object(item, safe_skip_list) for item in obj)
    elif isinstance(obj, dict):
        return {
            k: deep_copy_object(v, safe_skip_list) if k not in safe_skip_list else v
            for k, v in obj.items()
        }
    elif isinstance(obj, set):
        return {deep_copy_object(item, safe_skip_list) for item in obj}

    # Fallback: return original
    return obj


def list_all_days(start_date: datetime, end_date: datetime) -> list[datetime]:
    """Return a list of datetime.datetime objects for each day between dates.

    For each day between start_date and end_date (inclusive).

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


def display_width(text: str) -> int:
    """Return the monospace terminal display width of *text*.

    Emoji and other wide characters occupy 2 columns but Python's
    ``len()`` counts them as 1 (or 2 when a variation selector is
    present).  This helper compensates for that.

    Parameters
    ----------
    text : str
        The string whose display width is to be computed.

    Returns
    -------
    int
        The number of terminal columns the string would occupy.
    """
    import unicodedata

    w = 0
    for ch in text:
        cat = unicodedata.category(ch)
        # Zero-width: combining marks, enclosing marks, format chars
        if cat.startswith("M") or cat == "Cf":
            continue
        cp = ord(ch)
        eaw = unicodedata.east_asian_width(ch)
        if (
            eaw in ("W", "F")
            or 0x1F000 <= cp <= 0x1FFFF  # Supplemental Symbols & Pictographs
            or 0x1F900 <= cp <= 0x1F9FF  # Supplemental Symbols Extended-A
        ):
            w += 2
        else:
            w += 1
    return w
