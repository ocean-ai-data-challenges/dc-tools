# -*- coding: UTF-8 -*-

"""Debug, visualization and introspection utilities.

These functions are kept for ad-hoc debugging sessions but are not part of
the production pipeline and are excluded from test-coverage reporting.
"""

import os
import pickle
from collections import deque
from datetime import date, datetime, time
from typing import Any, Dict, List, Optional

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import xarray as xr
from cartopy import crs as ccrs
from dask.distributed import get_client, get_worker


def get_dates_from_startdate(start_date: str, ndays: int) -> List[str]:
    """Get dates of n days after start_date.

    Args:
        date (str): start date
        ndays (int): number of days after start_date

    Returns:
        List[str]: list of n dates.
    """
    list_days: List[Any] = []
    for nday in range(0, ndays):
        time_stamp = pd.to_datetime(start_date) + pd.DateOffset(days=nday)
        list_days.append(time_stamp.strftime("%Y-%m-%d"))
    return list_days


def visualize_netcdf_with_geometry(
    ds: xr.Dataset, geometry: gpd.GeoSeries, coordinates: Dict[str, str], variable_name: str = "zos"
):
    """Visualize a NetCDF dataset variable along with a geometry on a map.

    Args:
        ds (xr.Dataset): The dataset containing the variable to plot.
        geometry (gpd.GeoSeries): The geometry (polygon, point, etc.) to overlay.
        coordinates (Dict[str, str]): Mapping of coordinate names,
            e.g., {'lon': 'longitude', 'lat': 'latitude'}.
        variable_name (str, optional): The name of the variable to visualize. Defaults to "zos".
    """
    lon = ds[coordinates["lon"]]
    lat = ds[coordinates["lat"]]
    variable = ds[variable_name]

    gdf = gpd.GeoDataFrame({"geometry": [geometry]}, crs="EPSG:4326")

    fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
    ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], crs=ccrs.PlateCarree())  # type: ignore

    variable.plot(ax=ax, transform=ccrs.PlateCarree(), cmap="viridis")  # type: ignore
    gdf.plot(ax=ax, edgecolor="red", facecolor="none", linewidth=2, transform=ccrs.PlateCarree())

    plt.show()


def walk_obj(obj):
    """Recursively yield all leaf elements from a nested structure.

    Args:
        obj: The object to traverse (dict, list, tuple, set, or scalar).

    Yields:
        Any: Leaf elements of the structure.
    """
    if isinstance(obj, dict):
        for _key, value in obj.items():
            yield from walk_obj(value)
    elif isinstance(obj, (list, tuple, set)):
        for item in obj:
            yield from walk_obj(item)
    else:
        yield obj


def _safe_repr(x, maxlen=120):
    try:
        s = repr(x)
    except Exception:
        s = str(type(x))
    if len(s) > maxlen:
        return s[: maxlen - 3] + "..."
    return s


def print_structure_types(
    obj,
    indent: int = 0,
    max_depth: int = 6,
    max_items: int = 10,
    show_values: bool = False,
    follow_attrs: bool = False,
    visited: Optional[set] = None,
):
    """Print hierarchically the types contained in `obj` with some metadata.

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
    if obj is None or isinstance(obj, (bool, int, float, str)):
        val = _safe_repr(obj) if show_values else ""
        print(f"{prefix}- {t} {val}")
        return

    if isinstance(obj, np.generic):
        try:
            val = obj.item()
        except Exception:
            val = str(obj)
        val_str = f": {_safe_repr(val)}" if show_values else ""
        print(f"{prefix}- numpy scalar {obj.dtype}{val_str}")
        return

    if isinstance(obj, np.ndarray):
        dtype = getattr(obj, "dtype", None)
        shape = getattr(obj, "shape", None)
        print(f"{prefix}- numpy.ndarray dtype={dtype} shape={shape}")
        if obj.shape == ():
            try:
                sval = obj.item()
            except Exception:
                sval = None
            if show_values:
                print(f"{prefix}  scalar value: {_safe_repr(sval)}")
            return
        flat = obj.flat
        n = min(max_items, obj.size)
        for i, x in enumerate(flat):
            if i >= n:
                print(f"{prefix}  ... ({obj.size - n} more elements)")
                break
            print(f"{prefix}  index {i}:")
            print_structure_types(
                x, indent + 2, max_depth, max_items, show_values, follow_attrs, visited
            )
        return

    if isinstance(obj, (pd.Timestamp, np.datetime64)):
        print(f"{prefix}- pandas.Timestamp: {str(pd.to_datetime(obj))}")
        return
    if isinstance(obj, pd.Timedelta):
        print(f"{prefix}- pandas.Timedelta: {str(obj)}")
        return
    if isinstance(obj, pd.Interval):
        print(
            f"{prefix}- pandas.Interval left={_safe_repr(obj.left)} "
            f"right={_safe_repr(obj.right)} closed={obj.closed}"
        )
        return

    if isinstance(obj, pd.DataFrame):
        print(f"{prefix}- pandas.DataFrame shape={obj.shape}")
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

    if isinstance(obj, xr.Dataset):
        print(f"{prefix}- xarray.Dataset dims={dict(obj.dims)}")
        vars_list = list(obj.data_vars)[:max_items]
        print(
            f"{prefix}  data_vars: {vars_list} "
            f"(+{max(0, len(obj.data_vars) - len(vars_list))} more)"
        )
        return
    if isinstance(obj, xr.DataArray):
        print(f"{prefix}- xarray.DataArray dims={obj.dims} shape={obj.shape}")
        return

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
                print_structure_types(
                    v, indent + 2, max_depth, max_items, show_values, follow_attrs, visited
                )
        return

    if isinstance(obj, (list, tuple, set, frozenset, deque)):
        n = len(obj)
        print(f"{prefix}- {t.__name__} len={n}")
        for i, item in enumerate(list(obj)[:max_items]):
            print(f"{prefix}  index {i}:")
            if indent + 1 >= max_depth:
                print(f"{prefix}    - max depth reached")
            else:
                print_structure_types(
                    item, indent + 2, max_depth, max_items, show_values, follow_attrs, visited
                )
        if n > max_items:
            print(f"{prefix}  ... ({n - max_items} more elements)")
        return

    if isinstance(obj, (datetime, date, time)):
        print(f"{prefix}- {type(obj).__name__} {str(obj)}")
        return

    if isinstance(obj, (bytes, bytearray, memoryview)):
        print(f"{prefix}- {type(obj).__name__} len={len(obj)} repr={_safe_repr(obj, 80)}")
        return

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
                print_structure_types(
                    v, indent + 3, max_depth, max_items, show_values, follow_attrs, visited
                )
    else:
        print(f"{prefix}  repr: {_safe_repr(obj, 200)}")


def find_unpicklable_objects(obj, path=""):
    """Recursively finds unserializable objects."""
    try:
        pickle.dumps(obj)
        return []
    except Exception as e:
        print(f"Not serializable at {path}: {type(obj)} - {str(e)[:100]}")

        if isinstance(obj, dict):
            problematic: List[Any] = []
            for key, value in obj.items():
                try:
                    pickle.dumps(value)
                except Exception:
                    problematic.extend(find_unpicklable_objects(value, f"{path}.{key}"))
            return problematic
        elif isinstance(obj, (list, tuple)):
            problematic = []
            for i, item in enumerate(obj):
                try:
                    pickle.dumps(item)
                except Exception:
                    problematic.extend(find_unpicklable_objects(item, f"{path}[{i}]"))
            return problematic
        elif hasattr(obj, "__dict__"):
            problematic = []
            for attr_name, attr_value in obj.__dict__.items():
                try:
                    pickle.dumps(attr_value)
                except Exception:
                    problematic.extend(
                        find_unpicklable_objects(attr_value, f"{path}.{attr_name}")
                    )
            return problematic

        return [(path, type(obj), str(e))]


def log_memory(stage):
    """Log the current memory usage of the process.

    Args:
        stage (str): A label for the logging stage (e.g., 'Before processing').
    """
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / 1e6
    print(f"[{stage}] Memory usage: {mem_mb:.2f} MB")


def is_dask_worker():
    """Check via environment variables."""
    return any(
        [
            "DASK_WORKER_NAME" in os.environ,
            "DASK_SCHEDULER_ADDRESS" in os.environ,
            os.environ.get("DASK_WORKER", False),
        ]
    )


def ensure_timestamp(date_input):
    """Convert to Timestamp only if not already a Timestamp."""
    if isinstance(date_input, pd.Timestamp):
        return date_input
    else:
        return pd.to_datetime(date_input)


def get_active_workers_count():
    """Return the number of active Dask workers."""
    try:
        client = get_client()
        workers_info = client.scheduler_info()["workers"]
        active_workers = len(workers_info)
        return active_workers
    except Exception:
        return 0


def get_dask_config_workers():
    """Return Dask workers configuration."""
    try:
        client = get_client()
        return {
            "n_workers": len(client.scheduler_info()["workers"]),
            "threads_per_worker": client.scheduler_info().get("threads_per_worker", 1),
            "total_cores": sum(w["nthreads"] for w in client.scheduler_info()["workers"].values()),
        }
    except Exception:
        return {"n_workers": 0, "threads_per_worker": 0, "total_cores": 0}


def get_current_worker_id():
    """Get current worker ID from within a task."""
    try:
        worker = get_worker()
        return {
            "worker_id": worker.id,
            "worker_address": worker.address,
            "worker_name": getattr(worker, "name", "unknown"),
            "worker_threads": getattr(worker.state, "nthreads", "unknown"),
            "worker_memory_limit": getattr(worker.memory_manager, "memory_limit", "unknown"),
        }
    except Exception as e:
        return {"error": f"Not running in worker context: {e}"}


def show_worker_info():
    """Function executed on a Dask worker."""
    try:
        worker = get_worker()
        worker_id = worker.id
        worker_address = worker.address
        worker_name = getattr(worker, "name", "unknown")
        process_id = os.getpid()

        print(f"Worker ID: {worker_id}")
        print(f"Worker Name: {worker_name}")
        print(f"Worker Address: {worker_address}")
        print(f"Process ID: {process_id}")

    except ValueError as e:
        print(f"Not running in worker context: {e}")
        return f"Not in worker (PID: {os.getpid()})"
