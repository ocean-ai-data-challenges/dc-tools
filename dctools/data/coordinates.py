"""Coordinate system utilities and transformations."""

from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
import json
from typing import Any, Dict, List, DefaultDict, Optional, Union

from loguru import logger
import numpy as np

try:
    from oceanbench.core.lagrangian_trajectory import ZoneCoordinates
except Exception:

    @dataclass(frozen=True)
    class ZoneCoordinates:  # type: ignore[no-redef]
        """Fallback stub for ZoneCoordinates when oceanbench is unavailable."""

        minimum_latitude: float
        maximum_latitude: float
        minimum_longitude: float
        maximum_longitude: float


try:
    from oceanbench.core.rmsd import Variable
except Exception:

    class Variable(str, Enum):  # type: ignore[no-redef]
        """Fallback stub for Variable when oceanbench is unavailable."""

        SEA_SURFACE_HEIGHT_ABOVE_GEOID = "sea_surface_height_above_geoid"
        SEA_WATER_POTENTIAL_TEMPERATURE = "sea_water_potential_temperature"
        SEA_WATER_SALINITY = "sea_water_salinity"
        NORTHWARD_SEA_WATER_VELOCITY = "northward_sea_water_velocity"
        EASTWARD_SEA_WATER_VELOCITY = "eastward_sea_water_velocity"


import pandas as pd
from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry
from shapely import box, MultiPoint, simplify
import xarray as xr


# Coordinate map (matches)
from dctools_core.aliases import (  # noqa: F401  (moved to dctools-core, re-exported)
    COORD_ALIASES,
    COORD_STD_NAMES,
    VARIABLES_ALIASES,
    GEO_STD_COORDS,
    _match_var_by_std_name,
    _match_coord_by_std_name,
    get_standardized_var_name,
)


EVAL_VARIABLES_GLONET = [
    Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID,
    Variable.SEA_WATER_POTENTIAL_TEMPERATURE,
    Variable.SEA_WATER_SALINITY,
    Variable.NORTHWARD_SEA_WATER_VELOCITY,
    Variable.EASTWARD_SEA_WATER_VELOCITY,
]

GLOBAL_ZONE_COORDINATES = ZoneCoordinates(
    minimum_latitude=-90,
    maximum_latitude=90,
    minimum_longitude=-180,
    maximum_longitude=180,
)

# Possible names of coordinates that we want to check for

LIST_VARS_GLONET = ["thetao", "zos", "uo", "vo", "so", "depth", "lat", "lon", "time"]


def _as_dict(cfg: Any) -> Dict[str, Any]:
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return cfg
    try:
        return dict(vars(cfg))
    except Exception:
        return {}


def _build_axis_from_spec(spec: Any) -> Any:
    """Build a 1D axis from a YAML-friendly spec.

    Supported forms:
    - list/tuple: returned as-is
    - {start, stop, step}: returns np.arange(start, stop, step)
    """
    if spec is None:
        return None
    if isinstance(spec, (list, tuple, range, np.ndarray)):
        return spec
    if isinstance(spec, dict):
        if {"start", "stop", "step"}.issubset(spec.keys()):
            return np.arange(float(spec["start"]), float(spec["stop"]), float(spec["step"]))
    return spec


def get_target_dimensions(cfg: Any, *, surface: bool = False) -> Dict[str, Any]:
    """Return the target dimensions dict from global config.

    This replaces the former module-level constants (TARGET_DIM_RANGES, TARGET_DEPTH_VALS, ...)
    and keeps the grid definition in the YAML config.
    """
    cfg_dict = _as_dict(cfg)
    key = "target_dimensions_surface" if surface else "target_dimensions"
    spec = cfg_dict.get(key) or {}
    if not isinstance(spec, dict):
        return {}

    out: Dict[str, Any] = {}
    for axis_name, axis_spec in spec.items():
        out[axis_name] = _build_axis_from_spec(axis_spec)
    return out


def get_target_depth_values(cfg: Any, *, surface: bool = False) -> Optional[List[float]]:
    """Extract configured depth values from a global config dict."""
    dims = get_target_dimensions(cfg, surface=surface)
    depth = dims.get("depth")
    if depth is None:
        return None
    if isinstance(depth, np.ndarray):
        return [float(x) for x in depth.tolist()]
    if isinstance(depth, range):
        return [float(x) for x in list(depth)]
    if isinstance(depth, (list, tuple)):
        return [float(x) for x in depth]
    return None


def get_target_time_values(cfg: Any) -> Optional[List[Any]]:
    """Extract configured target time values from a global config dict."""
    cfg_dict = _as_dict(cfg)
    vals = cfg_dict.get("target_time_values")
    if vals is None:
        return None
    if isinstance(vals, range):
        return list(vals)
    if isinstance(vals, (list, tuple)):
        return list(vals)
    return None


"""GLONET_ENCODING = {"depth": {"dtype": "float32"},
                   "lat": {"dtype": "float64"},
                   "lon": {"dtype": "float64"},
                   "time": {"dtype": "str"},
                   "so": {"dtype": "float32"},
                   "thetao": {"dtype": "float32"},
                   "uo": {"dtype": "float32"},
                   "vo": {"dtype": "float32"},
                   "zos": {"dtype": "float32"},"""


class CoordinateSystem:
    """Class representing the coordinate system of a dataset."""

    def __init__(
        self,
        coord_type: str,  # "geographic", "polar", etc.
        coord_level: str,  # "grid", "point", "sparse", etc.
        coordinates: Dict[str, str],
        crs: str,
    ):
        self.coord_type = coord_type
        self.coord_level = coord_level
        self.coordinates = coordinates
        self.crs = crs

    def to_dict(self) -> dict:
        """Convert the CoordinateSystem instance to a dictionary."""
        return {
            "coord_type": self.coord_type,
            "coord_level": self.coord_level,
            "coordinates": self.coordinates,
            "crs": self.crs,
        }

    def is_polar(self) -> bool:
        """Check if the coordinate system is polar."""
        return "polar" in self.coord_type

    def is_geographic(self) -> bool:
        """Check if the coordinate system is geographic."""
        return "geographic" in self.coord_type

    def is_observation_dataset(self) -> bool:
        """Check if this coordinate system is for an ungridded observation dataset."""
        return self.coord_level != "L4"

    def toJSON(self):
        """Convert the object to a JSON string."""
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)

    @staticmethod
    def detect_data_level(data: object, names_dict: dict) -> str:
        """
        Infers the observation data level (L1, L2, L3, L4) from structure and content.

        Parameters
        ----------
        data : object
            Input data (DataFrame, GeoDataFrame, or xarray.Dataset)

        Returns
        -------
        str
            One of: "L1", "L2", "L3", "L4", or "unknown"
        """
        lat_name = names_dict.get("lat", None)
        lon_name = names_dict.get("lon", None)
        time_name = names_dict.get("time", None)
        # xarray.Dataset
        if isinstance(data, xr.Dataset):
            dims = set(data.dims)
            vars_ = set(data.variables)
            coords = set(data.coords)
            all_names = dims | vars_ | coords

            # L3/L4: Gridded data (lat/lon as dims)
            if {lat_name, lon_name} <= dims:
                if "time" in dims:
                    return "L4"
                else:
                    return "L3"

            # L2: Point observations (lat/lon/time as variables, not dims)
            if {lat_name, lon_name, time_name} <= all_names and lat_name and lon_name and time_name:
                lat = data[lat_name]
                lon = data[lon_name]
                time = data[time_name]
                # 1D arrays of same length, not dims
                if (
                    lat is not None
                    and lon is not None
                    and time is not None
                    and hasattr(lat, "ndim")
                    and hasattr(lon, "ndim")
                    and hasattr(time, "ndim")
                    and lat.ndim == lon.ndim == time.ndim == 1
                    and not {lat_name, lon_name, time_name} <= dims
                    and len(lat) == len(lon) == len(time)
                ):
                    # Optionally check for quality_flag
                    if "quality_flag" in all_names:
                        return "L2"
                    else:
                        return "L2"
            # L1: No clear lat/lon/time, or raw structure
            if not ({lat_name, lon_name} & all_names):
                return "L1"

        # DataFrame (GeoDataFrame is a pandas.DataFrame subclass)
        if isinstance(data, pd.DataFrame):
            cols = set(data.columns)
            # L2: lat/lon/time as columns
            if {lat_name, lon_name, time_name} <= cols:
                if "quality_flag" in cols:
                    return "L2"
                else:
                    return "L2"
            # L1: raw columns, no lat/lon
            if not ({lat_name, "lon"} & cols):
                return "L1"
            # L3/L4: gridded DataFrame (rare, but possible)
            if {lat_name, lon_name} <= cols and "time" not in cols:
                return "L4"
            if {lat_name, lon_name, time_name} <= cols:
                return "L3"

        # Fallback: try to detect from attributes or structure
        if hasattr(data, "attrs"):
            attrs = getattr(data, "attrs", {})
            if "level" in attrs:
                return str(attrs["level"])

        return "unknown"

    @staticmethod
    def get_coordinate_system(ds: Union[xr.Dataset, xr.DataArray]) -> "CoordinateSystem":
        """Detect and return the coordinate system from a dataset."""
        alias_map: Dict[Any, Any] = {}
        for std_name, aliases in COORD_ALIASES.items():
            for a in aliases:
                alias_map[a.lower()] = std_name

        coords_in_ds = list(ds.coords) + list(ds.dims)
        coords_lower = {str(c).lower(): c for c in coords_in_ds}
        standardized: Dict[Any, Any] = {}
        for name_lc, original in coords_lower.items():
            mapped_name = alias_map.get(name_lc)
            if mapped_name:
                standardized[mapped_name] = original

        # Search for lat/lon variables if they are not dimensions/coords
        var_names_lower = {str(v).lower(): v for v in ds.variables}
        for key in ["lat", "lon", "depth", "time"]:
            if key not in standardized:
                for alias in COORD_ALIASES[key]:
                    if alias.lower() in var_names_lower:
                        standardized[key] = var_names_lower[alias.lower()]

        # Fallback: match any still-undetected coordinate/variable by standard_name attribute
        all_coord_keys = set(COORD_ALIASES.keys())
        for var_name in ds.variables:
            try:
                std_name_attr = ds[var_name].attrs.get("standard_name", "")
            except Exception:
                std_name_attr = ""
            if not std_name_attr:
                continue
            coord_key = _match_coord_by_std_name(std_name_attr)
            if coord_key and coord_key not in standardized:
                standardized[coord_key] = var_name
                continue
            # Also catch variables that double as coordinates (lat, lon, depth, time)
            var_key = _match_var_by_std_name(std_name_attr)
            if var_key in all_coord_keys and var_key not in standardized:
                standardized[var_key] = var_name

        # dims = set(ds.dims)
        # has_depth_dim = "depth" in standardized and standardized["depth"] in dims

        has_lat = "lat" in standardized
        has_lon = "lon" in standardized
        has_x = "x" in standardized
        has_y = "y" in standardized

        # Detection of coordinate type
        if has_lat and has_lon:
            coord_type = "geographic"
        elif has_x and has_y:
            coord_type = "polar"
        else:
            coord_type = "unknown"

        # Detection of structure level
        coord_level = CoordinateSystem.detect_data_level(ds, standardized)

        crs = ds.attrs.get("crs", None)
        if crs is None:
            crs = ds.attrs.get("srid", None)
        return CoordinateSystem(
            coord_type=coord_type,
            coord_level=coord_level,
            coordinates=standardized,
            crs=crs or "",
        )

    @staticmethod
    def detect_oceanographic_variables(variables: dict) -> dict:
        """
        Detect oceanographic variables in an xarray Dataset based on standard_name and aliases.

        Parameters
        ----------
        variables : dict
            _description_

        Returns
        -------
        dict
            mapping variable type -> actual name in the dataset.

        Raises
        ------
        ValueError
            _description_
        """
        found: DefaultDict[str, Optional[str]] = defaultdict(lambda: None)
        try:
            for var_name in variables.keys():
                var = variables[var_name]
                std_name = var["std_name"].lower()

                name = var_name.lower()
                found_var = False

                for key, config in VARIABLES_ALIASES.items():
                    # Condition 1: exact standard_name match
                    condition1 = std_name and std_name in config["standard_names"]

                    # Condition 2: exact alias match on a word of the variable name
                    condition2 = any(alias.lower() == name for alias in config["aliases"])
                    if condition1 or condition2:
                        found[key] = var_name
                        found_var = True
                        break  # next var_name
                if not found_var:
                    logger.warning(f"Unknown variable alias. Ignoring variable: {var_name}.")
            return dict(found)
        except Exception as exc:
            logger.error(f"Error in variable detection: {repr(exc)}")
            raise ValueError("Failed to detect oceanographic variables.") from exc


def get_dataset_geometry(
    ds: xr.Dataset, coord_sys: CoordinateSystem, max_points: int = 50000
) -> BaseGeometry:
    """Robustly extract a geometry from a dataset, avoiding memory errors for huge point clouds.

    If the number of points is too large, subsample before computing the geometry.
    """
    coords = coord_sys.coordinates
    if coord_sys.is_polar():
        lat = ds.coords[coords.get("y")].values
        lon = ds.coords[coords.get("x")].values
    elif coord_sys.is_geographic():
        lat = ds.coords[coords.get("lat")].values
        lon = ds.coords[coords.get("lon")].values
    else:
        raise ValueError(f"Unknown coordinate system: {coord_sys.coord_type}")
    try:
        # Case 1: 1D coordinates of same size (individual points, e.g. Argo)
        if lat.ndim == 1 and lon.ndim == 1 and lat.shape == lon.shape:
            coords_arr = np.column_stack([lon, lat])
        # Case 2: 2D coordinates (regular grid)
        elif lat.ndim == 2 and lon.ndim == 2 and lat.shape == lon.shape:
            coords_arr = np.column_stack([lon.ravel(), lat.ravel()])
        elif lat.ndim == 1 and lon.ndim == 1:
            lon2d, lat2d = np.meshgrid(lon, lat)
            coords_arr = np.column_stack([lon2d.ravel(), lat2d.ravel()])
        else:
            raise ValueError("Unsupported coordinate dimensions or mismatched shapes.")

        n_points = coords_arr.shape[0]

        # Subsampling if too many points
        if n_points > max_points:
            idx = np.random.choice(n_points, size=max_points, replace=False)
            coords_arr = coords_arr[idx]

        # Cleanup: remove NaNs and duplicates
        coords_arr = coords_arr[~np.isnan(coords_arr).any(axis=1)]
        unique_points = np.unique(coords_arr, axis=0)

        # Check that unique_points is of shape (N, 2) and float type
        if not (
            isinstance(unique_points, np.ndarray)
            and unique_points.ndim == 2
            and unique_points.shape[1] == 2
        ):
            raise ValueError(
                f"unique_points malformed: shape={unique_points.shape}, type={type(unique_points)}"
            )

        # Detection grid or point cloud
        points: List[Any] = []
        for x, y in unique_points:
            try:
                pt = Point(float(x), float(y))
                if not pt.is_empty and pt.is_valid:
                    points.append(pt)

            except Exception as exc:
                logger.warning(f"Invalid point ({x}, {y}): {exc}")

        boundary = MultiPoint(points).convex_hull
        boundary = simplify(boundary, tolerance=0.1, preserve_topology=False)
        return boundary

    except Exception as exc:
        logger.error(f"Error in geometry extraction: {repr(exc)}")
        raise


def get_dataset_geometry_light(
    ds: xr.Dataset, coord_sys: CoordinateSystem
) -> Optional[BaseGeometry]:
    """Simplified version to avoid memory issues."""
    try:
        coords = coord_sys.coordinates
        lat_name = coords.get("lat", "y")
        lon_name = coords.get("lon", "x")

        # Some zarr conversions store lat/lon as plain data variables rather
        # than as xarray coordinates (e.g. Sentinel: ds.coords is empty and
        # "lat"/"lon" only appear in ds.data_vars). Check `ds.variables`
        # (coords + data_vars) instead of `ds.coords` alone so geometry is
        # still computed in that case.
        if lat_name in ds.variables and lon_name in ds.variables:
            import dask
            # Compute all four extremes in a single dask pass to halve I/O.
            # dask.compute() returns DataArrays; extract scalar via .values.
            _lat_min, _lat_max, _lon_min, _lon_max = dask.compute(
                ds[lat_name].min(),
                ds[lat_name].max(),
                ds[lon_name].min(),
                ds[lon_name].max(),
            )
            lat_min = float(_lat_min.values)
            lat_max = float(_lat_max.values)
            lon_min = float(_lon_min.values)
            lon_max = float(_lon_max.values)

            # Create a simple box
            bbox = box(lon_min, lat_min, lon_max, lat_max)
            return bbox

        return None

    except Exception as exc:
        logger.error(f"Safe geometry extraction failed: {repr(exc)}")
        from shapely.geometry import Point

        return Point(0, 0)

def detect_variables_in_dataset(ds: xr.Dataset) -> Dict[str, str]:
    """Detect all known oceanographic variables in an xarray Dataset.

    Uses three complementary strategies, applied in order of priority:

    1. Match the variable's ``standard_name`` attribute against CF standard names
       listed in :data:`VARIABLES_ALIASES`.
    2. Match the variable name against known aliases in :data:`VARIABLES_ALIASES`.
    3. Match the variable name against the generic key names themselves.

    Only the first variable found per generic key is kept.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset.

    Returns
    -------
    dict
        Mapping generic variable key -> actual variable name in the dataset.
    """
    found: Dict[str, str] = {}
    for var_name in ds.data_vars:
        try:
            std_name_attr: str = ds[var_name].attrs.get("standard_name", "")
        except Exception:
            std_name_attr = ""
        name_lc = str(var_name).lower()
        matched_key: Optional[str] = None
        # Strategy 1: CF standard_name attribute
        if std_name_attr:
            matched_key = _match_var_by_std_name(std_name_attr)
        # Strategy 2 & 3: alias / key-name match (logs at DEBUG, not WARNING)
        if matched_key is None:
            for key, config in VARIABLES_ALIASES.items():
                if name_lc in [a.lower() for a in config["aliases"]] or name_lc == key:
                    matched_key = key
                    break
        if matched_key is not None and matched_key not in found:
            found[matched_key] = str(var_name)
        elif matched_key is None:
            logger.debug(f"No match found for variable '{var_name}' in dataset.")
    return found