
from collections import defaultdict
from typing import Dict, Optional

import geopandas as gpd
import json
from loguru import logger
import numpy as np
from oceanbench.core.lagrangian_trajectory import ZoneCoordinates
from oceanbench.core.rmsd import rmsd, Variable
import pandas as pd
from shapely.geometry import Polygon, Point
from shapely import box, MultiPoint, simplify
import xarray as xr


# Dictionnaire de correspondances (coordonnées)
COORD_ALIASES = {
    "lat": {"lat", "latitude", "nav_lat"},
    "lon": {"lon", "longitude", "nav_lon"},
    "x": {"x", "xc", "x_center", "easting", "projection_x_coordinate", "grid_xt", "i"},
    "y": {"y", "yc", "y_center", "northing", "projection_y_coordinate", "grid_yt", "j"},
    "depth": {"depth", "z", "lev", "level", "bottom", "deptht", "isodepth"},
    "quadrant": {"quadrant", "sector"},
    "time": {"time", "date", "datetime", "valid_time", "forecast_time", "time_counter"},
}

# Dictionnaire des variables d'intérêt : {nom générique -> standard_name(s), alias courants}
VARIABLES_ALIASES = {
    "sla": {
        "standard_names": ["sea_surface_height_above_sea_level"],
        "aliases": ["sla", "data_01__ku__ssha", "ssha"]
    },
    "sst": {
        "standard_names": ["sea_surface_temperature"],
        "aliases": ["sst", "surface_temperature", "temperature_surface"]
    },
    "sss": {
        "standard_names": ["sea_surface_salinity"],
        "aliases": ["sss", "surface_salinity", "salinity_surface", "SSS", "SST_sal"]
    },
    "ssh": {
        "standard_names": [
            "sea_surface_height",
            "sea_surface_height_above_geoid",
            "sea_surface_height_above_reference_ellipsoid"
        ],
        "aliases": ["ssh", "sea_level", "surface_height", "ssha_filtered", "zos"]
    },
    "temperature": {
        "standard_names": ["sea_water_potential_temperature", "sea_water_temperature"],
        "aliases": ["temperature", "temp", "thetao"]
    },
    "salinity": {
        "standard_names": ["sea_water_salinity"],
        "aliases": ["salinity", "psu", "sal", "PSAL", "S", "salt", "so"]
    },
    "u_current": {
        "standard_names": ["eastward_sea_water_velocity"],
        "aliases": ["u", "uo", "u_velocity"]
    },
    "v_current": {
        "standard_names": ["northward_sea_water_velocity"],
        "aliases": ["v", "vo", "v_velocity"]
    },
    "w_current": {
        "standard_names": ["upward_sea_water_velocity"],
        "aliases": ["w", "wo", "w_velocity"]
    },
    "mld": {
        "standard_names": ["mixed_layer_depth"],
        "aliases": ["mld", "mix_layer_depth"]
    },
    "mdt": {
        "standard_names": ["mean_dynamic_topography_cnes_cls", "mean_dynamic_topography"],
        "aliases": ["mdt", "mean_dynamic_topography", "mean_dynamic_topography_cnes_cls"]
    }
}

GEO_STD_COORDS = {"lon": "lon", "lat": "lat", "depth": "depth", "time": "time"}


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

GLONET_DEPTH_VALS = [0.494025, 47.37369, 92.32607, 155.8507, 222.4752, 318.1274, 380.213, 
        453.9377, 541.0889, 643.5668, 763.3331, 902.3393, 1245.291, 1684.284, 
        2225.078, 3220.82, 3597.032, 3992.484, 4405.224, 4833.291, 5274.784]

GLONET_TIME_VALS = range(0, 10)

RANGES_GLONET = {
    "lat": np.arange(-78, 90, 0.25),
    "lon": np.arange(-180, 180, 0.25),
    "depth": GLONET_DEPTH_VALS,
    #"time": GLONET_TIME_VALS,
}

'''GLONET_ENCODING = {"depth": {"dtype": "float32"},
                   "lat": {"dtype": "float64"},
                   "lon": {"dtype": "float64"},
                   "time": {"dtype": "str"},
                   "so": {"dtype": "float32"},
                   "thetao": {"dtype": "float32"},
                   "uo": {"dtype": "float32"},
                   "vo": {"dtype": "float32"},
                   "zos": {"dtype": "float32"},'''


class CoordinateSystem:
    def __init__(
        self,
        coord_type: str,      # "geographic", "polar", etc.
        coord_level: str,     # "grid", "point", "sparse", etc.
        coordinates: Dict[str, str],
        crs: str,
        # is_observation: bool
    ):
        self.coord_type = coord_type
        self.coord_level = coord_level
        self.coordinates = coordinates
        self.crs = crs
        # self.is_observation = is_observation

    def to_dict(self) -> dict:
        """
        Convertit l'instance CoordinateSystem en dictionnaire.
        """
        return {
            "coord_type": self.coord_type,
            "coord_level": self.coord_level,
            "coordinates": self.coordinates,
            "crs": self.crs,
            # "is_observation": self.is_observation,
        }

    def is_polar(self) -> bool:
        return "polar" in self.coord_type

    def is_geographic(self) -> bool:
        return "geographic" in self.coord_type

    def is_observation_dataset(self) -> bool:
        """Check if this coordinate system is for an ungridded observation dataset."""
        return self.coord_level != "L4"

    def toJSON(self):
        return json.dumps(
            self,
            default=lambda o: o.__dict__, 
            sort_keys=True,
            indent=4
        )


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
        lat_name = names_dict["lat"]
        lon_name = names_dict["lon"]
        time_name = names_dict["time"]
        # --- xarray.Dataset ---
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
            if {lat_name, lon_name, time_name} <= all_names:
                lat = data[lat_name]
                lon = data[lon_name]
                time = data[time_name]
                # 1D arrays of same length, not dims
                if (
                    lat.ndim == lon.ndim == time.ndim == 1
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

        # --- DataFrame / GeoDataFrame ---
        if isinstance(data, (pd.DataFrame, gpd.GeoDataFrame)):
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

        # --- Fallback: try to detect from attributes or structure ---
        # L1: If no spatial/temporal info at all
        if hasattr(data, "attrs"):
            attrs = getattr(data, "attrs", {})
            if "level" in attrs:
                return attrs["level"]

        return "unknown"


    @staticmethod
    def get_coordinate_system(ds: xr.Dataset) -> "CoordinateSystem":
        std_keys = COORD_ALIASES.keys()
        alias_map = {}
        for std_name, aliases in COORD_ALIASES.items():
            for a in aliases:
                alias_map[a.lower()] = std_name

        coords_in_ds = list(ds.coords) + list(ds.dims)
        coords_lower = {c.lower(): c for c in coords_in_ds}
        standardized = {}
        for name_lc, original in coords_lower.items():
            std_name = alias_map.get(name_lc)
            if std_name:
                standardized[std_name] = original

        # Recherche des variables lat/lon si elles ne sont pas des dimensions/coords
        var_names_lower = {v.lower(): v for v in ds.variables}
        for key in ["lat", "lon", "depth", "time"]:
            if key not in standardized:
                for alias in COORD_ALIASES[key]:
                    if alias.lower() in var_names_lower:
                        standardized[key] = var_names_lower[alias.lower()]

        dims = set(ds.dims)
        has_lat_dim = "lat" in standardized and standardized["lat"] in dims
        has_lon_dim = "lon" in standardized and standardized["lon"] in dims
        has_time_dim = "time" in standardized and standardized["time"] in dims
        has_depth_dim = "depth" in standardized and standardized["depth"] in dims

        has_lat = "lat" in standardized
        has_lon = "lon" in standardized
        has_x = "x" in standardized
        has_y = "y" in standardized

        # Détection du type de coordonnées
        if has_lat and has_lon:
            coord_type = "geographic"
        elif has_x and has_y:
            coord_type = "polar"
        else:
            coord_type = "unknown"

        # Détection du niveau de structuration
        coord_level = CoordinateSystem.detect_data_level(ds, standardized)

        # N'inclure depth dans coordinates que si pertinent
        coords_out = {}
        for key in std_keys:
            if key in standardized:
                if key == "depth" and not (has_depth_dim or (coord_level in ["grid_3d", "point_3d"])):
                    continue
                coords_out[key] = standardized[key]

        crs = ds.attrs.get("crs", None)
        return CoordinateSystem(
            coord_type=coord_type,
            coord_level=coord_level,
            coordinates=coords_out,
            crs=crs,
            # is_observation=is_observation,
        )

    @staticmethod
    def detect_oceanographic_variables(variables: dict) -> dict:
        """
        Detect oceanographic variables in an xarray Dataset based on standard_name and aliases.
        Returns a dictionary mapping variable type -> actual name in the dataset.
        """
        try:
            found = defaultdict(lambda: None)

            for var_name in variables.keys():
                var = variables[var_name]
                std_name = var["std_name"].lower()

                name = var_name.lower()

                for key, config in VARIABLES_ALIASES.items():
                    # Condition 1 : standard_name exact
                    condition1 = std_name and std_name in config["standard_names"]

                    # Condition 2 : alias exact sur un mot du nom de variable
                    condition2 = any(alias.lower() == name for alias in config["aliases"])
                    if condition1 or condition2:
                        logger.debug(f"         Found variable: {var_name} for key: {key}\n\n\n")
                        found[key] = var_name
            return dict(found)
        except Exception as exc:
            logger.error(f"Error in variable detection: {repr(exc)}")
            raise ValueError("Failed to detect oceanographic variables.") from exc


def get_dataset_geometry(ds: xr.Dataset, coord_sys: dict, max_points: int = 50000) -> Polygon:
    """
    Robustly extract a geometry from a dataset, avoiding memory errors for huge point clouds.
    If the number of points is too large, subsample before computing the geometry.
    """
    # logger.debug("Extracting geometry from dataset coordinates.")
    coords = coord_sys.coordinates
    #logger.debug(f"Coordinates mapping: {coords}")
    if coord_sys.is_polar():
        lat = ds.coords[coords.get("y")].values
        lon = ds.coords[coords.get("x")].values
    elif coord_sys.is_geographic():
        lat = ds.coords[coords.get("lat")].values
        lon = ds.coords[coords.get("lon")].values
    else:
        raise ValueError(f"Unknown coordinate system: {coord_sys.coord_type}")


    try:
        # Cas 1 : coordonnées 1D de même taille (points individuels, ex: Argo)
        if lat.ndim == 1 and lon.ndim == 1 and lat.shape == lon.shape:
            coords_arr = np.column_stack([lon, lat])
        # Cas 2 : coordonnées 2D (grille régulière)
        elif lat.ndim == 2 and lon.ndim == 2 and lat.shape == lon.shape:
            coords_arr = np.column_stack([lon.ravel(), lat.ravel()])
        elif lat.ndim == 1 and lon.ndim == 1:
            lon2d, lat2d = np.meshgrid(lon, lat)
            coords_arr = np.column_stack([lon2d.ravel(), lat2d.ravel()])
        else:
            raise ValueError("Unsupported coordinate dimensions or mismatched shapes.")

        n_points = coords_arr.shape[0]

        # Sous-échantillonnage si trop de points
        if n_points > max_points:
            #logger.warning(f"Too many points ({n_points}), subsampling to {max_points} for geometry computation.")
            idx = np.random.choice(n_points, size=max_points, replace=False)
            coords_arr = coords_arr[idx]

        # Nettoyage : retirer les NaN et doublons
        coords_arr = coords_arr[~np.isnan(coords_arr).any(axis=1)]
        unique_points = np.unique(coords_arr, axis=0)

        # Vérifie que unique_points est bien de shape (N, 2) et de type float
        if not (isinstance(unique_points, np.ndarray) and unique_points.ndim == 2 and unique_points.shape[1] == 2):
            raise ValueError(f"unique_points mal formé: shape={unique_points.shape}, type={type(unique_points)}")

        # Détection grille ou nuage de points
        points = []
        for x, y in unique_points:
            try:
                pt = Point(float(x), float(y))
                if not pt.is_empty and pt.is_valid:
                    points.append(pt)

            except Exception as exc:
                logger.warning(f"Invalid point ({x}, {y}): {exc}")

        boundary = MultiPoint(points).convex_hull
        boundary = simplify(boundary, tolerance=0.1, preserve_topology=False)
        return boundary  #MultiPoint(points)

    except Exception as exc:
        logger.error(f"Error in geometry extraction: {repr(exc)}")
        raise

def is_rectangular_grid(points: np.ndarray, tol: float = 1e-5) -> bool:
    """
    Heuristic to determine if points form a rectangular grid (i.e., aligned lat/lon).
    """
    if len(points) < 4:
        return False
    minx, miny = points.min(axis=0)
    maxx, maxy = points.max(axis=0)
    corners = np.array([[minx, miny], [minx, maxy], [maxx, miny], [maxx, maxy]])
    match = sum(np.any(np.all(np.abs(points - c) < tol, axis=1)) for c in corners)
    return match == 4


import xarray as xr

def is_ungridded_observation_dataset(ds: xr.Dataset) -> bool:
    """
    Heuristically detect if a xarray.Dataset contains ungridded (point/profile) observations.

    Criteria:
    - Presence of 'lat' and 'lon' as 1D variables or coordinates.
    - Absence of 2D meshgrid for lat/lon.
    - Number of points much less than a typical grid (e.g. < 10_000).
    - Optionally, presence of 'profile', 'station', or 'obs' dimension.
    - Optionally, attribute 'is_observation' set to True.

    Returns
    -------
    bool
        True if likely ungridded observations, False otherwise.
    """

    # 2. Check for 1D lat/lon coordinates
    lat = ds.coords.get("lat", None)
    lon = ds.coords.get("lon", None)
    if lat is not None and lon is not None:
        # If both are 1D and have the same length, likely point data
        if lat.ndim == 1 and lon.ndim == 1 and lat.shape == lon.shape:
            # If number of points is small, likely obs
            if lat.size < 10000:
                return True

    # 3. Check for typical obs dimensions
    for dim in ds.dims:
        if dim.lower() in ("profile", "station", "obs", "trajectory"):
            return True

    # 4. Check for 2D lat/lon (meshgrid) => gridded, so return False
    if lat is not None and lon is not None:
        if lat.ndim == 2 or lon.ndim == 2:
            return False

    # 5. Fallback: if only 1 spatial dimension, likely obs
    spatial_dims = [d for d in ds.dims if d.lower() in ("lat", "lon", "latitude", "longitude")]
    if len(spatial_dims) == 1:
        return True

    return False