
from collections import defaultdict
from typing import Dict, Optional

import geopandas as gpd
import json
from loguru import logger
import numpy as np
from oceanbench.core.lagrangian_trajectory import ZoneCoordinates
from oceanbench.core.rmsd import rmsd, Variable
from shapely.geometry import Polygon, Point
from shapely import box, MultiPoint
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
    "sst": {
        "standard_names": ["sea_surface_temperature"],
        "aliases": ["sst", "surface_temperature", "temperature_surface"]
    },
    "sss": {
        "standard_names": ["sea_surface_salinity"],
        "aliases": ["sss", "surface_salinity", "salinity_surface"]
    },
    "ssh": {
        "standard_names": [
            "sea_surface_height",
            "sea_surface_height_above_geoid",
            "sea_surface_height_above_reference_ellipsoid"
        ],
        "aliases": ["ssh", "sea_level", "surface_height"]
    },
    "temperature": {
        "standard_names": ["sea_water_temperature"],
        "aliases": ["temperature", "temp"]
    },
    "salinity": {
        "standard_names": ["sea_water_salinity"],
        "aliases": ["salinity", "psu", "sal"]
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
    }
}

GEO_STD_COORDS = {"lon": "lon", "lat": "lat", "depth": "depth", "time": "time"}


EVAL_VARIABLES_GLONET = [
    Variable.HEIGHT,
    Variable.TEMPERATURE,
    Variable.SALINITY,
    Variable.NORTHWARD_VELOCITY,
    Variable.EASTWARD_VELOCITY,
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
        self, coord_type: str,
        coordinates: Dict[str, str],
        crs: str,
    ):
        self.coord_type = coord_type
        self.coordinates = coordinates
        self.crs = crs

    def is_polar(self) -> bool:
        return "polar" in self.coord_type

    def is_geographic(self) -> bool:
        return "geographic" in self.coord_type

    def toJSON(self):
        return json.dumps(
            self,
            default=lambda o: o.__dict__, 
            sort_keys=True,
            indent=4
        )

    @staticmethod
    def get_coordinate_system( ds: xr.Dataset) -> dict:
        """Detect coordinate system type and standardized spatial/temporal dimensions."""
        logger.debug("Detecting coordinate system from dataset dimensions and coordinates.")

        # Inversion du dictionnaire : nom -> standard_name
        std_keys = COORD_ALIASES.keys()   # ["lat", "lon", "x", "y", "depth", "quadrant", "time"]
        alias_map = {}
        for std_name, aliases in COORD_ALIASES.items():
            for a in aliases:
                alias_map[a.lower()] = std_name
        logger.debug(f"Alias map: {alias_map}")
        coords_in_ds = list(ds.coords) + list(ds.dims)  # on considère aussi les dimensions
        coords_lower = {c.lower(): c for c in coords_in_ds}  # mapping original
        logger.debug(f"Coordinates in dataset: {coords_lower}")
        standardized = {}
        for name_lc, original in coords_lower.items():
            std_name = alias_map.get(name_lc)
            if std_name:
                standardized[std_name] = original
        logger.debug(f"Standardized coordinates: {standardized}")
        # Détection du type de système
        #spatial_keys = {"latitude", "longitude", "x", "y"}
        has_latlon = {"lat", "lon"} <= standardized.keys()
        has_xy = {"x", "y"} <= standardized.keys()
        logger.debug(f"Has lat/lon: {has_latlon}, Has x/y: {has_xy}")
        epsg = None
        if has_latlon:
            coord_type = "geographic"
            epsg = "EPSG:4326"  # CRS standard géographique
        elif has_xy:
            coord_type = "polar"
            epsg = "EPSG:3413" # CRS arctique par défaut
        else:
            raise ValueError(
                "Unable to detect coordinate system from dimensions and coordinates."
            )

        crs = ds.attrs.get("crs", epsg)
        logger.debug(f"Detected coordinate system: {coord_type}, CRS: {crs}")
        return CoordinateSystem(
            coord_type=coord_type,
            coordinates={
                key: standardized.get(key) for key in std_keys if key in standardized
            },
            crs=crs,
        )

    @staticmethod
    def detect_oceanographic_variables(variables: dict) -> dict:
        """
        Detect oceanographic variables in an xarray Dataset based on standard_name and aliases.
        Returns a dictionary mapping variable type -> actual name in the dataset.
        """
        found = defaultdict(lambda: None)

        for var_name in variables.keys():
            var = variables[var_name]
            std_name = var["std_name"].lower()
            # long_name = var.attrs.get("long_name", "").lower()
            name = var_name.lower()
            name_parts = name.split("_")  # Ajout : split sur "_"

            for key, config in VARIABLES_ALIASES.items():
                if found[key] is not None:
                    continue  # déjà trouvé

                # Condition 1 : standard_name exact
                condition1 = std_name and std_name in config["standard_names"]

                # Condition 2 : alias exact sur un mot du nom de variable
                condition2 = any(alias.lower() in name_parts for alias in config["aliases"])

                if condition1 or condition2:
                    logger.debug(f"  Found variable: {var_name} for key: {key}")
                    found[key] = var_name

        return dict(found)


def get_dataset_geometry(ds: xr.Dataset, coord_sys: dict, max_points: int = 1000) -> Polygon:
    """
    Robustly extract a geometry from a dataset, avoiding memory errors for huge point clouds.
    If the number of points is too large, subsample before computing the geometry.
    """
    # logger.debug("Extracting geometry from dataset coordinates.")
    coords = coord_sys.coordinates
    logger.debug(f"Coordinates mapping: {coords}")
    if coord_sys.is_polar():
        lat = ds.coords[coords.get("y")].values
        lon = ds.coords[coords.get("x")].values
    elif coord_sys.is_geographic():
        lat = ds.coords[coords.get("lat")].values
        lon = ds.coords[coords.get("lon")].values
    else:
        raise ValueError(f"Unknown coordinate system: {coord_sys.coord_type}")

    logger.debug(f"Latitude shape: {lat.shape}, Longitude shape: {lon.shape}")

    try:
        # Cas 1 : coordonnées 1D de même taille (points individuels, ex: Argo)
        if lat.ndim == 1 and lon.ndim == 1 and lat.shape == lon.shape:
            logger.debug("Coordinates are 1D arrays of same length (point cloud).")
            coords_arr = np.column_stack([lon, lat])
        # Cas 2 : coordonnées 2D (grille régulière)
        elif lat.ndim == 2 and lon.ndim == 2 and lat.shape == lon.shape:
            logger.debug("Coordinates are 2D arrays with matching shapes.")
            coords_arr = np.column_stack([lon.ravel(), lat.ravel()])
        # Cas 3 : coordonnées 1D indépendantes (grille, meshgrid possible)
        elif lat.ndim == 1 and lon.ndim == 1:
            logger.debug("Coordinates are 1D arrays (meshgrid case).")
            lon2d, lat2d = np.meshgrid(lon, lat)
            coords_arr = np.column_stack([lon2d.ravel(), lat2d.ravel()])
        else:
            raise ValueError("Unsupported coordinate dimensions or mismatched shapes.")

        n_points = coords_arr.shape[0]
        logger.debug(f"Number of points for geometry: {n_points}")

        # Sous-échantillonnage si trop de points
        if n_points > max_points:
            logger.warning(f"Too many points ({n_points}), subsampling to {max_points} for geometry computation.")
            idx = np.random.choice(n_points, size=max_points, replace=False)
            coords_arr = coords_arr[idx]

        unique_points = np.unique(coords_arr, axis=0)
        logger.debug(f"Unique points extracted: {unique_points.shape[0]}")

        # Détection grille ou nuage de points
        if is_rectangular_grid(unique_points):
            logger.debug("Detected rectangular grid shape.")
            minx, miny = unique_points.min(axis=0)
            maxx, maxy = unique_points.max(axis=0)
            geometry = Polygon([
                (minx, miny),
                (minx, maxy),
                (maxx, maxy),
                (maxx, miny),
                (minx, miny),
            ])
        else:
            logger.debug("Detected complex shape, using convex hull.")
            points = [Point(x, y) for x, y in unique_points]
            multipoint = MultiPoint(points)
            geometry = multipoint.convex_hull

        if not geometry.is_valid:
            raise ValueError("Generated geometry is invalid.")
        logger.debug(f"Final geometry: {geometry.wkt}")
        return geometry

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
