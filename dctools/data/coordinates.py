
from typing import Dict, Optional

import geopandas as gpd
import json
from loguru import logger
import numpy as np
from pyproj import Transformer
from shapely.geometry import Polygon, Point
from shapely import box, MultiPoint
import xarray as xr

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


def get_coordinate_system(ds: xr.Dataset) -> dict:
    """Detect coordinate system type and standardized spatial/temporal dimensions."""

    # Dictionnaire de correspondances
    coord_aliases = {
        "lat": {"lat", "latitude", "nav_lat"},
        "lon": {"lon", "longitude", "nav_lon"},
        "x": {"x", "xc", "x_center", "easting", "projection_x_coordinate", "grid_xt", "i"},
        "y": {"y", "yc", "y_center", "northing", "projection_y_coordinate", "grid_yt", "j"},
        "depth": {"depth", "z", "lev", "level", "bottom", "deptht", "isodepth"},
        "quadrant": {"quadrant", "sector"},
        "time": {"time", "date", "datetime", "valid_time", "forecast_time", "time_counter"},
    }

    # Inversion du dictionnaire : nom -> standard_name
    alias_map = {}
    for std_name, aliases in coord_aliases.items():
        for a in aliases:
            alias_map[a.lower()] = std_name

    coords_in_ds = list(ds.coords) + list(ds.dims)  # on considère aussi les dimensions
    coords_lower = {c.lower(): c for c in coords_in_ds}  # mapping original

    standardized = {}
    for name_lc, original in coords_lower.items():
        std_name = alias_map.get(name_lc)
        if std_name:
            standardized[std_name] = original

    # Détection du type de système
    #spatial_keys = {"latitude", "longitude", "x", "y"}
    std_keys = ["lat", "lon", "x", "y", "depth", "quadrant", "time"]
    has_latlon = {"lat", "lon"} <= standardized.keys()
    has_xy = {"x", "y"} <= standardized.keys()

    epsg = None
    if has_latlon:
        coord_type = "geographic"
        epsg = "EPSG:4326"  # CRS standard géographique
    elif has_xy:
        coord_type = "polar"
        epsg = "EPSG:3413" # CRS arctique par défaut
    else:
        raise ValueError("Unable to detect coordinate system from dimensions and coordinates.")

    crs = ds.attrs.get("crs", epsg)

    return CoordinateSystem(
        coord_type=coord_type,
        coordinates={
            key: standardized.get(key) for key in std_keys if key in standardized
        },
        crs=crs,
    )

def get_dataset_geometry(
    ds: xr.Dataset, coord_sys: dict
) -> gpd.GeoSeries:
    """
    Extract a GeoSeries with the spatial geometry covering all data points
    from a dataset, using either convex hull or bounding box depending on shape.

    Args:
        ds: xarray.Dataset
        coord_map: dict mapping standard keys to actual dataset coordinate names.
                Example: {"system": "geographic", "latitude": "lat", "longitude": "lon"}

    Returns:
        gpd.GeoSeries with a single shapely geometry (Polygon)
    """
    coords = coord_sys.coordinates
    if coord_sys.is_polar():
        lat = ds.coords[coords.get("y")].values
        lon = ds.coords[coords.get("x")].values
    elif coord_sys.is_geographic():
        lat = ds.coords[coords.get("lat")].values
        lon = ds.coords[coords.get("lon")].values
    else:
        raise ValueError(
            f"Unknown coordinate system: {coord_sys.coord_type}"
        )

    try:
        # Handle both 1D and 2D coordinates
        if lat.ndim == 1 and lon.ndim == 1:
            lon2d, lat2d = np.meshgrid(lon, lat)
        elif lat.ndim == 2 and lon.ndim == 2 and lat.shape == lon.shape:
            lat2d = lat
            lon2d = lon
        else:
            raise ValueError("Unsupported coordinate dimensions or mismatched shapes.")
        coords = np.column_stack([lon2d.ravel(), lat2d.ravel()])
        unique_points = np.unique(coords, axis=0)
        # Check if the shape is rectangular (gridded) or complex (e.g. track)
        if is_rectangular_grid(unique_points):
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
            points = [Point(x, y) for x, y in unique_points]
            multipoint = MultiPoint(points)
            geometry = multipoint.convex_hull

        # Vérifier que la géométrie est valide
        if not geometry.is_valid:
            raise ValueError("Generated geometry is invalid.")
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
