"""Canonical names of ocean variables and coordinates, and the aliases met in real products.

Moved from ``dctools.data.coordinates`` (which re-exports these names). Pure data + three functions,
no dependency beyond the standard library, so that any reader can map ``ssha`` / ``adt`` / ``zos``
to ``ssh`` the same way DC2 does.
"""
from __future__ import annotations

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


COORD_ALIASES = {
    "lat": {"lat", "latitude", "nav_lat"},
    "lon": {"lon", "longitude", "nav_lon"},
    "x": {"x", "xc", "x_center", "easting", "projection_x_coordinate", "grid_xt", "i"},
    "y": {"y", "yc", "y_center", "northing", "projection_y_coordinate", "grid_yt", "j"},
    "depth": {
        "depth",
        "lev",
        "level",
        "bottom",  # "z" # z also used by `siconc`
        "deptht",
        "isodepth",
        # NOTE: for argo data pressure = depth (pres_adjusted)
    },
    "quadrant": {"quadrant", "sector"},
    "time": {"time", "date", "datetime", "valid_time", "forecast_time", "time_counter"},
    "n_points": {"n_points", "N_POINTS", "points", "obs"},
}

# CF standard names for coordinates (used for standard_name attribute lookup)
COORD_STD_NAMES: Dict[str, set] = {
    "lat": {"latitude"},
    "lon": {"longitude"},
    "depth": {"depth", "sea_floor_depth_below_sea_surface"},
    "time": {"time"},
    "x": {"projection_x_coordinate"},
    "y": {"projection_y_coordinate"},
    "n_points": set(),
    "quadrant": set(),
}

# Variable of interest dictionary: {generic name -> standard_name(s), common aliases}
VARIABLES_ALIASES = {
    # "sla": {   # TODO : check about computing sla from ssh
    #    "standard_names": ["sea_surface_height_above_sea_level"],
    #    "aliases": ["sla", "data_01__ku__ssha", "ssha"]
    # },
    # "sst": {
    #    "standard_names": ["sea_surface_temperature"],
    #    "aliases": ["sst", "surface_temperature", "temperature_surface"]
    # },
    # "sst_foundation": {
    #    "standard_names": ["sea_surface_foundation_temperature"],
    #    "aliases": [
    #        "sst_foundation", "sst_fnd",
    #        "sstfoundation", "sstfnd", "sst_ref",
    #        "foundation_temperature", "t_surf_foundation",
    #    ]
    # },
    # TODO : check mixing sst and sst_fnd
    "sst": {
        "standard_names": [
            "sea_surface_temperature",
            "sea_surface_foundation_temperature",
            "sea_surface_skin_temperature",
        ],
        "aliases": [
            "sst",
            "surface_temperature",
            "temperature_surface",
            "sst_foundation",
            "sst_fnd",
            "sstfoundation",
            "sstfnd",
            "sst_ref",
            "foundation_temperature",
            "t_surf_foundation",
            "adjusted_sea_surface_temperature",  # TODO : check this one
            "analysed_sst",
            "analysed_sea_surface_temperature",
        ],
    },
    "sss": {
        "standard_names": ["sea_surface_salinity", "sea_water_surface_salinity"],
        "aliases": [
            "sss",
            "surface_salinity",
            "salinity_surface",
            "SSS",
            "SST_sal",
            "Sea_Surface_Salinity_Rain_Corrected",  # TODO : check mixing validity
        ],
    },
    "ssh": {
        "standard_names": [
            "sea_surface_height",
            "sea_surface_height_above_geoid",
            "sea_surface_height_above_reference_ellipsoid",
            "sea_surface_height_above_mean_sea_level",
        ],
        "aliases": [
            "ssh",
            "sea_level",
            "surface_height",
            "ssha_filtered",
            "zos",
            "data_01__ku__ssha",
            "ssha",
            "adt",
            "sea_level_anomaly",
            "sla",
        ],
    },
    "temperature": {
        "standard_names": [
            "sea_water_potential_temperature",
            "sea_water_temperature",
            "sea_water_conservative_temperature",
        ],
        "aliases": ["temperature", "temp", "thetao", "temp_adjusted", "to", "theta"],
    },
    "salinity": {
        "standard_names": [
            "sea_water_salinity",
            "sea_water_practical_salinity",
            "sea_water_absolute_salinity",
        ],
        "aliases": ["salinity", "psu", "sal", "psal", "s", "salt", "so", "psal_adjusted"],
    },
    "u_current": {
        "standard_names": [
            "eastward_sea_water_velocity",
            "surface_geostrophic_eastward_sea_water_velocity",
        ],
        "aliases": ["u", "uo", "u_velocity", "ugos", "ugo", "u_curr", "ucur"],
    },
    "v_current": {
        "standard_names": [
            "northward_sea_water_velocity",
            "surface_geostrophic_northward_sea_water_velocity",
        ],
        "aliases": ["v", "vo", "v_velocity", "vgos", "vgo", "v_curr", "vcur"],
    },
    "w_current": {
        "standard_names": ["upward_sea_water_velocity"],
        "aliases": ["w", "wo", "w_velocity", "upward_velocity"],
    },
    "mld": {
        "standard_names": [
            "mixed_layer_depth",
            "ocean_mixed_layer_thickness_defined_by_sigma_theta",
        ],
        "aliases": ["mld", "mix_layer_depth", "mlotst", "zmld", "mld_003"],
    },
    "mean_dynamic_topography": {
        "standard_names": ["mean_dynamic_topography_cnes_cls", "mean_dynamic_topography"],
        "aliases": [
            "mdt",
            "mean_topography",
            "mean_dynamic_topography",
            "mean_dynamic_topography_cnes_cls",
            "data_01__mean_dynamic_topography",
            "mean_dyn_topo",
        ],  # TODO : check "mean_topography"
    },
    "mean_sea_surface": {
        "standard_names": ["mean_sea_surface_height"],
        "aliases": [
            "mean_sea_surface",
            "mss",
            "mean_sea_surface_height",
            "mss_cnes_clsXX",
            "data_01__mean_sea_surface_cnescls",
        ],
    },
    "quality_level": {"standard_names": ["quality_level"], "aliases": ["quality_level"]},
    # ARGO specific pressure
    "pressure": {
        "standard_names": ["sea_water_pressure"],
        "aliases": ["pres", "pres_adjusted", "pressure"],
    },
    # dimensions/coordinates are variables in some datasets
    "time": {
        "standard_names": ["time"],
        "aliases": [
            "time",
            "date",
            "datetime",
            "valid_time",
            "forecast_time",
            "time_counter",
            "data_01__time_tai",
            "profile_date",
        ],
    },
    "lat": {"standard_names": ["latitude"], "aliases": ["lat", "latitude", "nav_lat"]},
    "lon": {"standard_names": ["longitude"], "aliases": ["lon", "longitude", "nav_lon"]},
    "depth": {
        "standard_names": ["depth"],
        "aliases": [
            "depth",
            "lev",
            "level",
            "bottom",
            "deptht",
            "isodepth",
            "data_01__depth_or_elevation",
            "data_01__altitude",
        ],
        # NOTE: Removed "z" from aliases as it is already used by `siconc`
    },
    # Polar coordinates (EPSG:3413)
    "x": {
        "standard_names": ["x"],
        "aliases": ["x", "xc", "x_center", "easting", "projection_x_coordinate", "grid_xt", "i"],
    },
    "y": {
        "standard_names": ["y"],
        "aliases": ["y", "yc", "y_center", "northing", "projection_y_coordinate", "grid_yt", "j"],
    },
    "leadmap": {  # MODIS ArcLeads
        "standard_names": ["leadmap"],
        "aliases": ["leadmap"],
    },
    # IABP variables
    "barometric_pressure": {  # TODO: check for conflicts with sea_water_pressure
        "standard_names": ["barometric_pressure"],
        "aliases": ["bp", "BP"],
    },
    "barometric_pressure_tendency": {
        "standard_names": ["barometric_pressure_tendency"],
        "aliases": ["bpt", "BPT"],
    },
    "air_temperature": {"standard_names": ["air_temperature"], "aliases": ["Ta"]},
    "hull_temperature": {"standard_names": ["hull_temperature"], "aliases": ["Th"]},
    "surface_temperature": {"standard_names": ["surface_temperature"], "aliases": ["Ts"]},
    # DC3 (AMSR2) variables
    "siconc": {
        "standard_names": ["sea_ice_area_fraction"],
        "aliases": [
            "siconc",
            "sea_icea_area_fraction",
            "z",  # "z" used for siconc on AMSR2
            "ice_conc",
            "ice_concentration",
            "aice",
            "sic",
            "ice_area_fraction",
            "ci",
        ],
    },
    "n_points": {
        "standard_names": ["n_points"],
        "aliases": ["n_points", "N_POINTS", "points", "obs"],
    },
}

GEO_STD_COORDS = {"lon": "lon", "lat": "lat", "depth": "depth", "time": "time"}


def _match_var_by_std_name(std_name: str) -> Optional[str]:
    """Return the generic variable key whose ``standard_names`` list contains *std_name*, or None.

    The comparison is case-insensitive.
    """
    if not std_name:
        return None
    std_name_lc = std_name.lower()
    for key, config in VARIABLES_ALIASES.items():
        if std_name_lc in [s.lower() for s in config["standard_names"]]:
            return key
    return None


def _match_coord_by_std_name(std_name: str) -> Optional[str]:
    """Return the generic coordinate key whose ``COORD_STD_NAMES`` set contains *std_name*, or None.

    The comparison is case-insensitive.
    """
    if not std_name:
        return None
    std_name_lc = std_name.lower()
    for key, std_names in COORD_STD_NAMES.items():
        if std_name_lc in {s.lower() for s in std_names}:
            return key
    return None


def get_standardized_var_name(name: str, standard_name: Optional[str] = None) -> Optional[str]:
    """
    Return the standardized variable key from a CF standard_name attribute or a known alias.

    Strategy (in order):
    1. Match against CF ``standard_name`` attribute (if provided).
    2. Match variable name against known aliases.
    3. Match variable name against generic keys.

    Args:
        name (str): Original variable name.
        standard_name (str, optional): CF ``standard_name`` attribute of the variable.

    Returns:
        str: Standardized key if found, else None.
    """
    # Many observation products include bookkeeping/index variables that are
    # not meant to be standardized or evaluated.
    _ignorable = {
        "i_num_pixel",
        "i_num_line",
        "num_pixels",
        "num_lines",
        "num_nadir",
        "id",
        "platform_id",
        "platformid",
    }
    if name.lower() in _ignorable:
        logger.debug(f"Ignoring non-science variable: {name}.")
        return None
    # Priority 1: CF standard_name attribute
    if standard_name:
        key = _match_var_by_std_name(standard_name)
        if key is not None:
            return key
    # Priority 2: alias / generic-key name match
    for key, config in VARIABLES_ALIASES.items():
        list_aliases = config["aliases"]
        if name.lower() in list_aliases:
            return key
        if name.lower() == key:
            return key
    logger.warning(f"Unknown variable alias. Ignoring variable: {name}.")
    return None
