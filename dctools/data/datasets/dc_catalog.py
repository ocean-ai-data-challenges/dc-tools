"""Data catalog management for DC-tools."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime
import traceback
from typing import (
    Any, Callable, Dict, List,
    Optional, Union, Sequence
)

import geopandas as gpd
import json
from loguru import logger
import pandas as pd
from shapely import geometry
from shapely.geometry.base import BaseGeometry
from shapely.geometry import mapping, shape

from dctools.data.coordinates import CoordinateSystem
from dctools.utilities.misc_utils import (
    make_timestamps_serializable,
    serialize_structure,
)

GLOBAL_METADATA = [
    "coord_type", "crs", "dimensions", "keep_variables",
     "resolution", "variables", "variables_rename_dict",
]

ALL_METADATA = [
    "coord_type", "crs", "date_end", "date_start",
    "dimensions", "geometry", "keep_variables", "path",
    "resolution", "variables",
]

@dataclass
class CatalogEntry:
    """Represents a single entry in a dataset catalog with metadata."""

    path: str
    date_start: pd.Timestamp
    date_end: pd.Timestamp
    # variables: Dict[str, List[str]]
    geometry: gpd.GeoSeries

    def to_dict(self):
        """Convert catalog entry to dictionary."""
        try:
            dct = asdict(self)
            dct["date_start"] = self.date_start.isoformat()
            dct["date_end"] = self.date_end.isoformat()
            if self.geometry is not None:
                dct["geometry"] = mapping(self.geometry)
            else:
                dct["geometry"] = None
            return dct
        except Exception as exc:
            logger.error(f"Error converting CatalogEntry to dict: {exc}")
            traceback.print_exc()
            raise

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Create catalog entry from dictionary."""
        data_copy = data.copy()
        data_copy["date_start"] = pd.to_datetime(data_copy["date_start"])
        data_copy["date_end"] = pd.to_datetime(data_copy["date_end"])
        if data_copy["geometry"] is not None:
            data_copy["geometry"] = shape(data_copy["geometry"])
        return cls(**data_copy)


class DatasetCatalog:
    """Structured catalog to hold and filter dataset metadata entries."""

    def __init__(
        self,
        alias: str,
        global_metadata: Optional[Dict[str, Any]] = None,
        entries: Optional[Sequence[Union[CatalogEntry, Dict[str, Any]]]] = None,
        dataframe: Optional[gpd.GeoDataFrame] = None,
    ):
        """
        Initialize the catalog with a list of entries.

        Args:
            entries (List[Union[CatalogEntry, Dict[str, Any]]]): List of dataset metadata.
        """
        self.alias = alias
        self._global_metadata = global_metadata
        if dataframe is None:
            dataframe = gpd.GeoDataFrame()
        if entries is None and dataframe.empty:
            logger.warning("No entries or dataframe provided. Initializing empty catalog.")
            self.entries: List[CatalogEntry] = []
            self.gdf = gpd.GeoDataFrame()
            return

        if entries:
            self.entries = []
            for entry in entries:
                if isinstance(entry, CatalogEntry):
                    self.entries.append(entry)
                elif isinstance(entry, dict):
                    self.entries.append(CatalogEntry(**entry))
                else:
                    logger.warning(f"Ignoring invalid entry: {entry}")

            # Convert entries to GeoDataFrame
            data_dicts: List[Any] = []
            for entry in self.entries:
                entry_dict = asdict(entry)
                # Convert geometry dict back to Shapely object if needed
                geometry_data = entry_dict.get('geometry')
                if isinstance(geometry_data, dict):
                    entry_dict['geometry'] = shape(geometry_data)
                elif geometry_data is None:
                    entry_dict['geometry'] = None
                # If it's already a Shapely object, keep it as is
                data_dicts.append(entry_dict)

            self.gdf = gpd.GeoDataFrame(
                data_dicts,
                geometry="geometry",
                crs="EPSG:4326",
            )
        if not dataframe.empty:
            self.gdf = dataframe

        if not self.gdf.empty:
            self.gdf = self._clean_dataframe(self.gdf)


    def to_json(self, path: Optional[Optional[str]] = None) -> str:
        """
        Export the entire DatasetCatalog content to JSON format.

        Args:
            path (Optional[str]): Path to save the JSON file.

        Returns:
            str: Complete JSON representation of the instance.
        """
        def coord_system_to_dict(coord_system):
            if isinstance(coord_system, dict):
                return coord_system
            if hasattr(coord_system, "__dict__"):
                return {k: v for k, v in coord_system.__dict__.items() if not k.startswith("_")}
            return str(coord_system)
        try:
            # Copy and conversion coord_system BEFORE serialize_structure
            meta_copy = self._global_metadata.copy() if self._global_metadata else {}
            if "coord_system" in meta_copy:
                meta_copy["coord_system"] = coord_system_to_dict(meta_copy["coord_system"])
            serial_metadata = serialize_structure(meta_copy)
            gdf_serializable = make_timestamps_serializable(self.gdf)
            geojson_dict = json.loads(gdf_serializable.to_json())
            export_dict = {
                "global_metadata": serial_metadata,
                "features": geojson_dict.get("features", [])
            }
            json_str = json.dumps(export_dict, indent=2)
            if path:
                with open(path, "w") as f:
                    f.write(json_str)
            return json_str
        except Exception as exc:
            logger.error(f"Error while exporting JSON file: {repr(exc)}")
            raise

    @classmethod
    def from_json(
        cls, path: str, alias:str, limit: int, ignore_geometry = False
    ) -> 'DatasetCatalog':
        """Reconstruct a DatasetCatalog instance from a GeoJSON file."""
        def process_feature(feat):
            try:
                props = feat.get("properties", {})
                if not ignore_geometry:
                    if feat["geometry"] is not None:
                        geom = shape(feat["geometry"])
                    else:
                        geom = None

                    if geom is not None:
                        geom_obj = shape(geom)
                        if not isinstance(geom_obj, BaseGeometry):
                            logger.warning(f"Invalid geometry: {geom}")
                            return None
                        props["geometry"] = geom_obj
                    else:
                        logger.warning(f"Feature without geometry : {props['path']}, skipping.")
                return props
            except Exception as exc:
                logger.warning(f"Could not parse feature: {feat} ({exc})")
                traceback.print_exc()
                return None

        try:
            with open(path, "r") as f:
                data = json.load(f)
            features = data["features"]
            global_metadata = data.get("global_metadata", {})

            coord_system = global_metadata.get("coord_system")
            if isinstance(coord_system, str):
                try:
                    coord_system = json.loads(coord_system)
                except Exception as e:
                    # If not JSON, ignore or raise explicit error
                    raise ValueError(
                        "coord_system in global_metadata is a string but not valid JSON"
                    ) from e

            coord_sys = CoordinateSystem(
                coord_type=coord_system["coord_type"],
                coord_level=coord_system["coord_level"],
                coordinates=coord_system["coordinates"],
                crs=coord_system["crs"],
            )
            global_metadata["coord_system"] = coord_sys

            # Extract properties and geometry
            records: List[Any] = []

            with ThreadPoolExecutor(max_workers=16) as executor:
                records = list(executor.map(process_feature, features))

            # Create the GeoDataFrame
            if not ignore_geometry:
                gdf = gpd.GeoDataFrame(records, geometry="geometry")
            else:
                gdf = gpd.GeoDataFrame(records)
            #if "keep_variables" in gdf.columns:
            #    gdf["keep_variables"] = gdf["keep_variables"].apply(
            #        lambda x: json.loads(x) if isinstance(x, str) and x.startswith("[") else x
            #    )
            # Create instance with global_metadata

            instance = cls(alias=alias, global_metadata=global_metadata, dataframe=gdf)

            return instance
        except Exception:
            logger.error(f"Error while loading from JSON file: {traceback.format_exc()}")
            traceback.print_exc()
            raise


    def filter_attrs(
        self, filters: dict[str, Union[Callable[[Any], bool], gpd.GeoSeries]]
    ) -> None:
        """Filter catalog entries by attribute values using filter functions."""
        gdf = self.gdf.copy()
        for key, func in filters.items():
            if key not in gdf.columns:
                raise KeyError(f"{key} not found in catalog columns")
            elif key == "geometry":
                # Special case for geometry filtering
                roi = func(gdf[key])
                if not isinstance(roi, gpd.GeoSeries):
                    raise ValueError(
                        f"Expected a gpd.GeoSeries for {key}, got {type(roi)}"
                    )
                gdf = gdf[roi]

            if key not in gdf.columns:
                raise KeyError(f"{key} not found in catalog columns")
            elif key == "variables":
                # Special case for variables filtering
                gdf = gdf[func(key)]
            else:
                gdf = gdf[func(gdf[key])]

        self.gdf = gdf

    def get_dataframe(self) -> gpd.GeoDataFrame:
        """
        Return the internal GeoDataFrame.

        Returns:
            gpd.GeoDataFrame: The catalog GeoDataFrame.
        """
        return self.gdf

    def set_dataframe(self, gdf: gpd.GeoDataFrame) -> None:
        """
        Set the internal GeoDataFrame.

        Args:
            gdf (gpd.GeoDataFrame): The catalog GeoDataFrame.
        """
        self.gdf = self._clean_dataframe(gdf)

    def _clean_dataframe(self, gdf: pd.DataFrame) -> gpd.GeoDataFrame:
        """
        Clean and structure the DataFrame to ensure consistent metadata.

        Args:
            gdf (pd.DataFrame): Raw DataFrame.

        Returns:
            gpd.GeoDataFrame: Cleaned and structured GeoDataFrame.
        """
        gdf["date_start"] = pd.to_datetime(gdf["date_start"], errors="coerce")
        gdf["date_end"] = pd.to_datetime(gdf["date_end"], errors="coerce")

        return gdf

    def get_global_metadata(self):
        """Get global metadata for the catalog."""
        return self._global_metadata

    def append(self, metadata: Union[CatalogEntry, Dict[str, Any]]):
        """
        Append an entry to the catalog.

        Args:
            metadata (Union[CatalogEntry, Dict[str, Any]]): Metadata to append.
        """
        if isinstance(metadata, dict):
            metadata = CatalogEntry(**metadata)
        self.entries.append(metadata)
        row = pd.DataFrame([asdict(metadata)])
        row = self._clean_dataframe(row)
        self.gdf = pd.concat([self.gdf, row], ignore_index=True)
        if not isinstance(self.gdf, gpd.GeoDataFrame):
             self.gdf = gpd.GeoDataFrame(
                 self.gdf, geometry="geometry" if "geometry" in self.gdf.columns else None
             )

    def extend(self, other_catalog: 'DatasetCatalog'):
        """
        Extend the catalog with another catalog.

        Args:
            other_catalog (DatasetCatalog): Other catalog to merge.
        """
        self.gdf = pd.concat([self.gdf, other_catalog.gdf], ignore_index=True)
        if not isinstance(self.gdf, gpd.GeoDataFrame):
            self.gdf = gpd.GeoDataFrame(
                self.gdf,
                geometry="geometry" if "geometry" in self.gdf.columns else None
            )



    def filter_by_date(
        self,
        start: datetime | list[datetime],
        end: datetime | list[datetime],
    ) -> None:
        """
        Filter entries by time range.

        Args:
            start (datetime): Start date(s).
            end (datetime): End date(s).

        Returns:
            gpd.GeoDataFrame: Filtered GeoDataFrame.
        """
        if isinstance(start, datetime) and isinstance(end, datetime):
            mask = (self.gdf["date_end"] >= start) & (self.gdf["date_start"] < end)
        elif isinstance(start, list) and isinstance(end, list) \
            and bool(start) and all(isinstance(elem, datetime) for elem in start) \
            and bool(end) and all(isinstance(elem, datetime) for elem in end):
            if len(start) != len(end):
                logger.warning("Start and end must have the same number of elements.")
                return
            mask = (self.gdf["date_end"] >= start[0])
            for start_el, end_el in zip(start, end, strict=False):
                in_period = (
                    (self.gdf["date_end"] >= start_el) &
                    (self.gdf["date_start"] < end_el)
                    )
                mask = mask | in_period

        else:
            logger.warning(
                "Start and end dates must be datetime objects or lists of datetimes."
            )
            return

        self.gdf = self.gdf.loc[mask]

    def filter_by_region(self, region: Union[gpd.GeoSeries, geometry.base.BaseGeometry]) -> None:
        """
        Filter GeoDataFrame entries intersecting with the given region.

        Parameters
        ----------
        region : gpd.GeoSeries
            A GeoSeries containing a polygon or a collection of polygons.
        """
        if self.gdf.empty:
            logger.warning("GeoDataFrame is empty, nothing to filter")
            return

        # Verify if geometry column exists/is active
        try:
            _ = self.gdf.geometry
        except AttributeError:
             # Try to set it if column "geometry" exists
             if "geometry" in self.gdf.columns:
                 self.gdf = self.gdf.set_geometry("geometry")
             else:
                 logger.warning(
                     "GeoDataFrame has no active geometry column. Skipping spatial filtering."
                 )
                 return

        # Ensure GeoDataFrame has a defined CRS
        if not isinstance(self.gdf, gpd.GeoDataFrame):
            if "geometry" in self.gdf.columns:
                self.gdf = gpd.GeoDataFrame(self.gdf, geometry="geometry")
            else:
                self.gdf = gpd.GeoDataFrame(self.gdf)

        if self.gdf.crs is None:
            logger.info("Setting CRS to EPSG:4326")
            self.gdf.set_crs("EPSG:4326", inplace=True)

        # Log information before filtering
        initial_count = len(self.gdf)
        logger.debug(f"Initial GeoDataFrame length: {initial_count}")

        # Store bounds before filtering for diagnostics
        original_data_bounds = self.gdf.total_bounds if not self.gdf.empty else None
        region_bounds = region.bounds

        # Geometry diagnostics
        self.check_geometries_compatibility(self.gdf, region)

        # Apply spatial filter
        try:
            # Check that all geometries are valid
            invalid_geoms = ~self.gdf.geometry.is_valid
            if invalid_geoms.any():
                logger.warning(
                    f"Found {invalid_geoms.sum()} invalid geometries, attempting to fix them"
                )
                self.gdf.loc[invalid_geoms, 'geometry'] = self.gdf.loc[
                    invalid_geoms, 'geometry'
                ].buffer(0)

            # Check that the region is valid
            is_valid_check = region.is_valid
            if hasattr(is_valid_check, 'all'):
                 is_valid_bool = is_valid_check.all()
            else:
                 is_valid_bool = bool(is_valid_check)

            if not is_valid_bool:
                logger.warning("Region geometry is invalid, attempting to fix it")
                region = region.buffer(0)

            # Handle CRS conversion
            if hasattr(region, 'crs') and region.crs is not None:
                if self.gdf.crs != region.crs:
                    logger.info(f"Reprojecting region from {region.crs} to {self.gdf.crs}")
                    try:
                        region_gdf = gpd.GeoDataFrame(
                            geometry=[region] if isinstance(
                                region, (geometry.Polygon, geometry.base.BaseGeometry)
                            ) else region,
                            crs=region.crs
                        )
                        region_gdf = region_gdf.to_crs(self.gdf.crs)
                        region = region_gdf.geometry
                        logger.debug("Region successfully reprojected")
                    except Exception as e:
                        logger.error(f"Failed to reproject region: {e}")
                        logger.warning("Using original region without reprojection")

            logger.debug("Applying spatial filter using intersects method")
            if isinstance(region, (gpd.GeoSeries, gpd.GeoDataFrame)):
                region = region.union_all

            mask = self.gdf.intersects(region)
            self.gdf = self.gdf[mask]

            final_count = len(self.gdf)
            percentage = (final_count/initial_count*100) if initial_count > 0 else 0
            logger.info(f"Spatial filter applied: {initial_count} -> {final_count} entries "
                    f"({percentage:.1f}% retained)")

            # Diagnostic with original bounds
            if final_count == 0:
                logger.warning("No data points found in the specified region!")
                logger.warning(f"Original data bounds: {original_data_bounds}")
                logger.warning(f"Region bounds: {region_bounds}")

                # Additional diagnostic: check for overlap
                if original_data_bounds is not None:
                    data_minx, data_miny, data_maxx, data_maxy = original_data_bounds
                    reg_minx, reg_miny, reg_maxx, reg_maxy = region_bounds

                    overlap_x = not (data_maxx < reg_minx or data_minx > reg_maxx)
                    overlap_y = not (data_maxy < reg_miny or data_miny > reg_maxy)

                    if not overlap_x:
                        logger.warning(
                            f"No longitude overlap: data [{data_minx:.2f}, {data_maxx:.2f}] "
                            f"vs region [{reg_minx:.2f}, {reg_maxx:.2f}]"
                        )
                    if not overlap_y:
                        logger.warning(
                            f"No latitude overlap: data [{data_miny:.2f}, {data_maxy:.2f}] "
                            f"vs region [{reg_miny:.2f}, {reg_maxy:.2f}]"
                        )
            else:
                logger.debug(f"Filtered data bounds: {self.gdf.total_bounds}")
        except Exception as exc:
            logger.error(f"Error during spatial filtering: {exc}")
            logger.warning("Spatial filtering failed, keeping original data")

    def check_geometries_compatibility(
            self, gdf: gpd.GeoDataFrame,
            region: Union[gpd.GeoSeries, geometry.base.BaseGeometry]
        ):
        """Diagnostic with automatic corrections."""
        logger.debug("=== SPATIAL FILTER DIAGNOSTICS ===")

        # Check CRS
        logger.debug(f"GeoDataFrame CRS: {gdf.crs}")
        if hasattr(region, 'crs'):
            logger.debug(f"Region CRS: {region.crs}")
        else:
            logger.debug("Region CRS: None (shapely geometry, assumed EPSG:4326)")

        if gdf.crs is None:
            logger.warning("WARNING: GeoDataFrame has no CRS defined")

        # Check bounds
        if not gdf.empty:
            try:
                data_bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
                logger.debug(f"Data bounds: [{data_bounds[0]:.2f}, {data_bounds[1]:.2f}, "
                            f"{data_bounds[2]:.2f}, {data_bounds[3]:.2f}]")
            except Exception as e:
                logger.debug(f"Could not compute data bounds: {e}")

        try:
            region_bounds = region.bounds  # (minx, miny, maxx, maxy)
            logger.debug(f"Region bounds: [{region_bounds[0]:.2f}, {region_bounds[1]:.2f}, "
                        f"{region_bounds[2]:.2f}, {region_bounds[3]:.2f}]")
        except Exception as e:
            logger.debug(f"Could not compute region bounds: {e}")

        # Check geometry type
        if not gdf.empty:
            geom_types = gdf.geometry.geom_type.unique()
            logger.debug(f"Data geometry types: {geom_types}")
        logger.debug(f"Region geometry type: {region.geom_type}")

        # Check geometry validity
        if not gdf.empty:
            invalid_count = (~gdf.geometry.is_valid).sum()
            if invalid_count > 0:
                logger.warning(f"{invalid_count} invalid geometries in data")

        is_valid_check = region.is_valid
        if hasattr(is_valid_check, 'all'):
             is_valid_bool = is_valid_check.all()
        else:
             is_valid_bool = bool(is_valid_check)

        if not is_valid_bool:
            logger.warning("Region geometry is invalid")

        # Intersection test on a sample
        if not gdf.empty and len(gdf) > 0:
            try:
                # Test intersection on the first 5 points
                sample_size = min(5, len(gdf))
                sample_intersects = gdf.geometry.head(sample_size).intersects(region)
                intersect_count = sample_intersects.sum()
                logger.debug(
                    f"Sample intersection test: {intersect_count}/{sample_size} points intersect"
                )
            except Exception as e:
                logger.debug(f"Sample intersection test failed: {e}")

        logger.debug("=== END DIAGNOSTICS ===")



    def filter_by_variables(self, variables: List[str]) -> None:
        """
        Filter entries by variable list.

        Args:
            variables (List[str]): List of variables to filter.

        Returns:
            gpd.GeoDataFrame: Filtered GeoDataFrame.
        """
        if not variables or len(variables) == 0:
            logger.warning("No variables provided for filtering.")
            return
        self.gdf = self.gdf[
            self.gdf["variables"].apply(lambda vars: any(var in vars for var in variables))
        ]
        logger.debug(
            f"Filtered by keeping variables {variables}, remaining entries: {len(self.gdf)}"
        )

    def to_geodataframe(self) -> gpd.GeoDataFrame:
        """
        Return the complete GeoDataFrame.

        Returns:
            gpd.GeoDataFrame: Catalog GeoDataFrame.
        """
        return self.gdf.copy()



    def list_paths(self):
        """
        List file paths in the catalog.

        Returns:
            List[str]: List of paths.
        """
        return [entry["path"] for _, entry in self.gdf.iterrows()]
