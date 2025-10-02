

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime
import traceback
from typing import (
    Any, Callable, Dict, List,
    Optional, Tuple, Union,
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
    path: str
    date_start: pd.Timestamp
    date_end: pd.Timestamp
    # variables: Dict[str, List[str]]
    geometry: gpd.GeoSeries

    def to_dict(self):
        try:
            dct = asdict(self)
            dct["date_start"] = self.date_start.isoformat()
            dct["date_end"] = self.date_end.isoformat()
            dct["geometry"] = mapping(self.geometry)
            return dct
        except Exception as exc:
            logger.error(f"Error converting CatalogEntry to dict: {exc}")
            raise

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        data_copy = data.copy()
        data_copy["date_start"] = pd.to_datetime(data_copy["date_start"])
        data_copy["date_end"] = pd.to_datetime(data_copy["date_end"])
        data_copy["geometry"] = shape(data_copy["geometry"])
        return cls(**data_copy)


class DatasetCatalog:
    """Structured catalog to hold and filter dataset metadata entries."""

    def __init__(
        self,
        alias: str,
        global_metadata: Dict[str, Any] = None,
        entries: Optional[List[Union[CatalogEntry, Dict[str, Any]]]] = None,
        dataframe: Optional[gpd.GeoDataFrame] = gpd.GeoDataFrame(),
    ):
        """
        Initialise le catalogue avec une liste d'entrées.

        Args:
            entries (List[Union[CatalogEntry, Dict[str, Any]]]): Liste des métadonnées des datasets.
        """
        self.alias = alias
        self._global_metadata = global_metadata
        if entries is None and dataframe.empty:
            logger.warning("No entries or dataframe provided. Initializing empty catalog.")
            self.entries = []
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

            # Convertir les entrées en GeoDataFrame
            self.gdf = gpd.GeoDataFrame(
                [asdict(entry) for entry in self.entries],
                geometry="geometry",
                crs="EPSG:4326",
            )
        if not dataframe.empty:
            self.gdf = dataframe

        if not self.gdf.empty:
            self.gdf = self._clean_dataframe(self.gdf)


    def to_json(self, path: Optional[str] = None) -> str:
        """
        Exporte l'intégralité du contenu de DatasetCatalog au format JSON.

        Args:
            path (Optional[str]): Chemin pour sauvegarder le fichier JSON.

        Returns:
            str: Représentation JSON complète de l'instance.
        """
        def coord_system_to_dict(coord_system):
            if isinstance(coord_system, dict):
                return coord_system
            if hasattr(coord_system, "__dict__"):
                return {k: v for k, v in coord_system.__dict__.items() if not k.startswith("_")}
            return str(coord_system)
        try:
            # Copie et conversion coord_system AVANT serialize_structure
            meta_copy = self._global_metadata.copy()
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
            logger.error(f"Erreur lors de l'exportation en JSON : {repr(exc)}")
            raise

    @classmethod
    def from_json(cls, path: str, alias:str, limit: int, ignore_geometry = False) -> 'DatasetCatalog':
        """
        Reconstruit une instance de DatasetCatalog à partir d'un fichier GeoJSON.
        """
        def process_feature(feat):
            try:
                props = feat.get("properties", {})
                if alias == "glorys" and "path" in props:
                    try:
                        # Convertir le path string en datetime
                        path_value = props["path"]
                        if isinstance(path_value, str):
                            # Essayer différents formats de date
                            try:
                                # Format ISO
                                props["path"] = datetime.fromisoformat(path_value)
                            except ValueError:
                                try:
                                    # Format YYYY-MM-DD
                                    props["path"] = datetime.strptime(path_value, "%Y-%m-%d")
                                except ValueError:
                                    try:
                                        # Format YYYY-MM-DD HH:MM:SS
                                        props["path"] = datetime.strptime(path_value, "%Y-%m-%d %H:%M:%S")
                                    except ValueError:
                                        logger.warning(f"Could not parse path as datetime for glorys: {path_value}")
                                        # Garder la valeur originale si la conversion échoue
                                        traceback.print_exc()
                    except Exception as exc:
                        logger.warning(f"Error converting path to datetime for glorys: {exc}")
                        traceback.print_exc()

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
                except Exception:
                    # Si ce n'est pas du JSON, ignorer ou lever une erreur explicite
                    raise ValueError("coord_system in global_metadata is a string but not valid JSON")

            coord_sys = CoordinateSystem(
                coord_type=coord_system["coord_type"],
                coord_level=coord_system["coord_level"],
                coordinates=coord_system["coordinates"],
                crs=coord_system["crs"],
            )
            global_metadata["coord_system"] = coord_sys

            # Extraire les propriétés et la géométrie
            records = []

            with ThreadPoolExecutor(max_workers=16) as executor:
                records = list(executor.map(process_feature, features))
            '''n_feats = 0
            for feat in features:
                if limit and n_feats >= limit:
                    break
                props = process_feature(feat, ignore_geometry)
                records.append(props)
                n_feats += 1'''
            # Créer le GeoDataFrame
            if not ignore_geometry:
                gdf = gpd.GeoDataFrame(records, geometry="geometry")
            else:
                gdf = gpd.GeoDataFrame(records)
            #if "keep_variables" in gdf.columns:
            #    gdf["keep_variables"] = gdf["keep_variables"].apply(
            #        lambda x: json.loads(x) if isinstance(x, str) and x.startswith("[") else x
            #    )
            # Créer l'instance avec global_metadata

            instance = cls(alias=alias, global_metadata=global_metadata, dataframe=gdf)
        
            return instance
        except Exception as exc:
            logger.error(f"Erreur lors du chargement depuis JSON : {traceback.format_exc(exc)}")
            traceback.print_exc()
            raise


    def filter_attrs(
        self, filters: dict[str, Union[Callable[[Any], bool], gpd.GeoSeries]]
    ) -> None:
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
                #logger.debug(f"Filtered DataFrame shape: {gdf.shape}")

            if not key in gdf.columns:
                raise KeyError(f"{key} not found in catalog columns")
            elif key == "variables":
                # Special case for variables filtering
                gdf = gdf[func(key)]
            else:
                gdf = gdf[func(gdf[key])]

        self.gdf = gdf

    def get_dataframe(self) -> gpd.GeoDataFrame:
        """
        Retourne le GeoDataFrame interne.

        Returns:
            gpd.GeoDataFrame: Le GeoDataFrame du catalogue.
        """
        return self.gdf

    def set_dataframe(self, gdf: gpd.GeoDataFrame) -> None:
        """
        Retourne le GeoDataFrame interne.

        Returns:
            gpd.GeoDataFrame: Le GeoDataFrame du catalogue.
        """
        self.gdf = self._clean_dataframe(gdf)

    def _clean_dataframe(self, gdf: pd.DataFrame) -> gpd.GeoDataFrame:
        """
        Nettoie et structure le DataFrame pour garantir la cohérence des métadonnées.

        Args:
            gdf (pd.DataFrame): DataFrame brut.

        Returns:
            gpd.GeoDataFrame: GeoDataFrame nettoyé et structuré.
        """
        gdf["date_start"] = pd.to_datetime(gdf["date_start"], errors="coerce")
        gdf["date_end"] = pd.to_datetime(gdf["date_end"], errors="coerce")

        return gdf

    def get_global_metadata(self):
        return self._global_metadata

    def append(self, metadata: Union[CatalogEntry, Dict[str, Any]]):
        """
        Ajoute une entrée au catalogue.

        Args:
            metadata (Union[CatalogEntry, Dict[str, Any]]): Métadonnées à ajouter.
        """
        if isinstance(metadata, dict):
            metadata = CatalogEntry(**metadata)
        self.entries.append(metadata)
        row = pd.DataFrame([asdict(metadata)])
        row = self._clean_dataframe(row)
        self.gdf = pd.concat([self.gdf, row], ignore_index=True)

    def extend(self, other_catalog: 'DatasetCatalog'):
        """
        Étend le catalogue avec un autre catalogue.

        Args:
            other_catalog (DatasetCatalog): Autre catalogue à fusionner.
        """
        self.gdf = pd.concat([self.gdf, other_catalog.gdf], ignore_index=True)



    def filter_by_date(
        self,
        start: datetime | list[datetime],
        end: datetime | list[datetime],
    ) -> None:
        """
        Filtre les entrées par plage temporelle.

        Args:
            start (datetime): Date(s) de début.
            end (datetime): Date(s) de fin.

        Returns:
            gpd.GeoDataFrame: GeoDataFrame filtré.
        """
        
        if isinstance(start, datetime) and isinstance(end, datetime):
            mask = (self.gdf["date_end"] >= start) & (self.gdf["date_start"] < end)
        elif isinstance(start, list) and isinstance(end, list) \
            and bool(start) and all(isinstance(elem, datetime) for elem in start) \
            and bool(end) and all(isinstance(elem, datetime) for elem in end):
            if len(start) != len(end):
                logger.warning("start and end must have the same number of elements.")
                return
            mask = (self.gdf["date_end"] >= start[0])
            for start_el, end_el in zip(start, end):
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

    def filter_by_region(self, region: geometry.Polygon) -> None:
        """
        Filtre les entrées du GeoDataFrame qui intersectent avec la région donnée.

        Args:
            region (geometry.Polygon): Un polygone ou une collection de polygones.
        """
        if self.gdf.empty:
            logger.warning("GeoDataFrame is empty, nothing to filter")
            return
        
        # S'assurer que le GeoDataFrame a un CRS défini
        if self.gdf.crs is None:
            logger.info("Setting CRS to EPSG:4326")
            self.gdf.set_crs("EPSG:4326", inplace=True)
        
        # Log des informations avant filtrage
        initial_count = len(self.gdf)
        logger.debug(f"Initial GeoDataFrame length: {initial_count}")
        
        # Stocker les bounds avant le filtrage pour le diagnostic
        original_data_bounds = self.gdf.total_bounds if not self.gdf.empty else None
        region_bounds = region.bounds
        
        # Diagnostic des géométries
        self.check_geometries_compatibility(self.gdf, region)

        # Appliquer le filtre spatial
        try:
            # Vérifier que toutes les géométries sont valides
            invalid_geoms = ~self.gdf.geometry.is_valid
            if invalid_geoms.any():
                logger.warning(f"Found {invalid_geoms.sum()} invalid geometries, attempting to fix them")
                self.gdf.loc[invalid_geoms, 'geometry'] = self.gdf.loc[invalid_geoms, 'geometry'].buffer(0)
            
            # Vérifier que la région est valide
            if not region.is_valid:
                logger.warning("Region geometry is invalid, attempting to fix it")
                region = region.buffer(0)
            
            # Gérer la conversion CRS
            if hasattr(region, 'crs') and region.crs is not None:
                if self.gdf.crs != region.crs:
                    logger.info(f"Reprojecting region from {region.crs} to {self.gdf.crs}")
                    try:
                        import geopandas as gpd
                        region_gdf = gpd.GeoDataFrame([1], geometry=[region], crs=region.crs)
                        region_gdf = region_gdf.to_crs(self.gdf.crs)
                        region = region_gdf.geometry.iloc[0]
                        logger.debug(f"Region successfully reprojected")
                    except Exception as e:
                        logger.error(f"Failed to reproject region: {e}")
                        logger.warning("Using original region without reprojection")
            
            logger.debug("Applying spatial filter using intersects method")
            # mask = self.gdf.geometry.intersects(region)
            mask = self.gdf.intersects(region)
            self.gdf = self.gdf[mask]
            
            final_count = len(self.gdf)
            percentage = (final_count/initial_count*100) if initial_count > 0 else 0
            logger.info(f"Spatial filter applied: {initial_count} -> {final_count} entries "
                    f"({percentage:.1f}% retained)")
            
            # Diagnostic avec les bounds originales
            if final_count == 0:
                logger.warning("No data points found in the specified region!")
                logger.warning(f"Original data bounds: {original_data_bounds}")
                logger.warning(f"Region bounds: {region_bounds}")
                
                # Diagnostic supplémentaire : vérifier s'il y a un recouvrement
                if original_data_bounds is not None:
                    data_minx, data_miny, data_maxx, data_maxy = original_data_bounds
                    reg_minx, reg_miny, reg_maxx, reg_maxy = region_bounds
                    
                    overlap_x = not (data_maxx < reg_minx or data_minx > reg_maxx)
                    overlap_y = not (data_maxy < reg_miny or data_miny > reg_maxy)
                    
                    if not overlap_x:
                        logger.warning(f"No longitude overlap: data [{data_minx:.2f}, {data_maxx:.2f}] vs region [{reg_minx:.2f}, {reg_maxx:.2f}]")
                    if not overlap_y:
                        logger.warning(f"No latitude overlap: data [{data_miny:.2f}, {data_maxy:.2f}] vs region [{reg_miny:.2f}, {reg_maxy:.2f}]")
            else:
                logger.debug(f"Filtered data bounds: {self.gdf.total_bounds}")
        except Exception as exc:
            logger.error(f"Error during spatial filtering: {exc}")
            logger.warning("Spatial filtering failed, keeping original data")
        
    def check_geometries_compatibility(self, gdf: gpd.GeoDataFrame, region: geometry.Polygon):
        """diagnostic avec corrections automatiques."""
        
        logger.debug(f"=== SPATIAL FILTER DIAGNOSTICS ===")
        
        # Vérifier le CRS
        logger.debug(f"GeoDataFrame CRS: {gdf.crs}")
        if hasattr(region, 'crs'):
            logger.debug(f"Region CRS: {region.crs}")
        else:
            logger.debug("Region CRS: None (shapely geometry, assumed EPSG:4326)")

        if gdf.crs is None:
            logger.warning("WARNING: GeoDataFrame has no CRS defined")
        
        # Vérifier les bounds
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
        
        # Vérifier le type de géométrie
        if not gdf.empty:
            geom_types = gdf.geometry.geom_type.unique()
            logger.debug(f"Data geometry types: {geom_types}")
        logger.debug(f"Region geometry type: {region.geom_type}")

        # Vérifier la validité des géométries
        if not gdf.empty:
            invalid_count = (~gdf.geometry.is_valid).sum()
            if invalid_count > 0:
                logger.warning(f"WARNING: {invalid_count} invalid geometries in data")
        
        if not region.is_valid:
            logger.warning("WARNING: Region geometry is invalid")
        
        # Test d'intersection sur un échantillon
        if not gdf.empty and len(gdf) > 0:
            try:
                # Tester l'intersection sur les 5 premiers points
                sample_size = min(5, len(gdf))
                sample_intersects = gdf.geometry.head(sample_size).intersects(region)
                intersect_count = sample_intersects.sum()
                logger.debug(f"Sample intersection test: {intersect_count}/{sample_size} points intersect")
            except Exception as e:
                logger.debug(f"Sample intersection test failed: {e}")
        
        logger.debug(f"=== END DIAGNOSTICS ===")



    def filter_by_variables(self, variables: List[str]) -> None:
        """
        Filtre les entrées par liste de variables.

        Args:
            variables (List[str]): Liste des variables à filtrer.

        Returns:
            gpd.GeoDataFrame: GeoDataFrame filtré.
        """
        if not variables or len(variables) == 0:
            logger.warning("No variables provided for filtering.")
            return
        self.gdf = self.gdf[self.gdf["variables"].apply(lambda vars: any(var in vars for var in variables))]
        logger.debug(f"Filtered by variables {variables}, remaining entries: {len(self.gdf)}")

    def to_geodataframe(self) -> gpd.GeoDataFrame:
        """
        Retourne le GeoDataFrame complet.

        Returns:
            gpd.GeoDataFrame: GeoDataFrame du catalogue.
        """
        return self.gdf.copy()

 

    def list_paths(self):
        """
        Liste les chemins des fichiers dans le catalogue.

        Returns:
            List[str]: Liste des chemins.
        """
        return [entry["path"] for _, entry in self.gdf.iterrows()]
