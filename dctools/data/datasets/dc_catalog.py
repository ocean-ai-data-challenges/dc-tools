

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
    transform_in_place,
    make_serializable,
    make_timestamps_serializable,
    make_fully_serializable,
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
    variables: Dict[str, List[str]]
    # variables_dict: Dict[str, List[str]]
    # variables_rename_dict: Dict[str, str]
    # dimensions: Dict[str, str]
    # keep_variables: List[str]
    # coord_type: str
    # crs: str
    geometry: gpd.GeoSeries
    # resolution: Optional[Dict[str, float]] = None

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
        try:
            # Convertir le GeoDataFrame en GeoJSON (dict)

            # serial_metadata = transform_in_place(
            #    self._global_metadata.copy(), make_serializable
            # )
            serial_metadata = make_fully_serializable(self._global_metadata.copy())
            gdf_serializable = make_timestamps_serializable(self.gdf)
            geojson_dict = json.loads(gdf_serializable.to_json())
            # Ajouter les métadonnées globales
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
    def from_json(cls, path: str, alias:str) -> 'DatasetCatalog':
        """
        Reconstruit une instance de DatasetCatalog à partir d'un fichier GeoJSON.
        """
        try:
            with open(path, "r") as f:
                data = json.load(f)
            features = data["features"]
            global_metadata = data.get("global_metadata", {})


            # désérialiser coord_system si besoin ---
            coord_system = global_metadata.get("coord_system")
            if isinstance(coord_system, str):
                import json as _json
                try:
                    coord_system = _json.loads(coord_system)
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
            for feat in features:
                props = feat.get("properties", {})
                geom = shape(feat["geometry"])

                if geom is not None:
                    try:
                        geom_obj = shape(geom)
                        if not isinstance(geom_obj, BaseGeometry):
                            logger.warning(f"Invalid geometry: {geom}")
                            continue
                        props["geometry"] = geom_obj


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
                                                # Garder la valeur originale si la conversion échoue                        except Exception as exc:
                            except Exception as exc:
                                logger.warning(f"Error converting path to datetime for glorys: {exc}")

                        records.append(props)
                    except Exception as exc:
                        logger.warning(f"Could not parse geometry: {geom} ({exc})")
                        continue
                else:
                    logger.warning("Feature without geometry, skipping.")

            # Créer le GeoDataFrame
            gdf = gpd.GeoDataFrame(records, geometry="geometry")
            if "keep_variables" in gdf.columns:
                gdf["keep_variables"] = gdf["keep_variables"].apply(
                    lambda x: json.loads(x) if isinstance(x, str) and x.startswith("[") else x
                )
            # Créer l'instance avec global_metadata

            instance = cls(alias=alias, global_metadata=global_metadata, dataframe=gdf)
        
            return instance
        except Exception as exc:
            logger.error(f"Erreur lors du chargement depuis JSON : {traceback.format_exc(exc)}")
            raise


    def filter_attrs(
        self, filters: dict[str, Union[Callable[[Any], bool], gpd.GeoSeries]]
        #filters: Dict[str, Callable[[Any], bool]],
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
            mask = (self.gdf["date_end"] >= start[0]) # Initialize mask 
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

    def check_geometries_compatibility(self, gdf: gpd.GeoDataFrame, region: geometry.Polygon):
        # 1. Vérifier le CRS
        print(f"GeoDataFrame CRS: {gdf.crs}")
        if hasattr(region, 'crs'):
            print(f"Region CRS: {region.crs}")
        else:
            print("Region CRS: None (shapely geometry, assumed EPSG:4326)")

        if gdf.crs is None:
            print("WARNING: GeoDataFrame has no CRS defined.")
        if hasattr(region, 'crs') and region.crs is not None and gdf.crs != region.crs:
            print("WARNING: CRS mismatch between GeoDataFrame and region.")

        # 2. Vérifier le type de géométrie
        print(f"GeoDataFrame geometry type: {gdf.geometry.geom_type.unique()}")
        print(f"Region type: {type(region)}")

        # 3. Vérifier l’amplitude des longitudes
        if "lon" in gdf.columns:
            lon_min, lon_max = gdf["lon"].min(), gdf["lon"].max()
            print(f"GeoDataFrame longitude range: {lon_min} to {lon_max}")
        else:
            print("GeoDataFrame has no 'lon' column.")

        # 4. Vérifier l’amplitude des latitudes
        if "lat" in gdf.columns:
            lat_min, lat_max = gdf["lat"].min(), gdf["lat"].max()
            print(f"GeoDataFrame latitude range: {lat_min} to {lat_max}")
        else:
            print("GeoDataFrame has no 'lat' column.")

        # 5. Vérifier la validité des géométries
        if not gdf.geometry.is_valid.all():
            print("WARNING: Some geometries in GeoDataFrame are invalid.")
        if hasattr(region, 'is_valid') and not region.is_valid:
            print("WARNING: Region geometry is invalid.")

        # 6. Vérifier la plage de longitude du polygone
        if hasattr(region, 'bounds'):
            print(f"Region bounds: {region.bounds}")

        # 7. Vérifier le nombre de points dans le GeoDataFrame
        print(f"GeoDataFrame length: {len(gdf)}")

        print("---- End of geometry checks ----")



    def filter_by_region(self, region: geometry.Polygon) -> None:
        """
        Filtre les entrées du GeoDataFrame qui sont incluses dans la région donnée.

        Args:
            region (gpd.GeoSeries): Un polygone ou une collection de polygones.
        """
        self.gdf.set_crs("EPSG:4326", inplace=True)   # TODO : SET AT INIT
        self.check_geometries_compatibility(self.gdf, region)
        '''if not isinstance(region, shapely.MultiPoint):
            logger.warning("Region must be a GeoSeries.")
            return'''
        # Si region contient un seul polygone, prends-le directement
        '''print(self.gdf.crs)
        print(region.crs)
        region_geom = region.unary_union
        mask = self.gdf.geometry.within(region_geom)
        self.gdf = self.gdf[mask]'''
        # region_geometry = region.geometry  #.iloc[0]
        gdf_geometry0 = self.gdf.geometry.iloc[0]
        # self.gdf = self.gdf.explode(index_parts=False, ignore_index=True)
        # self.gdf = self.gdf[self.gdf.geometry.within(region.geometry.iloc[0])]
        self.gdf = self.gdf[self.gdf.within(region)]
        gdf_geometry1 = self.gdf.geometry  #.iloc[0]
        print(f"INTERSECT: {self.gdf}")

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








