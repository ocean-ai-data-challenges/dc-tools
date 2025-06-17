

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import (
    Any, Callable, Dict, List,
    Optional, Tuple, Union,
)

import geopandas as gpd
import json
from loguru import logger
import pandas as pd
from shapely.geometry import mapping, shape


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
        entries: Optional[List[Union[CatalogEntry, Dict[str, Any]]]] = None,
        dataframe: Optional[gpd.GeoDataFrame] = gpd.GeoDataFrame(),
    ):
        """
        Initialise le catalogue avec une liste d'entrées.

        Args:
            entries (List[Union[CatalogEntry, Dict[str, Any]]]): Liste des métadonnées des datasets.
        """
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
                    # logger.debug(f"Adding entry: {entry}")
                elif isinstance(entry, dict):
                    self.entries.append(CatalogEntry(**entry))
                    # logger.debug(f"Adding entry: {entry}")
                else:
                    logger.warning(f"Ignoring invalid entry: {entry}")

            # Convertir les entrées en GeoDataFrame
            self.gdf = gpd.GeoDataFrame(
                [asdict(entry) for entry in self.entries]
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
            self.gdf.to_file(path, driver="GeoJSON")

            #return json_str
        except Exception as exc:
            logger.error(f"Erreur lors de l'exportation en JSON : {repr(exc)}")
            raise


    @classmethod
    def from_json(cls, path: str) -> 'DatasetCatalog':
        """
        Reconstruit une instance de DatasetCatalog à partir d'un fichier GeoJSON.
        """
        try:
            with open(path, "r") as f:
                data = json.load(f)
            features = data["features"]
            # Extraire les propriétés et la géométrie
            records = []
            for feat in features:
                props = feat.get("properties", {})
                geom = shape(feat["geometry"])

                if geom is not None:
                    from shapely.geometry.base import BaseGeometry
                    try:
                        geom_obj = shape(geom)
                        if not isinstance(geom_obj, BaseGeometry):
                            logger.warning(f"Invalid geometry: {geom}")
                            continue
                        props["geometry"] = geom_obj
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
            # Créer l'instance
            instance = cls(dataframe=gdf)
            return instance
        except Exception as exc:
            import traceback
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

    def filter_by_region(self, region: gpd.GeoSeries) -> None:
        """
        Filtre les entrées par boîte englobante (bounding box).

        Args:
            bbox (Tuple[float, float, float, float]): (lon_min, lat_min, lon_max, lat_max).

        Returns:
            gpd.GeoDataFrame: GeoDataFrame filtré.
        """
        if not isinstance(region, gpd.GeoSeries):
            logger.warning("Region must be a GeoSeries.")
            return
        # logger.debug(f"region1 crs: {region.crs}")
        # logger.debug(f"fdg crs: {self.gdf.crs}")
        self.gdf = self.gdf[self.gdf.intersects(region)]

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








