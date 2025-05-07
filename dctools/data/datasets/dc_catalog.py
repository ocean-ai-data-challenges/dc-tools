

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import fsspec
import geopandas as gpd
import json
from loguru import logger
import pandas as pd
from shapely.geometry import box, Polygon


@dataclass
class CatalogEntry:
    path: str
    date_start: Optional[str]  # Format ISO 8601 (e.g., "2024-01-01")
    date_end: Optional[str]
    lat_min: Optional[float]
    lat_max: Optional[float]
    lon_min: Optional[float]
    lon_max: Optional[float]
    variables: Dict[str, List[str]]  # Nom des variables et leurs dimensions associées
    dimensions: Dict[str, str]  # Dimensions sous forme de {standard_name: real_name}
    spatial_resolution: Optional[Tuple[float, float]] = None  # Résolution (lat, lon) en degrés
    temporal_resolution: Optional[str] = None  # En format ISO 8601 (e.g., "P1D")
    geometry: Optional[Polygon] = None
    # alias: Optional[str] = None  # Ajout du champ alias


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
            self.df = gpd.GeoDataFrame()
            return
        #logger.info(f"\n\nENTRIES: {entries}\n\n")
        # Convertir les dictionnaires en CatalogEntry si nécessaire
        if entries:
            self.entries = []
            for entry in entries:
                """if isinstance(entry, dict):
                    self.entries.append(CatalogEntry(**entry))"""
                if isinstance(entry, CatalogEntry):
                    self.entries.append(entry)
                else:
                    logger.warning(f"Ignoring invalid entry: {entry}")
            # Convertir les entrées en GeoDataFrame
            self.df = gpd.GeoDataFrame([asdict(entry) for entry in self.entries])
        if not dataframe.empty:
            self.df = dataframe

        if not self.df.empty:
            self.df = self._clean_dataframe(self.df)

    def get_dataframe(self) -> gpd.GeoDataFrame:
        """
        Retourne le GeoDataFrame interne.

        Returns:
            gpd.GeoDataFrame: Le GeoDataFrame du catalogue.
        """
        return self.df

    def set_dataframe(self, df: gpd.GeoDataFrame) -> None:
        """
        Retourne le GeoDataFrame interne.

        Returns:
            gpd.GeoDataFrame: Le GeoDataFrame du catalogue.
        """
        self.df = self._clean_dataframe(df)

    def _clean_dataframe(self, df: pd.DataFrame) -> gpd.GeoDataFrame:
        """
        Nettoie et structure le DataFrame pour garantir la cohérence des métadonnées.

        Args:
            df (pd.DataFrame): DataFrame brut.

        Returns:
            gpd.GeoDataFrame: GeoDataFrame nettoyé et structuré.
        """
        required = ["path", "date_start", "date_end", "lat_min", "lat_max", "lon_min", "lon_max", "variables", "dimensions"]
        for col in required:
            if col not in df:
                df[col] = pd.NA
        df["date_start"] = pd.to_datetime(df["date_start"], errors="coerce")
        df["date_end"] = pd.to_datetime(df["date_end"], errors="coerce")
        df["geometry"] = df.apply(
            lambda row: box(row["lon_min"], row["lat_min"], row["lon_max"], row["lat_max"]), axis=1
        )
        return gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")

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
        self.df = pd.concat([self.df, row], ignore_index=True)

    def extend(self, other_catalog: 'DatasetCatalog'):
        """
        Étend le catalogue avec un autre catalogue.

        Args:
            other_catalog (DatasetCatalog): Autre catalogue à fusionner.
        """
        self.df = pd.concat([self.df, other_catalog.df], ignore_index=True)

    def filter_by_date(self, start: datetime, end: datetime) -> None:
        """
        Filtre les entrées par plage temporelle.

        Args:
            start (datetime): Date de début.
            end (datetime): Date de fin.

        Returns:
            gpd.GeoDataFrame: GeoDataFrame filtré.
        """
        if not isinstance(start, datetime) or not isinstance(end, datetime):
            logger.warning("Start and end dates must be datetime objects.")
            return
        # self.df = self.df[(self.df.date_end >= start) & (self.df.date_start < end)]
        self.df = self.df.loc[
            (self.df["date_end"] >= start) & (self.df["date_start"] < end)
        ]


    def filter_by_bbox(self, bbox: Tuple[float, float, float, float]) -> None:
        """
        Filtre les entrées par boîte englobante (bounding box).

        Args:
            bbox (Tuple[float, float, float, float]): (lon_min, lat_min, lon_max, lat_max).

        Returns:
            gpd.GeoDataFrame: GeoDataFrame filtré.
        """
        if not isinstance(bbox, tuple) or len(bbox) != 4:
            logger.warning("Bounding box must be a tuple of (lon_min, lat_min, lon_max, lat_max).")
            return
        region = box(*bbox)
        self.df = self.df[self.df.intersects(region)]

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
        self.df = self.df[self.df["variables"].apply(lambda vars: any(var in vars for var in variables))]

    def to_geodataframe(self) -> gpd.GeoDataFrame:
        """
        Retourne le GeoDataFrame complet.

        Returns:
            gpd.GeoDataFrame: GeoDataFrame du catalogue.
        """
        return self.df.copy()

    def to_file(self, path: Optional[str] = None) -> str:
        """
        Exporte l'intégralité du contenu de DatasetCatalog au format JSON.

        Args:
            path (Optional[str]): Chemin pour sauvegarder le fichier JSON.

        Returns:
            str: Représentation JSON complète de l'instance.
        """
        try:
            """df = self.df.copy()
            for col in ["date_start", "date_end"]:  # Colonnes contenant des dates
                if col in df.columns:
                    df[col] = df[col].dt.strftime("%Y-%m-%dT%H:%M:%SZ")  # Format ISO 8601"""

            self.df.to_file(path, driver="GeoJSON")

            #return json_str
        except Exception as exc:
            logger.error(f"Erreur lors de l'exportation en JSON : {repr(exc)}")
            raise


    @classmethod
    def from_json(cls, path: str) -> 'DatasetCatalog':
        """
        Reconstruit une instance de DatasetCatalog à partir d'un fichier JSON.

        Args:
            path (str): Chemin vers le fichier JSON.

        Returns:
            DatasetCatalog: Instance reconstruite.
        """
        try:
            # Charger le contenu JSON
            df = gpd.read_file(path)
            return cls(dataframe=df)
        except Exception as exc:
            logger.error(f"Erreur lors du chargement depuis JSON : {repr(exc)}")
            raise

    def list_paths(self):
        """
        Liste les chemins des fichiers dans le catalogue.

        Returns:
            List[str]: Liste des chemins.
        """
        return [entry["path"] for _, entry in self.df.iterrows()]
