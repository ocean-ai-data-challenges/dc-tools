

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import geopandas as gpd
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
    alias: Optional[str] = None  # Ajout du champ alias


class DatasetCatalog:
    """Structured catalog to hold and filter dataset metadata entries."""


    def __init__(self, entries: List[Union[CatalogEntry, Dict[str, Any]]]):
        """
        Initialise le catalogue avec une liste d'entrées.

        Args:
            entries (List[Union[CatalogEntry, Dict[str, Any]]]): Liste des métadonnées des datasets.
        """
        # Convertir les dictionnaires en CatalogEntry si nécessaire
        self.entries = []
        for entry in entries:
            if isinstance(entry, dict):
                self.entries.append(CatalogEntry(**entry))
            elif isinstance(entry, CatalogEntry):
                self.entries.append(entry)
            else:
                logger.warning(f"Ignoring invalid entry: {entry}")

        # Convertir les entrées en GeoDataFrame
        self.df = gpd.GeoDataFrame([asdict(entry) for entry in self.entries])
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

    def filter_by_date(self, start: datetime, end: datetime) -> gpd.GeoDataFrame:
        """
        Filtre les entrées par plage temporelle.

        Args:
            start (datetime): Date de début.
            end (datetime): Date de fin.

        Returns:
            gpd.GeoDataFrame: GeoDataFrame filtré.
        """
        self.df = self.df[(self.df.date_end >= start) & (self.df.date_start < end)]

    def filter_by_bbox(self, bbox: Tuple[float, float, float, float]) -> gpd.GeoDataFrame:
        """
        Filtre les entrées par boîte englobante (bounding box).

        Args:
            bbox (Tuple[float, float, float, float]): (lon_min, lat_min, lon_max, lat_max).

        Returns:
            gpd.GeoDataFrame: GeoDataFrame filtré.
        """
        region = box(*bbox)
        logger.debug(f"Filtering by bounding box: {bbox}")
        logger.debug(f"Region: {region}")
        logger.debug(f"Initial number of entries: {len(self.df)}")
        logger.debug(f"Entries before filtering: {self.df}")
        self.df = self.df[self.df.intersects(region)]

    def filter_by_variables(self, variables: List[str]) -> gpd.GeoDataFrame:
        """
        Filtre les entrées par liste de variables.

        Args:
            variables (List[str]): Liste des variables à filtrer.

        Returns:
            gpd.GeoDataFrame: GeoDataFrame filtré.
        """
        self.df = self.df[self.df["variables"].apply(lambda vars: any(var in vars for var in variables))]

    def to_geodataframe(self) -> gpd.GeoDataFrame:
        """
        Retourne le GeoDataFrame complet.

        Returns:
            gpd.GeoDataFrame: GeoDataFrame du catalogue.
        """
        return self.df.copy()

    def to_json(self, path: Optional[str] = None) -> str:
        """
        Exporte le catalogue au format GeoJSON.

        Args:
            path (Optional[str]): Chemin pour sauvegarder le fichier GeoJSON.

        Returns:
            str: Représentation GeoJSON du catalogue.
        """
        try:
            # Convertir les colonnes Timestamp en chaînes de caractères au format ISO 8601
            df = self.df.copy()

            for col in ["date_start", "date_end"]:  # Colonnes contenant des dates
                if col in df.columns:
                    df[col] = df[col].dt.strftime("%Y-%m-%dT%H:%M:%SZ")  # Format ISO 8601

            # Exporter au format GeoJSON
            json_str = df.to_json()

            # Sauvegarder dans un fichier si un chemin est fourni
            if path:
                with open(path, "w") as f:
                    f.write(json_str)

            return json_str
        except Exception as exc:
            logger.error(f"Erreur lors de l'exportation en GeoJSON : {repr(exc)}")
            raise

    @classmethod
    def from_json(cls, json_str: str) -> 'DatasetCatalog':
        """
        Charge un catalogue à partir d'une chaîne JSON.

        Args:
            json_str (str): Chaîne JSON.

        Returns:
            DatasetCatalog: Instance du catalogue.
        """
        df = pd.read_json(json_str, orient="records")
        return cls(df.to_dict(orient="records"))