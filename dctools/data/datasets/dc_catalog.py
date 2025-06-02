

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import (
    Any, Callable, Dict, List,
    Optional, Tuple, Union,
)


import copy
import geopandas as gpd
import json
from loguru import logger
import pandas as pd
from shapely.geometry import mapping, shape, Polygon


METADATA_VARIABLES = [
    "variables", "variables_rename_dict",
    "dimensions", "coord_type", "crs", "resolution"
]

@dataclass
class CatalogEntry:
    path: str
    date_start: pd.Timestamp
    date_end: pd.Timestamp
    variables: Dict[str, List[str]]
    variables_dict: Dict[str, List[str]]
    variables_rename_dict: Dict[str, str]
    dimensions: Dict[str, str]
    # dimensions_rename_dict: Dict[str, str]
    coord_type: str
    crs: str
    geometry: gpd.GeoSeries
    resolution: Optional[Dict[str, float]] = None

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


'''@dataclass
class SharedCatalogEntry:
    local: CatalogEntry
    shared: Dict[str, Any]

    def __getattr__(self, name: str) -> Any:
        if hasattr(self.local, name):
            val = getattr(self.local, name)
            shared_val = self.shared.get(name)
            return shared_val if val is None else val
        raise AttributeError(f"'CatalogEntry' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        if name in ("local", "shared"):
            super().__setattr__(name, value)
        else:
            setattr(self.local, name, value)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "local": self.local.to_dict()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], shared: Dict[str, Any]):
        return cls(local=CatalogEntry.from_dict(data["local"]), shared=shared)'''


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
        #logger.info(f"\n\nENTRIES: {entries}\n\n")
        # Convertir les dictionnaires en CatalogEntry si nécessaire
        if entries:
            self.entries = []
            for entry in entries:
                """if isinstance(entry, dict):
                    self.entries.append(CatalogEntry(**entry))"""
                if isinstance(entry, CatalogEntry):
                    self.entries.append(entry)
                    # logger.debug(f"Adding entry: {entry}")
                elif isinstance(entry, dict):
                    self.entries.append(CatalogEntry(**entry))
                    # logger.debug(f"Adding entry: {entry}")
                else:
                    logger.warning(f"Ignoring invalid entry: {entry}")
            # logger.debug(f"Entries0: {entries[0]}")
            '''if isinstance(entry, CatalogEntry):
                crs = entries[0].crs
            elif isinstance(entry, dict):
                crs = entries[0].get('crs')'''
            # Convertir les entrées en GeoDataFrame
            self.gdf = gpd.GeoDataFrame(
                [asdict(entry) for entry in self.entries]
            )
            #    geometry="geometry", crs=crs
            #)
        if not dataframe.empty:
            self.gdf = dataframe

        if not self.gdf.empty:
            self.gdf = self._clean_dataframe(self.gdf)
        

        # get global metadata
        first_row = self.gdf.iloc[0]
        self.global_metadata = {}
        for metadata_var in METADATA_VARIABLES:
            self.global_metadata[metadata_var] = first_row[metadata_var]

    def to_json(self, path: Optional[str] = None) -> str:
        """
        Exporte l'intégralité du contenu de DatasetCatalog au format JSON.

        Args:
            path (Optional[str]): Chemin pour sauvegarder le fichier JSON.

        Returns:
            str: Représentation JSON complète de l'instance.
        """
        try:
            """gdf = self.gdf.copy()
            for col in ["date_start", "date_end"]:  # Colonnes contenant des dates
                if col in gdf.columns:
                    gdf[col] = gdf[col].dt.strftime("%Y-%m-%dT%H:%M:%SZ")  # Format ISO 8601"""

            self.gdf.to_file(path, driver="GeoJSON")

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
            gdf = gpd.read_file(path)
            return cls(dataframe=gdf)
        except Exception as exc:
            logger.error(f"Erreur lors du chargement depuis JSON : {repr(exc)}")
            raise

    '''def to_json(self, path: str):
        data = {
            "shared": self.shared_fields,
            "entries": [
                {
                    "geometry": mapping(row.geometry),
                    "entry": row[self.entry_column].to_dict()
                }
                for _, row in self.gdf.iterrows()
            ]
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "DatasetCatalog":
        logger.info(f"Loading catalog from {path}")
        with open(path, "r") as jf:
            data = json.load(jf)

        entries = []

        for item in data:
            geom = shape(item["geometry"])
            item["geometry"] = geom
            entries.append(item)

        return cls(entries)'''

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
                #gdf = gdf[gdf[key].intersects(roi)]
                gdf = gdf[roi]
                logger.debug(f"Filtered DataFrame shape: {gdf.shape}")

            if not key in gdf.columns:
                raise KeyError(f"{key} not found in catalog columns")
            elif key == "variables":
                # Special case for variables filtering
                # gdf = df[df[key].apply(func)]
                gdf = gdf[func(key)]
                logger.debug(f"Filtered DataFrame shape: {gdf.shape}") 
            else:
                # gdf = df[df[key].apply(func)]
                gdf = gdf[func(gdf[key])]
                logger.debug(f"Filtered DataFrame: {gdf}")

        self.gdf = gdf
        # catalog[catalog.intersects(roi)]

    '''def __init__(self, entries: Union[List[CatalogEntry], gpd.GeoDataFrame], entry_column: str = "entry"):
        self.entry_column = entry_column
        self.shared_fields: Dict[str, Any] = {}

        if isinstance(entries, gpd.GeoDataFrame):
            self.gdf = entries.copy()
        elif isinstance(entries, list):
            self.entries = []
            # On construit un GeoDataFrame à partir de la liste de CatalogEntry
            for entry in entries:
                if isinstance(entry, CatalogEntry):
                    self.entries.append(entry)
                else:
                    logger.warning(f"Ignoring invalid entry: {entry}")
            self.gdf = gpd.GeoDataFrame([asdict(entry) for entry in self.entries])
            #geometries = [entry.geometry for entry in entries]
            #self.gdf = gpd.GeoDataFrame({entry_column: entries}, geometry=geometries)
            logger.debug(f"self.gdf: {self.gdf.to_markdown()}")
        else:
            raise TypeError("entries must be either a list of CatalogEntry or a GeoDataFrame")
        self.factorize()

    def factorize(self):
        """Identifie les champs communs et optimise les CatalogEntry via délégation."""
        entries: List[CatalogEntry] = self.gdf[self.entry_column].tolist()
        if not entries:
            return

        shared = {}
        field_names = [f for f in CatalogEntry.__dataclass_fields__.keys() if f != "geometry"]

        for field_name in field_names:
            values = [getattr(entry, field_name) for entry in entries]
            first = values[0]
            if all(v == first for v in values):
                shared[field_name] = copy.deepcopy(first)
                for entry in entries:
                    setattr(entry, field_name, None)

        self.shared_fields = shared
        self.gdf[self.entry_column] = [
            SharedCatalogEntry(local=entry, shared=self.shared_fields) for entry in entries
        ]'''

    '''def __init__(self, entries: Union[List[CatalogEntry], gpd.GeoDataFrame], entry_column: str = "entry"):
        logger.debug(f"entries: {entries}")
        self.entry_column = entry_column
        self.shared_fields: Dict[str, Any] = {}
        if isinstance(entries, gpd.GeoDataFrame):
            self.gdf = entries.copy()
        elif isinstance(entries, list):
            logger.debug(f"entries!")
            geometries = [entry.geometry for entry in entries]
            self.gdf = gpd.GeoDataFrame({entry_column: entries}, geometry=geometries)
            logger.debug(f"self.df 0: {self.gdf.to_markdown()}")
        else:
            raise TypeError("entries must be either a list of CatalogEntry or a GeoDataFrame")
        logger.debug(f"factorize!")
        self.factorize()
        logger.debug(f"\n\nself.df 1: {self.gdf.to_markdown()}\n\n")

    def factorize(self):
        """Identifie les champs communs et optimise les CatalogEntry via délégation."""
        entries: List[CatalogEntry] = self.gdf[self.entry_column].tolist()
        if not entries:
            return

        shared = {}
        field_names = [f for f in CatalogEntry.__dataclass_fields__.keys() if f != "geometry"]

        for field_name in field_names:
            values = [getattr(entry, field_name) for entry in entries]
            first = values[0]
            if all(v == first for v in values):
                shared[field_name] = copy.deepcopy(first)
                for entry in entries:
                    setattr(entry, field_name, None)

        self.shared_fields = shared
        self.gdf[self.entry_column] = [
            SharedCatalogEntry(local=entry, shared=self.shared_fields) for entry in entries
        ]


    def to_json(self, path: str):
        data = {
            "shared": self.shared_fields,
            "entries": [
                {
                    "geometry": mapping(row.geometry),
                    "entry": row[self.entry_column].to_dict()
                }
                for _, row in self.gdf.iterrows()
            ]
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def from_json(cls, path: str, entry_column: str = "entry") -> "DatasetCatalog":
        logger.info(f"Loading catalog from {path}")
        with open(path, "r") as jf:
            data = json.load(jf)

        shared = data["shared"]
        entries = []
        geometries = []

        for item in data["entries"]:
            geom = shape(item["geometry"])
            entry = SharedCatalogEntry.from_dict(item["entry"], shared)
            entries.append(entry)
            geometries.append(geom)

        gdf = gpd.GeoDataFrame({entry_column: entries}, geometry=geometries)
        collection = cls(gdf, entry_column=entry_column)
        collection.shared_fields = shared
        return collection

    @property
    def df(self) -> gpd.GeoDataFrame:
        """
        Returns a flat GeoDataFrame with all shared and local fields expanded
        for filtering, selection, etc.
        """
        rows = []
        for row in self.gdf.itertuples(index=False):
            entry = getattr(row, self.entry_column)
            flat = {}

            for field in CatalogEntry.__dataclass_fields__.keys():
                val = getattr(entry, field)
                if val is None and field in self.shared_fields:
                    val = self.shared_fields[field]
                flat[field] = val

            rows.append(flat)

        gdf = pd.DataFrame(rows)
        gdf["geometry"] = self.gdf.geometry.values
        return gpd.GeoDataFrame(gdf, geometry="geometry")

    @df.setter
    def df(self, value: gpd.GeoDataFrame):
        """
        Allows replacing the internal GeoDataFrame by a filtered one.
        Assumes that filtering did not break object structure.
        """
        new_gdf = value.copy()
        new_entries = []

        for row in new_gdf.itertuples(index=False):
            logger.debug(f"row: {row}")
            for f in CatalogEntry.__dataclass_fields__.keys():
                logger.debug(f"     f: {f}")
                ff = getattr(row.val, f)
                logger.debug(f"     ff: {ff}")
            data = {f: getattr(row, f) for f in CatalogEntry.__dataclass_fields__.keys()}
            data["geometry"] = getattr(row.val, "geometry")
            entry = CatalogEntry(**data)
            wrapped = SharedCatalogEntry(entry, self.shared_fields)
            new_entries.append(wrapped)

        self.gdf = gpd.GeoDataFrame({self.entry_column: new_entries}, geometry=new_gdf.geometry)'''

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
        required = [
            "path", "date_start", "date_end",
            "coord_type", "dimensions", "crs", 
            "variables", "resolution",
            "geometry",
        ]
        for col in required:
            if col not in gdf:
                gdf[col] = pd.NA
        gdf["date_start"] = pd.to_datetime(gdf["date_start"], errors="coerce")
        gdf["date_end"] = pd.to_datetime(gdf["date_end"], errors="coerce")
        # gdf["geometry"] = gdf.apply(
        #    lambda row: box(row["lon_min"], row["lat_min"], row["lon_max"], row["lat_max"]), axis=1
        #)
        return gdf  #gpd.GeoDataFrame(gdf, crs=gdf.crs)

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

        self.gdf = self.gdf.loc[
            (self.gdf["date_end"] >= start) & (self.gdf["date_start"] < end)
        ]


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








