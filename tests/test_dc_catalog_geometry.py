"""Regression tests for catalog geometry normalization."""

import geopandas as gpd
import pandas as pd
from shapely.geometry import box, mapping
from shapely.geometry.base import BaseGeometry

from dctools.data.datasets.dc_catalog import CatalogEntry, DatasetCatalog


def test_catalog_entry_to_dict_normalizes_geoseries_geometry():
    """CatalogEntry.to_dict should convert a GeoSeries geometry to a plain dict."""
    entry = CatalogEntry(
        path="dummy_path",
        date_start=pd.Timestamp("2024-01-01"),
        date_end=pd.Timestamp("2024-01-02"),
        geometry=gpd.GeoSeries([box(-10, -5, 10, 5)], crs="EPSG:4326"),
    )

    payload = entry.to_dict()

    assert payload["geometry"] is not None
    assert payload["geometry"]["type"] == "Polygon"


def test_dataset_catalog_accepts_featurecollection_geometry_dict():
    """DatasetCatalog should accept a GeoJSON FeatureCollection dict as geometry."""
    geom = mapping(box(-20, -10, 20, 10))
    feature_collection = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": geom,
                "properties": {},
            }
        ],
    }

    catalog = DatasetCatalog(
        alias="argo_profiles",
        global_metadata={},
        entries=[
            {
                "path": "2024_01",
                "date_start": "2024-01-01T00:00:00",
                "date_end": "2024-01-31T23:59:59",
                "geometry": feature_collection,
            }
        ],
    )

    gdf = catalog.get_dataframe()

    assert len(gdf) == 1
    assert isinstance(gdf.geometry.iloc[0], BaseGeometry)
