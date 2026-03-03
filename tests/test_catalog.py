"""Tests for data catalog functionality."""

import pandas as pd
from shapely.geometry import Polygon
from dctools.data.datasets.dc_catalog import CatalogEntry, DatasetCatalog


def test_catalog_entry_serialization():
    """Test serialization and deserialization of CatalogEntry."""
    # Setup
    start_date = pd.Timestamp("2024-01-01")
    end_date = pd.Timestamp("2024-01-02")
    geom = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])

    entry = CatalogEntry(
        path="/tmp/test.nc", date_start=start_date, date_end=end_date, geometry=geom
    )

    # Test to_dict
    entry_dict = entry.to_dict()
    assert entry_dict["path"] == "/tmp/test.nc"
    assert entry_dict["date_start"] == start_date.isoformat()
    assert entry_dict["geometry"]["type"] == "Polygon"

    # Test from_dict
    reconstructed = CatalogEntry.from_dict(entry_dict)
    assert reconstructed.path == entry.path
    assert reconstructed.date_start == entry.date_start
    assert reconstructed.date_end == entry.date_end
    # Geometry comparison
    assert reconstructed.geometry.equals(entry.geometry)


def test_dataset_catalog_init():
    """Test initialization of DatasetCatalog."""
    # Setup
    start_date = pd.Timestamp("2024-01-01")
    end_date = pd.Timestamp("2024-01-02")
    geom = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])

    entry = CatalogEntry(
        path="/tmp/test.nc", date_start=start_date, date_end=end_date, geometry=geom
    )

    entries = [entry]
    global_meta = {"resolution": "10km"}

    catalog = DatasetCatalog("test_alias", global_metadata=global_meta, entries=entries)

    assert len(catalog.entries) == 1
    assert catalog._global_metadata == global_meta
    assert catalog.alias == "test_alias"
