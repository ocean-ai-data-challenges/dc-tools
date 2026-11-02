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


def _daily_catalog(days):
    geom = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    return DatasetCatalog("obs", entries=[
        CatalogEntry(path=f"/tmp/{d:%Y%m%d}.zarr", date_start=d, date_end=d, geometry=geom)
        for d in days])


def test_a_gap_at_the_end_of_the_period_is_not_the_end_of_the_data():
    """SWOT's gap of 2023-05-20/21 ended a forecast window of 2023-05-11: not the data's end."""
    gap = _daily_catalog(list(pd.date_range("2023-05-01", "2023-05-19"))
                         + list(pd.date_range("2023-05-22", "2023-05-31")))
    gap.filter_by_date(pd.Timestamp("2023-05-11"), pd.Timestamp("2023-05-21"))
    assert gap.get_dataframe()["date_end"].max() == pd.Timestamp("2023-05-19")
    assert gap.available_until() == pd.Timestamp("2023-05-21")
    ended = _daily_catalog(pd.date_range("2023-05-01", "2023-05-19"))
    ended.filter_by_date(pd.Timestamp("2023-05-11"), pd.Timestamp("2023-05-21"))
    assert ended.available_until() == pd.Timestamp("2023-05-19")      # the data's own end
