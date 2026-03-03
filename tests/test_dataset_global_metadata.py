"""Regression tests for BaseDataset global metadata fallback behavior."""

from dctools.data.datasets.dataset import BaseDataset


class _DummyConnectionManager:
    def __init__(self, metadata):
        self._metadata = metadata

    def get_global_metadata(self):
        return self._metadata


def test_get_global_metadata_falls_back_when_cached_none():
    """get_global_metadata should fetch from the manager when the cache is None."""
    dataset = object.__new__(BaseDataset)
    dataset._global_metadata = None
    dataset.connection_manager = _DummyConnectionManager({"variables": {}})

    metadata = BaseDataset.get_global_metadata(dataset)

    assert metadata == {"variables": {}}
    assert dataset._global_metadata == {"variables": {}}


def test_get_global_metadata_returns_empty_dict_when_manager_none():
    """get_global_metadata should return an empty dict when the manager returns None."""
    dataset = object.__new__(BaseDataset)
    dataset._global_metadata = None
    dataset.connection_manager = _DummyConnectionManager(None)

    metadata = BaseDataset.get_global_metadata(dataset)

    assert metadata == {}
    assert dataset._global_metadata == {}


def test_build_catalog_rebuilds_when_catalog_missing_even_with_stale_type():
    """build_catalog should rebuild even when catalog is None but catalog_type is set."""
    dataset = object.__new__(BaseDataset)
    dataset.alias = "argo_profiles"
    dataset.catalog = None
    dataset.catalog_type = "from_catalog_file"
    dataset._global_metadata = {}
    dataset._metadata = [
        {
            "path": "2024_01",
            "date_start": "2024-01-01T00:00:00",
            "date_end": "2024-01-31T00:00:00",
            "geometry": None,
        }
    ]

    BaseDataset.build_catalog(dataset)

    assert dataset.catalog is not None
    assert dataset.catalog_is_empty() is False
