"""Tests for MultiSourceDatasetManager.get_transform argument compatibility."""

from types import SimpleNamespace

from dctools.data.datasets.dataset_manager import MultiSourceDatasetManager


class _DummyDataset:
    def __init__(self):
        self.keep_variables = ["TEMP"]

    def get_global_metadata(self):
        return {
            "coord_system": {
                "coordinates": {
                    "lat": "LATITUDE",
                    "lon": "LONGITUDE",
                    "time": "TIME",
                    "depth": "DEPTH",
                    "n_points": "N_POINTS",
                }
            },
            "variables_rename_dict": {"TEMP": "temperature"},
        }

    def get_connection_manager(self):
        return SimpleNamespace(get_global_metadata=lambda: self.get_global_metadata())

    def standardize_names(self, coords_rename_dict, vars_rename_dict):
        return None


def _build_manager(monkeypatch):
    import dctools.data.datasets.dataset_manager as dm_mod

    manager = object.__new__(MultiSourceDatasetManager)
    manager.datasets = {"glonet": _DummyDataset()}
    manager.dataset_processor = None

    def fake_get_dataset_transform(
        alias, metadata, dataset_processor, transform_name=None, config=None
    ):
        return {
            "alias": alias,
            "transform_name": transform_name,
            "config": config,
        }

    monkeypatch.setattr(dm_mod, "get_dataset_transform", fake_get_dataset_transform)
    return manager


def test_get_transform_keyword_dataset_alias(monkeypatch):
    """get_transform resolves the transform when dataset_alias is passed as a keyword."""
    manager = _build_manager(monkeypatch)

    transform = manager.get_transform(dataset_alias="glonet", reduce_precision=True)

    assert transform["alias"] == "glonet"
    assert transform["config"]["reduce_precision"] is True


def test_get_transform_backward_compat_style(monkeypatch):
    """get_transform supports the legacy positional-name + keyword-alias calling style."""
    manager = _build_manager(monkeypatch)

    transform = manager.get_transform("standardize", dataset_alias="glonet")

    assert transform["alias"] == "glonet"
    assert transform["transform_name"] == "standardize"


def test_get_transform_positional_alias(monkeypatch):
    """get_transform resolves the transform when alias is the first positional argument."""
    manager = _build_manager(monkeypatch)

    transform = manager.get_transform("glonet", transform_name="standardize")

    assert transform["alias"] == "glonet"
    assert transform["transform_name"] == "standardize"
