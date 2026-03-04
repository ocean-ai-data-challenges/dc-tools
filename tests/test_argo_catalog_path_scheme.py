"""Tests for ARGO catalog path scheme normalization helpers."""

from dctools.data.datasets.dataset import (
    _has_unknown_argo_catalog_paths,
    _is_legacy_argo_catalog_paths,
)


def test_detect_legacy_argo_paths():
    """Legacy path format wmo:cycle must be detected."""
    assert _is_legacy_argo_catalog_paths(["6901234:1", "6901235:2"]) is True
    assert _is_legacy_argo_catalog_paths(["2024_01", "2024_02"]) is False


def test_detect_unknown_argo_keys_against_master_index():
    """Unknown keys vs master index should trigger catalog rebuild."""
    assert _has_unknown_argo_catalog_paths(["2024_01"], ["2024_01", "2024_02"]) is False
    assert _has_unknown_argo_catalog_paths(["2024_03"], ["2024_01", "2024_02"]) is True
    assert _has_unknown_argo_catalog_paths(["2024_01"], []) is False
