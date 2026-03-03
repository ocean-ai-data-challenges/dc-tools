"""Pytest configuration and fixtures for DC-tools tests."""

import sys
from pathlib import Path

import pytest
from loguru import logger


INTEGRATION_TEST_FILES = {
    "test_argo_interface.py",
    "test_connection_manager.py",
    "test_dask_batch_cleanup.py",
    "test_dataloader_argo_coherence.py",
    "test_pipeline.py",
}


def pytest_configure():
    """Configure pytest logging."""
    logger.remove()
    logger.add(sys.stdout, level="TRACE")
    logger.configure(handlers=[{"sink": sys.stderr, "level": "TRACE"}])


def pytest_collection_modifyitems(config, items):
    """Assign default markers for test taxonomy.

    - Keep explicit markers already set in tests.
    - Mark known heavier files as ``integration``.
    - Mark the rest as ``unit``.
    """
    del config

    for item in items:
        if item.get_closest_marker("unit") or item.get_closest_marker("integration"):
            continue
        if item.get_closest_marker("slow"):
            continue

        filename = Path(str(item.fspath)).name
        if filename in INTEGRATION_TEST_FILES:
            item.add_marker(pytest.mark.integration)
        else:
            item.add_marker(pytest.mark.unit)
