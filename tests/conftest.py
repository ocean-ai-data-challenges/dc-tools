"""Pytest configuration and fixtures for DC-tools tests."""

import sys
from loguru import logger

def pytest_configure():
    """Configure pytest logging."""
    logger.remove()
    logger.add(sys.stdout, level="TRACE")
    logger.configure(
    handlers=[
        {"sink": sys.stderr, "level": "TRACE"}
    ]
)
