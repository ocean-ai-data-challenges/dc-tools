
import sys
import logging
from loguru import logger

def pytest_configure():
    logger.remove()
    logger.add(sys.stdout, level="TRACE")
    logger.configure(
    handlers=[
        {"sink": sys.stderr, "level": "TRACE"}
    ]
)