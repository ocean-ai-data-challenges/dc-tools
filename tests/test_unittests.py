#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Unit Tests."""

import os
from pathlib import Path

import numpy as np
import xarray as xr
import pytest

from dctools.dcio.dclogger import DCLogger
from dctools.dcio.loader import FileLoader
from dctools.dcio.saver import DataSaver
from dctools.processing.gridder import DataGridder
from dctools.utilities.errors import DCExceptionHandler
from dctools.utilities.file_utils import empty_folder

def get_sample_dataset():
    """Create a sample Xarray dataset."""
    # Create dimensions for the dataset
    lat = np.linspace(-90, 90, 3)  # 3 latitude points
    lon = np.linspace(-180, 180, 4)  # 4 longitude points

    # Create some random temperature data
    temperature = 15 + 8 * np.random.randn(len(lat), len(lon))

    # Define the data in an xarray dataset
    ds = xr.Dataset(
        {
            "temperature": (["lat", "lon"], temperature),
        },
        coords={
            "lat": lat,
            "lon": lon,
        },
    )
    # define dataset attributes
    ds.attrs.update(
        {
            "title": "Sea temperature",
            "institution": "IMT Atlantique",
            "source": "Test suite",
            "description": "Sample temperature data",
            "units": "Celsius",
        }
    )
    return ds

@pytest.fixture(scope='module')
def setup_data():
    """Setup test data."""
    data = get_sample_dataset()
    yield data


@pytest.fixture(scope='session')
def setup_filepath():
    """Setup test paths."""
    test_output_dir = os.path.join("tests", "test_output")
    test_file_path = os.path.join(test_output_dir, "test.nc")

    yield test_file_path

    # Teardown
    if os.path.exists(test_file_path):
        os.remove(test_file_path)

@pytest.fixture(scope='session', autouse=True)
def setup_output_dir():
    """Test path configuration."""
    test_output_dir = os.path.join("tests", "test_output")
    os.makedirs(test_output_dir, exist_ok=True)

    yield test_output_dir

    # Teardown
    if os.path.exists(test_output_dir):
        empty_folder(test_output_dir)
        os.rmdir(test_output_dir)


@pytest.fixture(scope='session', autouse=True)
def setup_logger():
    """Setup test logger."""
    # initialize_logger
    test_logger = DCLogger(
        name="Test Logger", logfile=None
    ).get_logger()
    yield test_logger

@pytest.fixture(scope='session', autouse=True)
def setup_exception_handler(setup_logger):
    """Setup exception handler."""
    # initialize exception handler
    exc_handler = DCExceptionHandler(setup_logger)
    yield exc_handler

def test_save_load_dataset(
    setup_data,
    setup_filepath,
    setup_logger,
    setup_exception_handler,
    ):
    """Test dataset loading."""
    setup_logger.info("Run Test dataset loading")
    DataSaver.save_dataset(
        setup_data, setup_filepath,
        setup_exception_handler, setup_logger,
    )
    loaded_ds = FileLoader.load_dataset(
        setup_filepath, setup_exception_handler, setup_logger
    )
    assert isinstance(loaded_ds, xr.Dataset)
    assert "temperature" in loaded_ds.variables

def test_load_error(setup_filepath, setup_logger, setup_exception_handler):
    """Test trying to load a non-existent file."""
    setup_logger.info("Run test_load_error")
    try:
        FileLoader.load_dataset(
            Path(setup_filepath).stem,
            setup_exception_handler, setup_logger,
            fail_on_error=False,
        )
    except Exception:
        pass


def test_regrid_data(setup_data, setup_logger):
    """Test regridding data."""
    setup_logger.info("Run test_regrid_data")
    gridded_ds = DataGridder.interpolate_to_2dgrid(setup_data)

    assert "temperature" in gridded_ds.variables
    assert gridded_ds.sizes["lon"] == 360
    assert gridded_ds.sizes["lat"] == 180


'''
from dctools.utilities.xarray_utils import filter_spatial_area, filter_time_interval

def test_filter_spatial_area_with_valid_data():
    # Create a sample dataset with latitude and longitude coordinates
    lat = [10, 20, 30]
    lon = [100, 110, 120]
    data = xr.DataArray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], coords=[lat, lon], dims=["lat", "lon"])
    ds = xr.Dataset({"data": data})

    # Filter the dataset for a specific spatial area
    result = filter_spatial_area(ds, 15, 25, 105, 115)

    assert result is not None
    assert result["data"].shape == (1, 1)  # Only one point should be in the specified area

def test_filter_spatial_area_no_data():
    # Create a sample dataset with latitude and longitude coordinates
    lat = [10, 20, 30]
    lon = [100, 110, 120]
    data = xr.DataArray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], coords=[lat, lon], dims=["lat", "lon"])
    ds = xr.Dataset({"data": data})

    # Filter the dataset for a spatial area that doesn't match any data
    result = filter_spatial_area(ds, 35, 45, 125, 135)

    assert result is None  # Should return None as no data matches

def test_filter_time_interval_with_valid_data():
    # Create a sample dataset with a time coordinate
    times = pd.date_range("2024-05-02", periods=10, freq="D")
    data = xr.DataArray(range(10), coords=[times], dims=["time"])
    ds = xr.Dataset({"data": data})

    # Filter the dataset for a specific time range
    result = filter_time_interval(ds, "2024-05-04", "2024-05-06")

    assert result is not None
    assert result["data"].shape[0] == 3  # Should have 3 time steps

def test_filter_time_interval_no_data():
    # Create a sample dataset with a time coordinate
    times = pd.date_range("2024-05-02", periods=10, freq="D")
    data = xr.DataArray(range(10), coords=[times], dims=["time"])
    ds = xr.Dataset({"data": data})

    # Filter the dataset for a time range that doesn't match any data
    result = filter_time_interval(ds, "2024-06-01", "2024-06-10")

    assert result is None  # Should return None as no data matches
'''
