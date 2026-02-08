#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Unit Tests."""

import os
import shutil
from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
import xarray as xr
import pytest

from dctools.dcio.loader import FileLoader
from dctools.dcio.saver import DataSaver
# from dctools.processing.gridder import DataGridder
from dctools.utilities.xarray_utils import (
    filter_spatial_area,
    filter_time_interval,
    filter_variables,
    filter_dataset_by_depth,
)

# TODO : Update all tests
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
        shutil.rmtree(test_output_dir, ignore_errors=True)


def test_save_load_dataset(
    # setup_logger,
    setup_data,
    setup_filepath,
    ):
    """Test dataset loading."""
    logger.info("Run Test dataset loading")
    DataSaver.save_dataset(
        setup_data, setup_filepath,
    )
    loaded_ds = FileLoader.open_dataset_auto(
        setup_filepath,
        engine="netcdf4"
    )
    assert isinstance(loaded_ds, xr.Dataset)
    assert "temperature" in loaded_ds.variables
    loaded_ds.close()

def test_load_error(setup_filepath):
    """Test trying to load a non-existent file."""
    logger.info("Run test_load_error")
    try:
        FileLoader.open_dataset_auto(
            Path(setup_filepath).stem
        )
    except Exception:
        pass



def test_filter_spatial_area_with_valid_data():
    """Test filtering by spatial area with valid data overlap."""
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
    """Test filtering by spatial area with no data overlap."""
    # Create a sample dataset with latitude and longitude coordinates
    lat = [10, 20, 30]
    lon = [100, 110, 120]
    data = xr.DataArray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], coords=[lat, lon], dims=["lat", "lon"])
    ds = xr.Dataset({"data": data})

    # Filter the dataset for a spatial area that doesn't match any data
    result = filter_spatial_area(ds, 35, 45, 125, 135)

    assert result is None  # Should return None as no data matches

def test_filter_time_interval_with_valid_data():
    """Test filtering by time interval with valid data overlap."""
    # Create a sample dataset with a time coordinate
    times = pd.date_range("2024-05-02", periods=10, freq="D")
    data = xr.DataArray(range(10), coords=[times], dims=["time"])
    ds = xr.Dataset({"data": data})

    # Filter the dataset for a specific time range
    result = filter_time_interval(ds, "2024-05-04", "2024-05-06")

    assert result is not None
    assert result["data"].shape[0] == 3  # Should have 3 time steps

def test_filter_time_interval_no_data():
    """Test filtering by time interval with no data overlap."""
    # Create a sample dataset with a time coordinate
    times = pd.date_range("2024-05-02", periods=10, freq="D")
    data = xr.DataArray(range(10), coords=[times], dims=["time"])
    ds = xr.Dataset({"data": data})

    # Filter the dataset for a time range that doesn't match any data
    result = filter_time_interval(ds, "2024-06-01", "2024-06-10")

    assert result is None  # Should return None as no data matches


def test_filter_variables():
    """Test filtering dataset to keep only selected variables."""
    # Setup
    ds = xr.Dataset(
        {"temp": (("x", "y"), [[1, 2], [3, 4]]),
         "salt": (("x", "y"), [[5, 6], [7, 8]])},
        coords={"x": [1, 2], "y": [10, 20]}
    )
    keep_vars = ["temp"]

    # Action
    filtered = filter_variables(ds, keep_vars)

    # Assert
    assert "temp" in filtered
    assert "salt" not in filtered
    assert filtered["temp"].equals(ds["temp"])

def test_filter_dataset_by_depth():
    """Test filtering dataset by depth."""
    # Setup
    ds = xr.Dataset(
        {"temp": (("time", "depth"), [[1, 2, 3], [4, 5, 6]])},
        coords={"time": [1, 2], "depth": [10, 100, 1000]}
    )
    target_depths = [100]

    # Action
    filtered = filter_dataset_by_depth(ds, target_depths, depth_tol=1)

    # Assert
    assert filtered is not None
    # Assuming filter_dataset_by_depth keeps only nearest depths or similar
    # Depending on implementation, it might return dataset with subset of depth
    assert 100 in filtered.depth
    assert 10 not in filtered.depth
    assert 1000 not in filtered.depth

