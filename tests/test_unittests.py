import os
from pathlib import Path

import numpy as np
import xarray as xr
import pytest

from dctools.dcio.loader import DataLoader
from dctools.dcio.saver import DataSaver
from dctools.processing.gridder import DataGridder
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

def test_save_load_dataset(
    setup_data,
    setup_filepath,
    ):
    """Test dataset loading."""
    DataSaver.save_dataset(setup_data, setup_filepath)
    loaded_ds = DataLoader.load_dataset(setup_filepath)
    assert isinstance(loaded_ds, xr.Dataset)
    assert "temperature" in loaded_ds.variables

def test_load_error(setup_filepath):
    """Test trying to load a non-existent file."""

    # Shouldn't this raise an error instead of returning `None`?
    assert DataLoader.load_dataset(Path(setup_filepath).stem) is None


def test_regrid_data(setup_data):
    """Test regridding data."""
    gridded_ds = DataGridder.interpolate_to_2dgrid(setup_data)

    assert "temperature" in gridded_ds.variables
    assert gridded_ds.sizes["lon"] == 360
    assert gridded_ds.sizes["lat"] == 180