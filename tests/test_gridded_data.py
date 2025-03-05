import numpy as np
import pandas as pd
import xarray as xr
import pytest

from dctools.processing.gridded_data import GriddedDataProcessor

def get_4D_gridded_dataset():
    """Create a sample Xarray dataset."""
    # Create dimensions for the dataset
    lat = np.linspace(-90, 90, 3)  # 3 latitude points
    lon = np.linspace(-180, 180, 4)  # 4 longitude points
    depth = np.linspace(0, 800, 5) # 5 depth levels
    time = pd.date_range(
        start = "2025-01-01",
        end = "2026-01-01",
        periods = 13, # One timestep at the start of every month
    )

    # Create some random temperature data
    temperature = 15 \
        + 8 * np.random.randn(len(lat), len(lon), len(depth), len(time))

    # Define the data in an xarray dataset
    ds = xr.Dataset(
        {
            "temperature": (["lat", "lon", "depth", "time"], temperature),
        },
        coords={
            "lat": lat,
            "lon": lon,
            "depth": depth,
            "time": time
        },
    )
    # define dataset attributes
    ds.attrs.update(
        {
            "title": "Fictituous ocean temperature",
            "institution": "IMT Atlantique",
            "source": "dc-tools",
            "description": "Generated temperature data",
            "units": "Celsius",
        }
    )
    return ds

@pytest.fixture(scope='module')
def setup_data():
    """Setup test data."""
    data = get_4D_gridded_dataset()
    yield data

def test_select_gridded_data(setup_data):
    """Test selecting gridded data."""
    selection = GriddedDataProcessor.subset_grid(
        setup_data,
        lat_range = (0, 90),
        lon_range = (0, 180),
        vert_range=(0, 300),
        time_range=("2025-01-01", "2025-12-31")
    )
    assert selection.sizes["lon"] == 2
    assert selection.sizes["lat"] == 2
    assert selection.sizes["depth"] == 2
    assert selection.sizes["time"] == 12
    assert selection.lat.min() == 0 # Edge case
    assert selection.lat.max() > 0
    # Using > just in case we change the test data

def test_coarsen_gridded_data(setup_data):
    """Test coarsening gridded data."""
    coarsened_ds = GriddedDataProcessor.coarsen_grid(
        setup_data,
        horizontal_window = 2,
        vertical_window = 4,
        time_window = 3
    )
    assert coarsened_ds.sizes["lon"] == 2
    assert coarsened_ds.sizes["lat"] == 2
    assert coarsened_ds.sizes["depth"] == 2
    assert coarsened_ds.sizes["time"] == 5
    assert coarsened_ds.isel(
        lon=-1,
        lat=-1,
        depth=-1,
        time=-1
    ) != np.nan

@pytest.mark.skip(reason="problems with testing when time_window is str")
def test_resampling_gridded_data(setup_data):
    """Test resampling gridded data in time."""
    coarsened_ds = GriddedDataProcessor.coarsen_grid(
        setup_data,
        time_window = "3ME" # TODO: Fix this
    )
    assert coarsened_ds.sizes["time"] == 5
    assert coarsened_ds.isel(
        lon=-1,
        lat=-1,
        depth=-1,
        time=-1
    ) != np.nan