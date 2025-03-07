#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Test Nadir data."""

import numpy as np
import pandas as pd
import xarray as xr
import pytest

from dctools.processing.nadir_data import NadirDataProcessor

def create_nadir_dataset(
    n_points: int,
    start_date: str,
    end_date: str
    ) -> xr.Dataset:
    """Create a sample Xarray Argo dataset."""
    # Create dimensions for the dataset
    lat = np.linspace(-90, 90, n_points)
    lon = np.linspace(-180, 180, n_points)
    time = pd.date_range(
        start = start_date,
        end = end_date,
        periods = n_points,
    )

    # Create some random temperature data
    ssha = np.random.randn(n_points)

    # Define the data in an xarray dataset
    ds = xr.Dataset(
        {
            "ssha": (["time"], ssha),
        },
        coords={
            "lat": ("time", lat),
            "lon": ("time", lon),
            "time": ("time", time),
        },
    )
    # define dataset attributes
    ds.attrs.update(
        {
            "title": "Fictitious nadir altimetry data",
            "institution": "IMT Atlantique",
            "source": "dc-tools",
            "description": "Generated SSHA data",
        }
    )
    return ds

@pytest.fixture(scope='module')
def setup_data():
    """Setup test data."""
    data = create_nadir_dataset(
        19,
        "2025-01-01",
        "2026-01-01"
    )
    yield data

def test_subset_nadir(setup_data):
    """Test subsetting Argo data."""
    subset = NadirDataProcessor.subset_nadir(
        setup_data,
        lat_range = (0,90),
        lon_range = (0, 180),
        time_range = (
            np.datetime64("2025-01-01"),
            np.datetime64("2025-12-31")
            )
    )
    assert subset.sizes["time"] == 9
