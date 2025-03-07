#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Test ARGO data."""

import numpy as np
import pandas as pd
import xarray as xr
import pytest

from dctools.processing.argo_data import ArgoDataProcessor

def create_argo_dataset(
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
    temperature = 15 + 8 * np.random.randn(n_points)

    # Define the data in an xarray dataset
    ds = xr.Dataset(
        {
            "temperature": (["N_POINTS"], temperature),
        },
        coords={
            "LATITUDE": ("N_POINTS", lat),
            "LONGITUDE": ("N_POINTS", lon),
            "TIME": ("N_POINTS", time),
            "N_POINTS": np.arange(n_points)
        },
    )
    # define dataset attributes
    ds.attrs.update(
        {
            "title": "Fictitious Argo floats",
            "institution": "IMT Atlantique",
            "source": "dc-tools",
            "description": "Generated Argo data",
        }
    )
    return ds

@pytest.fixture(scope='module')
def setup_data():
    """Setup test data."""
    data = create_argo_dataset(
        19,
        "2025-01-01",
        "2026-01-01"
    )
    yield data

def test_subset_argo(setup_data):
    """Test subsetting Argo data."""
    subset = ArgoDataProcessor.subset_argo(
        setup_data,
        lat_range = (0,90),
        lon_range = (0, 180),
        time_range = (
            np.datetime64("2025-01-01"),
            np.datetime64("2025-12-31")
            )
    )
    assert subset.sizes["N_POINTS"] == 9
