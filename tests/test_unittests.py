import os
import unittest

import numpy as np
import xarray as xr

from dctools.dcio.loader import DataLoader
from dctools.dcio.saver import DataSaver
from dctools.processing.gridder import DataGridder
from dctools.utilities.file_utils import empty_folder


class TestDCTools(unittest.TestCase):
    """Unit tests for DCTools library."""

    test_dataset = xr.Dataset()

    def setUp(self):
        """Tests configuration."""
        self.test_output_dir = os.path.join("tests", "test_output")
        os.makedirs(self.test_output_dir, exist_ok=True)
        self.test_dataset = self.get_sample_dataset()
        self.test_file_path = os.path.join(self.test_output_dir, "test.nc")

    def tearDown(self):
        """Clear after tests."""
        if os.path.exists(self.test_file_path):
            os.remove(self.test_file_path)
        if os.path.exists(self.test_output_dir):
            empty_folder(self.test_output_dir)
            os.rmdir(self.test_output_dir)

    def get_sample_dataset(self):
        # Step 1: Create a sample NetCDF dataset
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
        # define dataset attribures
        self.test_dataset.attrs.update(
            {
                "title": "Sea temperature",
                "institution": "IMT Atlantique",
                "source": "Sentinel Satellite",
                "description": "Sample temperature data",
                "units": "Celsius",
            }
        )
        return ds

    def test_save_load_dataset(self):
        """Test dataset loading."""
        DataSaver.save_dataset(self.test_dataset, self.test_file_path)
        loaded_ds = DataLoader.load_dataset(self.test_file_path)
        self.assertIsInstance(loaded_ds, xr.Dataset)
        self.assertIn("temperature", loaded_ds.variables)

    def test_grid_data(self):
        """Test gridding data."""
        gridded_ds = DataGridder.interpolate_to_2dgrid(self.test_dataset)

        self.assertTrue("temperature" in gridded_ds.variables)
        self.assertEqual(gridded_ds.sizes["lon"], 360)
        self.assertEqual(gridded_ds.sizes["lat"], 180)


if __name__ == "__main__":
    unittest.main()
