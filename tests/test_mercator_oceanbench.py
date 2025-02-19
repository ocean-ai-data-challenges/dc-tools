import os
import time
import unittest

import numpy as np
import xarray as xr
import wget

from dctools.dcio.loader import DataLoader
from dctools.dcio.saver import DataSaver
from dctools.processing.gridder import DataGridder
from dctools.utils.file_utils import empty_folder
from dctools.third_party.mercator_oceanbench import mercator_plot


class TestOceanBenchTools(unittest.TestCase):
    """Unit tests for Mercator's oceanbench library."""

    test_dataset = xr.Dataset()
    ref_dataset = xr.Dataset()

    def setUp(self):
        """Tests configuration."""
        self.glonet_dir = os.path.join("data", "glonet")
        self.glorys_dir = os.path.join("data", "glorys")
        os.makedirs(self.glonet_dir, exist_ok=True)
        os.makedirs(self.glorys_dir, exist_ok=True)
        self.test_file_path = os.path.join(self.glonet_dir, "test.nc")
        self.ref_file_path = os.path.join(self.glorys_dir, "ref.nc")
        self.plot_file = os.path.join(self.glonet_dir, "plot.png")
        url_test = "https://project-oceanbench-708263-0.lab.dive.edito.eu/lab/tree/data/glonet/2024-01-03.nc"
        url_ref = "https://project-oceanbench-708263-0.lab.dive.edito.eu/lab/tree/data/glorys14/2024-01-03.nc"
        self.download_file(url_test, self.test_file_path)
        self.download_file(url_ref, self.ref_file_path)

    def tearDown(self):
        try:
            """Clear after tests."""
            if os.path.exists(self.test_file_path):
                os.remove(self.test_file_path)
            if os.path.exists(self.ref_file_path):
                os.remove(self.ref_file_path)
            if os.path.exists(self.plot_file):
                os.remove(self.plot_file)
            if os.path.exists(self.glonet_dir):
                empty_folder(self.glonet_dir)
                os.rmdir(self.glonet_dir)
            if os.path.exists(self.glorys_dir):
                empty_folder(self.glorys_dir)
                os.rmdir(self.glorys_dir)
        except Exception as e:
            print("Error while removing temporary data: ", e)

    def download_file(self, url, path):
        wget.download(url, out=path)

    def load_dataset(self, filepath):
        """dataset loading."""
        loaded_ds = DataLoader.load_dataset(filepath)
        self.assertIsInstance(loaded_ds, xr.Dataset)
        self.assertIn("temperature", loaded_ds.variables)
        return loaded_ds

    def test_mercator_ocean_bench(self):
        nparray = oceanbench.evaluate.pointwise_evaluation(
            glonet_datasets_path=self.glonet_dir,
            glorys_datasets_path=self.glorys_dir,
        )
        mercator_plot.plot_pointwise_evaluation(nparray, 2, self.plot_file, True)


if __name__ == "__main__":
    unittest.main()
