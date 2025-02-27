import os
import unittest

from pathlib import Path
from pyunpack import Archive
import xarray as xr
import wget

from dctools.dcio.loader import DataLoader
from dctools.third_party.mercator_oceanbench import oceanbench_plotting
import oceanbench


class TestOceanBench(unittest.TestCase):
    """Unit tests for Mercator's oceanbench library."""

    def __init__(self, *args, **kwargs):
        """Init class. Tests configuration."""
        super(TestOceanBench, self).__init__(*args, **kwargs)
        self.glonet_dir = os.path.join("data", "glonet")
        self.glorys_dir = os.path.join("data", "glorys")
        """self.test_file_path = os.path.join(self.glonet_dir, "test.nc")
        self.ref_file_path = os.path.join(self.glorys_dir, "ref.nc")
        os.makedirs(self.glonet_dir, exist_ok=True)
        os.makedirs(self.glorys_dir, exist_ok=True)
        url_test = "ftp://project-oceanbench-708263-0.lab.dive.edito.eu/lab/tree/data/glonet/2024-01-03.nc"
        url_ref = "ftp://project-oceanbench-708263-0.lab.dive.edito.eu/lab/tree/data/glorys14/2024-01-03.nc"
        if not (Path(self.test_file_path).is_file()):
            self.download_file(url_test, self.test_file_path)
        if not (Path(self.ref_file_path).is_file()):
            self.glorys_data = self.download_file(url_ref, self.ref_file_path)
        self.glonet_data = self.load_dataset(self.test_file_path)
        self.glorys_data = self.load_dataset(self.ref_file_path)"""
        test_file_archive = os.path.join(self.glonet_dir, "2024-01-03.7z")
        ref_file_archive = os.path.join(self.glorys_dir, "2024-01-03.7z")
        self.test_file_path = os.path.join(self.glonet_dir, "2024-01-03.nc")
        self.ref_file_path = os.path.join(self.glorys_dir, "2024-01-03.nc")
        assert(Path(test_file_archive).is_file())
        assert(Path(ref_file_archive).is_file())
        if not (Path(self.test_file_path).is_file()):
            Archive(test_file_archive).extractall(self.glonet_dir)
        if not (Path(self.ref_file_path).is_file()):
            Archive(ref_file_archive).extractall(self.glorys_dir)
        assert(Path(self.test_file_path).is_file())
        assert(Path(self.ref_file_path).is_file())

        self.glonet_data = self.load_dataset(self.test_file_path)
        self.glorys_data = self.load_dataset(self.ref_file_path)

    def setUp(self):
        """Tests configuration."""
        # self.plot_file = os.path.join(self.glonet_dir, "plot.png")
        pass

    def tearDown(self):
        '''try:
            """Clear after tests."""
            if os.path.exists(self.test_file_path):
                os.remove(self.test_file_path)
            if os.path.exists(self.ref_file_path):
                os.remove(self.ref_file_path)
            # if os.path.exists(self.plot_file):
            #    os.remove(self.plot_file)
            if os.path.exists(self.glonet_dir):
                empty_folder(self.glonet_dir)
                os.rmdir(self.glonet_dir)
            if os.path.exists(self.glorys_dir):
                empty_folder(self.glorys_dir)
                os.rmdir(self.glorys_dir)
        except Exception as e:
            print("Error while removing temporary data: ", e)'''
        pass

    def download_file(self, url, path):
        wget.download(url, out=path)

    def load_dataset(self, filepath):
        """dataset loading."""
        loaded_ds = DataLoader.load_dataset(filepath)
        self.assertIsInstance(loaded_ds, xr.Dataset)
        self.assertIn("depth", loaded_ds.variables)
        return loaded_ds

    def test_oceanbench_rmse_evaluation(self):
        """Test RMSE."""
        nparray = oceanbench.evaluate.pointwise_evaluation(
            glonet_datasets=[self.glonet_data],
            glorys_datasets=[self.glorys_data],
        )
        # plot_file = os.path.join(self.glonet_dir, "plot1.png")
        oceanbench_plotting.plot_pointwise_evaluation(rmse_dataarray=nparray, depth=2)

        #  plot_file = os.path.join(self.glonet_dir, "plot2.png")
        oceanbench_plotting.plot_pointwise_evaluation_for_average_depth(
            rmse_dataarray=nparray
        )

        # plot_file = os.path.join(self.glonet_dir, "plot3.png")
        oceanbench_plotting.plot_pointwise_evaluation_depth_for_average_time(
            rmse_dataarray=nparray, dataset_depth_values=self.glonet_data.depth.values
        )

    def test_oceanbench_mld_analysis(self):
        """Test MLD."""
        dataset = oceanbench.process.calc_mld(
            dataset=self.glonet_data,
            lead=1,
        )
        oceanbench.plot.plot_mld(dataset=dataset)

    def test_oceanbench_geo_analysis(self):
        """Test Geo."""
        dataset = oceanbench.process.calc_geo(
            dataset=self.glonet_data,
            lead=1,
            variable="zos",
        )
        oceanbench.plot.plot_geo(dataset=dataset)

    def test_oceanbench_density_analysis(self):
        """Test density."""
        dataarray = oceanbench.process.calc_density(
            dataset=self.glonet_data,
            lead=1,
            minimum_longitude=-100,
            maximum_longitude=-40,
            minimum_latitude=-15,
            maximum_latitude=50,
        )
        oceanbench.plot.plot_density(dataarray=dataarray)

    def test_oceanbench_euclid_dist_analysis(self):
        """Test density."""
        euclidean_distance = oceanbench.evaluate.get_euclidean_distance(
            first_dataset=self.glonet_data,
            second_dataset=self.glorys_data,
            minimum_latitude=466,
            maximum_latitude=633,
            minimum_longitude=400,
            maximum_longitude=466,
        )
        oceanbench.plot.plot_euclidean_distance(euclidean_distance)

    def test_oceanbench_energy_cascad_analysis(self):
        """Test energy cascading."""
        _, gglonet_sc = oceanbench.evaluate.analyze_energy_cascade(
            self.glonet_data, "uo", 0, 1 / 4
        )
        oceanbench.plot.plot_energy_cascade(gglonet_sc)

    def test_oceanbench_kinetic_energy_analysis(self):
        """Test kinetic energy."""
        oceanbench.plot.plot_kinetic_energy(self.glonet_data)

    def test_oceanbench_vorticity_analysis(self):
        """Test vorticity."""
        oceanbench.plot.plot_vortocity(self.glonet_data)

    def test_oceanbench_mass_conservation_analysis(self):
        """Test mass conservation."""
        mean_div_time_series = oceanbench.process.mass_conservation(
            self.glonet_data, 0, deg_resolution=0.25
        )  # should be close to zero
        print(mean_div_time_series.data)  # time-dependent scores


if __name__ == "__main__":
    unittest.main()
