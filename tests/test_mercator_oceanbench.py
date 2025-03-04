from pathlib import Path

import pytest
from pyunpack import Archive
import xarray as xr
import wget
from urllib.error import URLError

from dctools.dcio.loader import DataLoader
from dctools.third_party.mercator_oceanbench import oceanbench_plotting
import oceanbench


def load_dataset(filepath):
    """Dataset loading."""
    loaded_ds = DataLoader.load_dataset(filepath)
    assert isinstance(loaded_ds, xr.Dataset)
    assert "depth" in loaded_ds.variables
    return loaded_ds

@pytest.fixture(scope="module")
def setup_glorys_data():
    """Setup GLORYS data."""
    glorys_dir = Path("data") / "glorys"
    ref_file_archive = glorys_dir / "2024-01-03.7z"
    ref_file_path = glorys_dir / "2024-01-03.nc"

    # Get data
    url_ref = "ftp://project-oceanbench-708263-0.lab.dive.edito.eu/lab/tree/data/glorys14/2024-01-03.nc"
    if not ref_file_path.is_file():
        print("No existing GloNet data found.")
        if ref_file_archive.is_file():
            print("Found GloNet archive. Extracting...")
            Archive(ref_file_archive).extractall(glorys_dir)
            print("Done!")
        else:
            print("Downloading GloNet data...")
            try:
                wget.download(url_ref, out=str(ref_file_path))
                print("Done!")
            except URLError:
                print("Problem downloading file.")
    
    assert ref_file_path.is_file()
    glorys_data = load_dataset(ref_file_path)
    yield glorys_data

@pytest.fixture(scope="module")
def setup_glonet_data():
    """Setup GloNet data."""
    glonet_dir = Path("data") / "glonet"
    test_file_archive = glonet_dir / "2024-01-03.7z"
    test_file_path = glonet_dir / "2024-01-03.nc"

    # Get data
    url_test = "ftp://project-oceanbench-708263-0.lab.dive.edito.eu/lab/tree/data/glonet/2024-01-03.nc"
    if not test_file_path.is_file():
        print("No existing GloNet data found.")
        if test_file_archive.is_file():
            print("Found GloNet archive. Extracting...")
            Archive(test_file_archive).extractall(glonet_dir)
            print("Done!")
        else:
            print("Downloading GloNet data...")
            try:
                wget.download(url_test, out=str(test_file_path))
                print("Done!")
            except URLError:
                print("Problem downloading file.")
    
    assert test_file_path.is_file()
    glonet_data = load_dataset(test_file_path)
    yield glonet_data


# Testing functions
# ===================================================================

def test_oceanbench_rmse_evaluation(
    setup_glonet_data,
    setup_glorys_data,
    ):
    """Test RMSE."""
    nparray = oceanbench.evaluate.pointwise_evaluation(
        glonet_datasets=[setup_glonet_data],
        glorys_datasets=[setup_glorys_data],
    )
    # plot_file = os.path.join(self.glonet_dir, "plot1.png")
    oceanbench_plotting.plot_pointwise_evaluation(rmse_dataarray=nparray, depth=2)

    #  plot_file = os.path.join(self.glonet_dir, "plot2.png")
    oceanbench_plotting.plot_pointwise_evaluation_for_average_depth(
        rmse_dataarray=nparray
    )

    # plot_file = os.path.join(self.glonet_dir, "plot3.png")
    oceanbench_plotting.plot_pointwise_evaluation_depth_for_average_time(
        rmse_dataarray=nparray,
        dataset_depth_values=setup_glonet_data.depth.values
    )

def test_oceanbench_mld_analysis(setup_glonet_data):
    """Test MLD."""
    dataset = oceanbench.process.calc_mld(
        dataset=setup_glonet_data,
        lead=1,
    )
    oceanbench.plot.plot_mld(dataset=dataset)

def test_oceanbench_geo_analysis(setup_glonet_data):
    """Test Geo."""
    dataset = oceanbench.process.calc_geo(
        dataset=setup_glonet_data,
        lead=1,
        variable="zos",
    )
    oceanbench.plot.plot_geo(dataset=dataset)

def test_oceanbench_density_analysis(setup_glonet_data):
    """Test density."""
    dataarray = oceanbench.process.calc_density(
        dataset=setup_glonet_data,
        lead=1,
        minimum_longitude=-100,
        maximum_longitude=-40,
        minimum_latitude=-15,
        maximum_latitude=50,
    )
    oceanbench.plot.plot_density(dataarray=dataarray)

def test_oceanbench_euclid_dist_analysis(
    setup_glonet_data,
    setup_glorys_data,
    ):
    """Test density."""
    euclidean_distance = oceanbench.evaluate.get_euclidean_distance(
        first_dataset=setup_glonet_data,
        second_dataset=setup_glorys_data,
        minimum_latitude=466,
        maximum_latitude=633,
        minimum_longitude=400,
        maximum_longitude=466,
    )
    oceanbench.plot.plot_euclidean_distance(euclidean_distance)

def test_oceanbench_energy_cascad_analysis(setup_glonet_data):
    """Test energy cascading."""
    _, gglonet_sc = oceanbench.evaluate.analyze_energy_cascade(
        setup_glonet_data, "uo", 0, 1 / 4
    )
    oceanbench.plot.plot_energy_cascade(gglonet_sc)

def test_oceanbench_kinetic_energy_analysis(setup_glonet_data):
    """Test kinetic energy."""
    oceanbench.plot.plot_kinetic_energy(setup_glonet_data)

def test_oceanbench_vorticity_analysis(setup_glonet_data):
    """Test vorticity."""
    oceanbench.plot.plot_vortocity(setup_glonet_data)

def test_oceanbench_mass_conservation_analysis(setup_glonet_data):
    """Test mass conservation."""
    mean_div_time_series = oceanbench.process.mass_conservation(
        setup_glonet_data, 0, deg_resolution=0.25
    )  # should be close to zero
    print(mean_div_time_series.data)  # time-dependent scores