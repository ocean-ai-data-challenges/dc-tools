import os

import boto3
from botocore import UNSIGNED
from botocore.config import Config
import copernicusmarine
from pathlib import Path
import pytest
import xarray as xr
import xesmf as xe
import wget

from dctools.dcio.loader import DataLoader
from dctools.dcio.saver import DataSaver
from dctools.processing.gridded_data import GriddedDataProcessor
from dctools.third_party.mercator_oceanbench import oceanbench_plotting_funcs, oceanbench_processing_funcs, oceanbench_evaluate_funcs
from dctools.utilities.file_utils import list_files_with_extension
from dctools.utilities.xarray_utils import rename_coordinates, DICT_RENAME_CMEMS
from dctools.utilities.net_utils import download_s3_file, S3Url



class TestVars:
    """Class with variables for testing."""

    glonet_dir = os.path.join("data", "glonet")
    glorys_dir = os.path.join("data", "glorys")
    glonet_data = None
    glorys_data = None

    test_file_path = os.path.join(glonet_dir, "2024-01-03.nc")
    tmp_ref_file_path = os.path.join(glorys_dir, "2024-01-03_tmp.nc")
    ref_file_path = os.path.join(glorys_dir, "2024-01-03.nc")
    s3_client = None
    cmems_credentials = ""
    # initialized = False
    
    # Oceanbench functions
    oceanbench_processing_funcs = oceanbench_processing_funcs()
    oceanbench_plotting_funcs = oceanbench_plotting_funcs()
    oceanbench_evaluate_funcs = oceanbench_evaluate_funcs()


class TestOceanBench:
    """Unit tests for Mercator's oceanbench library."""     
        
    @pytest.fixture(scope="class")
    def test_vars(self):
        c_v = TestVars()
        yield c_v
        

    # @classmethod
    def cmems_login(self, test_vars):
        home_path = Path.home()
        test_vars.cmems_credentials = os.path.join(
            home_path,
            ".copernicusmarine/",
            ".copernicusmarine-credentials",
        )
        if not (Path(test_vars.cmems_credentials).is_file()):
            copernicusmarine.login()
        test_vars.s3_client = boto3.client(
            "s3",
            config=Config(signature_version=UNSIGNED),
            endpoint_url="https://minio.dive.edito.eu",
        )

    # @classmethod
    @pytest.fixture(autouse=True, scope="class")
    def create_datasets(self, test_vars):
        os.makedirs(test_vars.glonet_dir, exist_ok=True)
        os.makedirs(test_vars.glorys_dir, exist_ok=True)
        print('START TEST')
        n_days_forecast = 10
        # aws  --no-sign-request  s3 cp  s3://project-glonet/public/glonet_reforecast_2024/2024-01-03.nc /home/k24aitmo/IMT/software/dc-tools/data/glonet/2024-01-03.nc
        if not (Path(test_vars.test_file_path).is_file()):
            self.cmems_login(test_vars)
            test_url = "s3://project-glonet/public/glonet_reforecast_2024/2024-05-01.nc"
            s3_test_url = S3Url(test_url)
            download_s3_file(
                test_vars.s3_client,
                s3_test_url.bucket,
                s3_test_url.key,
                test_vars.test_file_path,
            )

        assert(Path(test_vars.test_file_path).is_file())
        #glonet_data = DataLoader.load_dataset(cls.test_file_path)
        self.glonet_data = DataLoader.lazy_load_dataset(test_vars.test_file_path)

        if not (Path(test_vars.tmp_ref_file_path).is_file()):
            self.cmems_login(test_vars)
            list_nc_files = list_files_with_extension(test_vars.glorys_dir, ".nc")
            list_mercator_files = [ncf for ncf in list_nc_files if ncf.startswith("mercatorglorys")]
            if len(list_mercator_files) != n_days_forecast:
                copernicusmarine.get(
                    # dataset_id="cmems_mod_glo_phy_my_0.083deg_P1D-m",
                    dataset_id="cmems_mod_glo_phy_myint_0.083deg_P1D-m",
                    filter="*/2024/05/*_2024050[1-9]_*.nc",
                    output_directory=test_vars.glorys_dir,
                    no_directories = True,
                    credentials_file=test_vars.cmems_credentials,
                )
                copernicusmarine.get(
                    # dataset_id="cmems_mod_glo_phy_my_0.083deg_P1D-m",
                    dataset_id="cmems_mod_glo_phy_myint_0.083deg_P1D-m",
                    filter="*/2024/05/*_20240510_*.nc",
                    output_directory=test_vars.glorys_dir,
                    no_directories = True,
                    credentials_file=test_vars.cmems_credentials,
            )
            list_nc_files = list_files_with_extension(test_vars.glorys_dir, ".nc")
            ref_data = xr.Dataset()
            time_step = 0
            for fname in list_nc_files:
                fpath = os.path.join(test_vars.glorys_dir, fname)
                tmp_ds = DataLoader.lazy_load_dataset(fpath)
                # print("TIME: ",  tmp_ds.coords['time'])
                tmp_ds = tmp_ds.drop_vars(
                    ["bottomT", "usi", "vsi", "mlotst", "siconc", "sithick"]
                )
                tmp_ds = tmp_ds.assign_coords(time=[time_step])
                tmp_ds['time'].attrs = {"units": "days since 2024-01-04 00:00:00", "calendar": "proleptic_gregorian"}
                if time_step == 0:
                    ref_data = tmp_ds
                else:
                    ref_data = GriddedDataProcessor.concatenate(
                        ref_data, tmp_ds, dim='time'
                    )
                tmp_ds.close()
                time_step += 1

            ref_data = xr.open_mfdataset(os.path.join(test_vars.glorys_dir, 'mercatorglorys*.nc'), parallel=True)
            time_attrs = {'units': "days since 2024-01-04 00:00:00", 'calendar': "proleptic_gregorian"}
            ref_data = ref_data.assign_coords({'time': ('time', [i for i in range(0,n_days_forecast)], time_attrs)})
            # ref_data = ref_data.assign_coords(time=[i for i in range(0,n_days_forecast)])
            ref_data = rename_coordinates(ref_data, DICT_RENAME_CMEMS)
            DataSaver.save_dataset(ref_data, test_vars.tmp_ref_file_path)
            ref_data.close()

        test_vars.glonet_data = DataLoader.lazy_load_dataset(test_vars.test_file_path)
        if not (Path(test_vars.ref_file_path).is_file()):
            print('NO REF FILE')
            assert(Path(test_vars.tmp_ref_file_path).is_file())
            test_vars.glorys_data = DataLoader.lazy_load_dataset(test_vars.tmp_ref_file_path)
            depth_vals = test_vars.glonet_data.coords['depth'].values

            depth_indices = [idx for idx in range(0, test_vars.glorys_data.depth.values.size) if test_vars.glorys_data.depth.values[idx] in depth_vals]
            test_vars.glorys_data = test_vars.glorys_data.isel(
                depth=depth_indices
            )

            regridder = xe.Regridder(
                test_vars.glorys_data, test_vars.glonet_data, method='bilinear', unmapped_to_nan=True
            )
            test_vars.glorys_data = regridder(test_vars.glorys_data)
            DataSaver.save_dataset(test_vars.glorys_data, test_vars.ref_file_path)
            test_vars.glorys_data = DataLoader.lazy_load_dataset(test_vars.ref_file_path)
        else:
            test_vars.glorys_data = DataLoader.lazy_load_dataset(test_vars.ref_file_path)

    def download_file(self, url, path):
        wget.download(url, out=path)

    def load_dataset(self, filepath):
        """dataset loading."""
        loaded_ds = DataLoader.load_dataset(filepath)
        self.assertIn("lon", loaded_ds.variables)
        return loaded_ds

    def test_oceanbench_rmse_evaluation(self, test_vars):
        """Test RMSE."""
        print("Test RMSE.")
        nparray = test_vars.oceanbench_evaluate_funcs.pointwise_evaluation(
            glonet_datasets=[test_vars.glonet_data],
            glorys_datasets=[test_vars.glorys_data],
        )
        test_vars.oceanbench_plotting_funcs.plot_pointwise_evaluation(rmse_dataarray=nparray, depth=2)

        test_vars.oceanbench_plotting_funcs.plot_pointwise_evaluation_for_average_depth(
            rmse_dataarray=nparray
        )

        test_vars.oceanbench_plotting_funcs.plot_pointwise_evaluation_depth_for_average_time(
            rmse_dataarray=nparray, dataset_depth_values=test_vars.glonet_data.depth.values
        )

    def test_oceanbench_mld_analysis(self, test_vars):
        """Test MLD."""
        print('Test MLD.')
        dataset = test_vars.oceanbench_processing_funcs.calc_mld(
            dataset=test_vars.glonet_data.compute(),
            lead=1,
        )
        test_vars.oceanbench_plotting_funcs.plot_mld(dataset=dataset)

    def test_oceanbench_geo_analysis(self, test_vars):
        """Geo analysis."""
        print("Test Geo analysis.")
        dataset = test_vars.oceanbench_processing_funcs.calc_geo(
            dataset=test_vars.glonet_data,
            lead=1,
            variable="zos",
        )
        test_vars.oceanbench_plotting_funcs.plot_geo(dataset=dataset)

    def test_oceanbench_density_analysis(self, test_vars):
        """Test density."""
        print("Test density.")
        dataarray = test_vars.oceanbench_processing_funcs.calc_density(
            dataset=test_vars.glonet_data,
            lead=1,
            minimum_longitude=-100,
            maximum_longitude=-40,
            minimum_latitude=-15,
            maximum_latitude=50,
        )
        test_vars.oceanbench_plotting_funcs.plot_density(dataset=dataarray)

    def test_oceanbench_euclid_dist_analysis(self, test_vars):
        """Euclid dist analysis."""
        print("Euclid dist analysis.")
        euclidean_distance = test_vars.oceanbench_evaluate_funcs.get_euclidean_distance(
            first_dataset=test_vars.glonet_data,
            second_dataset=test_vars.glorys_data,
            minimum_latitude=466,
            maximum_latitude=633,
            minimum_longitude=400,
            maximum_longitude=466,
        )
        test_vars.oceanbench_plotting_funcs.plot_euclidean_distance(euclidean_distance)

    def test_oceanbench_energy_cascad_analysis(self, test_vars):
        """Test energy cascading."""
        print("Test energy cascading.")
        _, gglonet_sc = test_vars.oceanbench_evaluate_funcs.analyze_energy_cascade(
            test_vars.glonet_data, "uo", 0, 1 / 4
        )
        test_vars.oceanbench_plotting_funcs.plot_energy_cascade(gglonet_sc)

    def test_oceanbench_kinetic_energy_analysis(self, test_vars):
        """Test kinetic energy."""
        print("Test kinetic energy.")
        test_vars.oceanbench_plotting_funcs.plot_kinetic_energy(test_vars.glonet_data)

    def test_oceanbench_vorticity_analysis(self, test_vars):
        """Test vorticity."""
        print("Test vorticity.")
        test_vars.oceanbench_plotting_funcs.plot_vorticity(test_vars.glonet_data)

    def test_oceanbench_mass_conservation_analysis(self, test_vars):
        """Test mass conservation."""
        print("Test mass conservation.")
        mean_div_time_series = test_vars.oceanbench_processing_funcs.mass_conservation(
            test_vars.glonet_data, 0, deg_resolution=0.25
        )  # should be close to zero
        print(mean_div_time_series.data)  # time-dependent scores