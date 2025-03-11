#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Tests for the Mercator Oceanbench library."""

import os

import boto3
from botocore import UNSIGNED
from botocore.config import Config
from pathlib import Path
import pytest

from dctools.dcio.loader import DataLoader
from dctools.dcio.saver import DataSaver
from dctools.dcio.dclogger import DCLogger
from dctools.metrics.oceanbench_metrics import OceanbenchMetrics
from dctools.utilities.errors import DCExceptionHandler
from dctools.utilities.file_utils import get_list_filter_files
from dctools.utilities.net_utils import download_s3_file, S3Url, CMEMSManager
from dctools.processing.cmems_data import create_glorys_ndays_forecast


class TestVars:
    """Class with variables for testing."""

    glonet_dir = os.path.join("tests", "data", "glonet")
    glorys_dir = os.path.join("tests", "data", "glorys")
    glonet_data = None
    glorys_data = None

    start_date = "2024-05-01"
    filename = start_date + '.nc'
    test_file_path = os.path.join(glonet_dir, filename)
    glorys_file_path = os.path.join(glorys_dir, filename)

    test_url = "s3://project-glonet/public/glonet_reforecast_2024/2024-05-01.nc"
    edito_url = "https://minio.dive.edito.eu"
    dclogger = None
    logfile = os.path.join("tests", "test.log")


class TestOceanBench:
    """Unit tests for Mercator's oceanbench library."""

    def init_log_exc(self, test_vars):
        """Initialize logger and exception handler."""
        # initialize_logger
        test_vars.dclogger = DCLogger(
            name="Test Logger", logfile=test_vars.logfile
        ).get_logger()
        # initialize exception handler
        test_vars.exception_handler = DCExceptionHandler(test_vars.dclogger)
        test_vars.dclogger.info("Tests started.")
        test_vars.oceanbench_metrics = OceanbenchMetrics(
            test_vars.dclogger, test_vars.exception_handler
        )

    @pytest.fixture(scope="class")
    def test_vars(self):
        """Get test variables."""
        c_v = TestVars()
        yield c_v

    @pytest.fixture(autouse=True, scope="class")
    def create_datasets(self, test_vars):
        """Create Glorys and Glonet datasets for testing."""
        self.init_log_exc(test_vars)
        test_vars.dclogger.info("Creating tests datasets.")
        os.makedirs(test_vars.glonet_dir, exist_ok=True)
        os.makedirs(test_vars.glorys_dir, exist_ok=True)
        n_days_forecast = 10

        if not (Path(test_vars.test_file_path).is_file() and \
                Path(test_vars.glorys_file_path).is_file()):
            cmems_manager = CMEMSManager(
                test_vars.dclogger, test_vars.exception_handler
            )
            s3_client = boto3.client(
                "s3",
                config=Config(signature_version=UNSIGNED),
                endpoint_url=test_vars.edito_url,
            )

        if not (Path(test_vars.test_file_path).is_file()):
            test_vars.dclogger.info("No Glonet forecast available." \
                "Starting download from CMEMS.")
            cmems_manager.cmems_login()
            s3_test_url = S3Url(test_vars.test_url)
            # equivalent to: aws  --no-sign-request  s3 \
            #      cp  s3://project-glonet/public/glonet_reforecast_2024/2024-01-03.nc \
            #     /home/k24aitmo/IMT/software/dc-tools/data/glonet/2024-01-03.nc
            download_s3_file(
                s3_client=s3_client,
                bucket_name=s3_test_url.bucket,
                file_name=s3_test_url.key,
                local_file_path=test_vars.test_file_path,
                dclogger=test_vars.dclogger,
                exception_handler=test_vars.exception_handler,
            )

        assert(Path(test_vars.test_file_path).is_file())
        test_vars.glonet_data = DataLoader.lazy_load_dataset(
            test_vars.test_file_path, test_vars.exception_handler
        )

        if not (Path(test_vars.glorys_file_path).is_file()):
            cmems_manager.cmems_login()
            list_mercator_files = get_list_filter_files(
                test_vars.glorys_dir,
                extension='.nc',
                regex="mercatorglorys",
                prefix=True,
            )
            if len(list_mercator_files) != n_days_forecast:
                test_vars.dclogger.info(
                    "Missing Glorys data. Starting download from CMEMS."
                )
                cmems_manager.cmems_download(
                    product_id="cmems_mod_glo_phy_myint_0.083deg_P1D-m",
                    output_dir=test_vars.glorys_dir,
                    filter="*/2024/05/*_2024050[1-9]_*.nc",
                )
                cmems_manager.cmems_download(
                    product_id="cmems_mod_glo_phy_myint_0.083deg_P1D-m",
                    output_dir=test_vars.glorys_dir,
                    filter="*/2024/05/*_20240510_*.nc",
                )
                list_mercator_files = get_list_filter_files(
                    test_vars.glorys_dir,
                    extension='.nc',
                    regex="mercatorglorys",
                    prefix=True,
                )
            glorys_data = create_glorys_ndays_forecast(
                test_vars.glorys_dir,
                list_mercator_files,
                test_vars.glonet_data,
                test_vars.start_date,
                test_vars.dclogger,
                test_vars.exception_handler
            )

            test_vars.dclogger.info("Save Glorys data on disk")
            DataSaver.save_dataset(
                glorys_data, test_vars.glorys_file_path, test_vars.exception_handler
            )
            glorys_data.close()

        assert(Path(test_vars.glorys_file_path).is_file())
        test_vars.glorys_data = DataLoader.lazy_load_dataset(
            test_vars.glorys_file_path, test_vars.exception_handler
        )

    def test_oceanbench_rmse_evaluation(self, test_vars):
        """Test RMSE."""
        test_vars.oceanbench_metrics.compute_metric(
            'rmse', test_vars.glonet_data, test_vars.glorys_data, True
        )

    def test_oceanbench_mld_analysis(self, test_vars):
        """Test MLD."""
        test_vars.oceanbench_metrics.compute_metric(
            'mld', test_vars.glonet_data, True
        )

    def test_oceanbench_geo_analysis(self, test_vars):
        """Geo analysis."""
        test_vars.oceanbench_metrics.compute_metric(
            'geo', test_vars.glonet_data, True
        )

    def test_oceanbench_density_analysis(self, test_vars):
        """Test density."""
        test_vars.oceanbench_metrics.compute_metric(
            'density', test_vars.glonet_data, True
        )

    def test_oceanbench_euclid_dist_analysis(self, test_vars):
        """Euclid dist analysis."""
        test_vars.oceanbench_metrics.compute_metric(
            'euclid_dist', test_vars.glonet_data, test_vars.glorys_data, True
        )

    def test_oceanbench_energy_cascad_analysis(self, test_vars):
        """Test energy cascading."""
        test_vars.oceanbench_metrics.compute_metric(
            'energy_cascad', test_vars.glonet_data, True
        )


    def test_oceanbench_kinetic_energy_analysis(self, test_vars):
        """Test kinetic energy."""
        test_vars.oceanbench_metrics.compute_metric(
            'kinetic_energy', test_vars.glonet_data, True
        )


    def test_oceanbench_vorticity_analysis(self, test_vars):
        """Test vorticity."""
        test_vars.oceanbench_metrics.compute_metric(
            'vorticity', test_vars.glonet_data
        )

    def test_oceanbench_mass_conservation_analysis(self, test_vars):
        """Test mass conservation."""
        test_vars.oceanbench_metrics.compute_metric(
            'mass_conservation', test_vars.glonet_data
        )

