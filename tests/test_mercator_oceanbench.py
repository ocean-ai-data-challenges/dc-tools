#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Tests for the Mercator Oceanbench library."""

from types import SimpleNamespace
import os

#import numpy as np
import pytest

from dctools.data.dataloader import DatasetLoader
from dctools.data.dataset import CmemsGlorysDataset, GlonetDataset
#from dctools.dcio.loader import FileLoader
#from dctools.dcio.saver import DataSaver
from dctools.dcio.dclogger import DCLogger
from dctools.metrics.evaluator import Evaluator
from dctools.metrics.metrics import MetricComputer
from dctools.data.transforms import CustomTransforms
from dctools.utilities.init_dask import setup_dask
from dctools.utilities.errors import DCExceptionHandler
from dctools.utilities.xarray_utils import DICT_RENAME_CMEMS,\
    LIST_VARS_GLONET, RANGES_GLONET, GLONET_DEPTH_VALS


class TestOceanBench:
    """Unit tests for Mercator's oceanbench library."""

    def init_log_exc(self, test_vars):
        """Initialize logger and exception handler."""
        # initialize_logger
        test_vars.logger_instance = DCLogger(
            name="Test Logger", logfile=test_vars.logfile,
            jsonfile=test_vars.jsonfile
        )
        test_vars.dclogger = test_vars.logger_instance.get_logger()
        test_vars.json_logger = test_vars.logger_instance.get_json_logger()
        # initialize exception handler
        test_vars.exception_handler = DCExceptionHandler(test_vars.dclogger)
        test_vars.dclogger.info("Tests started.")

    @pytest.fixture(scope="class")
    def test_vars(self):
        """Get test variables."""
        startdate = "2024-05-01"
        list_dates = []
        list_dates.append(startdate)
        params_dir = os.path.join("weights")
        glonetdir = os.path.join("tests", "data", "glonet")
        glorysdir = os.path.join("tests", "data", "glorys")
        glonet_file_name = startdate + '.nc'
        glorys_file_name = startdate + '.zarr'
        c_v = SimpleNamespace(
            glonet_data_dir = glonetdir,
            glorys_data_dir = glorysdir,
            glonet_data = None,
            glorys_data = None,
            model_name = "glonet",
            start_date = startdate,
            #filename = file_name,
            regridder_weights = os.path.join(params_dir, "weights"),
            test_file_path = os.path.join(glonetdir, glonet_file_name),
            glorys_file_path = os.path.join(glorysdir, glorys_file_name),

            # test_url = "s3://project-glonet/public/glonet_reforecast_2024/2024-05-01.nc",
            glonet_base_url = "https://minio.dive.edito.eu",
            glonet_s3_bucket = "project-glonet",
            s3_glonet_folder = "public/glonet_reforecast_2024",
            glorys_cmems_product_name = "cmems_mod_glo_phy_myint_0.083deg_P1D-m",
            glonet_n_days_forecast = 10,
            list_glonet_start_dates = list_dates,
            logger_instance=None,
            dclogger = None,
            json_logger = None,
            logfile = os.path.join("tests", "logs", "test.log"),
            jsonfile = os.path.join("tests", "logs", "results2.json"),
        )
        yield c_v

    @pytest.fixture(autouse=True, scope="class")
    def create_datasets(self, test_vars):
        os.makedirs(os.path.join("tests", "logs"), exist_ok=True)
        """Create Glorys and Glonet datasets for testing."""
        self.init_log_exc(test_vars)
        test_vars.dclogger.info("Creating tests datasets.")

    @pytest.fixture(autouse=True, scope="class")
    def setup_evaluator(self, test_vars):
        dask_client = setup_dask(test_vars)
        glonet_data_dir = test_vars.glonet_data_dir
        glorys_data_dir = test_vars.glorys_data_dir
 
        transf_glorys = CustomTransforms(
            transform_name="glorys_to_glonet",
            dict_rename=DICT_RENAME_CMEMS,
            list_vars=LIST_VARS_GLONET,
            depth_coord_vals=GLONET_DEPTH_VALS,
            interp_ranges = RANGES_GLONET,
            weights_path=test_vars.regridder_weights,
        )

        dataset_glonet = GlonetDataset(
            conf_args=test_vars,
            root_data_dir= glonet_data_dir,
            list_dates=test_vars.list_glonet_start_dates,
            transform_fct=None,
        )

        dataset_glorys = CmemsGlorysDataset(
            conf_args=test_vars,
            root_data_dir= glorys_data_dir,
            cmems_product_name=test_vars.glorys_cmems_product_name,
            cmems_file_prefix="mercatorglorys",
            list_dates=test_vars.list_glonet_start_dates,
            transform_fct=transf_glorys,
            save_after_preprocess=False,
            file_format="zarr",
        )
        # 1. Chargement des données de référence et des prédictions avec DatasetLoader
        glonet_vs_glorys_loader = DatasetLoader(
            pred_dataset=dataset_glonet,
            ref_dataset=dataset_glorys
        )

        # 3. Exécution de l’évaluation sur plusieurs modèles
        evaluator = Evaluator(
            test_vars, 
            dask_client=dask_client, metrics=None,
            data_container={'glonet': glonet_vs_glorys_loader},
        )
        vars(test_vars)['evaluator'] = evaluator
        vars(test_vars)['dataloader'] = glonet_vs_glorys_loader

    def test_oceanbench_metrics(self, test_vars):
        """Test MLD."""
        test_vars.dclogger.info("Test Mercator's Oceanbench Metrics.")

        metrics = [
            MetricComputer(
                dc_logger=test_vars.dclogger,
                exc_handler=test_vars.exception_handler,
                metric_name='rmse', plot_result=True,
            ),

            MetricComputer(
                dc_logger=test_vars.dclogger,
                exc_handler=test_vars.exception_handler,
                metric_name='energy_cascad',
                plot_result=True,
                var="uo", depth=2,
            ),
        ]
        ''' TODO : check error on oceanbench : why depth = 0 ? -> crash
            MetricComputer(
                dc_logger=test_vars.dclogger,
                exc_handler=test_vars.exception_handler,
                metric_name='euclid_dist',
                plot_result=True,
                minimum_latitude=0,
                maximum_latitude=10,
                minimum_longitude=0,
                maximum_longitude=10,
            ),'''
        test_vars.evaluator.set_metrics(metrics)
        results = test_vars.evaluator.evaluate()
        test_vars.dclogger.info(f"Computed Metrics: {results}.")

