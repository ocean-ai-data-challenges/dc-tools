
import os
import pytest
from types import SimpleNamespace

from datetime import datetime
from loguru import logger
import numpy as np
import pandas as pd
import xarray as xr

from dctools.data.connection.config import (
    S3ConnectionConfig,
    WasabiS3ConnectionConfig,
    CMEMSConnectionConfig,
    LocalConnectionConfig,
    GlonetConnectionConfig
)
#from dctools.data.connection.connection_manager import (
#    S3Manager,
#    CMEMSManager,
#)
#from dctools.data.datasets.factory import DatasetFactory
from dctools.data.datasets.dataset import RemoteDataset
from dctools.data.datasets.dataset import DatasetConfig
from dctools.data.datasets.dataset import RemoteDataset, LocalDataset
#from dctools.data.datasets.dataloader import EvaluationDataloader
from dctools.data.datasets.dataset_manager import MultiSourceDatasetManager
#from dctools.data.datasets.dc_catalog import DatasetCatalog
from dctools.data.transforms import CustomTransforms
from dctools.metrics.evaluator import Evaluator
from dctools.metrics.metrics import MetricComputer
#from dctools.processing.cmems_data import extract_dates_from_filename
from dctools.utilities.init_dask import setup_dask
from dctools.utilities.file_utils import load_config_file
from dctools.utilities.xarray_utils import (
    DICT_RENAME_CMEMS,
    LIST_VARS_GLONET,
    RANGES_GLONET,
    GLONET_DEPTH_VALS,
)


class TestPipeline:

    @pytest.fixture(scope="class")
    def test_config(self):
        """Fixture pour configurer les variables de test."""
        config_file = os.path.join("tests", "config", "test_config.yaml")

        # Charger le fichier YAML
        config = load_config_file(config_file)

        # Ajouter des chemins spécifiques pour les tests
        config["glonet_data_dir"] = os.path.join("tests", "data", "glonet")
        config["glorys_data_dir"] = os.path.join("tests", "data", "glorys")
        config["regridder_weights"] = os.path.join("tests", "data", "weights")
        if os.path.exists(config["regridder_weights"]):
            os.remove(config["regridder_weights"])

        return SimpleNamespace(**config)

    @pytest.fixture(scope="class")
    def setup_datasets(self, test_config):
        """Fixture pour configurer les datasets."""

        glorys_config = DatasetConfig(
            name="glorys",
            connection_config=CMEMSConnectionConfig(
                local_root=test_config.glorys_data_dir,
                dataset_id=test_config.glorys_cmems_product_name,
                max_samples=test_config.max_samples,
            ),
        )
       # Configuration des datasets
        '''glonet_local_config = DatasetConfig(
            name="glonet_local",
            connection_config=LocalConnectionConfig(
                local_root=test_config.glonet_local_dir,
                max_samples=test_config.max_samples,
            ),
        )'''

        glonet_config = DatasetConfig(
            name="glonet",
            connection_config=GlonetConnectionConfig(
                local_root=test_config.glonet_data_dir,
                endpoint_url=test_config.glonet_base_url,
                max_samples=test_config.max_samples,
            ),
        )

        glonet_wasabi_config = DatasetConfig(
            name="glonet_wasabi",
            connection_config=WasabiS3ConnectionConfig(
                local_root=test_config.glonet_data_dir,
                bucket=test_config.wasabi_bucket,
                bucket_folder=test_config.wasabi_glonet_folder,
                key=test_config.wasabi_key,
                secret_key=test_config.wasabi_secret_key,
                endpoint_url=test_config.wasabi_endpoint_url,
                max_samples=test_config.max_samples,
            ),
        )
        # Création des datasets
        glorys_dataset = RemoteDataset(glorys_config)
        #glonet_local_dataset = LocalDataset(glonet_local_config)
        glonet_dataset = RemoteDataset(glonet_config)
        glonet_wasabi_dataset = RemoteDataset(glonet_wasabi_config)

        return {
            "glonet": glonet_dataset,
            #"glonet_local": glonet_local_dataset,
            "glonet_wasabi": glonet_wasabi_dataset,
            "glorys": glorys_dataset,
        }

    @pytest.fixture(scope="class")
    def setup_dataloader(self, setup_datasets, test_config):

        manager = MultiSourceDatasetManager()

        logger.debug(f"Setup datasets")
        # Ajouter les datasets avec des alias
        manager.add_dataset("glonet", setup_datasets["glonet"])
        manager.add_dataset("glorys", setup_datasets["glorys"])
        #manager.add_dataset("glonet_local", setup_datasets["glonet_local"])
        manager.add_dataset("glonet_wasabi", setup_datasets["glonet_wasabi"])

        # Construire le catalogue
        logger.debug(f"Build catalog")
        manager.build_catalogs()

        # Appliquer les filtres temporels
        #manager.filter_all_by_date(
        #    start=pd.to_datetime(test_config.start_times[0]),
        #    end=pd.to_datetime(test_config.end_times[0]),
        #)
        # Appliquer les filtres spatiaux
        manager.filter_all_by_bbox(
            bbox=(test_config.min_lon, test_config.min_lat, test_config.max_lon, test_config.max_lat)
        )
        # Appliquer les filtres sur les variables
        manager.filter_all_by_variable(variables=test_config.target_vars)

        manager.all_to_json(output_dir=os.path.join("tests", "data"))


        glonet_transform = CustomTransforms(
            transform_name="glorys_to_glonet",
            weights_path=test_config.regridder_weights,
            depth_coord_vals=GLONET_DEPTH_VALS,
            interp_ranges=RANGES_GLONET,
        )
        # Configurer les transformations
        """pred_transform = CustomTransforms(
            transform_name="rename_subset_vars",
            dict_rename={"longitude": "lon", "latitude": "lat"},
            list_vars=["uo", "vo", "zos"],
        )

        ref_transform = CustomTransforms(
            transform_name="interpolate",
            interp_ranges={"lat": np.arange(-10, 10, 0.25), "lon": np.arange(-10, 10, 0.25)},
        )"""

        # Créer un dataloader
        """dataloader = manager.get_dataloader(
            pred_alias="glonet",
            ref_alias="glorys",
            batch_size=8,
            pred_transform=glonet_transform,
            ref_transform=glonet_transform,
        )"""
        dataloader = manager.get_dataloader(
            pred_alias="glonet_wasabi",
            ref_alias=None,
            batch_size=8,
            pred_transform=None,
            ref_transform=None,
        )
        for batch in dataloader:
            # Vérifier que le batch contient les clés attendues
            assert "pred_data" in batch[0]
            assert "ref_data" in batch[0]
            # Vérifier que les données sont de type xarray.Dataset
            assert isinstance(batch[0]["pred_data"], str)  #xr.Dataset)
            #assert isinstance(batch[0]["ref_data"], xr.Dataset)
            # Vérifier que les dimensions sont correctes
            #logger.debug(f"Batch pred dims: {list(batch[0]['pred_data'].coords.keys())}")
            #logger.debug(f"Batch pred: {batch[0]['pred_data']}")
            #logger.debug(f"Batch ref: {batch[0]['ref_data']}")
            #assert set(list(batch[0]["pred_data"].dims)) == set(["time", "depth", "lat", "lon"])
            #assert set(list(batch[0]["ref_data"].dims)) == set(["time", "depth", "lat", "lon"])
        return dataloader

    @pytest.fixture(scope="class")
    def setup_evaluator(self, setup_dataloader, test_config):
        """Fixture pour configurer l'évaluateur."""
        dask_cluster = setup_dask(test_config)

        metrics = [
            MetricComputer(metric_name="rmse"),
            #MetricComputer(metric_name="euclid_dist"),
            #MetricComputer(metric_name="energy_cascad"),
        ]

        return Evaluator(
            dask_cluster=dask_cluster,
            metrics=metrics,
            dataloader=setup_dataloader,
        )


    def test_evaluation(self, setup_evaluator):
        """Test de l'évaluation des métriques."""
        results = setup_evaluator.evaluate()

        # Vérifier que les résultats existent
        assert len(results) > 0

        # Vérifier que chaque résultat contient les champs attendus
        for result in results:
            assert "date" in result
            assert "metric" in result
            assert "result" in result
        logger.info(f"Test Results: {results}")
