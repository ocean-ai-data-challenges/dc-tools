
import os
import pytest
from types import SimpleNamespace

from datetime import datetime
import geopandas as gpd
from loguru import logger
import numpy as np
import pandas as pd
from shapely import geometry
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
from dctools.data.datasets.dataloader import EvaluationDataloader
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
        config["catalog_dir"] = os.path.join("tests", "data")
        config["regridder_weights"] = os.path.join("tests", "data", "weights")
        if os.path.exists(config["regridder_weights"]):
            os.remove(config["regridder_weights"])
        return SimpleNamespace(**config)

    @pytest.fixture(scope="function")
    def setup_datasets(
        self, test_config: SimpleNamespace,
        use_json_catalog: bool,
    ):
        """Fixture pour configurer les datasets."""

        '''ds_test = xr.open_dataset(
            f"https://minio.dive.edito.eu/project-glonet/public/glonet_reforecast_2024/2024-01-03.zarr",
            engine="zarr",
        )
        logger.debug(f"ds_test: {ds_test}")'''

        glorys_dataset_name = "glorys"
        glonet_dataset_name = "glonet"
        glonet_wasabi_dataset_name = "glonet_wasabi"
        glorys_catalog_path = os.path.join(
            test_config.catalog_dir, glorys_dataset_name + ".json"
        )
        glonet_catalog_path = os.path.join(
            test_config.catalog_dir, glonet_dataset_name + ".json"
        )
        glonet_wasabi_catalog_path = os.path.join(
            test_config.catalog_dir, glonet_wasabi_dataset_name + ".json"
        )

        # Configurer les datasets
        # Glorys
        glorys_connection_config = CMEMSConnectionConfig(
            local_root=test_config.glorys_data_dir,
            dataset_id=test_config.glorys_cmems_product_name,
            max_samples=test_config.max_samples,
        )
        if os.path.exists(glorys_catalog_path) and use_json_catalog:
            # Load dataset metadata from catalog
            glorys_config = DatasetConfig(
                alias=glorys_dataset_name,
                connection_config=glorys_connection_config,
                catalog_options={"catalog_path": glorys_catalog_path}
            )
        else:
            # create dataset
            glorys_config = DatasetConfig(
                alias=glorys_dataset_name,
                connection_config=glorys_connection_config,
            )
        # Création du dataset
        glorys_dataset = RemoteDataset(glorys_config)

        # Glonet (source Wasabi)
        glonet_wasabi_connection_config = WasabiS3ConnectionConfig(
            local_root=test_config.glonet_data_dir,
            bucket=test_config.wasabi_bucket,
            bucket_folder=test_config.wasabi_glonet_folder,
            key=test_config.wasabi_key,
            secret_key=test_config.wasabi_secret_key,
            endpoint_url=test_config.wasabi_endpoint_url,
            max_samples=test_config.max_samples,
        )
        if os.path.exists(glonet_wasabi_catalog_path) and use_json_catalog:
            glonet_wasabi_config = DatasetConfig(
                alias=glonet_wasabi_dataset_name,
                connection_config=glonet_wasabi_connection_config,
                catalog_options={"catalog_path": glonet_wasabi_catalog_path},
            )
        else:
            glonet_wasabi_config = DatasetConfig(
                alias=glonet_wasabi_dataset_name,
                connection_config=glonet_wasabi_connection_config,        
            )
        glonet_wasabi_dataset = RemoteDataset(glonet_wasabi_config)


        # Glonet
        '''glonet_connection_config = GlonetConnectionConfig(
            local_root=test_config.glonet_data_dir,
            endpoint_url=test_config.glonet_base_url,
            glonet_s3_bucket=test_config.glonet_s3_bucket,
            s3_glonet_folder=test_config.s3_glonet_folder,
            max_samples=test_config.max_samples,
        )
        if os.path.exists(glonet_catalog_path) and use_json_catalog:
            glonet_config = DatasetConfig(
                alias=glorys_dataset_name,
                connection_config=glonet_connection_config,
                catalog_options={"catalog_path": glonet_catalog_path}
            )
        else:
            # create dataset
            glonet_config = DatasetConfig(
                alias=glonet_dataset_name,
                connection_config=glonet_connection_config,
            )
        glonet_dataset = RemoteDataset(glonet_config)'''


        '''glonet_local_config = DatasetConfig(
            name="glonet_local",
            connection_config=LocalConnectionConfig(
                local_root=test_config.glonet_local_dir,
                max_samples=test_config.max_samples,
            ),
        )'''
        #glonet_local_dataset = LocalDataset(glonet_local_config)

        return {
            #"glonet": glonet_dataset,
            #"glonet_local": glonet_local_dataset,
            "glonet_wasabi": glonet_wasabi_dataset,
            "glorys": glorys_dataset,
        }

    def filter_data(
        self, manager: MultiSourceDatasetManager,
        test_config: SimpleNamespace,
    ):
        #min_time = pd.to_datetime(test_config.start_times[0])
        #max_time = pd.to_datetime(test_config.end_times[0])
        filter_region = gpd.GeoSeries(geometry.Polygon((
            (test_config.min_lon,test_config.min_lat),
            (test_config.min_lon,test_config.max_lat),
            (test_config.max_lon,test_config.min_lat),
            (test_config.max_lon,test_config.max_lat),
            (test_config.min_lon,test_config.min_lat),
            )), crs="EPSG:4326")
        '''manager.filter_attrs(
            filters={
                "date_start": lambda dt: min_time <= dt,
                "date_end": lambda dt: dt < max_time,
                #  "coord_system": lambda c_t: c_t.coord_type in ("polar", "geographic"),
                "variables": lambda vars: var in test_config.target_vars for var in vars,
                "geometry": lambda reg: reg.intersects(filter_region),
            }
        )'''
        # Appliquer les filtres temporels
        manager.filter_all_by_date(
            start=pd.to_datetime(test_config.start_times[0]),
            end=pd.to_datetime(test_config.end_times[0]),
            #start=test_config.start_times[0],
            #end=test_config.end_times[0],
        )
        # Appliquer les filtres spatiaux
        manager.filter_all_by_region(
            region=filter_region #=(test_config.min_lon, test_config.min_lat, test_config.max_lon, test_config.max_lat)
        )
        # Appliquer les filtres sur les variables
        manager.filter_all_by_variable(variables=test_config.target_vars)
        return manager


    @pytest.fixture(scope="function")
    def setup_manager(
        self, setup_datasets: dict,
        test_config: SimpleNamespace,
    ):

        manager = MultiSourceDatasetManager()

        logger.info("Setup datasets manager")
        # Ajouter les datasets avec des alias
        # manager.add_dataset("glonet", setup_datasets["glonet"])
        manager.add_dataset("glorys", setup_datasets["glorys"])
        #manager.add_dataset("glonet_local", setup_datasets["glonet_local"])
        manager.add_dataset("glonet_wasabi", setup_datasets["glonet_wasabi"])

        # Construire le catalogue
        logger.info("Build catalog")
        manager.build_catalogs()

        manager.all_to_file(output_dir=test_config.catalog_dir)
        manager = self.filter_data(manager, test_config)
        return manager


    @pytest.fixture(scope="function")
    def setup_transforms(
        self, test_config: SimpleNamespace
    ):
        """Fixture pour configurer les transformations."""
        # Configurer les transformations
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
        return {"glonet": glonet_transform}

    @pytest.fixture(scope="function")
    def setup_dataloader(
        self,
        setup_transforms: dict,
        test_config: SimpleNamespace,
        setup_manager: MultiSourceDatasetManager,
    ):
        """Fixture pour configurer le dataloader avec les datasets."""
        transform_glonet = setup_transforms["glonet"]
        # Créer un dataloader
        """dataloader = manager.get_dataloader(
            pred_alias="glonet",
            ref_alias="glorys",
            batch_size=8,
            pred_transform=glonet_transform,
            ref_transform=glonet_transform,
        )"""
        dataloader = setup_manager.get_dataloader(
            pred_alias="glonet_wasabi",
            ref_alias=None,
            batch_size=test_config.batch_size,
            pred_transform=None,
            ref_transform=None,
        )

        # Vérifier le dataloader
        self.check_dataloader(dataloader)
        return dataloader



    def check_dataloader(
        self,
        dataloader: EvaluationDataloader,
    ):
        for batch in dataloader:
            # Vérifier que le batch contient les clés attendues
            assert "pred_data" in batch[0]
            assert "ref_data" in batch[0]
            # Vérifier que les données sont de type str (paths)
            assert isinstance(batch[0]["pred_data"], str)
            if batch[0]["ref_data"]:
                assert isinstance(batch[0]["ref_data"], str)

    @pytest.fixture(scope="function")
    def setup_evaluator(
        self,
        setup_dataloader: EvaluationDataloader,
        test_config: SimpleNamespace,
    ):
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

    @pytest.mark.parametrize('use_json_catalog', [False, True])
    def test_evaluation(
        self,
        setup_evaluator: Evaluator,
    ):
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


'''import folium
import geopandas as gpd

m = folium.Map(zoom_start=2)
for _, row in catalog.iterrows():
    geojson = gpd.GeoSeries([row.geometry]).to_json()
    folium.GeoJson(geojson, tooltip=row.path).add_to(m)
'''
