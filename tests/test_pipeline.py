
import os
import pytest
import sys
from types import SimpleNamespace

import geopandas as gpd
from loguru import logger
import pandas as pd
from shapely import geometry

from dctools.data.datasets.dataset import get_dataset_from_config
from dctools.data.datasets.dataloader import EvaluationDataloader
from dctools.data.datasets.dataset_manager import MultiSourceDatasetManager
from dctools.data.transforms import CustomTransforms
from dctools.metrics.evaluator import Evaluator
from dctools.metrics.metrics import MetricComputer
from dctools.utilities.init_dask import setup_dask
from dctools.utilities.file_utils import load_config_file
from dctools.data.coordinates import (
    RANGES_GLONET,
    GLONET_DEPTH_VALS,
)

class TestPipeline:

    @pytest.fixture(scope="class")
    def test_config(self):
        logger.remove()  # Supprime les handlers existants
        logger.add(sys.stderr, level="INFO")  # N'affiche que INFO et plus grave (masque DEBUG)
        """Fixture pour configurer les variables de test."""
        config_file = os.path.join("tests", "config", "test_config.yaml")

        # Charger le fichier YAML
        config = load_config_file(config_file)

        # Ajouter des chemins spécifiques pour les tests
        config["data_directory"] = os.path.join("tests", "data")
        config["glonet_data_dir"] = os.path.join("tests", "data", "glonet")
        config["glorys_data_dir"] = os.path.join("tests", "data", "glorys")
        config["catalog_dir"] = os.path.join("tests", "data")
        config["regridder_weights"] = os.path.join("tests", "data", "weights")


        # logger.debug(f"Test config loaded: {config.keys()}")
        keep_vars = {}
        for source in config["sources"]:
            source_name = source['dataset']
            keep_vars[source_name] = source['keep_variables']
        config["keep_vars"] = keep_vars

        if not os.path.exists(config["data_directory"]):
            os.mkdir(config["data_directory"])
        if not os.path.exists(config["catalog_dir"]):
            os.mkdir(config["catalog_dir"])
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
            f"https://minio.dive.edito.eu/project-glonet/public/glonet_refull_2024/20240103.zarr",
            engine="zarr",
        )
        logger.debug(f"ds_test: {ds_test}")'''

        for source in test_config.sources:
            source_name = source['dataset']
            file_pattern = source['file_pattern']
            logger.debug(f"\n\n\t\t\t\t\t**************  LOADING DATASET: {source_name}  ***************\n\n")
            if source_name == "glorys":
                glorys_dataset = get_dataset_from_config(
                    source,
                    test_config.data_directory,
                    test_config.catalog_dir,
                    test_config.max_samples,
                    file_pattern,
                    use_json_catalog,
                )

            elif source_name == "glonet":
                glonet_dataset = get_dataset_from_config(
                    source,
                    test_config.data_directory,
                    test_config.catalog_dir,
                    test_config.max_samples,
                    file_pattern,
                    use_json_catalog
                )
            elif source_name == "glonet_wasabi":
                glonet_wasabi_dataset = get_dataset_from_config(
                    source,
                    test_config.data_directory,
                    test_config.catalog_dir,
                    test_config.max_samples,
                    file_pattern,
                    use_json_catalog,
                )
            logger.debug(f"\n\n\t\t\t\t\t**************  LOADED DATASET: {source_name}  ***************\n\n")

        return {
            "glonet": glonet_dataset,
            #"glonet_local": glonet_local_dataset,
            "glonet_wasabi": glonet_wasabi_dataset,
            "glorys": glorys_dataset,
        }

    def filter_data(
        self, manager: MultiSourceDatasetManager,
        test_config: SimpleNamespace,
    ):
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

        )
        # Appliquer les filtres spatiaux
        manager.filter_all_by_region(
            region=filter_region
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
        manager.add_dataset("glonet", setup_datasets["glonet"])
        manager.add_dataset("glorys", setup_datasets["glorys"])
        #manager.add_dataset("glonet_local", setup_datasets["glonet_local"])
        manager.add_dataset("glonet_wasabi", setup_datasets["glonet_wasabi"])

        # Construire le catalogue
        logger.info("Build catalog")
        manager.build_catalogs()

        manager.all_to_json(output_dir=test_config.catalog_dir)
        manager = self.filter_data(manager, test_config)
        return manager


    @pytest.fixture(scope="function")
    def setup_transforms(
        self,
        test_config: SimpleNamespace,
        setup_datasets: dict,
    ):
        """Fixture pour configurer les transformations."""
        global_metadata = setup_datasets["glonet"].get_catalog().global_metadata
        logger.debug(f"Metadata variables: {global_metadata}")
        coords_rename_dict = global_metadata.get("dimensions")
        vars_rename_dict = global_metadata.get("variables_rename_dict")
        glonet_vars = test_config.keep_vars["glonet"]
        # Configurer les transformations
        glonet_transform = CustomTransforms(
            transform_name="glorys_to_glonet",
            weights_path=test_config.regridder_weights,
            depth_coord_vals=GLONET_DEPTH_VALS,
            interp_ranges=RANGES_GLONET,
        )

        pred_transform_glonet = CustomTransforms(
            transform_name="standardize_dataset",
            list_vars=glonet_vars,
            coords_rename_dict=coords_rename_dict,
            vars_rename_dict=vars_rename_dict,
        )

        """ref_transform = CustomTransforms(
            transform_name="interpolate",
            interp_ranges={"lat": np.arange(-10, 10, 0.25), "lon": np.arange(-10, 10, 0.25)},
        )"""
        return {"glonet": pred_transform_glonet}

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
        dataloader = setup_manager.get_dataloader(
            pred_alias="glonet",
            ref_alias=None,
            batch_size=test_config.batch_size,
            pred_transform=transform_glonet,
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
        add_noise = False  # True = create fake_results
        metrics = [
            MetricComputer(
                metric_name="lagrangian",
                add_noise=add_noise,
                eval_variables=setup_dataloader.eval_variables,
            ),
            MetricComputer(
                metric_name="rmsd_geostrophic_currents",
                add_noise=add_noise,
                eval_variables=setup_dataloader.eval_variables,
            ),
            MetricComputer(
                metric_name="rmsd_mld",
                add_noise=add_noise,
                eval_variables=setup_dataloader.eval_variables,
            ),
            MetricComputer(
                metric_name="rmsd",
                add_noise=add_noise,
                eval_variables=setup_dataloader.eval_variables,
            ),
        ]
        return Evaluator(
            dask_cluster=dask_cluster,
            metrics=metrics,
            dataloader=setup_dataloader,
            json_path=os.path.join(test_config.catalog_dir, "test_results.json"),
        )

    @pytest.mark.parametrize('use_json_catalog', [False, True])
    def test_evaluation(
        self,
        setup_evaluator: Evaluator,
    ):
        """Test de l'évaluation des métriques."""
        logger.debug(f"\n\n\t\t\t\t\t\t********************  STARTING EVALUATION  ********************\n\n")
        results = setup_evaluator.evaluate()
        logger.debug(f"\n\n\t\t\t\t\t\t********************  ENDED EVALUATION  ********************\n\n")

        # Vérifier que les résultats existent
        assert len(results) > 0

        # Vérifier que chaque résultat contient les champs attendus
        for result in results:
            assert "date" in result
            assert "metric" in result
            assert "result" in result
        for res in results:
            logger.info(f"Test Results: {res}")
