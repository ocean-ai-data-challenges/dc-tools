
import os
import pytest
import sys
from types import SimpleNamespace
from typing import List

import geopandas as gpd
from loguru import logger
import pandas as pd
from shapely import geometry
from torchvision import transforms

from tests.conftest import pytest_configure
from dctools.data.datasets.dataset import get_dataset_from_config
from dctools.data.datasets.dataloader import EvaluationDataloader
from dctools.data.datasets.dataset_manager import MultiSourceDatasetManager

from dctools.metrics.evaluator import Evaluator
from dctools.metrics.metrics import MetricComputer
from dctools.utilities.init_dask import setup_dask
from dctools.utilities.file_utils import load_config_file


# TODO Update all tests
@pytest.fixture(scope="function")
def test_config():
    pytest_configure()
    #logger.remove()  # Supprime les handlers existants
    #logger.add(sys.stderr, level="INFO")  # N'affiche que INFO et plus grave (masque DEBUG)
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

def filter_data(
    manager: MultiSourceDatasetManager,
    test_config: SimpleNamespace,
):
    filter_region = gpd.GeoSeries(geometry.Polygon((
        (test_config.min_lon,test_config.min_lat),
        (test_config.min_lon,test_config.max_lat),
        (test_config.max_lon,test_config.min_lat),
        (test_config.max_lon,test_config.max_lat),
        (test_config.min_lon,test_config.min_lat),
    )), crs="EPSG:4326")

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
    # manager.filter_all_by_variable(variables=test_config.target_vars)
    return manager


def check_dataloader(
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
def setup_dataset_manager(
    test_config: SimpleNamespace,
    use_json_catalog: bool,
):
    """Fixture pour configurer les datasets."""

    '''ds_test = xr.open_dataset(
        f"https://minio.dive.edito.eu/project-glonet/public/glonet_refull_2024/20240103.zarr",
        engine="zarr",
    )
    logger.debug(f"ds_test: {ds_test}")'''
    manager = MultiSourceDatasetManager()

    for source in test_config.sources:
        source_name = source['dataset']
        if source_name != "glonet":
                logger.warning(f"Dataset {source_name} is not supported yet, skipping.")
                continue
        logger.debug(f"SOURCES: {[source["dataset"] for source in test_config.sources]}")

        kwargs = {}
        kwargs["source"] = source
        kwargs["root_data_folder"] = test_config.data_directory
        kwargs["root_catalog_folder"] = test_config.catalog_dir
        kwargs["max_samples"] = test_config.max_samples
        kwargs["use_catalog"] = use_json_catalog
    
        logger.info(f"\n\n\t\t\t**************  LOADING DATASET: {source_name}  ***************\n\n")
        dataset = get_dataset_from_config(
            **kwargs
        )
        manager.add_dataset(source_name, dataset)
        logger.info(f"\n\n\t\t\t**************  LOADED DATASET: {source_name}  ***************\n\n")

    # Construire le catalogue
    logger.info("Build catalog")
    manager.build_catalogs()
    manager.all_to_json(output_dir=test_config.catalog_dir)

    return manager



#@pytest.mark.usefixtures("setup_datasets", "test_config")
#class TestPipeline:

@pytest.mark.parametrize('use_json_catalog', [False, True])
def test_evaluation(
    test_config: SimpleNamespace,
    setup_dataset_manager: MultiSourceDatasetManager,
):
    """Proceed to evaluation."""
    aliases = setup_dataset_manager.datasets.keys()
    dask_cluster = setup_dask(test_config)


    transforms_dict = {}
    logger.debug(f"setup_dataset_manager.datasets: {setup_dataset_manager}")

    aliases = setup_dataset_manager.datasets.keys()
    '''if "jason3" in aliases:
        logger.warning("Jason3 dataset is not available, skipping its transform setup.")
        transforms_dict["jason3"] = setup_dataset_manager.get_transform(
            "standardize",
            dataset_alias="jason3",
        )'''
    if "glonet" in aliases:
        transforms_dict["glonet"] = setup_dataset_manager.get_transform(
            "standardize",
            dataset_alias="glonet",
        )

    dataloaders = {}
    metrics_names = {}
    metrics = {}
    evaluators = {}
    models_results = {}

    for alias in setup_dataset_manager.datasets.keys():
        logger.debug(f"\n\n Get dataloader for {alias}")
        logger.debug(f"Transform: {transforms_dict.get(alias)}\n\n")
        pred_transform = transforms_dict.get(alias)
        ref_transform = transforms_dict.get(alias)

        if alias != 'glonet':
            ref_transform = transforms_dict.get(alias)
            ref_alias=alias
        else:
            ref_transform = None
            ref_alias=None
        dataloaders[alias] = setup_dataset_manager.get_dataloader(
            pred_alias=alias,
            ref_alias=ref_alias,
            batch_size=test_config.batch_size,
            pred_transform=pred_transform,
            ref_transform=ref_transform,
        )

        # Vérifier le dataloader
        check_dataloader(dataloaders[alias])

    for alias in setup_dataset_manager.datasets.keys():
        metrics_names[alias] = [
            "rmsd",
        ]
        metrics_kwargs = {}
        metrics_kwargs[alias] = {"add_noise": False,
            "eval_variables": setup_dataset_manager.datasets[alias].eval_variables,
        }
        metrics[alias] = [
            MetricComputer(metric_name=metric, **metrics_kwargs[alias])
            for metric in metrics_names[alias]
        ]

        evaluators[alias] = Evaluator(
            dask_cluster=dask_cluster,
            metrics=metrics[alias],
            dataloader=dataloaders[alias],
            json_path=os.path.join(test_config.catalog_dir, f"test_results_{alias}.json"),
        )

        models_results[alias] = evaluators[alias].evaluate()


    # Vérifier que chaque résultat contient les champs attendus, afficher
    for dataset_alias, results in models_results.items():
        # Vérifier que les résultats existent
        assert len(results) > 0
        logger.info(f"\n\n\n\t\t\t************ Results for {dataset_alias}  ***************")
        for result in results:
            assert "date" in result
            assert "metric" in result
            assert "result" in result
            logger.info(f"Test Result: {result}")

