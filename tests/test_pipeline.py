"""Tests for data processing pipeline."""

import os
import pytest
from types import SimpleNamespace

import geopandas as gpd
from loguru import logger
import pandas as pd
from shapely import geometry

from tests.conftest import pytest_configure
from dctools.data.datasets.dataset import get_dataset_from_config
from dctools.data.datasets.dataloader import EvaluationDataloader
from dctools.data.datasets.dataset_manager import MultiSourceDatasetManager
from oceanbench.core.distributed import DatasetProcessor

from dctools.metrics.evaluator import Evaluator
from dctools.metrics.metrics import MetricComputer
from dctools.utilities.init_dask import setup_dask
from dctools.utilities.file_utils import load_config_file


# TODO Update all tests
@pytest.fixture(scope="function")
def test_config():
    """Fixture to configure test variables."""
    pytest_configure()
    #logger.remove()  # Remove existing handlers
    #logger.add(sys.stderr, level="INFO")  # Only show INFO+ (hide DEBUG)
    config_file = os.path.join("tests", "config", "test_config.yaml")

    # Load YAML file
    config = load_config_file(config_file)

    # Add test-specific paths
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
    """Filter data based on configuration (time and region)."""
    filter_region = gpd.GeoSeries(geometry.Polygon((
        (test_config.min_lon,test_config.min_lat),
        (test_config.min_lon,test_config.max_lat),
        (test_config.max_lon,test_config.min_lat),
        (test_config.max_lon,test_config.max_lat),
        (test_config.min_lon,test_config.min_lat),
    )), crs="EPSG:4326")

    # Apply time filters
    manager.filter_all_by_date(
        start=pd.to_datetime(test_config.start_times[0]),
        end=pd.to_datetime(test_config.end_times[0]),
    )
    # Apply spatial filters
    manager.filter_all_by_region(
        region=filter_region
    )
    # Apply variable filters
    # manager.filter_all_by_variable(variables=test_config.target_vars)
    return manager


def check_dataloader(
    dataloader: EvaluationDataloader,
):
    """Check that dataloader yields valid batches."""
    for batch in dataloader:
        # Check that the batch contains the expected keys
        assert "pred_data" in batch[0]
        assert "ref_data" in batch[0]
        # Check that values are strings (paths)
        assert isinstance(batch[0]["pred_data"], str)
        if batch[0]["ref_data"]:
            assert isinstance(batch[0]["ref_data"], str)

@pytest.fixture(scope="function")
def setup_dataset_manager(
    test_config: SimpleNamespace,
    use_json_catalog: bool,
):
    """Fixture to configure datasets."""
    '''ds_test = xr.open_dataset(
        f"https://minio.dive.edito.eu/project-glonet/public/glonet_refull_2024/20240103.zarr",
        engine="zarr",
    )
    logger.debug(f"ds_test: {ds_test}")'''

    # Create simple dependencies for MultiSourceDatasetManager
    # distributed=True forces creation of a local Dask client, avoiding AttributeError later
    processor = DatasetProcessor(n_workers=1, memory_limit="2GB", distributed=True)
    target_dimensions = {
        'lat': (-90, 90),
        'lon': (-180, 180),
        'depth': (0, 0)
    }

    manager = MultiSourceDatasetManager(
        dataset_processor=processor,
        target_dimensions=target_dimensions,
        time_tolerance=pd.Timedelta("1h")
    )

    for source in test_config.sources:
        source_name = source['dataset']
        if source_name != "glonet":
                logger.warning(f"Dataset {source_name} is not supported yet, skipping.")
                continue
        logger.debug(f"SOURCES: {[source['dataset'] for source in test_config.sources]}")

        kwargs = {}
        kwargs["source"] = source
        kwargs["root_data_folder"] = test_config.data_directory
        kwargs["root_catalog_folder"] = test_config.catalog_dir

        # Pass the dataset_processor instance
        kwargs["dataset_processor"] = processor

        # Pass filter_values to avoid AttributeError
        kwargs["filter_values"] = {}

        kwargs["max_samples"] = test_config.max_samples
        kwargs["use_catalog"] = use_json_catalog

        logger.info(
            f"\n\n\t\t\t**************  LOADING DATASET: {source_name}  ***************\n\n"
        )
        dataset = get_dataset_from_config(
            **kwargs
        )
        manager.add_dataset(source_name, dataset)
        logger.info(f"\n\n\t\t\t**************  LOADED DATASET: {source_name}  ***************\n\n")

    # Build the catalog
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
    setup_dask(test_config)


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
            dataset_alias="glonet",
            transform_name="standardize",
        )

    dataloaders = {}
    metrics_names = {}
    metrics = {}
    evaluators = {}

    for alias in setup_dataset_manager.datasets.keys():
        setup_dataset_manager.build_forecast_index(
            alias,
            init_date=test_config.start_times[0],
            end_date=test_config.end_times[0],
            n_days_forecast=1,
            n_days_interval=1,
        )

        logger.debug(f"\n\n Get dataloader for {alias}")
        logger.debug(f"Transform: {transforms_dict.get(alias)}\n\n")
        pred_transform = transforms_dict.get(alias)
        ref_transform = transforms_dict.get(alias)

        # Force self-comparison for testing
        ref_transform = transforms_dict.get(alias)
        ref_alias = alias

        ref_transforms_dict = {ref_alias: ref_transform} if ref_alias and ref_transform else {}

        dataloaders[alias] = setup_dataset_manager.get_dataloader(
                pred_alias=alias,
                ref_aliases=[ref_alias] if ref_alias else [],
                batch_size=test_config.batch_size,
                pred_transform=pred_transform,
                ref_transforms=ref_transforms_dict,
                forecast_mode=True,
            )        # Check the dataloader
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

        ref_alias = alias # Force self-comparison for testing

        # Construct dictionaries keyed by ref_alias
        metrics_dict = {ref_alias: metrics[alias]} if ref_alias else {}
        ref_aliases_list = [ref_alias] if ref_alias else []

        evaluators[alias] = Evaluator(
            dataset_manager=setup_dataset_manager,
            metrics=metrics_dict,
            dataloader=dataloaders[alias],
            ref_aliases=ref_aliases_list,
            dataset_processor=setup_dataset_manager.dataset_processor,
            results_dir=test_config.catalog_dir,
        )
        evaluators[alias].evaluate()

        # models_results[alias] = evaluators[alias].evaluate()


    # Check that each result contains the expected fields (and log it)
    import glob
    import json

    for dataset_alias in setup_dataset_manager.datasets.keys():
        # Check for generated result files
        pattern = os.path.join(test_config.catalog_dir, f"results_{dataset_alias}_batch_*.json")
        result_files = glob.glob(pattern)

        logger.info(f"Checking results for {dataset_alias} in {pattern}")
        logger.info(f"Found files: {result_files}")

        # We expect at least one batch file
        assert len(result_files) > 0, f"No result files found for {dataset_alias}"

        for r_file in result_files:
            with open(r_file, "r") as f:
                results = json.load(f)

            logger.info(f"Results in {r_file}: {len(results)} items")
            assert isinstance(results, list)
            # If results are not empty, check structure
            for result in results:
                assert "result" in result
                metrics_list = result["result"]
                if metrics_list is not None:
                    assert isinstance(metrics_list, list)
                    for m in metrics_list:
                        assert "Metric" in m
                        assert "Variable" in m
                        assert "Value" in m
                logger.info(f"Test Result: {result}")

    '''
    for dataset_alias, results in models_results.items():
        # Check that results exist
        assert len(results) > 0
        logger.info(f"\n\n\n\t\t\t************ Results for {dataset_alias}  ***************")
        for result in results:
            assert "date" in result
            assert "metric" in result
            assert "result" in result
            logger.info(f"Test Result: {result}")
    '''

