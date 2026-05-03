# dctools Package Documentation

## Overview

**dctools** is a flexible and extensible Python framework for building ocean data challenges. It provides unified tools for:

1. **Data Management** - Loading data from multiple sources (CMEMS, Argo, S3, local files)
2. **Data Processing** - Coordinate system handling, interpolation, transformations
3. **Evaluation Metrics** - Computing RMSE, MAE, bias, and class4 validation
4. **Distributed Computing** - Built-in Dask support for scalable evaluation
5. **Configuration-Driven Workflows** - YAML-based evaluation pipelines

This package is developed as part of the [PPR Océan & Climat](https://www.ocean-climat.fr/) initiative and provides the core infrastructure used by all Ocean Data Challenges.

## Key Features

### 🔄 Multi-Source Data Loading

Load data seamlessly from different sources:
- **CMEMS**: Model outputs and reanalysis data
- **Argo**: In-situ temperature/salinity profiles
- **S3**: Cloud-hosted datasets
- **Local files**: NetCDF, Zarr, HDF5

### 📍 Flexible Coordinate Systems

Automatic detection and normalization of:
- Variable coordinate names (`lat` → `latitude`, `nav_lat`, etc.)
- Depth naming conventions (`depth`, `z_c`, `level`)
- Time coordinate handling
- Longitude normalization ([-180, 180] vs [0, 360])

### ⚡ Distributed Evaluation

- Built-in Dask integration for cluster computing
- Automatic worker memory management
- Remote dataset caching
- Per-bin aggregation for regional statistics

### 📊 Comprehensive Metrics

Compute multiple validation metrics:
- **RMSE** (Root Mean Square Error)
- **MAE** (Mean Absolute Error)
- **Bias**
- **Class4** (in-situ cross-validation)
- **Per-region aggregations**
- **Physical range masking**

### 🔌 Extensible Architecture

- Base classes for custom implementations
- Configuration-driven workflows
- Modular component design
- Integration with OceanBench library

## Module Structure

### `dctools.data` - Data Management

Handles data loading, coordinate systems, and transformations.

**Key Classes:**
- `EvaluationDataloader` - Multi-source data loader
- `CoordinateSystem` - Coordinate normalization
- `Transform` classes - Data transformations

**Features:**
- Automatic coordinate alias detection
- Dataset caching
- Multiple source backends (CMEMS, Argo, S3)

### `dctools.metrics` - Evaluation Metrics

Computes validation metrics and performance statistics.

**Key Classes:**
- `MetricComputer` - Core metric computation
- `Evaluator` - Distributed evaluation engine
- `OceanBenchMetrics` - Integration with OceanBench

**Features:**
- RMSE, MAE, bias computation
- Regional aggregation
- Dask parallelization
- Physical range masking

### `dctools.processing` - Data Processing

Processes and prepares data for evaluation.

**Key Classes:**
- `BaseDCEvaluation` - Base evaluation pipeline
- `interpolate_dataset()` - Grid interpolation (xESMF)
- `run_from_cli()` - CLI entry point

**Features:**
- Grid interpolation (bilinear, nearest)
- Dask pipeline orchestration
- Catalog management
- Per-bin aggregation

### `dctools.dcio` - Input/Output

Handles data reading and writing.

**Key Functions:**
- `DataSaver.save_dataset()` - Save to NetCDF/Zarr
- `DataLoader` - Load various formats
- `choose_chunks_automatically()` - Optimize chunking

**Formats supported:**
- NetCDF4 (with compression)
- Zarr (with append mode)
- HDF5
- Parquet (for columnar data)

### `dctools.utilities` - General Utilities

Helper functions and configurations.

**Submodules:**
- `args_config.py` - Command-line argument parsing
- `file_utils.py` - File operations
- `format_converter.py` - Data format conversions
- `init_dask.py` - Dask cluster initialization

### `dctools.debug` - Debugging Tools

Development and debugging utilities.

## Typical Workflow

```
1. Configure Sources
   ↓
2. Load Data → EvaluationDataloader
   ↓
3. Normalize Coordinates → CoordinateSystem
   ↓
4. Process Data → Interpolation, Transforms
   ↓
5. Setup Evaluation → Dask cluster
   ↓
6. Compute Metrics → MetricComputer/Evaluator
   ↓
7. Aggregate Results → Per-region statistics
   ↓
8. Save Output → JSON, NetCDF, Zarr
```

## Getting Started

- **[Installation Guide](installation.md)** - Set up the package
- **[Quick Start Guide](../usage/quickstart.md)** - 5-minute introduction
- **[API Reference](api.md)** - Complete module documentation

## Common Use Cases

### Use Case 1: Simple Point Validation

Validate model output against Argo profiles:

```python
from dctools.data import EvaluationDataloader
from dctools.metrics import MetricComputer

loader = EvaluationDataloader()
model = loader.load_dataset(source="cmems")
observations = loader.load_dataset(source="argo")

computer = MetricComputer()
metrics = computer.compute(model, observations)
```

### Use Case 2: Large-Scale Evaluation

Evaluate on a Dask cluster across multiple regions:

```python
from dask.distributed import Client
from dctools.metrics import Evaluator

with Client(n_workers=10):
    evaluator = Evaluator()
    results = evaluator.evaluate(
        datasets=[ds1, ds2, ds3],
        reference=reference
    )
```

### Use Case 3: Data Challenge Setup

Configure and run a complete evaluation pipeline:

```bash
# config.yaml
sources:
  model:
    path: s3://bucket/model.nc
  observations:
    path: ./data/argo/

evaluation:
  variables: [SO, THETAO]
  metrics: [rmse, mae]
  per_region: true
```

```bash
poetry run python -m dctools run config.yaml
```

## Architecture Patterns

### Lazy Evaluation

All data operations use lazy Dask arrays for memory efficiency. Data is only computed when explicitly requested or saved.

### Configuration-Driven

Workflows are specified in YAML, allowing reproducibility and version control of evaluation procedures.

### Modular Design

Components can be used independently or chained together for complex pipelines.

### Distributed-First

Built with Dask from the start, enabling seamless scaling from laptop to cluster.

## Dependencies

**Core dependencies:**
- `xarray` - Multi-dimensional arrays with labeled coordinates
- `dask[distributed]` - Distributed computing
- `xESMF` - Interpolation on unstructured grids
- `netcdf4`, `zarr`, `h5py` - Data I/O

**Data sources:**
- `argopy` - Argo data access
- `copernicusmarine` - CMEMS data access
- `s3fs` - S3 cloud storage

**Analysis:**
- `torch`, `torchgeo` - Deep learning utilities
- `scipy`, `scikit-learn` - Scientific computing
- `cartopy`, `geopandas` - Geospatial analysis

See [pyproject.toml](../../pyproject.toml) for complete list and versions.

## Related Repositories

- **[PPR Ocean IA](https://github.com/ppr-ocean-ia)** - Main organization
- **[Ocean AI Data Challenges](https://github.com/ocean-ai-data-challenges)** - Challenge repositories
- **[OceanBench](https://github.com/mercator-ocean/oceanbench)** - Metrics library

## Contributing

Contributions are welcome! Please refer to the main repository's contributing guidelines.

## License

GPL-3.0 - See [LICENSE](../../LICENSE) for details.

## Authors

- Kamel Ait Mohand
- Guillermo Cossio

---

```{toctree}
:maxdepth: 2

installation.md
../usage/index.md
api.md
```
