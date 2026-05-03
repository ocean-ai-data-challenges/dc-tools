# dc-tools: Ocean Data Challenges Framework

![Logo du PPR Océan & Climat](docs/source/_static/Logo_PPR.jpg)

**dc-tools** is a comprehensive Python framework for developing and evaluating ocean data challenges as part of the [PPR Océan & Climat](https://www.ocean-climat.fr/) initiative. It provides unified tools for data management, processing, evaluation, and distributed computation.

## ✨ Features

- **Multi-source data loading**: Seamlessly integrate data from CMEMS, Argo, S3, and local sources
- **Flexible coordinate system**: Auto-detect and normalize various coordinate naming conventions
- **Distributed evaluation**: Built-in Dask support for scalable evaluation on clusters
- **Flexible evaluation pipelines**: Configuration-driven evaluation workflows via YAML
- **Extensible architecture**: Base classes for implementing custom evaluation logic
- **Memory optimization**: Automatic chunking, memory-aware processing, and worker caching
- **Comprehensive metrics**: Calculate RMSE, MAE, and class4 validation metrics

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Testing](#testing)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [Documentation](#documentation)

## 📦 Installation

### Requirements

- Python 3.11+
- Git
- Conda (recommended for managing complex dependencies)

### Step 1: Clone the repository

```bash
git clone git@github.com:ocean-ai-data-challenges/dc-tools.git
cd dc-tools
```

### Step 2: Set up the Python environment

We strongly recommend using Conda for dependency management due to complex system-level dependencies (especially for `xESMF`):

```bash
conda create -n dctools python=3.13 esmpy poetry -c conda-forge
conda activate dctools
```

### Step 3: Install the package and dependencies

```bash
poetry install --with dev
```

If you encounter import errors related to `xESMF`, consider reinstalling the environment from scratch. This is a known issue related to how `xESMF` interacts with the build system.

### Optional: Documentation dependencies

To build the Sphinx documentation locally:

```bash
poetry install --with docs
```

## 🚀 Quick Start

### Basic usage example

Here's a simple example to get started:

```python
from dctools.data import EvaluationDataloader
from dctools.metrics import Evaluator
import xarray as xr

# Load data from multiple sources
dataloader = EvaluationDataloader()
ds = dataloader.load_dataset(source="argo")  # or "cmems", "s3", "local"

# Evaluate against reference data
evaluator = Evaluator()
metrics = evaluator.evaluate(ds, reference_ds)

print(f"RMSE: {metrics['rmse']}")
print(f"MAE: {metrics['mae']}")
```

### Running from configuration

Define your evaluation in `config.yaml`:

```yaml
sources:
  - name: argo
    type: profiles
    path: s3://bucket/argo/

  - name: reference
    type: gridded
    path: /local/path/reference/

evaluation:
  metrics: [rmse, mae]
  variables: [SO, THETAO]
  grid: 0.25
```

Then run via CLI:

```bash
poetry run python -m dctools run config.yaml
```

## 📁 Project Structure

```
dc-tools/
├── dctools/                    # Main package
│   ├── data/                   # Data management and loading
│   │   ├── coordinates.py      # Coordinate system utilities
│   │   ├── transforms.py       # Data transformations
│   │   ├── datasets/           # Dataset implementations
│   │   └── connection/         # Remote data connections
│   ├── metrics/                # Metric computation
│   │   ├── evaluator.py        # Distributed evaluation engine
│   │   ├── metrics.py          # Metric implementations
│   │   └── oceanbench_metrics.py  # OceanBench integration
│   ├── processing/             # Data processing pipelines
│   │   ├── base.py             # Base evaluation class
│   │   ├── interpolation.py    # Grid interpolation
│   │   ├── runner.py           # CLI runner
│   │   └── nadir_data.py       # Nadir data processing
│   ├── dcio/                   # Input/Output operations
│   │   ├── loader.py           # Dataset loader
│   │   └── saver.py            # Dataset saver
│   ├── debug/                  # Debugging utilities
│   └── utilities/              # General utilities
├── dc2/                        # DC2 specific implementation
│   ├── evaluation/             # DC2 evaluation logic
│   ├── config/                 # DC2 configuration files
│   └── leaderboard_results/    # DC2 leaderboard results
├── docs/                       # Sphinx documentation
├── scripts/                    # Standalone utility scripts
├── tests/                      # Test suite
└── pyproject.toml              # Project configuration
```

## 🔧 Usage

### Loading Data

```python
from dctools.data import EvaluationDataloader, CoordinateSystem

# Initialize data loader
loader = EvaluationDataloader()

# Load from different sources
argo_data = loader.load_dataset(source="argo", region="north_atlantic")
cmems_data = loader.load_dataset(source="cmems", variables=["SO", "THETAO"])

# Automatic coordinate detection and normalization
cs = CoordinateSystem(argo_data)
normalized = cs.get_target_dimensions()
```

### Processing Data

```python
from dctools.processing import interpolate_dataset

# Interpolate to a standard grid
target_grid = xr.open_dataset("target_grid.nc")
interpolated = interpolate_dataset(
    cmems_data,
    target_grid,
    method="bilinear"
)
```

### Computing Metrics

```python
from dctools.metrics import MetricComputer

# Compute evaluation metrics
metric_computer = MetricComputer(variables=["SO", "THETAO"])
results = metric_computer.compute(
    prediction=interpolated,
    reference=argo_data,
    metrics=["rmse", "mae", "bias"]
)
```

### Distributed Evaluation with Dask

```python
from dctools.metrics import Evaluator
from dask.distributed import Client

# Setup Dask cluster
with Client(n_workers=4, threads_per_worker=2):
    evaluator = Evaluator()
    
    # Evaluation automatically uses Dask parallelism
    results = evaluator.evaluate(
        datasets=[ds1, ds2, ds3],
        reference=reference_ds
    )
```

### Saving Results

```python
from dctools.dcio import DataSaver

saver = DataSaver()

# Save to NetCDF
saver.save_dataset(results, "results.nc", format="netcdf")

# Save to Zarr (preferred for large datasets)
saver.save_dataset(results, "results.zarr", format="zarr")
```

## 🧪 Testing

The test suite is organized around three pytest markers for different test profiles:

| Marker | Purpose | Speed | Examples |
|--------|---------|-------|----------|
| `unit` | Fast, deterministic tests | < 1s each | Module unit tests, utility functions |
| `integration` | Multiple components (I/O, Dask, external libs) | 5-30s | Data loading, pipeline tests |
| `slow` | Long-running tests (network, large I/O) | > 30s | Network tests, large file handling |

### Running tests

```bash
# Fast PR profile (recommended for local development)
poetry run poe test-fast

# Fast profile with coverage report
poetry run poe test-coverage

# Only slow/integration tests
poetry run poe test-slow

# Full test suite (unit + integration)
poetry run poe test-coverage

# Complete profile with all tests (includes slow tests)
poetry run poe all-with-slow

# Strict profile (high coverage requirement - CI only)
poetry run poe all-strict

# Run all checks (lint + types + coverage)
poetry run poe all
```

### Code quality

```bash
# Run type checking
poetry run poe types

# Run linter
poetry run poe lint

# Run all linters and type checks
poetry run poe lint && poetry run poe types
```

### Coverage strategy

The project follows an incremental coverage improvement strategy:

1. **Fast PR gate**: `test-fast` with `--cov-fail-under=20` to track trends
2. **Gradual improvement**: Focus on stable modules: `dctools/data/datasets`, `dctools/metrics`, `dctools/data/connection`
3. **Incremental thresholds**: Raise coverage requirements by ~5% per cycle once sustained
4. **Nightly runs**: `slow` tests in scheduled CI, failures inform backlog but don't block PRs

## ⚙️ Configuration

### YAML Configuration Format

Evaluation workflows are configured via YAML. See [Configuration Guide](docs/source/usage/config.md) for detailed documentation.

### Example configuration structure

```yaml
# Data sources
sources:
  - name: model_output
    type: gridded
    path: s3://bucket/model/forecast.nc
    
  - name: observations
    type: profiles
    path: ./data/argo_profiles/

# Evaluation settings
evaluation:
  grid_spacing: 0.25  # degrees
  variables: [SO, THETAO, ZOS]
  regions: ["Atlantic", "Pacific", "Indian"]
  depth_levels: [0, 100, 500, 1000, 2000]
  
# Output settings
output:
  format: json
  per_bin_statistics: true
  mask_land: true
```

## 📚 Documentation

Full documentation is available in the [docs/](docs/) directory:

- **[Installation Guide](docs/source/package_docs/installation.md)** - Detailed setup instructions
- **[Quick Start Guide](docs/source/usage/quickstart.md)** - Getting started examples
- **[API Reference](docs/source/package_docs/api.md)** - Complete API documentation
- **[Data Challenges](docs/source/data_challenges/)** - DC1-DC5 specific documentation

To build documentation locally:

```bash
cd docs/
make html
open build/html/index.html
```

## 🛠️ Development Workflow

### Setting up for development

```bash
# Clone repository
git clone git@github.com:ocean-ai-data-challenges/dc-tools.git
cd dc-tools

# Create environment
conda create -n dctools-dev python=3.13 esmpy poetry -c conda-forge
conda activate dctools-dev
poetry install --with dev --with docs

# Verify setup
poetry run poe all
```

### Making changes

1. Create a feature branch: `git checkout -b feature/my-feature`
2. Make your changes and commit: `git commit -am "Add my feature"`
3. Run tests locally: `poetry run poe test-fast`
4. Push to GitHub: `git push origin feature/my-feature`
5. Create a Pull Request and ensure CI passes

## 🤝 Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch
3. Follow the code style (enforced by `ruff` and `mypy`)
4. Add tests for new functionality
5. Submit a Pull Request

See [CONTRIBUTING.md](CONTRIBUTING.md) if available for more details.

## 📦 Dependencies

Key dependencies include:

- **Data handling**: `xarray`, `netcdf4`, `zarr`, `h5py`
- **Processing**: `dask[distributed]`, `xESMF` (interpolation), `scipy`
- **Data sources**: `argopy` (Argo), `copernicusmarine` (CMEMS)
- **ML**: `torch`, `torchvision`, `torchgeo`
- **Geospatial**: `cartopy`, `geopandas`, `shapely`
- **Metrics**: Custom fork of `oceanbench`

See [pyproject.toml](pyproject.toml) for complete dependency list and version constraints.

## 📖 Related Resources

- **[PPR Océan & Climat](https://www.ocean-climat.fr/)** - Main initiative website
- **[Ocean Data Challenges Organization](https://github.com/ocean-ai-data-challenges)** - Additional challenge repositories
- **[OceanBench](https://github.com/mercator-ocean/oceanbench)** - Upstream metrics library
- **[Xarray](https://xarray.pydata.org/)** - Data structure documentation
- **[Dask](https://dask.org/)** - Distributed computing framework

## 📝 License

This project is licensed under the GPL-3.0 license - see [LICENSE](LICENSE) file for details.

## 👥 Authors

- Kamel Ait Mohand
- Guillermo Cossio

## ⚡ Quick Links

- 📄 [Full Documentation](docs/source/index.md)
- 🐛 [Report Issues](https://github.com/ppr-ocean-ia/dc-tools/issues)
- 💬 [Discussions](https://github.com/ppr-ocean-ia/dc-tools/discussions)
- 📊 [Data Challenges](docs/source/data_challenges/dc_index.md)
