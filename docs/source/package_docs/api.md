# API Reference

Complete API documentation for all dctools modules.

## Data Management

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :template: custom-module-template.rst
   :recursive:

   dctools.data
```

### Key Classes

- `EvaluationDataloader` - Load data from multiple sources
- `CoordinateSystem` - Handle coordinate transformations
- `Transform` - Apply data transformations

## Input/Output

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :template: custom-module-template.rst
   :recursive:

   dctools.dcio
```

### Key Functions

- `DataSaver.save_dataset()` - Save data to various formats
- `FileLoader` - Load various file formats (NetCDF, Zarr, HDF5)
- `choose_chunks_automatically()` - Optimize data chunking

## Metrics

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :template: custom-module-template.rst
   :recursive:

   dctools.metrics
```

### Key Classes

- `MetricComputer` - Compute evaluation metrics (RMSE, MAE, bias)
- `Evaluator` - Distributed evaluation with Dask
- `OceanBenchMetrics` - Integration with OceanBench library

## Data Processing

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :template: custom-module-template.rst
   :recursive:

   dctools.processing
```

### Key Classes

- `BaseDCEvaluation` - Base evaluation pipeline class
- `run_from_cli()` - Command-line interface runner

### Key Functions

- `interpolate_dataset()` - Interpolate data to target grid

## Utilities

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :template: custom-module-template.rst
   :recursive:

   dctools.utilities
```

### Submodules

- `args_config.py` - Command-line argument parsing
- `file_utils.py` - File system operations
- `format_converter.py` - Data format conversions
- `init_dask.py` - Dask cluster initialization

## Debug Tools

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :template: custom-module-template.rst
   :recursive:

   dctools.debug
```

### Key Functions

- `debug_utils.py` - Development and debugging utilities

---

## Common Imports

Here are the typical imports for common tasks:

```python
# Data loading
from dctools.data.coordinates import CoordinateSystem
from dctools.data.datasets.dataloader import EvaluationDataloader

# Metrics computation
from dctools.metrics.evaluator import Evaluator
from dctools.metrics.metrics import MetricComputer

# Data processing
from dctools.processing.base import BaseDCEvaluation
from dctools.processing.interpolation import interpolate_dataset

# I/O operations
from dctools.dcio.loader import FileLoader, choose_chunks_automatically
from dctools.dcio.saver import DataSaver

# Utilities
from dctools.utilities.args_config import load_args_and_config
```

## Module Overview

| Module | Purpose | Key Classes |
|--------|---------|-------------|
| `dctools.data` | Data loading & coordinate handling | `EvaluationDataloader`, `CoordinateSystem` |
| `dctools.dcio` | Data I/O operations | `DataSaver`, `FileLoader` |
| `dctools.metrics` | Metric computation | `MetricComputer`, `Evaluator` |
| `dctools.processing` | Data processing pipelines | `BaseDCEvaluation`, `interpolate_dataset` |
| `dctools.utilities` | General utilities | Various helper functions |
| `dctools.debug` | Debugging tools | Various debug utilities |
