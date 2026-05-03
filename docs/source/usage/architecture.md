# Architecture Guide

This document describes the architecture and design patterns used in dc-tools.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   User Applications                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────┐ │
│  │ YAML Config  │  │ Python Code  │  │  Challenge-       │ │
│  │ Interface    │  │  Interface   │  │  Specific Code    │ │
│  └──────────────┘  └──────────────┘  └───────────────────┘ │
│         │                 │                    │            │
└─────────┼─────────────────┼────────────────────┼────────────┘
          │                 │                    │
         ▼                 ▼                   ▼
┌─────────────────────────────────────────────────────────────┐
│            dctools Evaluation Framework                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Data Management (data/)                                    │
│  ├─ MultiSource DataLoader                                 │
│  ├─ CoordinateSystem Normalization                         │
│  └─ Transform Pipeline                                     │
│                                                              │
│  Processing (processing/)                                   │
│  ├─ Interpolation (xESMF-based)                            │
│  ├─ Coordinate Conformance                                 │
│  └─ BaseDCEvaluation Pipeline                              │
│                                                              │
│  Metrics (metrics/)                                         │
│  ├─ MetricComputer (RMSE, MAE, ...)                        │
│  ├─ Evaluator (Distributed)                                │
│  └─ OceanBench Integration                                 │
│                                                              │
│  I/O (dcio/)                                                │
│  ├─ DataSaver (NetCDF/Zarr)                                │
│  └─ DataLoader                                              │
│                                                              │
│  Utilities (utilities/)                                     │
│  ├─ Configuration Parsing                                  │
│  ├─ Dask Initialization                                    │
│  └─ Helper Functions                                        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
          ▲                 ▲                    ▲
          │                 │                    │
     ┌────┴─────────────────┴────────────────────┴─────┐
     │  External Dependencies (xarray, Dask, etc)      │
     │  External Data Sources (CMEMS, Argo, S3, ...)   │
     └────────────────────────────────────────────────┘
```

## Core Design Principles

### 1. **Lazy Evaluation**

All operations use Dask arrays for memory efficiency:

```
Data → Dask Array (lazy) → Compute only on save/explicit call
```

**Benefits**:
- Handle datasets larger than RAM
- Parallelize naturally across clusters
- Avoid unnecessary computations

### 2. **Configuration-Driven**

Workflows defined in YAML, separate from code:

```
config.yaml → Parser → Task Graph → Execution
```

**Benefits**:
- Reproducibility
- Version control of experiments
- Parameter sweeps without code changes
- Non-technical users can configure

### 3. **Coordinate Flexibility**

Automatic detection of various coordinate naming conventions:

```
latitude ──┐
nav_lat   ──├─ CoordinateSystem ─→ standardized
lat_c     ──┤  (lat, lon, depth,
lat       ──┘   time)
```

**Benefits**:
- Works with heterogeneous data sources
- No manual coordinate renaming
- Robust against naming variations

### 4. **Distributed-First**

Built with Dask from the start:

```
Single machine (threads)
        ↕
Dask Local Cluster (processes)
        ↕
Dask Distributed (cloud/HPC cluster)
```

**No code changes needed - same code runs everywhere.**

### 5. **Modular Components**

Each module has clear responsibilities and interfaces:

| Module | Input | Output | Responsibility |
|--------|-------|--------|-----------------|
| `data` | Config | xr.Dataset | Load & normalize |
| `processing` | xr.Dataset | xr.Dataset | Transform & interpolate |
| `metrics` | 2× xr.Dataset | dict | Compute metrics |
| `dcio` | xr.Dataset | File | Save to disk |

## Data Flow

### Basic Workflow

```
1. Load Data
   └─ EvaluationDataloader
      ├─ ConnectionManager (handles source type)
      ├─ MultiSourceDatasetManager (fetches data)
      └─ CoordinateSystem (normalizes coordinates)

2. Process Data
   └─ Interpolation
      ├─ xESMF (grid interpolation)
      └─ Transform pipeline

3. Evaluate
   └─ MetricComputer / Evaluator
      ├─ Compute metrics (vectorized)
      └─ Aggregate results

4. Save
   └─ DataSaver
      ├─ NetCDF (with compression)
      └─ Zarr (with chunking)
```

### Class Relationships

```
BaseDCEvaluation (abstract)
│
├─ DC1Evaluation (SSH challenge)
├─ DC2Evaluation (SST challenge)
└─ DC3Evaluation (Dynamics challenge)

Each extends with challenge-specific:
├─ Variable mappings
├─ Metric configurations
├─ Validation ranges
└─ Regional configurations
```

## Component Details

### Data Module (`dctools/data/`)

**Responsibilities**:
- Discover available data sources
- Connect to remote/local backends
- Load data into Xarray datasets
- Normalize coordinates

**Key Classes**:

```python
class EvaluationDataloader:
    """Main entry point for data loading"""
    def load_dataset(source: str) → xr.Dataset
    def _transform_coordinates(ds) → xr.Dataset

class CoordinateSystem:
    """Detect and normalize coordinates"""
    def get_target_dimensions() → dict
    def get_target_depth_values() → np.array

class MultiSourceDatasetManager:
    """Manage multiple data sources"""
    def fetch(source_config) → xr.Dataset
```

**Coordinate Normalization**:

```python
# Input: various naming conventions
ds.dims = {'latitude': 100, 'nav_lon': 200, 'z_c': 50, 'time': 365}

# Process through CoordinateSystem
cs = CoordinateSystem(ds)

# Output: standardized names
ds.dims = {'lat': 100, 'lon': 200, 'depth': 50, 'time': 365}
```

### Processing Module (`dctools/processing/`)

**Responsibilities**:
- Orchestrate evaluation pipelines
- Interpolate to target grid
- Apply coordinate conformance
- Manage Dask cluster

**Key Classes**:

```python
class BaseDCEvaluation:
    """Base evaluation pipeline"""
    def run() → dict  # Main entry point
    def _setup_dask()  # Initialize cluster
    def _load_data()
    def _process()
    def _evaluate()
    def _save()

def interpolate_dataset(ds, target_grid, method='bilinear') → xr.Dataset:
    """xESMF-based interpolation"""
```

**Pipeline Orchestration**:

```
1. Setup Dask
   ├─ Create workers
   ├─ Configure memory
   └─ Initialize scheduler

2. Load Data
   ├─ Fetch from sources
   ├─ Validate variables
   └─ Create Dask arrays

3. Process
   ├─ Normalize coordinates
   ├─ Interpolate
   └─ Apply transforms

4. Evaluate
   ├─ Create task graph
   ├─ Distribute to workers
   └─ Aggregate results

5. Cleanup
   ├─ Close client
   └─ Release resources
```

### Metrics Module (`dctools/metrics/`)

**Responsibilities**:
- Compute validation metrics
- Support distributed evaluation
- Aggregate regional statistics
- Interface with OceanBench

**Key Classes**:

```python
class MetricComputer:
    """Compute evaluation metrics"""
    def compute(prediction, reference, metrics) → dict
    def _apply_masks()
    def _vectorized_metrics()

class Evaluator:
    """Distributed evaluation engine"""
    def evaluate(datasets, reference) → dict
    def _setup_cluster()
    def _cache_remote_datasets()
    def _compute_with_dask()
```

**Metric Computation**:

```python
# Input: 2 xr.Dataset objects (lazy Dask arrays)
prediction = xr.open_zarr('model.zarr')  # ~1 TB
reference = xr.open_dataset('argo.nc')    # ~10 GB

# Dask handles memory automatically
rmse = MetricComputer().compute(prediction, reference, ['rmse'])

# Result: computed value
print(rmse)  # {variables: {SO: 0.5, THETAO: 0.8}}
```

### I/O Module (`dctools/dcio/`)

**Responsibilities**:
- Read data from various formats
- Save with optimization
- Handle compression
- Chunk automatically

**Key Functions**:

```python
def save_dataset(ds: xr.Dataset, path: str, format: str = "netcdf"):
    """Save with automatic optimization"""
    # Handles: chunks, compression, format conversion

def choose_chunks_automatically(ds, target_memory="1GB") → dict:
    """Suggest optimal chunks based on memory target"""
```

## Data Models

### Configuration Object

```yaml
config:
  sources:
    - name: str
      type: gridded | profiles | generic
      path: str
      variables: List[str]
      ...
  
  evaluation:
    variables: List[str]
    metrics: List[str]
    grid: {spacing: float} | {path: str}
    regions: List[{name: str, bounds: dict}]
    ...
  
  output:
    path: str
    format: json | netcdf | zarr
    ...
```

### Evaluation Result

```python
{
    "metadata": {
        "timestamp": "2024-01-01T12:00:00Z",
        "config_hash": "abc123",
        "n_samples": 1000,
        ...
    },
    "summary": {
        "SO": {"rmse": 0.5, "mae": 0.3, ...},
        "THETAO": {"rmse": 0.8, "mae": 0.6, ...},
    },
    "per_region": {
        "tropical": {
            "SO": {"rmse": 0.4, ...},
            ...
        },
        ...
    },
    "per_depth": {
        "0": {"SO": {"rmse": 0.6, ...}, ...},
        ...
    },
}
```

## Concurrency & Distribution

### Local Execution (default)

```python
# Uses thread pool (no GIL bottleneck for Dask)
loader = EvaluationDataloader()
metrics = loader.evaluate()  # Synchronous
```

### Distributed Execution

```python
from dask.distributed import Client

with Client(n_workers=10, threads_per_worker=2):
    # Same code runs on cluster
    loader = EvaluationDataloader()
    metrics = loader.evaluate()  # Distributed
```

**No code changes needed - Dask handles distribution.**

## Memory Management

### Chunking Strategy

```
Dataset size: 100 GB
Target chunk size: 500 MB
Chunks per dimension: 10 × 10 × 10 = 1,000 chunks

Dask distributes chunks across:
├─ Threads (single machine, shared memory)
├─ Processes (single machine, separate memory)
└─ Worker nodes (cluster, separate machines)
```

### Cache Management

```
┌─────────────────────────┐
│  Remote dataset         │
│  (CMEMS, S3, Argo)      │
└──────────┬──────────────┘
           │ (first access)
           ▼
      Cache locally
      (with LRU eviction)
           │
           ▼
      Reuse on next access
```

## Error Handling

### Graceful Degradation

```python
try:
    # Primary source
    data = load_from_cmems()
except CopernicusMarineError:
    # Fallback source
    data = load_from_s3()
except S3Error:
    # Local cache
    data = load_from_local_cache()
```

### Validation

```python
# Automatic validation
├─ Check variable existence
├─ Check coordinate conformance
├─ Check value ranges
└─ Report issues before compute
```

## Extension Points

Developers can extend dc-tools at several points:

### 1. Custom Data Source

```python
from dctools.data import ConnectionManager

class CustomBackend(ConnectionManager):
    def connect(config):
        # Custom connection logic
        return xr.Dataset(...)
```

### 2. Custom Metric

```python
from dctools.metrics import MetricComputer

class CustomMetric(MetricComputer):
    def _custom_metric(pred, ref):
        # Custom metric computation
        return value
```

### 3. Custom Evaluation

```python
from dctools.processing import BaseDCEvaluation

class CustomEvaluation(BaseDCEvaluation):
    def _process(self):
        # Custom processing pipeline
        pass
```

### 4. Custom Challenge

Each challenge (DC1-DC5) is a subclass:

```python
class DC2Evaluation(BaseDCEvaluation):
    VARIABLES = ['THETAO', 'SO']
    METRICS = ['rmse', 'mae']
    REGIONS = {...}  # DC2-specific regions
```

## Testing Strategy

### Test Organization

```
tests/
├─ unit/           # Fast, isolated (< 1s each)
├─ integration/    # Multiple components (5-30s)
└─ slow/           # Long-running (> 30s)
```

### Test Levels

| Level | Scope | Example |
|-------|-------|---------|
| Unit | Single function | Coordinate normalization |
| Integration | Multiple modules | Load + interpolate |
| Slow | Full pipeline | Download + evaluate on cluster |

## Performance Characteristics

### Typical Benchmarks

| Operation | Time | Memory |
|-----------|------|--------|
| Load 1GB gridded data | 2s | 50MB (lazy) |
| Interpolate to 0.25° grid | 10s | 200MB (Dask) |
| Compute RMSE (1000 samples) | 5s | 100MB |
| **Total (small example)** | **~30s** | **< 1GB** |

### Scaling Characteristics

```
1 machine, 8 cores: ~30 seconds
4 workers, 2 cores each: ~15 seconds (2× speedup)
16 workers, 2 cores each: ~4 seconds (7.5× speedup)

(Linear scaling with proper chunking)
```

## Troubleshooting Architecture Issues

### Issue: "Out of Memory"

**Root causes**:
- Dataset not lazy loaded
- Too-large chunks
- Cached results not released

**Solution**:
```python
# Verify lazy loading
print(data.THETAO)  # Should show "dask.array"

# Configure chunks
data = data.chunk({"time": 30, "lat": 100, "lon": 100})

# Monitor Dask memory
from dask.diagnostics import ProgressBar
```

### Issue: "Slow Performance"

**Root causes**:
- Synchronous execution (not using Dask)
- Too-small chunks (excessive overhead)
- Data not colocated with workers

**Solution**:
```python
# Use distributed cluster
from dask.distributed import Client
with Client():  # Distributed scheduler
    evaluate()

# Optimize chunks
ds = ds.chunk(optimal_chunk_size)
```

## See Also

- [Quick Start](quickstart.md) - Practical examples
- [API Reference](../package_docs/api.md) - Detailed interfaces
- [Configuration Guide](config.md) - Workflow configuration
