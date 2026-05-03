# Configuration Guide

This guide explains how to configure data challenges using YAML files.

## Overview

dc-tools uses YAML configuration files to define reproducible evaluation workflows. This approach provides:

- **Version control**: Configuration is text-based and git-trackable
- **Reproducibility**: Same results from same configuration
- **Parameterization**: Easily test different settings
- **Separation of concerns**: Configuration separate from code

## Configuration Structure

A typical configuration file has three main sections:

```yaml
# Sources: where to load data from
sources:
  ...

# Evaluation: what to evaluate and how
evaluation:
  ...

# Output: where and how to save results
output:
  ...
```

## Sources Section

Defines data sources for the evaluation.

### Gridded Data (Model Output)

```yaml
sources:
  model:
    type: gridded
    path: s3://bucket/cmems/model_output.nc
    # or local: ./data/model.nc
    # or zarr: file:///data/model.zarr
    
    variables:
      - SO          # Salinity
      - THETAO      # Temperature
      - ZOS         # Sea Surface Height
    
    # Optional filters
    time_range: ["2023-01-01", "2023-12-31"]
    region: "north_atlantic"
    
    # Optional preprocessing
    lazy: true  # Use Dask for lazy loading
    chunks:
      time: 30
      lat: 100
      lon: 100
```

### Profile Data (Observations)

```yaml
sources:
  observations:
    type: profiles
    
    # Argo float profiles
    backend: argo
    
    # Optional filters
    time_range: ["2023-01-01", "2023-12-31"]
    region: "global"
    profiler_types: [float, glider]
    
    # Caching
    cache: true
    cache_dir: ~/.cache/argo_profiles
```

### Generic Data Source

```yaml
sources:
  my_data:
    type: generic
    path: /path/to/data
    format: netcdf  # or zarr, hdf5, parquet
    
    # Preprocessing
    lazy: true
    variables: [T, S]
```

## Evaluation Section

Defines what to evaluate and how.

### Metrics

```yaml
evaluation:
  # Metrics to compute
  metrics: [rmse, mae, bias, correlation, r2]
  
  # Variables to evaluate (must exist in both sources)
  variables: [SO, THETAO, ZOS]
  
  # Evaluation grid
  grid:
    spacing: 0.25  # degrees
    # Or specify explicitly:
    # path: ./grids/target_grid.nc
  
  # Interpolation method
  interpolation:
    method: bilinear  # or nearest, cubic
    bounds_error: False
    fill_value: nan
  
  # Regional breakdown
  regions:
    - name: global
    - name: north_atlantic
      bounds:
        lat: [30, 60]
        lon: [-80, 0]
    - name: tropical
      bounds:
        lat: [-20, 20]
        lon: [-180, 180]
  
  # Depth levels
  depth_levels: [0, 10, 50, 100, 500, 1000, 2000]
  
  # Physical constraints
  mask_land: true
  valid_range:
    SO: [20, 40]        # Salinity valid range (PSU)
    THETAO: [-2, 35]    # Temperature valid range (°C)
    ZOS: [-2, 2]        # SSH valid range (m)
```

### Distributed Computing

```yaml
evaluation:
  # Dask configuration
  dask:
    scheduler: distributed  # or threads, synchronous
    n_workers: 4
    threads_per_worker: 2
    memory_per_worker: 4GB
    
    # Worker configuration
    worker_class: threads  # or processes
    worker_timeout: 300
    
    # Dashboard (distributed scheduler)
    dashboard: true
    dashboard_port: 8787
    
    # Memory management
    memory_target_fraction: 0.8
    spill_to_disk: true
```

### Output Options

```yaml
evaluation:
  # Per-bin aggregation
  per_region: true
  per_depth: true
  
  # Temporal aggregation
  time_aggregation: monthly  # or daily, weekly, seasonal
  
  # Confidence intervals
  confidence_intervals: true
  ci_method: bootstrap  # or percentile
  
  # Additional outputs
  save_intermediate: false
  save_predictions: true
```

## Output Section

Defines where and how to save results.

### Basic Output

```yaml
output:
  # Output directory
  path: ./results/
  
  # Output format
  format: json  # or netcdf, zarr, csv, jsonl
  
  # Compression (for binary formats)
  compression: gzip  # or lz4, blosc
  compression_level: 4
  
  # File naming
  prefix: evaluation_
  suffix: _results
```

### JSON Output

```yaml
output:
  format: json
  path: ./results/metrics.json
  
  # JSON structure options
  include_metadata: true
  include_timestamp: true
  pretty_print: true
  
  # What to include
  metrics_summary: true
  per_region_metrics: true
  per_depth_metrics: true
  confidence_intervals: true
```

### NetCDF Output

```yaml
output:
  format: netcdf
  path: ./results/evaluation.nc
  
  # NetCDF options
  format: netcdf4  # or netcdf3_classic
  compression: true
  compression_level: 4
  
  # Metadata
  include_metadata: true
  global_attrs:
    title: "Ocean Data Challenge Evaluation"
    author: "Your Name"
    institution: "Your Institution"
```

### Zarr Output

```yaml
output:
  format: zarr
  path: ./results/evaluation.zarr
  
  # Zarr options
  mode: w       # w=write, r+=read-write, w-=write-exclusive
  chunk_size:
    time: 30
    lat: 100
    lon: 100
  
  # Encoding
  encoding:
    time: int64
    lat: float32
    lon: float32
```

## Complete Example

Here's a complete example configuration:

```yaml
# Ocean Data Challenge Configuration
# DC2: Sea Surface Temperature Prediction

sources:
  model:
    type: gridded
    path: s3://ppr-ocean/dc2/model_forecast.zarr
    variables: [THETAO, SO, ZOS]
    time_range: ["2023-06-01", "2023-08-31"]
    lazy: true
  
  observations:
    type: profiles
    backend: argo
    time_range: ["2023-06-01", "2023-08-31"]
    profiler_types: [float]

evaluation:
  # Challenge settings
  variables: [THETAO, SO]
  metrics: [rmse, mae, bias, correlation]
  
  # Grid specification
  grid:
    spacing: 0.25
  
  # Interpolation
  interpolation:
    method: bilinear
    bounds_error: false
  
  # Evaluation regions
  regions:
    - name: global
    - name: tropical
      bounds:
        lat: [-20, 20]
        lon: [-180, 180]
    - name: northern_hemisphere
      bounds:
        lat: [20, 60]
        lon: [-180, 180]
  
  # Depth levels
  depth_levels: [0, 10, 50, 100, 500, 1000]
  
  # Physical constraints
  mask_land: true
  valid_range:
    THETAO: [-2, 35]
    SO: [20, 40]
  
  # Dask configuration
  dask:
    scheduler: distributed
    n_workers: 8
    threads_per_worker: 2
    memory_per_worker: 4GB
  
  # Aggregations
  per_region: true
  per_depth: true
  time_aggregation: monthly

output:
  path: ./results/dc2_evaluation/
  format: jsonl
  
  metrics_summary: true
  per_region_metrics: true
  per_depth_metrics: true
  confidence_intervals: true
  
  # Metadata
  include_metadata: true
  include_timestamp: true
```

## Running Configuration

Execute evaluation using the configuration file:

```bash
# Basic usage
poetry run python -m dctools run config.yaml

# With custom output directory
poetry run python -m dctools run config.yaml --output-dir ./custom_results/

# Verbose output
poetry run python -m dctools run config.yaml --verbose

# Dry run (show what would be executed)
poetry run python -m dctools run config.yaml --dry-run
```

## Environment Variables

Override configuration with environment variables:

```bash
# Dask configuration
export DASK_SCHEDULER=distributed
export DASK_N_WORKERS=4

# Data source paths
export S3_ENDPOINT=s3.example.com
export ARGO_CACHEDIR=/custom/cache

# Run evaluation
poetry run python -m dctools run config.yaml
```

## Advanced Patterns

### Multiple Evaluation Runs

```yaml
evaluation_runs:
  - name: baseline
    variables: [THETAO]
    interpolation:
      method: bilinear
  
  - name: nearest_neighbor
    variables: [THETAO]
    interpolation:
      method: nearest
  
  - name: multi_variable
    variables: [THETAO, SO]
    interpolation:
      method: bilinear
```

### Conditional Execution

```yaml
evaluation:
  # Only run if enough data available
  min_samples: 100
  
  # Skip certain regions if insufficient coverage
  skip_low_coverage: true
  min_coverage_threshold: 0.5
  
  # Time-based filters
  exclude_periods:
    - start: 2023-01-01
      end: 2023-02-28
      reason: insufficient data
```

### Custom Metrics

```yaml
evaluation:
  custom_metrics:
    - name: weighted_rmse
      weights: ocean_area  # field name in dataset
      
    - name: class4_validation
      reference_type: argo
      prediction_type: gridded
```

## Troubleshooting

### Common Issues

**Issue**: Configuration not found
```bash
# Make sure path is correct
poetry run python -m dctools run --help
poetry run python -m dctools run ./config.yaml  # Use ./
```

**Issue**: Invalid variable names
```yaml
# Check available variables in source
evaluation:
  variables: [THETAO, SO, ZOS]  # Case-sensitive
```

**Issue**: Region out of bounds
```yaml
evaluation:
  regions:
    - name: atlantic
      bounds:
        lat: [30, 60]    # Must be within [-90, 90]
        lon: [-80, 0]    # Must be within [-180, 180]
```

## See Also

- **[Quick Start](quickstart.md)** - Basic usage examples
- **[API Reference](../package_docs/api.md)** - Programmatic configuration
- **[Data Challenges](../data_challenges/dc_index.md)** - Challenge-specific configurations
