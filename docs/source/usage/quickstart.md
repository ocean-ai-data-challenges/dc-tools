# Quick Start Guide

This guide will help you get started with dc-tools in 5 minutes. For detailed information, refer to the [API Reference](../package_docs/api.md).

## 1. Basic Setup

After [installing dc-tools](../package_docs/installation.md), activate your environment:

```bash
conda activate dctools
python -c "import dctools; print('✓ dc-tools ready')"
```

## 2. Loading Data

### Load from Argo (in-situ profiles)

```python
from dctools.data import EvaluationDataloader

loader = EvaluationDataloader()

# Load Argo data
argo_profiles = loader.load_dataset(
    source="argo",
    region="north_atlantic",  # optional: specify region
    time_range=("2023-01-01", "2023-12-31")
)

print(argo_profiles)
```

### Load from CMEMS (model output)

```python
# Load CMEMS data
model_data = loader.load_dataset(
    source="cmems",
    dataset="global_forecast",
    variables=["SO", "THETAO", "ZOS"],  # Salinity, Temperature, Sea Surface Height
    region="global"
)

print(model_data)
```

### Load from local files

```python
import xarray as xr

# Load from NetCDF
data = xr.open_dataset("yourfile.nc")

# Or from Zarr
data = xr.open_zarr("yourfile.zarr")
```

## 3. Data Inspection & Coordinate Handling

### Inspect dataset

```python
from dctools.data import CoordinateSystem

# View dataset info
print(argo_profiles)
print(f"Variables: {list(argo_profiles.data_vars)}")
print(f"Coordinates: {list(argo_profiles.coords)}")

# Auto-detect and normalize coordinates
cs = CoordinateSystem(argo_profiles)
standardized = cs.get_target_dimensions()
print(f"Standard coordinates: {standardized}")
```

### Get depth levels

```python
depth_levels = cs.get_target_depth_values()
print(f"Available depths: {depth_levels}")
```

## 4. Processing Data

### Simple transformation

```python
from dctools.data import EvaluationDataloader

# Normalize longitude to [-180, 180]
processed = loader._transform_coordinates(model_data)
```

### Interpolate to a regular grid

```python
from dctools.processing import interpolate_dataset

# Interpolate model output to observation grid
interpolated = interpolate_dataset(
    model_data,
    target_grid=argo_profiles,
    method="bilinear",
    variables=["SO", "THETAO"]
)

print(f"Interpolated shape: {interpolated.dims}")
```

## 5. Computing Metrics

### Calculate basic metrics

```python
from dctools.metrics import MetricComputer

# Compute RMSE and MAE
metric_computer = MetricComputer(
    variables=["SO", "THETAO"],
    mask_land=True
)

metrics = metric_computer.compute(
    prediction=interpolated,
    reference=argo_profiles,
    metrics=["rmse", "mae", "bias"]
)

print(f"RMSE: {metrics['rmse']}")
print(f"MAE: {metrics['mae']}")
print(f"Bias: {metrics['bias']}")
```

### Per-region evaluation

```python
# Define regions (lat, lon boxes)
regions = {
    "north_atlantic": {
        "lat": (30, 60),
        "lon": (-80, 0)
    },
    "tropical": {
        "lat": (-20, 20),
        "lon": (-180, 180)
    }
}

# Evaluate per region
for region_name, bounds in regions.items():
    regional_metrics = metric_computer.compute(
        prediction=interpolated.sel(
            lat=slice(bounds["lat"][0], bounds["lat"][1]),
            lon=slice(bounds["lon"][0], bounds["lon"][1])
        ),
        reference=argo_profiles.sel(
            lat=slice(bounds["lat"][0], bounds["lat"][1]),
            lon=slice(bounds["lon"][0], bounds["lon"][1])
        ),
        metrics=["rmse"]
    )
    print(f"{region_name}: RMSE = {regional_metrics['rmse']}")
```

## 6. Distributed Evaluation (with Dask)

### Setup Dask cluster

```python
from dask.distributed import Client

# Local cluster (4 workers)
client = Client(n_workers=4, threads_per_worker=2, memory_limit='4GB')

# Import dctools after Dask is set up
from dctools.metrics import Evaluator

# Evaluation now uses Dask automatically
evaluator = Evaluator()
results = evaluator.evaluate(
    datasets=[model_data],
    reference=argo_profiles
)

print(results)
client.close()
```

## 7. Saving Results

### Save to NetCDF

```python
from dctools.dcio import DataSaver

saver = DataSaver()

saver.save_dataset(
    interpolated,
    "results.nc",
    format="netcdf4"
)
```

### Save to Zarr (better for large datasets)

```python
saver.save_dataset(
    interpolated,
    "results.zarr",
    format="zarr",
    mode="w-"
)
```

### Save evaluation results as JSON

```python
import json

with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2, default=str)
```

## 8. Configuration-Driven Workflows

For more complex evaluations, use YAML configuration:

### Create `config.yaml`

```yaml
sources:
  model:
    type: gridded
    path: s3://bucket/model_output.nc
    variables: [SO, THETAO, ZOS]
    
  observations:
    type: profiles
    path: ./data/argo_profiles/
    
evaluation:
  grid_spacing: 0.25
  regions: [global, north_atlantic, tropical]
  metrics: [rmse, mae, bias]
  mask_land: true
  
output:
  format: json
  path: ./results/
```

### Run evaluation

```bash
poetry run python -m dctools run config.yaml
```

## 9. Common Tasks

### Check available variables

```python
print(model_data.data_vars)
print(argo_profiles.data_vars)
```

### Subset by time

```python
subset = model_data.sel(
    time=slice("2023-06-01", "2023-08-31")
)
```

### Subset by coordinates

```python
atlantic = model_data.sel(
    lat=slice(30, 60),
    lon=slice(-80, 0)
)
```

### Check and increase memory

```python
import dask
dask.config.set({'distributed.scheduler.worker-ttl': '1 hour'})
dask.config.set({'memory.target-fraction': 0.8})
```

## 10. Next Steps

- **[Installation Guide](../package_docs/installation.md)** - Detailed setup instructions
- **[API Reference](../package_docs/api.md)** - Full API documentation
- **[Data Challenges](../data_challenges/dc_index.md)** - Challenge-specific documentation
- **[GitHub Issues](https://github.com/ppr-ocean-ia/dc-tools/issues)** - Report bugs or ask questions

## 📚 Useful Resources

- [Xarray Documentation](https://xarray.pydata.org/) - Working with labeled arrays
- [Dask Documentation](https://dask.org/) - Distributed computing
- [OceanBench](https://github.com/mercator-ocean/oceanbench) - Metrics library
- [PPR Océan & Climat](https://www.ocean-climat.fr/) - Main initiative
