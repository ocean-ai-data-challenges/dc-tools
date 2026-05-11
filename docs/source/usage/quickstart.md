# Quick Start Guide

This guide shows a minimal workflow aligned with the current package layout.
For detailed behavior and configuration fields, see the
[API Reference](../package_docs/api.md).

## 1. Basic Setup

After [installing dc-tools](../package_docs/installation.md), activate your environment:

```bash
conda activate dctools
python -c "import dctools; print('dc-tools ready')"
```

## 2. Load Datasets

```python
import xarray as xr

# Load from local files (NetCDF or Zarr)
pred_ds = xr.open_dataset("prediction.nc")
ref_ds = xr.open_dataset("reference.nc")
```

If you need automatic format/group handling, use:

```python
from dctools.dcio.loader import FileLoader

pred_ds = FileLoader.open_dataset_auto("prediction.nc")
ref_ds = FileLoader.open_dataset_auto("reference.nc")
```

## 3. Inspect Coordinates

```python
from dctools.data.coordinates import CoordinateSystem

pred_coords = CoordinateSystem(pred_ds)
ref_coords = CoordinateSystem(ref_ds)

print(pred_coords.get_target_dimensions())
print(ref_coords.get_target_dimensions())
```

## 4. Optional Interpolation

```python
from dctools.processing.interpolation import interpolate_dataset

# Target grid is a dictionary of coordinate vectors.
target_grid = {
    "lat": ref_ds["lat"].values,
    "lon": ref_ds["lon"].values,
}

pred_interp = interpolate_dataset(
    pred_ds,
    target_grid=target_grid,
    interpolation_lib="pyinterp",
)
```

## 5. Compute Metrics

```python
from dctools.metrics.metrics import MetricComputer

metric_computer = MetricComputer(eval_variables=["so", "thetao"])

results = metric_computer.compute(
    pred_data=pred_interp,
    ref_data=ref_ds,
    pred_coords=pred_coords,
    ref_coords=ref_coords,
)

print(results)
```

## 6. Save Outputs

```python
from dctools.dcio.saver import DataSaver

DataSaver.save_dataset(pred_interp, "results.nc", file_format="netcdf")
DataSaver.save_dataset(pred_interp, "results.zarr", file_format="zarr", mode="w")
```

## 7. Run the DC2 Pipeline from Config

The challenge entrypoint is script-based:

```bash
poetry run python dc2/evaluate.py -d ./data -c dc2
```

Notes:
- `-d/--data_directory` is required.
- `-c/--config_name` selects `dc2/config/<config_name>.yaml`.

## 8. Next Steps

- **[Installation Guide](../package_docs/installation.md)** - Full setup and troubleshooting
- **[Configuration Guide](config.md)** - YAML options and patterns
- **[API Reference](../package_docs/api.md)** - Module-level documentation
- **[Data Challenges](../data_challenges/dc_index.md)** - Challenge-specific pages
