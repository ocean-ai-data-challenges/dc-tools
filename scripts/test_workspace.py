#!/usr/bin/env python3
"""Quick smoke test for _prepare_workspace config merge."""
import tempfile, os, yaml
from dctools.submission.submission import ModelSubmission
import xarray as xr
import numpy as np

tmpdir = tempfile.mkdtemp()
zarr_path = os.path.join(tmpdir, "test.zarr")
ds = xr.Dataset(
    {"zos": (["time", "lat", "lon"], np.zeros((10, 672, 1440), dtype="float32"))},
    coords={
        "time": np.arange(10),
        "lat": np.linspace(-60, 60, 672),
        "lon": np.linspace(0, 359.75, 1440),
    },
)
ds.to_zarr(zarr_path, mode="w", consolidated=True)

sub = ModelSubmission(model_name="TestModel", data_path=zarr_path, dc_config="dc2")
data_dir = os.path.join(tmpdir, "output")
result = sub._prepare_workspace(data_dir)

if result is None:
    print("ERROR: _prepare_workspace returned None")
    exit(1)

merged_path, args_dict = result
print(f"Config path: {merged_path}")
print(f"File exists: {os.path.exists(merged_path)}")

with open(merged_path) as f:
    config = yaml.safe_load(f)

sources = config.get("sources", [])
src_names = [s.get("dataset") for s in sources]
print(f"Sources: {src_names}")
print(f"dataset_references: {config.get('dataset_references')}")

pred_sources = [s for s in sources if not s.get("observation_dataset", False)]
print(f"Prediction sources: {[s['dataset'] for s in pred_sources]}")

assert "glonet" not in src_names, "GLONET should NOT be in sources!"
assert "TestModel" in src_names, "TestModel MUST be in sources!"
print("ALL CHECKS PASSED")
