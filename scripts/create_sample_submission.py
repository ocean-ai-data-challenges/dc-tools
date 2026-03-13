#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Generate a minimal compliant submission dataset for DC2.

Creates a directory of synthetic per-date Zarr stores (Option B from the
submission template) that passes all validation checks, so participants can
immediately test the submission pipeline before spending time on their real
prediction output.

Usage
-----
::

    # 1. Generate the sample dataset
    python scripts/create_sample_submission.py --output /tmp/sample_model

    # 2. Validate it
    dc-submit validate /tmp/sample_model --model-name SampleModel

    # 3. (Optional) Run the full pipeline
    dc-submit run /tmp/sample_model --model-name SampleModel \
        --data-directory /tmp/dc2_output

Notes
-----
The generated dataset uses random noise and is only meant for **testing the
submission workflow**. Replace it with actual model predictions for a real
benchmark evaluation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


# ── DC2 grid specification ───────────────────────────────────────────
DC2_LAT = np.arange(-78.0, 90.0, 0.25)        # 672 points
DC2_LON = np.arange(-180.0, 180.0, 0.25)       # 1440 points
DC2_DEPTH = np.array([
    0.494025, 47.37369, 92.32607, 155.8507, 222.4752,
    318.1274, 380.213, 453.9377, 541.0889, 643.5668,
    763.3333, 902.3393, 1245.291, 1684.284, 2225.078,
    3220.820, 3597.032, 3992.484, 4405.224, 4833.291,
    5274.784,
], dtype=np.float64)  # 21 levels

# DC2 evaluation period and forecast horizon
DC2_START = "2024-01-01"
DC2_END = "2024-02-01"
DC2_N_DAYS_FORECAST = 10   # lead times 0..9
DC2_N_DAYS_INTERVAL = 7    # one forecast every 7 days


def create_sample_dataset(
    output_path: str | Path,
    *,
    variables: list[str] | None = None,
    seed: int = 42,
    compress: bool = True,
) -> Path:
    """Create a minimal DC2-compliant submission directory.

    Creates a directory of per-date Zarr stores (Option B), each containing a
    10-day forecast on the DC2 grid.

    Parameters
    ----------
    output_path : str or Path
        Directory where per-date .zarr stores will be written.
    variables : list[str], optional
        Ocean variables to include (default: all 5 DC2 variables).
    seed : int
        Random seed for reproducibility.
    compress : bool
        Use Zarr compression (smaller on disk).

    Returns
    -------
    Path
        Path to the created output directory.
    """
    if variables is None:
        variables = ["zos", "thetao", "so", "uo", "vo"]

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    # Generate one forecast file per n_days_interval
    init_dates = pd.date_range(DC2_START, DC2_END, freq=f"{DC2_N_DAYS_INTERVAL}D")
    lead_times = np.arange(DC2_N_DAYS_FORECAST)

    n_files = 0
    for init_date in init_dates:
        # Time axis: actual forecast valid dates = init_date + lead_time
        forecast_times = pd.date_range(
            init_date, periods=DC2_N_DAYS_FORECAST, freq="1D"
        )

        data_vars: dict = {}
        encoding: dict = {}

        for var in variables:
            if var == "zos":
                shape = (len(lead_times), len(DC2_LAT), len(DC2_LON))
                dims = ["time", "lat", "lon"]
            else:
                shape = (len(lead_times), len(DC2_DEPTH), len(DC2_LAT), len(DC2_LON))
                dims = ["time", "depth", "lat", "lon"]

            data = rng.standard_normal(shape).astype(np.float32)
            data_vars[var] = (dims, data)

            if compress:
                encoding[var] = {"chunks": tuple(min(s, 64) for s in shape)}

        ds = xr.Dataset(
            data_vars,
            coords={
                "lat": DC2_LAT,
                "lon": DC2_LON,
                "depth": DC2_DEPTH,
                "time": forecast_times,
            },
        )

        # CF-like metadata
        ds.lat.attrs = {"units": "degrees_north", "long_name": "Latitude"}
        ds.lon.attrs = {"units": "degrees_east", "long_name": "Longitude"}
        ds.depth.attrs = {"units": "m", "long_name": "Depth", "positive": "down"}
        ds.time.attrs = {"long_name": "Forecast valid time", "axis": "T"}

        date_str = init_date.strftime("%Y%m%d")
        store_path = output_path / f"{date_str}.zarr"
        ds.to_zarr(str(store_path), mode="w", consolidated=True,
                   encoding=encoding or None)
        n_files += 1
        ds.close()

    print(f"Sample submission created: {output_path}")
    print(f"  Format    : {n_files} per-date Zarr stores (Option B)")
    print(f"  Variables : {variables}")
    print(f"  Grid      : {len(DC2_LAT)} lat × {len(DC2_LON)} lon × {len(DC2_DEPTH)} depth")
    print(f"  Lead times: {lead_times.tolist()} days")
    print(f"  Period    : {DC2_START} to {DC2_END} (every {DC2_N_DAYS_INTERVAL} days)")

    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a minimal DC2-compliant sample submission.",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="sample_submission",
        help="Output directory for the per-date Zarr stores (default: sample_submission).",
    )
    parser.add_argument(
        "--variables",
        nargs="+",
        default=None,
        metavar="VAR",
        help="Variables to include (default: zos thetao so uo vo).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    args = parser.parse_args()

    create_sample_dataset(
        args.output,
        variables=args.variables,
        seed=args.seed,
    )

    print(f"\nNext steps:")
    print(f"  dc-submit validate {args.output} --model-name SampleModel")
    print(f"  dc-submit run {args.output} --model-name SampleModel --data-directory ./output")

    return 0


if __name__ == "__main__":
    sys.exit(main())
