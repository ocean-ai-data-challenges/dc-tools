#!/usr/bin/env python3

"""Recursively convert NetCDF files in a directory tree to Zarr.

This script is meant to run on Datarmor after the local staging step.
It reads every ``*.nc`` file below a source directory and writes a matching
``.zarr`` store under the target directory, preserving the relative layout.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Set HDF5/NetCDF env vars BEFORE any import/use of netCDF backends.
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
os.environ.setdefault("NETCDF4_DEACTIVATE_MPI", "1")
os.environ.setdefault("NETCDF4_USE_FILE_LOCKING", "FALSE")
os.environ.setdefault("HDF5_DISABLE_VERSION_CHECK", "1")

# Ensure the repository root is importable when running the script directly
# from the scripts/ directory.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import xarray as xr

from dctools.utilities.xarray_utils import netcdf_to_zarr


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recursively convert all NetCDF files under a directory to Zarr."
    )
    parser.add_argument("source_dir", help="Root directory containing NetCDF files")
    parser.add_argument("output_dir", help="Root directory for generated Zarr stores")
    parser.add_argument(
        "--pattern",
        default="**/*.nc",
        help="Glob pattern used to find NetCDF files below source_dir",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .zarr stores if they already exist",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List conversions without writing any output",
    )
    return parser


def convert_tree(source_dir: Path, output_dir: Path, pattern: str, overwrite: bool, dry_run: bool) -> None:
    source_dir = source_dir.resolve()
    output_dir = output_dir.resolve()
    if not source_dir.is_dir():
        raise NotADirectoryError(f"Source directory does not exist: {source_dir}")

    nc_files = sorted(source_dir.glob(pattern))
    print(f"Found {len(nc_files)} NetCDF files under {source_dir}")

    converted = 0
    skipped = 0
    failed = 0

    for nc_path in nc_files:
        if not nc_path.is_file():
            continue

        rel = nc_path.relative_to(source_dir)
        out_path = (output_dir / rel).with_suffix(".zarr")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if out_path.exists() and not overwrite:
            print(f"[skip] {nc_path} -> {out_path} (already exists)")
            skipped += 1
            continue

        print(f"[convert] {nc_path} -> {out_path}")
        if dry_run:
            converted += 1
            continue

        try:
            with xr.open_dataset(nc_path, engine="h5netcdf") as ds:
                netcdf_to_zarr(ds, str(out_path), overwrite=True)
            converted += 1
        except Exception as exc:
            failed += 1
            print(f"[error] {nc_path}: {exc!r}")

    print(
        f"Summary: converted={converted}, skipped={skipped}, failed={failed}, "
        f"source_dir={source_dir}, output_dir={output_dir}"
    )


def main() -> None:
    args = build_parser().parse_args()
    convert_tree(
        Path(args.source_dir),
        Path(args.output_dir),
        args.pattern,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
