#!/usr/bin/env python3

"""Backfill DC2 datasets with an explicit local staging workflow.

This script intentionally does not upload anything to object storage.
It reads NetCDF files from a local staging folder, converts selected files to
Zarr, and writes a conversion manifest for manual review and upload.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import xarray as xr
import yaml


DT_RANGE_RE = re.compile(r"(\d{8}T\d{6}).*?(\d{8}T\d{6})")
DATE_RE = re.compile(r"(\d{8})")


@dataclass(frozen=True)
class DateRange:
    start: date
    end: date

    def overlaps(self, other: "DateRange") -> bool:
        return not (self.end < other.start or self.start > other.end)


def parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def parse_filename_coverage(path: Path) -> DateRange | None:
    """Infer date coverage from common file naming patterns.

    Supported patterns:
    - YYYYMMDDTHHMMSS ... YYYYMMDDTHHMMSS
    - any YYYYMMDD token (single-day fallback)
    """
    name = path.name
    m = DT_RANGE_RE.search(name)
    if m:
        d0 = datetime.strptime(m.group(1), "%Y%m%dT%H%M%S").date()
        d1 = datetime.strptime(m.group(2), "%Y%m%dT%H%M%S").date()
        if d1 < d0:
            d0, d1 = d1, d0
        return DateRange(d0, d1)

    dates = []
    for token in DATE_RE.findall(name):
        try:
            dates.append(datetime.strptime(token, "%Y%m%d").date())
        except ValueError:
            continue
    if dates:
        d = min(dates)
        return DateRange(d, d)
    return None


def load_plan(plan_path: Path) -> dict[str, Any]:
    with plan_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid plan format in {plan_path}")
    return raw


def dataset_missing_ranges(cfg: dict[str, Any]) -> list[DateRange]:
    ranges = []
    for item in cfg.get("missing_ranges", []) or []:
        if not isinstance(item, dict):
            continue
        try:
            ranges.append(DateRange(parse_date(item["start"]), parse_date(item["end"])))
        except Exception:
            continue
    return ranges


def iter_netcdf_files(root: Path, pattern: str) -> list[Path]:
    return sorted(root.glob(pattern))


def convert_netcdf_to_zarr(nc_path: Path, out_path: Path, overwrite: bool) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and overwrite:
        import shutil

        shutil.rmtree(out_path)
    elif out_path.exists() and not overwrite:
        raise FileExistsError(f"Target exists: {out_path}")

    # h5netcdf is robust for most NetCDF4/HDF5 files while keeping memory low.
    with xr.open_dataset(nc_path, engine="h5netcdf") as ds:
        ds.to_zarr(out_path, mode="w", consolidated=True)


def selected_datasets(plan: dict[str, Any], only: set[str] | None) -> dict[str, dict[str, Any]]:
    datasets = plan.get("datasets", {}) or {}
    if only is None:
        return dict(datasets)
    return {k: v for k, v in datasets.items() if k in only}


def run(args: argparse.Namespace) -> None:
    plan = load_plan(Path(args.plan))
    root = Path(args.staging_root).resolve()
    out_root = Path(args.output_root).resolve()
    manifest_path = Path(args.manifest).resolve()
    target_window = DateRange(parse_date(args.target_start), parse_date(args.target_end))

    only = set(args.dataset) if args.dataset else None
    datasets = selected_datasets(plan, only)
    if not datasets:
        raise ValueError("No dataset selected. Check --dataset values or plan file.")

    records: list[dict[str, Any]] = []
    summary: dict[str, dict[str, int]] = {}

    for dataset, cfg in datasets.items():
        mode = str(cfg.get("mode", "netcdf_to_zarr"))
        if mode != "netcdf_to_zarr":
            records.append(
                {
                    "dataset": dataset,
                    "status": "skipped",
                    "reason": f"mode={mode}",
                }
            )
            continue

        stage_subdir = str(cfg.get("staging_subdir", "")).strip("/")
        file_pattern = str(cfg.get("file_pattern", "**/*.nc"))
        missing = dataset_missing_ranges(cfg)
        if not missing:
            missing = [target_window]

        ds_root = root / stage_subdir
        nc_files = iter_netcdf_files(ds_root, file_pattern)
        summary[dataset] = {
            "seen": len(nc_files),
            "selected": 0,
            "converted": 0,
            "failed": 0,
        }

        for nc in nc_files:
            cov = parse_filename_coverage(nc)
            if cov is None and not args.include_unknown_dates:
                continue

            should_take = False
            if cov is None:
                should_take = True
            else:
                if not cov.overlaps(target_window):
                    should_take = False
                else:
                    should_take = any(cov.overlaps(m) for m in missing)

            if not should_take:
                continue

            summary[dataset]["selected"] += 1

            rel = nc.relative_to(ds_root)
            out = (out_root / dataset / rel).with_suffix(".zarr")
            rec = {
                "dataset": dataset,
                "input": str(nc),
                "output": str(out),
                "coverage": None if cov is None else {"start": cov.start.isoformat(), "end": cov.end.isoformat()},
                "status": "pending",
            }

            if args.dry_run:
                rec["status"] = "dry_run"
                records.append(rec)
                continue

            try:
                convert_netcdf_to_zarr(nc, out, overwrite=args.overwrite)
                rec["status"] = "converted"
                summary[dataset]["converted"] += 1
            except Exception as exc:
                rec["status"] = "failed"
                rec["error"] = repr(exc)
                summary[dataset]["failed"] += 1
            records.append(rec)

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "plan": str(Path(args.plan).resolve()),
        "staging_root": str(root),
        "output_root": str(out_root),
        "target_window": {"start": target_window.start.isoformat(), "end": target_window.end.isoformat()},
        "dry_run": bool(args.dry_run),
        "summary": summary,
        "records": records,
    }
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    print(f"Manifest written: {manifest_path}")
    for ds, stats in summary.items():
        print(
            f"{ds}: seen={stats['seen']} selected={stats['selected']} "
            f"converted={stats['converted']} failed={stats['failed']}"
        )
    if args.dry_run:
        print("Dry-run mode: no Zarr output was written.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Local staging NetCDF->Zarr backfill helper for DC2 datasets (no upload)."
    )
    parser.add_argument(
        "--plan",
        default="scripts/dc2_backfill_plan.yaml",
        help="YAML plan file defining datasets, staging subdirs, and missing ranges.",
    )
    parser.add_argument(
        "--staging-root",
        required=True,
        help="Local staging root containing raw NetCDF files.",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Local output root where converted Zarr stores will be written.",
    )
    parser.add_argument(
        "--manifest",
        default="output_ais_drifter_comparison/dc2_backfill_manifest.json",
        help="Manifest JSON path written after conversion.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        help="Dataset key from plan (repeatable). If omitted, process all.",
    )
    parser.add_argument("--target-start", default="2023-01-01")
    parser.add_argument("--target-end", default="2024-12-31")
    parser.add_argument("--dry-run", action="store_true", help="Build manifest only.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing .zarr stores.")
    parser.add_argument(
        "--include-unknown-dates",
        action="store_true",
        help="Also process files where date coverage cannot be inferred from filename.",
    )
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
