"""Dataset validation engine for Data Challenge submissions.

Performs systematic checks on a participant's prediction dataset to ensure
it conforms to the Data Challenge specification before evaluation is launched.

Checks implemented (inspired by WeatherBench2 / OceanBench):

1. **File readability** – Can the dataset be opened with xarray?
2. **Required variables** – Are expected ocean variables present?
3. **Coordinate axes** – Are lat/lon/depth/time dimensions present?
4. **Spatial resolution** – Does the grid match the target resolution?
5. **Spatial extent** – Does the domain cover the expected region?
6. **Depth levels** – Do the vertical levels match the target grid?
7. **Temporal coverage** – Does the time axis cover the challenge period?
8. **Lead-time axis** – (Forecast mode) Is the forecast horizon correct?
9. **Data integrity** – NaN fraction check per variable.
10. **Data types** – Are variables stored with appropriate precision?
11. **File size** – Sanity check on total data volume.

Usage
-----
>>> from dctools.submission.validator import SubmissionValidator
>>> v = SubmissionValidator.from_dc_config("dc2")
>>> report = v.validate("/path/to/my_model_output.zarr")
>>> print(report.pretty())
"""

from __future__ import annotations

import glob as _glob_mod
import os
import re
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from loguru import logger

from dctools.submission.report import (
    CheckResult,
    CheckStatus,
    ValidationReport,
)

# ---------------------------------------------------------------------------
# Progress bar helper (uses rich which is already a project dependency)
# ---------------------------------------------------------------------------
try:
    from rich.progress import (
        Progress,
        BarColumn,
        TextColumn,
        TimeRemainingColumn,
        MofNCompleteColumn,
    )
    _HAS_RICH = True
except ImportError:  # pragma: no cover
    _HAS_RICH = False

# ---------------------------------------------------------------------------
# Lazy xarray import (heavy dependency)
# ---------------------------------------------------------------------------
try:
    import xarray as xr
except ImportError:  # pragma: no cover
    xr = None  # type: ignore[assignment]


@dataclass
class InputInfo:
    """Resolved layout of the submitted prediction data.

    Attributes
    ----------
    mode : str
        ``'single'`` when the submission is a single Zarr / NetCDF covering
        the full period, ``'multi'`` when it is a directory (or glob) of
        individual forecast files.
    files : list[str]
        All discovered file paths.
    sample_path : str
        Path used for opening the *sample* dataset (structural checks).
    root_dir : str
        Root directory that contains all the files.
    format : str
        ``'zarr'``, ``'netcdf'``, ``'mixed'``, or ``'unknown'``.
    original_path : str
        The path originally provided by the user.
    """

    mode: str
    files: List[str]
    sample_path: str
    root_dir: str
    format: str
    original_path: str


class SubmissionValidator:
    """Validates a prediction dataset against the Data Challenge specification.

    Parameters
    ----------
    target_lat : array-like
        Expected latitude values.
    target_lon : array-like
        Expected longitude values.
    target_depth : array-like
        Expected depth levels (meters).
    target_time_values : list[int] | None
        Expected forecast lead-time steps (e.g. ``[0, 1, ..., 9]``) for
        forecast mode.  ``None`` for hindcast / analysis datasets.
    start_time : str
        Start of the challenge temporal window (ISO format).
    end_time : str
        End of the challenge temporal window (ISO format).
    required_variables : list[str]
        Variable names that *must* be present in the submission.
    optional_variables : list[str]
        Variable names that are checked but whose absence only triggers a
        warning (not a failure).
    max_nan_fraction : float
        Maximum tolerated NaN fraction per variable (0-1).
    model_name : str
        Human-readable identifier for the submitted model.
    n_days_forecast : int
        Forecast horizon in days.
    spatial_atol : float
        Absolute tolerance for spatial coordinate matching (degrees).
    depth_atol : float
        Absolute tolerance for depth level matching (meters).
    """

    # Class-level defaults ↓
    DEFAULT_MAX_NAN_FRACTION = 0.10
    DEFAULT_SPATIAL_ATOL = 0.01  # degrees
    DEFAULT_DEPTH_ATOL = 1.0  # metres

    def __init__(
        self,
        *,
        target_lat: Any = None,
        target_lon: Any = None,
        target_depth: Any = None,
        target_time_values: Optional[List[int]] = None,
        start_time: str = "2024-01-01",
        end_time: str = "2025-01-01",
        required_variables: Optional[List[str]] = None,
        optional_variables: Optional[List[str]] = None,
        max_nan_fraction: float = DEFAULT_MAX_NAN_FRACTION,
        model_name: str = "unnamed_model",
        n_days_forecast: int = 10,
        n_days_interval: int = 7,
        spatial_atol: float = DEFAULT_SPATIAL_ATOL,
        depth_atol: float = DEFAULT_DEPTH_ATOL,
    ) -> None:
        self.target_lat = np.asarray(target_lat) if target_lat is not None else None
        self.target_lon = np.asarray(target_lon) if target_lon is not None else None
        self.target_depth = np.asarray(target_depth) if target_depth is not None else None
        self.target_time_values = target_time_values
        self.start_time = start_time
        self.end_time = end_time
        self.required_variables = required_variables or []
        self.optional_variables = optional_variables or []
        self.max_nan_fraction = max_nan_fraction
        self.model_name = model_name
        self.n_days_forecast = n_days_forecast
        self.n_days_interval = n_days_interval
        self.spatial_atol = spatial_atol
        self.depth_atol = depth_atol

    # =================================================================
    # Factory: build from a DC YAML config name
    # =================================================================
    @classmethod
    def from_dc_config(
        cls,
        config_name: str = "dc2",
        *,
        model_name: str = "unnamed_model",
        max_nan_fraction: float = DEFAULT_MAX_NAN_FRACTION,
        variables: Optional[List[str]] = None,
    ) -> "SubmissionValidator":
        """Build a validator from an existing DC YAML config file.

        Parameters
        ----------
        config_name : str
            Name of the config (without ``.yaml``), looked up in
            ``dc/config/<name>.yaml``.
        model_name : str
            Human-readable model identifier.
        max_nan_fraction : float
            Maximum NaN ratio tolerated per variable.
        variables : list[str] or None
            If given, override the required variables list (partial
            submission).  Only the listed variables will be validated;
            all others become optional.

        Returns
        -------
        SubmissionValidator
        """
        from dctools.utilities.coordinates import get_target_dimensions, get_target_time_values
        from dctools.utilities.args_config import load_args_and_config

        # Resolve config path
        config_dir = Path(__file__).resolve().parents[2] / "dc" / "config"
        config_path = config_dir / f"{config_name}.yaml"
        if not config_path.exists():
            raise FileNotFoundError(
                f"DC config not found: {config_path}. "
                f"Available configs: {[p.stem for p in config_dir.glob('*.yaml')]}"
            )

        # Build a minimal Namespace to avoid CLI arg parsing
        from argparse import Namespace as _Ns
        _dummy_args = _Ns(data_directory="/tmp/dc_validate", logfile=None, jsonfile=None,
                          metric=None, config_name=config_name)
        args = load_args_and_config(str(config_path), args=_dummy_args)
        if args is None:
            raise RuntimeError(f"Failed to load config from {config_path}")

        target_dims = get_target_dimensions(args)
        target_time_vals = get_target_time_values(args)

        # Collect required variables from all prediction sources
        required_vars: List[str] = []
        optional_vars: List[str] = []
        for source in getattr(args, "sources", []) or []:
            if not isinstance(source, dict):
                continue
            is_obs = source.get("observation_dataset", False)
            if is_obs:
                continue
            eval_vars = source.get("eval_variables") or source.get("keep_variables") or []
            # Filter out coordinate names
            coord_names = {"time", "lat", "lon", "depth", "latitude", "longitude"}
            data_vars = [v for v in eval_vars if v.lower() not in coord_names]
            required_vars.extend(data_vars)

        # Deduplicate: keep raw names but drop those whose standardized
        # form already appears via an earlier entry.
        from dctools.utilities.coordinates import get_standardized_var_name as _std
        seen_std: set = set()
        deduped: List[str] = []
        for v in required_vars:
            std = _std(v) or v
            if std not in seen_std:
                seen_std.add(std)
                deduped.append(v)
        required_vars = deduped

        # --- Partial submission: caller specifies a subset of variables ---
        if variables is not None:
            from dctools.utilities.coordinates import get_standardized_var_name as _std2
            # Standardize the caller's variable list
            requested_std = {_std2(v) or v for v in variables}
            # Move unrequested required vars --> optional (warning, not failure)
            new_required: List[str] = []
            demoted: List[str] = []
            for v in required_vars:
                v_std = _std2(v) or v
                if v_std in requested_std:
                    new_required.append(v)
                else:
                    demoted.append(v)
            # Also accept caller names that may not be in config (e.g. aliases)
            for v in variables:
                v_std = _std2(v) or v
                if v_std not in {_std2(r) or r for r in new_required}:
                    new_required.append(v)
            required_vars = new_required
            optional_vars = optional_vars + demoted
            logger.info(
                f"Partial submission: validating {required_vars} "
                f"({len(demoted)} variable(s) demoted to optional)."
            )

        return cls(
            target_lat=target_dims.get("lat"),
            target_lon=target_dims.get("lon"),
            target_depth=target_dims.get("depth"),
            target_time_values=target_time_vals,
            start_time=getattr(args, "start_time", "2024-01-01"),
            end_time=getattr(args, "end_time", "2025-01-01"),
            required_variables=required_vars,
            optional_variables=optional_vars,
            max_nan_fraction=max_nan_fraction,
            model_name=model_name,
            n_days_forecast=int(getattr(args, "n_days_forecast", 10)),
            n_days_interval=int(getattr(args, "n_days_interval", 7)),
            spatial_atol=cls.DEFAULT_SPATIAL_ATOL,
            depth_atol=cls.DEFAULT_DEPTH_ATOL,
        )

    # =================================================================
    # Main validate method
    # =================================================================
    def validate(
        self,
        data_path: str | Path,
        *,
        quick: bool = False,
    ) -> ValidationReport:
        """Run all validation checks on a submitted dataset.

        Supported input layouts:

        - Single ``.zarr`` directory (one store covering the full period).
        - Single ``.nc`` / ``.nc4`` file.
        - A **directory** containing one ``.zarr`` or ``.nc`` file per forecast
          init date (e.g. ``20240103.zarr``, ``20240110.zarr``, …).
        - A **glob pattern** matching multiple files.

        For multi-file inputs the structural checks (variables, grid, depth,
        lead-time, dtypes) are performed on a **sample** file and the temporal
        coverage is verified across **all** files.

        Parameters
        ----------
        data_path : str or Path
            Path (local or S3) to the submitted prediction dataset.
        quick : bool
            If ``True``, skip expensive checks (NaN scan on full data).

        Returns
        -------
        ValidationReport
        """
        data_path = str(data_path)
        report = ValidationReport(
            model_name=self.model_name,
            data_path=data_path,
        )

        # 0. Resolve input layout (single file vs directory of files).
        input_info = self._resolve_input(data_path, report)
        if input_info is None:
            return report

        # 1. Open the (sample) dataset.
        ds = self._open_sample(input_info, report)
        if ds is None:
            return report

        # 2-6. Structural checks (on the sample dataset).
        _checks: List[tuple] = [
            ("Required variables", lambda: self._check_required_variables(ds, report)),
            ("Coordinate axes", lambda: self._check_coordinate_axes(ds, report)),
            ("Spatial resolution", lambda: self._check_spatial_resolution(ds, report)),
            ("Spatial extent", lambda: self._check_spatial_extent(ds, report)),
            ("Depth levels", lambda: self._check_depth_levels(ds, report)),
        ]

        # 7. Temporal coverage.
        if input_info.mode == "multi":
            _checks.append(("Temporal coverage", lambda: self._check_temporal_coverage_multi(input_info, report)))
        else:
            _checks.append(("Temporal coverage", lambda: self._check_temporal_coverage(ds, report)))

        # 8. Lead-time (on sample).
        _checks.append(("Lead-time axis", lambda: self._check_lead_time(ds, report)))

        # 9. NaN (on sample).
        if not quick:
            _checks.append(("NaN fraction", lambda: self._check_nan_fraction(ds, report)))

        # 10. Dtypes (on sample).
        _checks.append(("Data types", lambda: self._check_dtypes(ds, report)))

        # 11. File size (across all files).
        _checks.append(("File size", lambda: self._check_file_size_from_input(input_info, report)))

        # Run all checks with an optional progress bar
        if _HAS_RICH:
            with Progress(
                TextColumn("[bold cyan]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                transient=True,
            ) as progress:
                task = progress.add_task("Validation checks", total=len(_checks))
                for name, fn in _checks:
                    progress.update(task, description=f"Checking: {name}")
                    fn()
                    progress.advance(task)
        else:
            for _name, fn in _checks:
                fn()

        try:
            ds.close()
        except Exception:
            pass

        return report

    # =================================================================
    # Input resolution
    # =================================================================

    @staticmethod
    def _is_zarr_store(path: Path) -> bool:
        """Return *True* if *path* looks like a Zarr store directory."""
        if not path.is_dir():
            return False
        return (
            path.suffix == ".zarr"
            or (path / ".zmetadata").exists()
            or (path / ".zarray").exists()
            or (path / ".zattrs").exists()
        )

    def _resolve_input(
        self, data_path: str, report: ValidationReport
    ) -> Optional[InputInfo]:
        """Detect the data layout and list all files.

        Returns an :class:`InputInfo` or ``None`` (with a FAIL added to the
        report) when the path cannot be resolved.
        """
        path = Path(data_path)

        # -- Glob pattern ------------------------------------------
        if "*" in data_path or "?" in data_path:
            files = sorted(_glob_mod.glob(data_path, recursive=True))
            if not files:
                report.add(CheckResult(
                    name="Input resolution",
                    status=CheckStatus.FAIL,
                    message=f"No files matched glob pattern: {data_path}",
                ))
                return None
            fmt = "zarr" if files[0].endswith(".zarr") else "netcdf"
            root = str(Path(files[0]).parent)
            info = InputInfo("multi", files, files[0], root, fmt, data_path)
            report.add(CheckResult(
                name="Input resolution",
                status=CheckStatus.PASS,
                message=(
                    f"Glob pattern matched {len(files)} file(s) "
                    f"(format: {fmt})."
                ),
                details={"n_files": len(files), "sample": files[0]},
            ))
            return info

        # -- Single Zarr store (.zarr directory) -------------------
        if self._is_zarr_store(path):
            info = InputInfo(
                "single", [data_path], data_path,
                str(path.parent), "zarr", data_path,
            )
            report.add(CheckResult(
                name="Input resolution",
                status=CheckStatus.PASS,
                message="Single Zarr store detected.",
            ))
            return info

        # -- Single file (.nc, .nc4, .hdf5, …) --------------------
        if path.is_file():
            info = InputInfo(
                "single", [data_path], data_path,
                str(path.parent), "netcdf", data_path,
            )
            report.add(CheckResult(
                name="Input resolution",
                status=CheckStatus.PASS,
                message=f"Single file detected ({path.suffix}).",
            ))
            return info

        # -- Directory containing multiple forecast files ----------
        if path.is_dir():
            zarr_stores = sorted(
                str(d) for d in path.iterdir()
                if d.is_dir() and self._is_zarr_store(d)
            )
            nc_files = sorted(
                str(f) for f in path.rglob("*")
                if f.is_file() and f.suffix in (".nc", ".nc4", ".netcdf")
            )
            all_files = zarr_stores + nc_files
            if not all_files:
                report.add(CheckResult(
                    name="Input resolution",
                    status=CheckStatus.FAIL,
                    message=(
                        f"Directory '{data_path}' contains no Zarr stores "
                        f"or NetCDF files."
                    ),
                ))
                return None

            if zarr_stores and nc_files:
                fmt = "mixed"
            elif zarr_stores:
                fmt = "zarr"
            else:
                fmt = "netcdf"

            info = InputInfo(
                "multi", all_files, all_files[0], data_path, fmt, data_path,
            )
            report.add(CheckResult(
                name="Input resolution",
                status=CheckStatus.PASS,
                message=(
                    f"Directory with {len(all_files)} forecast file(s) detected "
                    f"(format: {fmt})."
                ),
                details={
                    "n_files": len(all_files),
                    "n_zarr": len(zarr_stores),
                    "n_netcdf": len(nc_files),
                    "sample": all_files[0],
                },
            ))
            return info

        # -- Path does not exist -----------------------------------
        report.add(CheckResult(
            name="Input resolution",
            status=CheckStatus.FAIL,
            message=f"Path does not exist or is not readable: {data_path}",
        ))
        return None

    # =================================================================
    # Individual checks
    # =================================================================

    def _open_sample(
        self, input_info: InputInfo, report: ValidationReport
    ) -> Optional[Any]:
        """Open the sample dataset and add a readability check to the report.

        For single-file inputs the dataset is opened directly.  For multi-file
        inputs the *first* discovered file is opened so that structural checks
        can be performed on a representative sample.
        """
        if xr is None:
            report.add(CheckResult(
                name="Dataset readability",
                status=CheckStatus.FAIL,
                message="xarray is not installed.",
            ))
            return None

        sample = input_info.sample_path

        try:
            if sample.endswith(".zarr") or self._is_zarr_store(Path(sample)):
                ds = xr.open_zarr(sample, consolidated=True)
            elif input_info.mode == "single" and (
                "*" in sample or "?" in sample
            ):
                ds = xr.open_mfdataset(sample, lock=False)
            else:
                # Try multiple engines in order of preference
                _engines = ["h5netcdf", "netcdf4", "scipy", None]
                ds = None
                last_exc = None
                for engine in _engines:
                    try:
                        kw: Dict[str, Any] = {"lock": False}
                        if engine is not None:
                            kw["engine"] = engine
                        ds = xr.open_dataset(sample, **kw)
                        break
                    except Exception as exc:
                        last_exc = exc
                        continue
                if ds is None:
                    raise last_exc  # type: ignore[misc]

            suffix = ""
            if input_info.mode == "multi":
                suffix = (
                    f" (sample: {Path(sample).name}, "
                    f"{len(input_info.files)} files total)"
                )

            report.add(CheckResult(
                name="Dataset readability",
                status=CheckStatus.PASS,
                message=(
                    f"Successfully opened dataset ({len(ds.data_vars)} variables, "
                    f"{dict(ds.sizes)} dimensions){suffix}."
                ),
                details={
                    "variables": list(ds.data_vars),
                    "dims": dict(ds.sizes),
                },
            ))
            return ds

        except Exception as exc:
            report.add(CheckResult(
                name="Dataset readability",
                status=CheckStatus.FAIL,
                message=f"Cannot open dataset: {exc}",
            ))
            return None

    def _check_required_variables(
        self, ds: Any, report: ValidationReport
    ) -> None:
        """Check 2: Required ocean variables."""
        from dctools.utilities.coordinates import get_standardized_var_name

        ds_vars = set(ds.data_vars)
        ds_coords = set(ds.coords)
        all_names = ds_vars | ds_coords

        missing: List[str] = []
        found: List[str] = []
        for var in self.required_variables:
            # Check direct name or via alias resolution
            if var in all_names:
                found.append(var)
                continue
            std = get_standardized_var_name(var)
            alias_found = False
            for ds_var in all_names:
                if get_standardized_var_name(ds_var) == std:
                    found.append(f"{var} (as {ds_var})")
                    alias_found = True
                    break
            if not alias_found:
                missing.append(var)

        if missing:
            report.add(CheckResult(
                name="Required variables",
                status=CheckStatus.FAIL,
                message=f"Missing variables: {missing}",
                details={"missing": missing, "found": found, "available": sorted(ds_vars)},
            ))
        else:
            report.add(CheckResult(
                name="Required variables",
                status=CheckStatus.PASS,
                message=f"All {len(self.required_variables)} required variable(s) found.",
                details={"found": found},
            ))

        # Optional variables --> warnings
        for var in self.optional_variables:
            if var not in all_names:
                std = get_standardized_var_name(var)
                alias_found = any(
                    get_standardized_var_name(dv) == std for dv in all_names
                )
                if not alias_found:
                    report.add(CheckResult(
                        name=f"Optional variable: {var}",
                        status=CheckStatus.WARN,
                        message=f"Optional variable '{var}' not found.",
                    ))

    def _check_coordinate_axes(
        self, ds: Any, report: ValidationReport
    ) -> None:
        """Check 3: Required coordinate axes."""
        required_axes = ["lat", "lon", "depth", "time"]
        _aliases = {
            "lat": ["lat", "latitude", "nav_lat", "y"],
            "lon": ["lon", "longitude", "nav_lon", "x"],
            "depth": ["depth", "lev", "level", "deptht", "z"],
            "time": ["time", "valid_time", "forecast_time", "lead_time", "date"],
        }

        ds_names = set(str(c) for c in ds.coords) | set(str(d) for d in ds.dims)

        for axis in required_axes:
            aliases = _aliases.get(axis, [axis])
            found_name = None
            for alias in aliases:
                if alias in ds_names:
                    found_name = alias
                    break

            if found_name:
                report.add(CheckResult(
                    name=f"Coordinate axis: {axis}",
                    status=CheckStatus.PASS,
                    message=f"Found as '{found_name}'.",
                ))
            else:
                report.add(CheckResult(
                    name=f"Coordinate axis: {axis}",
                    status=CheckStatus.FAIL,
                    message=f"Axis '{axis}' not found. Looked for: {aliases}. "
                            f"Available: {sorted(ds_names)}",
                ))

    def _find_coord(self, ds: Any, axis: str) -> Optional[np.ndarray]:
        """Resolve a coordinate axis in the dataset, trying common aliases."""
        _aliases = {
            "lat": ["lat", "latitude", "nav_lat", "y"],
            "lon": ["lon", "longitude", "nav_lon", "x"],
            "depth": ["depth", "lev", "level", "deptht", "z"],
            "time": ["time", "valid_time", "forecast_time", "lead_time", "date"],
        }
        for alias in _aliases.get(axis, [axis]):
            if alias in ds.coords:
                return np.asarray(ds[alias].values)
            if alias in ds.dims:
                try:
                    return np.asarray(ds[alias].values)
                except Exception:
                    pass
        return None

    def _check_spatial_resolution(
        self, ds: Any, report: ValidationReport
    ) -> None:
        """Check 4: Grid resolution matches target."""
        for axis_name, target in [("lat", self.target_lat), ("lon", self.target_lon)]:
            if target is None:
                report.add(CheckResult(
                    name=f"Spatial resolution: {axis_name}",
                    status=CheckStatus.SKIP,
                    message="No target grid configured.",
                ))
                continue

            actual = self._find_coord(ds, axis_name)
            if actual is None:
                report.add(CheckResult(
                    name=f"Spatial resolution: {axis_name}",
                    status=CheckStatus.FAIL,
                    message=f"Axis '{axis_name}' not found in dataset.",
                ))
                continue

            target_arr = np.sort(np.asarray(target, dtype=float))
            actual_arr = np.sort(np.asarray(actual, dtype=float))

            # Check step (resolution)
            if len(target_arr) > 1 and len(actual_arr) > 1:
                expected_step = np.median(np.diff(target_arr))
                actual_step = np.median(np.diff(actual_arr))
                step_match = abs(expected_step - actual_step) <= self.spatial_atol

                if step_match:
                    report.add(CheckResult(
                        name=f"Spatial resolution: {axis_name}",
                        status=CheckStatus.PASS,
                        message=f"Step = {actual_step:.4f}° (expected {expected_step:.4f}°).",
                    ))
                else:
                    report.add(CheckResult(
                        name=f"Spatial resolution: {axis_name}",
                        status=CheckStatus.FAIL,
                        message=f"Step = {actual_step:.4f}° but expected {expected_step:.4f}°.",
                        details={
                            "expected_step": float(expected_step),
                            "actual_step": float(actual_step),
                        },
                    ))
            else:
                report.add(CheckResult(
                    name=f"Spatial resolution: {axis_name}",
                    status=CheckStatus.WARN,
                    message=f"Cannot verify resolution (only {len(actual_arr)} point(s)).",
                ))

    def _check_spatial_extent(
        self, ds: Any, report: ValidationReport
    ) -> None:
        """Check 5: Spatial domain covers the expected region."""
        for axis_name, target in [("lat", self.target_lat), ("lon", self.target_lon)]:
            if target is None:
                continue
            actual = self._find_coord(ds, axis_name)
            if actual is None:
                continue  # already flagged elsewhere

            target_arr = np.asarray(target, dtype=float)
            actual_arr = np.asarray(actual, dtype=float)

            t_min, t_max = float(np.nanmin(target_arr)), float(np.nanmax(target_arr))
            a_min, a_max = float(np.nanmin(actual_arr)), float(np.nanmax(actual_arr))

            covers_min = a_min <= t_min + self.spatial_atol
            covers_max = a_max >= t_max - self.spatial_atol

            if covers_min and covers_max:
                report.add(CheckResult(
                    name=f"Spatial extent: {axis_name}",
                    status=CheckStatus.PASS,
                    message=f"Covers [{a_min:.2f}, {a_max:.2f}] "
                            f"(target [{t_min:.2f}, {t_max:.2f}]).",
                ))
            else:
                parts: List[str] = []
                if not covers_min:
                    parts.append(f"min={a_min:.2f} > target_min={t_min:.2f}")
                if not covers_max:
                    parts.append(f"max={a_max:.2f} < target_max={t_max:.2f}")
                report.add(CheckResult(
                    name=f"Spatial extent: {axis_name}",
                    status=CheckStatus.FAIL,
                    message=f"Incomplete coverage: {'; '.join(parts)}.",
                    details={
                        "actual_range": [a_min, a_max],
                        "target_range": [t_min, t_max],
                    },
                ))

    def _check_depth_levels(
        self, ds: Any, report: ValidationReport
    ) -> None:
        """Check 6: Depth levels match the target grid."""
        if self.target_depth is None:
            report.add(CheckResult(
                name="Depth levels",
                status=CheckStatus.SKIP,
                message="No target depth levels configured.",
            ))
            return

        actual = self._find_coord(ds, "depth")
        if actual is None:
            report.add(CheckResult(
                name="Depth levels",
                status=CheckStatus.FAIL,
                message="Depth axis not found in dataset.",
            ))
            return

        target_arr = np.sort(np.asarray(self.target_depth, dtype=float))
        actual_arr = np.sort(np.asarray(actual, dtype=float))

        # Check each target level has a match in actual
        missing_levels: List[float] = []
        for t_val in target_arr:
            if actual_arr.size == 0:
                missing_levels.append(float(t_val))
                continue
            closest_idx = int(np.argmin(np.abs(actual_arr - t_val)))
            if abs(actual_arr[closest_idx] - t_val) > self.depth_atol:
                missing_levels.append(float(t_val))

        extra_levels: List[float] = []
        for a_val in actual_arr:
            if target_arr.size == 0:
                extra_levels.append(float(a_val))
                continue
            closest_idx = int(np.argmin(np.abs(target_arr - a_val)))
            if abs(target_arr[closest_idx] - a_val) > self.depth_atol:
                extra_levels.append(float(a_val))

        if not missing_levels and not extra_levels:
            report.add(CheckResult(
                name="Depth levels",
                status=CheckStatus.PASS,
                message=f"All {len(target_arr)} target depth level(s) matched.",
            ))
        elif missing_levels:
            report.add(CheckResult(
                name="Depth levels",
                status=CheckStatus.FAIL,
                message=f"{len(missing_levels)} target depth level(s) missing.",
                details={
                    "missing": missing_levels[:10],
                    "n_expected": len(target_arr),
                    "n_actual": len(actual_arr),
                },
            ))
        else:
            # Extra levels only --> warning
            report.add(CheckResult(
                name="Depth levels",
                status=CheckStatus.WARN,
                message=f"{len(extra_levels)} extra depth level(s) present "
                        f"(not in target grid).",
                details={"extra": extra_levels[:10]},
            ))

    def _check_temporal_coverage(
        self, ds: Any, report: ValidationReport
    ) -> None:
        """Check 7: Time axis covers the challenge temporal window."""
        time_vals = self._find_coord(ds, "time")
        if time_vals is None:
            report.add(CheckResult(
                name="Temporal coverage",
                status=CheckStatus.FAIL,
                message="Time axis not found in dataset.",
            ))
            return

        if time_vals.size == 0:
            report.add(CheckResult(
                name="Temporal coverage",
                status=CheckStatus.FAIL,
                message="Time axis is empty.",
            ))
            return

        # Detect numeric lead-time axis (small integers like [0,1,...,9])
        # vs actual datetime timestamps.
        if np.issubdtype(time_vals.dtype, np.integer) or (
            np.issubdtype(time_vals.dtype, np.floating)
            and time_vals.size > 0
            and float(np.nanmax(np.abs(time_vals))) < 1e6
        ):
            # Treat as lead-time indices, not calendar dates.
            # Temporal range is verified by the lead-time check instead.
            report.add(CheckResult(
                name="Temporal coverage",
                status=CheckStatus.PASS,
                message=(
                    f"Time axis is numeric (likely lead-time indices, "
                    f"{time_vals.size} steps). Temporal range validation "
                    f"deferred to the lead-time check."
                ),
                details={"dtype": str(time_vals.dtype), "n_steps": int(time_vals.size)},
            ))
            return

        try:
            time_arr = np.asarray(time_vals, dtype="datetime64[ns]")
        except Exception:
            # Cannot interpret as datetime; still acceptable
            report.add(CheckResult(
                name="Temporal coverage",
                status=CheckStatus.WARN,
                message="Time axis is not datetime-like; cannot verify temporal range.",
                details={"dtype": str(time_vals.dtype) if hasattr(time_vals, "dtype") else "unknown"},
            ))
            return

        # Guard against epoch-zero timestamps (integer lead times that were
        # silently cast to datetime64 nanoseconds from 1970-01-01).
        _epoch_start = np.datetime64("1970-01-02")
        if np.nanmax(time_arr) < _epoch_start:
            report.add(CheckResult(
                name="Temporal coverage",
                status=CheckStatus.PASS,
                message=(
                    f"Time axis values are near epoch zero ({time_arr.size} steps); "
                    f"likely numeric lead-time indices. Temporal range validation "
                    f"deferred to the lead-time check."
                ),
                details={"dtype": str(time_vals.dtype), "n_steps": int(time_arr.size)},
            ))
            return

        t_start = np.datetime64(self.start_time)
        t_end = np.datetime64(self.end_time)
        actual_start = np.nanmin(time_arr)
        actual_end = np.nanmax(time_arr)

        # Allow 1-day tolerance
        _1day = np.timedelta64(1, "D")
        covers_start = actual_start <= t_start + _1day
        covers_end = actual_end >= t_end - _1day - np.timedelta64(self.n_days_forecast, "D")

        if covers_start and covers_end:
            report.add(CheckResult(
                name="Temporal coverage",
                status=CheckStatus.PASS,
                message=f"Covers {actual_start} to {actual_end} "
                        f"(target: {self.start_time} to {self.end_time}).",
            ))
        else:
            parts: List[str] = []
            if not covers_start:
                parts.append(f"starts at {actual_start} (expected ≤ {self.start_time})")
            if not covers_end:
                parts.append(f"ends at {actual_end} (expected ≥ ~{self.end_time})")
            report.add(CheckResult(
                name="Temporal coverage",
                status=CheckStatus.FAIL,
                message=f"Insufficient temporal coverage: {'; '.join(parts)}.",
                details={
                    "actual_range": [str(actual_start), str(actual_end)],
                    "target_range": [self.start_time, self.end_time],
                },
            ))

    def _check_temporal_coverage_multi(
        self, input_info: InputInfo, report: ValidationReport
    ) -> None:
        """Check 7 (multi-file): temporal coverage across all forecast files.

        Strategy:

        1. Try to extract init dates from **filenames** (``YYYYMMDD`` pattern).
        2. Fallback: open each file lazily and read the first time value.
        3. Verify the extracted dates cover ``[start_time, end_time)``.
        """
        _date_re = re.compile(r"(\d{8})")

        # --- Step 1: Try filename-based date extraction ---------------
        file_dates: List[datetime] = []
        for fpath in input_info.files:
            stem = Path(fpath).stem
            m = _date_re.search(stem)
            if m:
                try:
                    dt = datetime.strptime(m.group(1), "%Y%m%d")
                    # Sanity: discard if clearly outside 1950-2100
                    if 1950 <= dt.year <= 2100:
                        file_dates.append(dt)
                except ValueError:
                    pass

        # --- Step 2: Fallback – open files to read time ---------------
        if not file_dates:
            logger.info(
                "Could not extract dates from filenames; "
                "reading time axis from each file (may be slow)."
            )
            file_iter: Any = input_info.files
            if _HAS_RICH and len(input_info.files) > 5:
                _progress = Progress(
                    TextColumn("[bold blue]Reading dates"),
                    BarColumn(),
                    MofNCompleteColumn(),
                    TimeRemainingColumn(),
                    transient=True,
                )
                _progress.start()
                _task = _progress.add_task("files", total=len(input_info.files))
            else:
                _progress = None
                _task = None

            for fpath in input_info.files:
                try:
                    if fpath.endswith(".zarr") or self._is_zarr_store(Path(fpath)):
                        ds_tmp = xr.open_zarr(fpath, consolidated=True)
                    else:
                        ds_tmp = xr.open_dataset(fpath, lock=False)
                    time_vals = self._find_coord(ds_tmp, "time")
                    ds_tmp.close()
                    if time_vals is None or time_vals.size == 0:
                        if _progress is not None:
                            _progress.advance(_task)
                        continue
                    # If time is numeric lead-time, we cannot get a calendar date
                    if np.issubdtype(time_vals.dtype, np.number):
                        if _progress is not None:
                            _progress.advance(_task)
                        continue
                    t0 = np.datetime64(time_vals.flat[0], "ns")
                    file_dates.append(
                        datetime.utcfromtimestamp(
                            (t0 - np.datetime64("1970-01-01T00:00:00"))
                            / np.timedelta64(1, "s")
                        )
                    )
                except Exception as exc:
                    logger.debug(f"Cannot read time from {fpath}: {exc}")
                if _progress is not None:
                    _progress.advance(_task)

            if _progress is not None:
                _progress.stop()

        if not file_dates:
            report.add(CheckResult(
                name="Temporal coverage",
                status=CheckStatus.WARN,
                message=(
                    "Could not extract calendar dates from filenames or file "
                    "contents.  Temporal coverage cannot be verified."
                ),
                details={"n_files": len(input_info.files)},
            ))
            return

        file_dates.sort()
        actual_start = file_dates[0]
        actual_end = file_dates[-1]

        t_start = datetime.strptime(self.start_time, "%Y-%m-%d")
        t_end = datetime.strptime(self.end_time, "%Y-%m-%d")

        _1day = timedelta(days=1)
        covers_start = actual_start <= t_start + _1day
        covers_end = actual_end >= t_end - _1day - timedelta(days=self.n_days_forecast)

        # -- Gap detection ----------------------------------------
        # Warn if maximal gap between consecutive dates exceeds twice the
        # expected interval.
        gaps: List[Tuple[str, str, int]] = []
        max_gap_days = self.n_days_interval * 2
        for i in range(1, len(file_dates)):
            gap = (file_dates[i] - file_dates[i - 1]).days
            if gap > max_gap_days:
                gaps.append((
                    file_dates[i - 1].strftime("%Y-%m-%d"),
                    file_dates[i].strftime("%Y-%m-%d"),
                    gap,
                ))

        msg_parts: List[str] = []
        status = CheckStatus.PASS

        if not covers_start:
            msg_parts.append(
                f"starts at {actual_start:%Y-%m-%d} "
                f"(expected ≤ {self.start_time})"
            )
            status = CheckStatus.FAIL
        if not covers_end:
            msg_parts.append(
                f"ends at {actual_end:%Y-%m-%d} "
                f"(expected ≥ ~{self.end_time})"
            )
            status = CheckStatus.FAIL
        if gaps:
            if status != CheckStatus.FAIL:
                status = CheckStatus.WARN
            msg_parts.append(
                f"{len(gaps)} gap(s) > {max_gap_days} days detected"
            )

        if status == CheckStatus.PASS:
            report.add(CheckResult(
                name="Temporal coverage",
                status=CheckStatus.PASS,
                message=(
                    f"{len(file_dates)} file(s) covering "
                    f"{actual_start:%Y-%m-%d} to {actual_end:%Y-%m-%d} "
                    f"(target: {self.start_time} to {self.end_time})."
                ),
                details={
                    "n_files": len(file_dates),
                    "actual_range": [
                        actual_start.strftime("%Y-%m-%d"),
                        actual_end.strftime("%Y-%m-%d"),
                    ],
                },
            ))
        else:
            report.add(CheckResult(
                name="Temporal coverage",
                status=status,
                message=(
                    f"Temporal coverage issue: {'; '.join(msg_parts)}.  "
                    f"({len(file_dates)} files, "
                    f"{actual_start:%Y-%m-%d} to {actual_end:%Y-%m-%d})"
                ),
                details={
                    "n_files": len(file_dates),
                    "actual_range": [
                        actual_start.strftime("%Y-%m-%d"),
                        actual_end.strftime("%Y-%m-%d"),
                    ],
                    "target_range": [self.start_time, self.end_time],
                    "gaps": gaps[:10] if gaps else [],
                },
            ))

    def _check_lead_time(
        self, ds: Any, report: ValidationReport
    ) -> None:
        """Check 8: Forecast lead-time axis (if applicable)."""
        if self.target_time_values is None:
            report.add(CheckResult(
                name="Lead-time axis",
                status=CheckStatus.SKIP,
                message="No lead-time values configured (analysis/hindcast mode).",
            ))
            return

        # Look for a lead_time or forecast_time axis
        lead_vals = None
        for candidate in ["lead_time", "forecast_time", "step"]:
            if candidate in ds.coords or candidate in ds.dims:
                lead_vals = np.asarray(ds[candidate].values)
                break

        # If no dedicated lead_time axis, check the time axis length
        if lead_vals is None:
            time_vals = self._find_coord(ds, "time")
            if time_vals is not None and np.issubdtype(time_vals.dtype, np.number):
                lead_vals = time_vals

        if lead_vals is None:
            report.add(CheckResult(
                name="Lead-time axis",
                status=CheckStatus.WARN,
                message="No explicit lead_time axis found. "
                        "This is acceptable if each file represents one forecast init date.",
            ))
            return

        expected = np.asarray(self.target_time_values)
        if len(lead_vals) == len(expected):
            report.add(CheckResult(
                name="Lead-time axis",
                status=CheckStatus.PASS,
                message=f"Lead-time axis has {len(lead_vals)} steps "
                        f"(expected {len(expected)}).",
            ))
        else:
            report.add(CheckResult(
                name="Lead-time axis",
                status=CheckStatus.FAIL,
                message=f"Lead-time axis has {len(lead_vals)} steps "
                        f"but expected {len(expected)}.",
                details={
                    "expected": expected.tolist(),
                    "actual_count": len(lead_vals),
                },
            ))

    def _check_nan_fraction(
        self, ds: Any, report: ValidationReport
    ) -> None:
        """Check 9: NaN fraction per variable."""
        from dctools.utilities.coordinates import get_standardized_var_name

        all_vars = list(self.required_variables) + list(self.optional_variables)
        if not all_vars:
            # Check all data vars
            all_vars = list(ds.data_vars)

        high_nan_vars: List[Tuple[str, float]] = []

        for var in all_vars:
            # Find the actual variable name in the dataset
            actual_name = None
            if var in ds.data_vars:
                actual_name = var
            else:
                std = get_standardized_var_name(var)
                for dv in ds.data_vars:
                    if get_standardized_var_name(dv) == std:
                        actual_name = dv
                        break
            if actual_name is None:
                continue

            try:
                # Sample-based NaN check for efficiency
                da = ds[actual_name]
                total_size = int(np.prod(da.shape))
                if total_size == 0:
                    continue

                # For large arrays, use a random sample
                if total_size > 1_000_000:
                    # Take a slice instead of loading everything
                    slices = {dim: slice(None, min(s, 100)) for dim, s in zip(da.dims, da.shape)}
                    sample = da.isel(**slices).values
                else:
                    sample = da.values

                nan_count = int(np.isnan(sample).sum()) if np.issubdtype(sample.dtype, np.floating) else 0
                nan_frac = nan_count / sample.size if sample.size > 0 else 0.0

                if nan_frac > self.max_nan_fraction:
                    high_nan_vars.append((actual_name, nan_frac))
            except Exception as exc:
                logger.debug(f"NaN check failed for '{var}': {exc}")
                continue

        if high_nan_vars:
            report.add(CheckResult(
                name="Data integrity (NaN fraction)",
                status=CheckStatus.WARN,
                message=f"{len(high_nan_vars)} variable(s) exceed "
                        f"{self.max_nan_fraction:.0%} NaN threshold.",
                details={
                    "high_nan_variables": {
                        v: f"{frac:.1%}" for v, frac in high_nan_vars
                    },
                },
            ))
        else:
            report.add(CheckResult(
                name="Data integrity (NaN fraction)",
                status=CheckStatus.PASS,
                message=f"All variables within {self.max_nan_fraction:.0%} NaN tolerance.",
            ))

    def _check_dtypes(
        self, ds: Any, report: ValidationReport
    ) -> None:
        """Check 10: Data types."""
        problematic: List[Tuple[str, str]] = []
        for var in ds.data_vars:
            dtype = ds[var].dtype
            # Flag integer-only representations of physical ocean variables
            if np.issubdtype(dtype, np.integer):
                problematic.append((str(var), str(dtype)))

        if problematic:
            report.add(CheckResult(
                name="Data types",
                status=CheckStatus.WARN,
                message=f"{len(problematic)} variable(s) stored as integers "
                        f"(expected float32/float64).",
                details={"integer_vars": dict(problematic)},
            ))
        else:
            report.add(CheckResult(
                name="Data types",
                status=CheckStatus.PASS,
                message="All data variables use floating-point types.",
            ))

    def _check_file_size_from_input(
        self, input_info: InputInfo, report: ValidationReport
    ) -> None:
        """Check 11: File / directory size sanity check.

        For multi-file inputs the total size is the sum across all files.
        """
        try:
            total_bytes = 0
            for fpath in input_info.files:
                p = Path(fpath)
                if p.is_dir():
                    total_bytes += sum(
                        f.stat().st_size for f in p.rglob("*") if f.is_file()
                    )
                elif p.is_file():
                    total_bytes += p.stat().st_size
                # else: remote paths — skip

            if total_bytes == 0:
                report.add(CheckResult(
                    name="File size",
                    status=CheckStatus.SKIP,
                    message="Cannot determine file size (remote or glob path).",
                ))
                return

            size_gb = total_bytes / (1024**3)

            extra = ""
            if input_info.mode == "multi":
                extra = f" ({len(input_info.files)} files)"

            if size_gb < 0.001:
                report.add(CheckResult(
                    name="File size",
                    status=CheckStatus.WARN,
                    message=f"Dataset is very small ({size_gb:.4f} GB){extra}. "
                            f"This may indicate incomplete data.",
                    details={"size_bytes": total_bytes, "size_gb": round(size_gb, 4),
                             "n_files": len(input_info.files)},
                ))
            elif size_gb > 500:
                report.add(CheckResult(
                    name="File size",
                    status=CheckStatus.WARN,
                    message=f"Dataset is very large ({size_gb:.1f} GB){extra}. "
                            f"Consider using chunked Zarr format.",
                    details={"size_bytes": total_bytes, "size_gb": round(size_gb, 1),
                             "n_files": len(input_info.files)},
                ))
            else:
                report.add(CheckResult(
                    name="File size",
                    status=CheckStatus.PASS,
                    message=f"Dataset size: {size_gb:.2f} GB{extra}.",
                    details={"size_bytes": total_bytes, "size_gb": round(size_gb, 2),
                             "n_files": len(input_info.files)},
                ))
        except Exception as exc:
            report.add(CheckResult(
                name="File size",
                status=CheckStatus.SKIP,
                message=f"Cannot determine file size: {exc}",
            ))
