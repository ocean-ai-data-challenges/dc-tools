"""Tests for the model submission and validation system.

Covers:
- ValidationReport construction and serialization
- SubmissionValidator individual checks
- SubmissionValidator.from_dc_config factory
- ModelSubmission metadata and validate workflow
- CLI argument parsing
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import numpy as np
import pytest
import xarray as xr

from dctools.submission.report import (
    CheckResult,
    CheckStatus,
    ValidationReport,
)
from dctools.submission.validator import SubmissionValidator
from dctools.submission.submission import ModelSubmission, quick_validate


# =====================================================================
# Helpers — synthetic dataset creation
# =====================================================================

def _make_compliant_dataset(
    tmp_path: Path,
    *,
    fmt: str = "zarr",
    lat_start: float = -78.0,
    lat_stop: float = 90.0,
    lat_step: float = 0.25,
    lon_start: float = -180.0,
    lon_stop: float = 180.0,
    lon_step: float = 0.25,
    depth_levels: list | None = None,
    n_time: int = 10,
    variables: list | None = None,
    all_nan_var: str | None = None,
    int_var: str | None = None,
) -> Path:
    """Create a small synthetic dataset that passes all DC2 checks."""
    lat = np.arange(lat_start, lat_stop, lat_step)
    lon = np.arange(lon_start, lon_stop, lon_step)
    if depth_levels is None:
        depth_levels = [0.494025, 47.37369, 92.32607]
    depth = np.array(depth_levels, dtype=float)
    time = np.arange(n_time)  # lead time

    if variables is None:
        variables = ["zos", "thetao", "so", "uo", "vo"]

    rng = np.random.default_rng(42)
    data_vars: Dict[str, Any] = {}
    for v in variables:
        if v == "zos":
            # Surface-only  → (time, lat, lon)
            vals = rng.standard_normal((n_time, len(lat), len(lon))).astype(np.float32)
        else:
            # 3D → (time, depth, lat, lon)
            vals = rng.standard_normal((n_time, len(depth), len(lat), len(lon))).astype(np.float32)

        if all_nan_var and v == all_nan_var:
            vals[:] = np.nan

        if int_var and v == int_var:
            vals = vals.astype(np.int32)

        if v == "zos":
            data_vars[v] = (["time", "lat", "lon"], vals)
        else:
            data_vars[v] = (["time", "depth", "lat", "lon"], vals)

    ds = xr.Dataset(
        data_vars,
        coords={
            "lat": lat,
            "lon": lon,
            "depth": depth,
            "time": time,
        },
    )

    if fmt == "zarr":
        out_path = tmp_path / "test_model.zarr"
        ds.to_zarr(str(out_path), mode="w", consolidated=True)
    else:
        out_path = tmp_path / "test_model.nc"
        ds.to_netcdf(str(out_path))

    return out_path


# =====================================================================
# Tests — ValidationReport
# =====================================================================


class TestValidationReport:
    def test_add_pass(self):
        report = ValidationReport(model_name="test", data_path="/tmp/x")
        report.add(CheckResult("check1", CheckStatus.PASS, "ok"))
        assert report.overall_pass is True
        assert report.n_pass == 1
        assert report.n_fail == 0

    def test_add_fail_sets_overall_false(self):
        report = ValidationReport(model_name="test", data_path="/tmp/x")
        report.add(CheckResult("check1", CheckStatus.PASS, "ok"))
        report.add(CheckResult("check2", CheckStatus.FAIL, "bad"))
        assert report.overall_pass is False
        assert report.n_fail == 1

    def test_to_dict_and_json(self, tmp_path):
        report = ValidationReport(model_name="m", data_path="/data/x")
        report.add(CheckResult("c1", CheckStatus.PASS, "ok"))
        report.add(CheckResult("c2", CheckStatus.WARN, "hmm"))

        d = report.to_dict()
        assert d["summary"]["pass"] == 1
        assert d["summary"]["warn"] == 1
        assert d["overall_pass"] is True

        json_path = tmp_path / "report.json"
        report.save_json(json_path)
        loaded = json.loads(json_path.read_text())
        assert loaded["model_name"] == "m"

    def test_pretty(self):
        report = ValidationReport(model_name="MyModel", data_path="/data/x.zarr")
        report.add(CheckResult("check_a", CheckStatus.PASS, "all good"))
        report.add(CheckResult("check_b", CheckStatus.FAIL, "no bueno"))
        txt = report.pretty(use_color=False)
        assert "MyModel" in txt
        assert "PASS" in txt
        assert "FAIL" in txt


# =====================================================================
# Tests — SubmissionValidator individual checks
# =====================================================================


class TestSubmissionValidator:
    """Test each validation check in isolation with synthetic data."""

    def _make_validator(self, **kwargs) -> SubmissionValidator:
        defaults = dict(
            target_lat=np.arange(-78, 90, 0.25),
            target_lon=np.arange(-180, 180, 0.25),
            target_depth=[0.494025, 47.37369, 92.32607],
            target_time_values=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            start_time="2024-01-01",
            end_time="2025-01-01",
            required_variables=["zos", "thetao", "so", "uo", "vo"],
            model_name="test_model",
            n_days_forecast=10,
        )
        defaults.update(kwargs)
        return SubmissionValidator(**defaults)

    def test_full_compliant_dataset_passes(self, tmp_path):
        """A fully compliant dataset should pass all checks."""
        ds_path = _make_compliant_dataset(tmp_path)
        v = self._make_validator()
        report = v.validate(ds_path, quick=True)
        assert report.overall_pass is True, report.pretty(use_color=False)

    def test_missing_variables_detected(self, tmp_path):
        """Missing required variables trigger FAIL."""
        ds_path = _make_compliant_dataset(tmp_path, variables=["zos", "thetao"])
        v = self._make_validator()
        report = v.validate(ds_path, quick=True)
        fails = [c for c in report.checks if c.status == CheckStatus.FAIL]
        var_fail = [c for c in fails if "variable" in c.name.lower()]
        assert len(var_fail) > 0

    def test_wrong_resolution_detected(self, tmp_path):
        """Wrong spatial resolution triggers FAIL."""
        ds_path = _make_compliant_dataset(tmp_path, lat_step=0.5, lon_step=0.5)
        v = self._make_validator()
        report = v.validate(ds_path, quick=True)
        fails = [c for c in report.checks if c.status == CheckStatus.FAIL]
        res_fail = [c for c in fails if "resolution" in c.name.lower()]
        assert len(res_fail) > 0

    def test_incomplete_spatial_extent_detected(self, tmp_path):
        """Dataset covering partial region triggers FAIL for extent."""
        ds_path = _make_compliant_dataset(tmp_path, lat_start=0, lat_stop=45)
        v = self._make_validator()
        report = v.validate(ds_path, quick=True)
        fails = [c for c in report.checks if c.status == CheckStatus.FAIL]
        ext_fail = [c for c in fails if "extent" in c.name.lower()]
        assert len(ext_fail) > 0

    def test_missing_depth_levels_detected(self, tmp_path):
        """Missing depth levels trigger FAIL."""
        ds_path = _make_compliant_dataset(tmp_path, depth_levels=[0.5])
        v = self._make_validator(target_depth=[0.494025, 47.37369, 92.32607])
        report = v.validate(ds_path, quick=True)
        fails = [c for c in report.checks if c.status == CheckStatus.FAIL]
        depth_fail = [c for c in fails if "depth" in c.name.lower()]
        assert len(depth_fail) > 0

    def test_wrong_lead_time_count_detected(self, tmp_path):
        """Wrong number of lead-time steps triggers FAIL."""
        ds_path = _make_compliant_dataset(tmp_path, n_time=5)
        v = self._make_validator(target_time_values=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        report = v.validate(ds_path, quick=True)
        fails = [c for c in report.checks if c.status == CheckStatus.FAIL]
        lt_fail = [c for c in fails if "lead" in c.name.lower()]
        assert len(lt_fail) > 0

    def test_high_nan_fraction_detected(self, tmp_path):
        """Variable with 100% NaN triggers WARN."""
        ds_path = _make_compliant_dataset(tmp_path, all_nan_var="zos")
        v = self._make_validator(max_nan_fraction=0.10)
        report = v.validate(ds_path, quick=False)
        warns = [c for c in report.checks if c.status == CheckStatus.WARN]
        nan_warn = [c for c in warns if "nan" in c.name.lower()]
        assert len(nan_warn) > 0

    def test_integer_dtype_triggers_warning(self, tmp_path):
        """Integer-encoded variable triggers WARN."""
        ds_path = _make_compliant_dataset(tmp_path, int_var="zos")
        v = self._make_validator()
        report = v.validate(ds_path, quick=True)
        warns = [c for c in report.checks if c.status == CheckStatus.WARN]
        dtype_warn = [c for c in warns if "type" in c.name.lower()]
        assert len(dtype_warn) > 0

    def test_unreadable_file_fails(self, tmp_path):
        """Non-existent file triggers FAIL at input resolution."""
        v = self._make_validator()
        report = v.validate(tmp_path / "nonexistent.zarr", quick=True)
        assert report.overall_pass is False
        assert report.checks[0].status == CheckStatus.FAIL
        assert "input" in report.checks[0].name.lower() or "read" in report.checks[0].name.lower()

    def test_netcdf_format(self, tmp_path):
        """NetCDF format is accepted."""
        # Create a small NC file (fewer points to keep file small)
        lat = np.arange(-78, -77, 0.25)
        lon = np.arange(-180, -179, 0.25)
        depth = np.array([0.494025], dtype=float)
        time = np.arange(10)
        rng = np.random.default_rng(42)
        ds = xr.Dataset(
            {"zos": (["time", "lat", "lon"], rng.standard_normal((10, len(lat), len(lon))).astype(np.float32))},
            coords={"lat": lat, "lon": lon, "depth": depth, "time": time},
        )
        nc_path = tmp_path / "test.nc"
        ds.to_netcdf(str(nc_path))
        v = self._make_validator(required_variables=["zos"])
        report = v.validate(nc_path, quick=True)
        assert report.checks[0].status == CheckStatus.PASS  # readability


# =====================================================================
# Tests — ModelSubmission
# =====================================================================


class TestModelSubmission:
    def test_metadata_structure(self):
        sub = ModelSubmission(
            model_name="TestModel",
            data_path="/tmp/test.zarr",
            model_description="A test model",
            team_name="Team A",
        )
        meta = sub.get_metadata()
        assert meta["model_name"] == "TestModel"
        assert meta["team_name"] == "Team A"
        assert "submission_time" in meta

    def test_save_metadata(self, tmp_path):
        sub = ModelSubmission(
            model_name="TestModel",
            data_path="/tmp/test.zarr",
        )
        meta_path = sub.save_metadata(tmp_path)
        assert meta_path.exists()
        loaded = json.loads(meta_path.read_text())
        assert loaded["model_name"] == "TestModel"

    def test_validate_with_compliant_data(self, tmp_path):
        ds_path = _make_compliant_dataset(tmp_path)
        sub = ModelSubmission(
            model_name="GoodModel",
            data_path=str(ds_path),
            dc_config="dc2",
        )
        # Manually set the validator to avoid loading the full DC config
        sub._validator = SubmissionValidator(
            target_lat=np.arange(-78, 90, 0.25),
            target_lon=np.arange(-180, 180, 0.25),
            target_depth=[0.494025, 47.37369, 92.32607],
            target_time_values=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            start_time="2024-01-01",
            end_time="2025-01-01",
            required_variables=["zos", "thetao", "so", "uo", "vo"],
            model_name="GoodModel",
            n_days_forecast=10,
        )
        report = sub._validator.validate(str(ds_path), quick=True)
        assert report.overall_pass is True

    def test_is_valid_false_on_bad_data(self, tmp_path):
        ds_path = _make_compliant_dataset(tmp_path, variables=["zos"])
        sub = ModelSubmission(
            model_name="BadModel",
            data_path=str(ds_path),
        )
        sub._validator = SubmissionValidator(
            target_lat=np.arange(-78, 90, 0.25),
            target_lon=np.arange(-180, 180, 0.25),
            target_depth=[0.494025, 47.37369, 92.32607],
            required_variables=["zos", "thetao", "so", "uo", "vo"],
            model_name="BadModel",
        )
        report = sub._validator.validate(str(ds_path), quick=True)
        assert report.overall_pass is False


# =====================================================================
# Tests — quick_validate utility
# =====================================================================


class TestQuickValidate:
    def test_quick_validate_compliant(self, tmp_path):
        """quick_validate should return a passing report for compliant data."""
        ds_path = _make_compliant_dataset(tmp_path)
        # We cannot call quick_validate directly because it loads DC config,
        # so we test the validator directly
        v = SubmissionValidator(
            target_lat=np.arange(-78, 90, 0.25),
            target_lon=np.arange(-180, 180, 0.25),
            target_depth=[0.494025, 47.37369, 92.32607],
            target_time_values=list(range(10)),
            required_variables=["zos", "thetao", "so", "uo", "vo"],
            model_name="quick_test",
        )
        report = v.validate(ds_path, quick=True)
        assert report.overall_pass is True


# =====================================================================
# Tests — CLI parser
# =====================================================================


class TestCLIParser:
    def test_validate_args(self):
        from dc.submit import _build_parser

        parser = _build_parser()
        args = parser.parse_args([
            "validate", "/data/model.zarr",
            "--model-name", "TestModel",
            "--config", "dc2",
            "--quick",
        ])
        assert args.command == "validate"
        assert args.data_path == "/data/model.zarr"
        assert args.model_name == "TestModel"
        assert args.quick is True

    def test_run_args(self):
        from dc.submit import _build_parser

        parser = _build_parser()
        args = parser.parse_args([
            "run", "/data/model.zarr",
            "--model-name", "MyModel",
            "--data-directory", "/output",
            "--force",
        ])
        assert args.command == "run"
        assert args.force is True
        assert args.data_directory == "/output"

    def test_info_args(self):
        from dc.submit import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["info", "--config", "dc2"])
        assert args.command == "info"
        assert args.config == "dc2"


# =====================================================================
# Tests — Multi-file validation
# =====================================================================

def _make_per_date_zarr_dir(
    tmp_path: Path,
    *,
    dates: list | None = None,
    variables: list | None = None,
    n_time: int = 10,
    lat_start: float = -78.0,
    lat_step: float = 0.25,
    lat_stop: float = 90.0,
    lon_start: float = -180.0,
    lon_step: float = 0.25,
    lon_stop: float = 180.0,
    depth_levels: list | None = None,
) -> Path:
    """Create a directory of per-date Zarr stores (like glonet layout)."""
    if dates is None:
        # 52 weekly init dates over 2024, starting from Jan 1
        from datetime import datetime, timedelta
        start = datetime(2024, 1, 1)
        dates = [(start + timedelta(days=7 * i)).strftime("%Y%m%d") for i in range(52)]
    if variables is None:
        variables = ["zos", "thetao", "so", "uo", "vo"]
    if depth_levels is None:
        depth_levels = [0.494025, 47.37369, 92.32607]

    out_dir = tmp_path / "multi_model"
    out_dir.mkdir(parents=True, exist_ok=True)

    lat = np.arange(lat_start, lat_stop, lat_step)
    lon = np.arange(lon_start, lon_stop, lon_step)
    depth = np.array(depth_levels, dtype=float)
    time = np.arange(n_time)

    rng = np.random.default_rng(42)

    for date_str in dates:
        data_vars = {}
        for v in variables:
            if v == "zos":
                vals = rng.standard_normal((n_time, len(lat), len(lon))).astype(np.float32)
                data_vars[v] = (["time", "lat", "lon"], vals)
            else:
                vals = rng.standard_normal((n_time, len(depth), len(lat), len(lon))).astype(np.float32)
                data_vars[v] = (["time", "depth", "lat", "lon"], vals)

        ds = xr.Dataset(
            data_vars,
            coords={"lat": lat, "lon": lon, "depth": depth, "time": time},
        )
        zarr_path = out_dir / f"{date_str}.zarr"
        ds.to_zarr(str(zarr_path), mode="w", consolidated=True)

    return out_dir


def _make_per_date_nc_dir(
    tmp_path: Path,
    *,
    dates: list | None = None,
    variables: list | None = None,
    n_time: int = 10,
) -> Path:
    """Create a directory of per-date NetCDF files."""
    if dates is None:
        from datetime import datetime, timedelta
        start = datetime(2024, 1, 1)
        dates = [(start + timedelta(days=7 * i)).strftime("%Y%m%d") for i in range(5)]
    if variables is None:
        variables = ["zos"]

    out_dir = tmp_path / "multi_nc"
    out_dir.mkdir(parents=True, exist_ok=True)

    lat = np.arange(-78, -77, 0.25)
    lon = np.arange(-180, -179, 0.25)
    depth = np.array([0.494025], dtype=float)
    time = np.arange(n_time)

    rng = np.random.default_rng(42)

    for date_str in dates:
        data_vars = {}
        for v in variables:
            if v == "zos":
                vals = rng.standard_normal((n_time, len(lat), len(lon))).astype(np.float32)
                data_vars[v] = (["time", "lat", "lon"], vals)
            else:
                vals = rng.standard_normal((n_time, len(depth), len(lat), len(lon))).astype(np.float32)
                data_vars[v] = (["time", "depth", "lat", "lon"], vals)

        ds = xr.Dataset(
            data_vars,
            coords={"lat": lat, "lon": lon, "depth": depth, "time": time},
        )
        nc_path = out_dir / f"{date_str}.nc"
        ds.to_netcdf(str(nc_path))

    return out_dir


class TestMultiFileValidation:
    """Tests for multi-file (directory of per-date files) validation."""

    def _make_validator(self, **kwargs) -> SubmissionValidator:
        defaults = dict(
            target_lat=np.arange(-78, 90, 0.25),
            target_lon=np.arange(-180, 180, 0.25),
            target_depth=[0.494025, 47.37369, 92.32607],
            target_time_values=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            start_time="2024-01-01",
            end_time="2025-01-01",
            required_variables=["zos", "thetao", "so", "uo", "vo"],
            model_name="test_model",
            n_days_forecast=10,
            n_days_interval=7,
        )
        defaults.update(kwargs)
        return SubmissionValidator(**defaults)

    def test_resolve_input_directory_of_zarr(self, tmp_path):
        """A directory of .zarr stores is detected as multi-file."""
        from dctools.submission.validator import InputInfo
        from dctools.submission.report import ValidationReport

        dir_path = _make_per_date_zarr_dir(tmp_path, dates=["20240103", "20240110", "20240117"])
        v = self._make_validator()
        report = ValidationReport(model_name="test", data_path=str(dir_path))
        info = v._resolve_input(str(dir_path), report)

        assert info is not None
        assert info.mode == "multi"
        assert info.format == "zarr"
        assert len(info.files) == 3

    def test_resolve_input_directory_of_nc(self, tmp_path):
        """A directory of .nc files is detected as multi-file."""
        from dctools.submission.validator import InputInfo
        from dctools.submission.report import ValidationReport

        dir_path = _make_per_date_nc_dir(tmp_path, dates=["20240103", "20240110"])
        v = self._make_validator()
        report = ValidationReport(model_name="test", data_path=str(dir_path))
        info = v._resolve_input(str(dir_path), report)

        assert info is not None
        assert info.mode == "multi"
        assert info.format == "netcdf"
        assert len(info.files) == 2

    def test_resolve_input_single_zarr(self, tmp_path):
        """A single .zarr store is detected as single."""
        from dctools.submission.validator import InputInfo
        from dctools.submission.report import ValidationReport

        ds_path = _make_compliant_dataset(tmp_path, fmt="zarr")
        v = self._make_validator()
        report = ValidationReport(model_name="test", data_path=str(ds_path))
        info = v._resolve_input(str(ds_path), report)

        assert info is not None
        assert info.mode == "single"
        assert info.format == "zarr"
        assert len(info.files) == 1

    def test_multi_zarr_full_validation_passes(self, tmp_path):
        """A directory with 52 compliant Zarr files passes validation."""
        dir_path = _make_per_date_zarr_dir(tmp_path)
        v = self._make_validator()
        report = v.validate(dir_path, quick=True)
        # Should pass: input resolution, readability, variables, coords, temporal
        assert report.overall_pass is True, report.pretty(use_color=False)

    def test_multi_zarr_incomplete_temporal_coverage(self, tmp_path):
        """A directory with only 3 files covering Jan should fail temporal coverage."""
        dir_path = _make_per_date_zarr_dir(tmp_path, dates=["20240103", "20240110", "20240117"])
        v = self._make_validator()
        report = v.validate(dir_path, quick=True)
        # Temporal coverage should fail (missing most of the year)
        temporal_checks = [c for c in report.checks if "temporal" in c.name.lower()]
        assert len(temporal_checks) > 0
        assert temporal_checks[0].status == CheckStatus.FAIL

    def test_multi_zarr_gap_detection(self, tmp_path):
        """Large gaps between file dates trigger a warning."""
        # Create files with a 30-day gap (> 2 * 7 = 14 day threshold)
        dates = ["20240103", "20240110", "20240210", "20240917", "20241218"]
        dir_path = _make_per_date_zarr_dir(tmp_path, dates=dates)
        v = self._make_validator()
        report = v.validate(dir_path, quick=True)
        temporal = [c for c in report.checks if "temporal" in c.name.lower()]
        assert len(temporal) > 0
        # Should have gaps warning or fail
        assert temporal[0].details is not None
        assert "gaps" in temporal[0].details or temporal[0].status != CheckStatus.PASS

    def test_multi_nc_readability(self, tmp_path):
        """NetCDF files in a directory can be opened."""
        dir_path = _make_per_date_nc_dir(tmp_path, dates=["20240103", "20240110"])
        v = self._make_validator(required_variables=["zos"])
        report = v.validate(dir_path, quick=True)
        readability_checks = [c for c in report.checks if "readability" in c.name.lower()]
        assert len(readability_checks) > 0
        assert readability_checks[0].status == CheckStatus.PASS

    def test_empty_directory_fails(self, tmp_path):
        """An empty directory triggers FAIL for input resolution."""
        empty_dir = tmp_path / "empty_model"
        empty_dir.mkdir()
        v = self._make_validator()
        report = v.validate(empty_dir, quick=True)
        assert report.overall_pass is False
        # First check should be input resolution FAIL
        assert report.checks[0].status == CheckStatus.FAIL
        assert "input" in report.checks[0].name.lower()

    def test_file_size_sums_all_files(self, tmp_path):
        """File size check sums across all files in multi-file mode."""
        dir_path = _make_per_date_zarr_dir(tmp_path, dates=["20240103", "20240110"])
        v = self._make_validator()
        report = v.validate(dir_path, quick=True)
        size_checks = [c for c in report.checks if "size" in c.name.lower()]
        assert len(size_checks) > 0
        # Should report combined size
        if size_checks[0].details and "n_files" in size_checks[0].details:
            assert size_checks[0].details["n_files"] == 2


# =====================================================================
# Tests — Partial submission (--variables)
# =====================================================================


class TestPartialSubmission:
    """Tests for partial variable submission support."""

    def _make_validator(self, **kwargs) -> SubmissionValidator:
        defaults = dict(
            target_lat=np.arange(-78, 90, 0.25),
            target_lon=np.arange(-180, 180, 0.25),
            target_depth=[0.494025, 47.37369, 92.32607],
            target_time_values=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            start_time="2024-01-01",
            end_time="2025-01-01",
            required_variables=["zos", "thetao", "so", "uo", "vo"],
            model_name="test_model",
            n_days_forecast=10,
        )
        defaults.update(kwargs)
        return SubmissionValidator(**defaults)

    def test_partial_only_zos_passes(self, tmp_path):
        """Dataset with only zos passes when --variables zos is specified."""
        ds_path = _make_compliant_dataset(tmp_path, variables=["zos"])
        # Full validator would fail (missing thetao, so, uo, vo)
        v_full = self._make_validator()
        report_full = v_full.validate(ds_path, quick=True)
        assert report_full.overall_pass is False

        # Partial validator should pass
        v_partial = self._make_validator(
            required_variables=["zos"],
            optional_variables=["thetao", "so", "uo", "vo"],
        )
        report_partial = v_partial.validate(ds_path, quick=True)
        # Required var check should pass (only zos required)
        var_checks = [c for c in report_partial.checks if "variable" in c.name.lower()]
        var_fails = [c for c in var_checks if c.status == CheckStatus.FAIL]
        assert len(var_fails) == 0, report_partial.pretty(use_color=False)

    def test_partial_cli_args_variables(self):
        """CLI parser accepts --variables flag."""
        from dc.submit import _build_parser
        parser = _build_parser()
        args = parser.parse_args([
            "validate", "/data/model.zarr",
            "--model-name", "TestModel",
            "--variables", "zos", "ssh",
        ])
        assert args.variables == ["zos", "ssh"]

    def test_partial_cli_run_args_variables(self):
        """CLI run subcommand also accepts --variables."""
        from dc.submit import _build_parser
        parser = _build_parser()
        args = parser.parse_args([
            "run", "/data/model.zarr",
            "--model-name", "TestModel",
            "--variables", "zos",
        ])
        assert args.variables == ["zos"]

    def test_model_submission_variables_kwarg(self):
        """ModelSubmission accepts a variables parameter."""
        sub = ModelSubmission(
            model_name="PartialModel",
            data_path="/tmp/test.zarr",
            variables=["zos", "thetao"],
        )
        assert sub.variables == ["zos", "thetao"]


# =====================================================================
# Tests — Output / save report (--output / -o)
# =====================================================================


class TestOutputReport:
    def test_cli_output_alias(self):
        """--output is accepted as alias for --save-report."""
        from dc.submit import _build_parser
        parser = _build_parser()
        args = parser.parse_args([
            "validate", "/data/model.zarr",
            "--model-name", "Test",
            "-o", "/tmp/report.json",
        ])
        assert args.output == "/tmp/report.json"

    def test_save_report_still_works(self):
        """--save-report is still accepted."""
        from dc.submit import _build_parser
        parser = _build_parser()
        args = parser.parse_args([
            "validate", "/data/model.zarr",
            "--save-report", "/tmp/report.json",
        ])
        assert args.save_report == "/tmp/report.json"

    def test_validation_report_save_json(self, tmp_path):
        """ValidationReport.save_json creates a valid JSON file."""
        report = ValidationReport(model_name="test", data_path="/data/x")
        report.add(CheckResult("c1", CheckStatus.PASS, "ok"))
        report.add(CheckResult("c2", CheckStatus.FAIL, "bad"))

        out = tmp_path / "report.json"
        report.save_json(out)

        import json
        loaded = json.loads(out.read_text())
        assert loaded["overall_pass"] is False
        assert loaded["summary"]["fail"] == 1
        assert len(loaded["checks"]) == 2


# =====================================================================
# Tests — Sample submission script
# =====================================================================


class TestSampleSubmission:
    def test_create_sample_dataset(self, tmp_path):
        """create_sample_submission.py generates a valid submission directory."""
        from scripts.create_sample_submission import create_sample_dataset

        out_path = create_sample_dataset(tmp_path / "sample_model")
        assert out_path.exists()
        assert out_path.is_dir()

        zarr_stores = sorted(out_path.glob("*.zarr"))
        assert len(zarr_stores) >= 1

        ds = xr.open_zarr(str(zarr_stores[0]))
        assert "zos" in ds.data_vars
        assert "thetao" in ds.data_vars
        assert len(ds.lat) == 672
        assert len(ds.lon) == 1440
        assert len(ds.depth) == 21
        ds.close()

    def test_create_sample_partial(self, tmp_path):
        """create_sample_submission.py can generate partial datasets."""
        from scripts.create_sample_submission import create_sample_dataset

        out_path = create_sample_dataset(
            tmp_path / "partial_model", variables=["zos"]
        )
        zarr_stores = sorted(out_path.glob("*.zarr"))
        assert len(zarr_stores) >= 1
        ds = xr.open_zarr(str(zarr_stores[0]))
        assert list(ds.data_vars) == ["zos"]
        ds.close()

    def test_sample_passes_validation(self, tmp_path):
        """Generated sample passes all validation checks."""
        from scripts.create_sample_submission import create_sample_dataset

        out_path = create_sample_dataset(tmp_path / "valid_model")
        v = SubmissionValidator(
            target_lat=np.arange(-78, 90, 0.25),
            target_lon=np.arange(-180, 180, 0.25),
            target_depth=[
                0.494025, 47.37369, 92.32607, 155.8507, 222.4752,
                318.1274, 380.213, 453.9377, 541.0889, 643.5668,
                763.3333, 902.3393, 1245.291, 1684.284, 2225.078,
                3220.820, 3597.032, 3992.484, 4405.224, 4833.291,
                5274.784,
            ],
            target_time_values=list(range(10)),
            required_variables=["zos", "thetao", "so", "uo", "vo"],
            model_name="sample_test",
            # Match the sample's time range (only Jan 2024)
            start_time="2024-01-01",
            end_time="2024-02-01",
        )
        report = v.validate(out_path, quick=True)
        assert report.overall_pass is True, report.pretty(use_color=False)
