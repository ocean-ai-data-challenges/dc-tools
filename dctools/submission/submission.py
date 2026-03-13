"""High-level submission interface for Data Challenge participants.

This module provides a user-friendly ``ModelSubmission`` class that guides
participants through the complete workflow:

1. **Register** a model with its metadata.
2. **Validate** the prediction dataset against the DC specification.
3. **Launch** the full evaluation pipeline and leaderboard generation.

Design principles (inspired by WeatherBench2, OceanBench):

- *Convention over configuration*: sensible defaults from the DC YAML config.
- *Progressive disclosure*: simple ``submit()`` one-liner for common cases,
  with fine-grained methods for advanced users.
- *Fail-fast with clear messages*: validation runs before expensive evaluation.
- *Reproducible*: every submission creates a metadata record with timestamps,
  config snapshot, and validation report.

Usage
-----
::

    from dctools.submission import ModelSubmission

    sub = ModelSubmission(
        model_name="MyOceanModel_v2",
        data_path="/data/my_model_output.zarr",
        model_description="Global 1/4° forecast model, 10-day horizon",
    )

    # Quick validation only
    report = sub.validate()
    print(report.pretty())

    # Full pipeline: validate --> evaluate --> leaderboard
    sub.submit(data_directory="/data/dc2_output")
"""

from __future__ import annotations

import json
import os
import shutil
import time as _time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from loguru import logger

from dctools.submission.report import CheckStatus, ValidationReport
from dctools.submission.validator import SubmissionValidator


class ModelSubmission:
    """Manages the complete lifecycle of a model submission.

    Parameters
    ----------
    model_name : str
        Short identifier for the model (used in filenames, leaderboard).
    data_path : str or Path
        Path to the prediction dataset (.zarr, .nc, or glob pattern).
    dc_config : str
        Data Challenge config name (default ``"dc2"``).
    model_description : str
        Free-text description shown in metadata.
    model_url : str, optional
        Link to the model's paper / repository.
    team_name : str, optional
        Submitting team identifier.
    contact_email : str, optional
        Contact email for the submitting team.
    max_nan_fraction : float
        Maximum tolerated NaN fraction per variable during validation.
    """

    def __init__(
        self,
        model_name: str,
        data_path: str | Path,
        *,
        dc_config: str = "dc2",
        model_description: str = "",
        model_url: str = "",
        team_name: str = "",
        contact_email: str = "",
        max_nan_fraction: float = 0.10,
        variables: Optional[List[str]] = None,
    ) -> None:
        self.model_name = model_name
        self.data_path = str(data_path)
        self.dc_config = dc_config
        self.model_description = model_description
        self.model_url = model_url
        self.team_name = team_name
        self.contact_email = contact_email
        self.max_nan_fraction = max_nan_fraction
        self.variables = variables

        # Populated after validation
        self._validator: Optional[SubmissionValidator] = None
        self._report: Optional[ValidationReport] = None
        self._metadata: Dict[str, Any] = {}

    # ----------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------

    def validate(self, *, quick: bool = False) -> ValidationReport:
        """Validate the submitted dataset against the DC specification.

        Parameters
        ----------
        quick : bool
            Skip expensive NaN-scan on full data; useful for a fast first pass.

        Returns
        -------
        ValidationReport
            Structured report with per-check results and overall pass/fail.
        """
        logger.info(f"Validating submission '{self.model_name}' ...")
        self._validator = SubmissionValidator.from_dc_config(
            self.dc_config,
            model_name=self.model_name,
            max_nan_fraction=self.max_nan_fraction,
            variables=self.variables,
        )
        self._report = self._validator.validate(self.data_path, quick=quick)
        return self._report

    def is_valid(self, *, quick: bool = False) -> bool:
        """Return ``True`` if the dataset passes all mandatory checks."""
        if self._report is None:
            self.validate(quick=quick)
        assert self._report is not None
        return self._report.overall_pass

    def submit(
        self,
        *,
        data_directory: Optional[str] = None,
        skip_validation: bool = False,
        quick_validation: bool = False,
        force: bool = False,
    ) -> int:
        """Run the complete submission pipeline.

        Workflow:

        1. Validate the dataset (unless ``skip_validation=True``).
        2. Prepare the evaluation workspace (copy/symlink data,
           generate a custom YAML config).
        3. Run the evaluation pipeline (Dask-based).
        4. Generate the leaderboard.

        Parameters
        ----------
        data_directory : str, optional
            Root directory for evaluation artefacts (catalogs, results, etc.).
            Defaults to ``./output``.
        skip_validation : bool
            If ``True``, bypass validation checks entirely.
        quick_validation : bool
            If ``True``, run a fast validation (skip NaN full scan).
        force : bool
            If ``True``, proceed with evaluation even when validation fails.

        Returns
        -------
        int
            ``0`` on success, ``1`` on failure.
        """
        _sep = "═" * 72
        print(f"\n╔{_sep}╗")
        print(f"║{'  DC MODEL SUBMISSION':^72}║")
        print(f"╠{_sep}╣")
        print(f"║  Model : {self.model_name:<61}║")
        _dp = self.data_path
        if len(_dp) > 58:
            _dp = "..." + _dp[-55:]
        print(f"║  Data  : {_dp:<61}║")
        print(f"║  Config: {self.dc_config:<61}║")
        print(f"╚{_sep}╝\n")

        # -- Step 1: Validation -------------------------------------
        if not skip_validation:
            print("━" * 74)
            print("  STEP 1/3 — Validating dataset against DC specification")
            print("━" * 74)

            report = self.validate(quick=quick_validation)
            print(report.pretty())

            if not report.overall_pass:
                if not force:
                    logger.error(
                        "Validation FAILED. Fix the issues above or pass --force to override."
                    )
                    return 1
                else:
                    logger.warning(
                        "Validation FAILED but --force was set. Proceeding anyway."
                    )
        else:
            logger.info("Validation skipped (--skip-validation).")

        # -- Step 2: Prepare workspace ------------------------------
        print("\n" + "━" * 74)
        print("  STEP 2/3 — Preparing evaluation workspace")
        print("━" * 74 + "\n")

        if data_directory is None:
            data_directory = os.path.join(os.getcwd(), "output")

        workspace = self._prepare_workspace(data_directory)
        if workspace is None:
            return 1

        config_path, args_override = workspace

        # -- Step 3: Run evaluation ---------------------------------
        print("\n" + "━" * 74)
        print("  STEP 3/3 — Running evaluation and generating leaderboard")
        print("━" * 74 + "\n")

        return self._run_evaluation(config_path, data_directory)

    # ----------------------------------------------------------------
    # Metadata
    # ----------------------------------------------------------------

    def get_metadata(self) -> Dict[str, Any]:
        """Return submission metadata as a dict."""
        return {
            "model_name": self.model_name,
            "model_description": self.model_description,
            "model_url": self.model_url,
            "team_name": self.team_name,
            "contact_email": self.contact_email,
            "data_path": self.data_path,
            "dc_config": self.dc_config,
            "submission_time": datetime.now().isoformat(),
            "validation_passed": (
                self._report.overall_pass if self._report else None
            ),
        }

    def save_metadata(self, output_dir: str | Path) -> Path:
        """Write submission metadata to a JSON file."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        meta_path = output_dir / f"submission_{self.model_name}.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self.get_metadata(), f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"Submission metadata saved to {meta_path}")
        return meta_path

    # ----------------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------------

    def _prepare_workspace(
        self,
        data_directory: str,
    ) -> Optional[tuple]:
        """Set up the evaluation workspace and generate a merged submission config.

        The method loads the base DC config, replaces prediction sources with
        the submitted model, keeps observation sources, and writes a standalone
        config that ``run_from_config`` can consume directly.

        Returns (config_path, args_dict) or None on failure.
        """
        try:
            os.makedirs(data_directory, exist_ok=True)
            results_dir = os.path.join(data_directory, "results")
            catalogs_dir = os.path.join(data_directory, "catalogs")
            os.makedirs(results_dir, exist_ok=True)
            os.makedirs(catalogs_dir, exist_ok=True)

            # Save submission metadata
            self.save_metadata(results_dir)

            # Save validation report if available
            if self._report is not None:
                report_path = os.path.join(
                    results_dir, f"validation_report_{self.model_name}.json"
                )
                self._report.save_json(report_path)
                logger.info(f"Validation report saved to {report_path}")

            # -- Load the base DC config -------------------------------
            config_dir = Path(__file__).resolve().parents[2] / "dc" / "config"
            base_config_path = config_dir / f"{self.dc_config}.yaml"
            if not base_config_path.exists():
                logger.error(f"Base config not found: {base_config_path}")
                return None

            with open(base_config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}

            # -- Build the submitted model source entry ----------------
            overlay = self._generate_config_overlay()
            model_source = overlay["model_source"]

            # -- Replace prediction sources with the submitted model ---
            old_sources = config.get("sources", [])
            obs_sources = [
                s for s in old_sources
                if isinstance(s, dict) and s.get("observation_dataset", False)
            ]
            obs_names = [
                s["dataset"] for s in obs_sources
                if isinstance(s, dict) and "dataset" in s
            ]

            config["sources"] = [model_source] + obs_sources

            # Configure dataset_references so the evaluation loop uses the
            # submitted model as prediction against the observation datasets.
            config["dataset_references"] = {
                self.model_name: obs_names,
            }

            # Attach submission metadata inside the config for traceability.
            config["submission"] = overlay.get("submission", {})

            # -- Write standalone merged config ------------------------
            merged_path = os.path.join(
                data_directory, f"submission_{self.model_name}.yaml"
            )
            with open(merged_path, "w", encoding="utf-8") as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            logger.info(f"Merged submission config written to {merged_path}")

            logger.success("Workspace prepared successfully.")
            return str(merged_path), {"data_directory": data_directory}

        except Exception as exc:
            logger.error(f"Failed to prepare workspace: {exc}")
            return None

    def _generate_config_overlay(self) -> Dict[str, Any]:
        """Generate a YAML overlay that adds the participant's model as a source.

        This overlay is designed to be merged with the base DC config so that
        the evaluation pipeline picks up the new model alongside existing
        reference and observation datasets.
        """
        data_path = self.data_path
        p = Path(data_path)

        # Determine root_path and file_pattern depending on input layout.
        if p.is_file():
            # Single file --> root is the parent dir, pattern matches the file.
            root_path = str(p.parent)
            if p.suffix in (".nc", ".nc4", ".netcdf"):
                file_pattern = p.name
            else:
                file_pattern = "**/*.zarr"
            connection_type = "local"
        elif p.is_dir():
            # Could be a single zarr store or a directory of files.
            from dctools.submission.validator import SubmissionValidator
            if SubmissionValidator._is_zarr_store(p):
                # Single zarr store
                root_path = str(p.parent)
                file_pattern = p.name
            else:
                # Directory of forecast files
                root_path = data_path
                # Detect predominant format
                has_zarr = any(
                    d.is_dir() and SubmissionValidator._is_zarr_store(d)
                    for d in p.iterdir()
                )
                has_nc = any(
                    f.suffix in (".nc", ".nc4", ".netcdf")
                    for f in p.rglob("*") if f.is_file()
                )
                if has_zarr:
                    file_pattern = "*.zarr"
                elif has_nc:
                    file_pattern = "**/*.nc"
                else:
                    file_pattern = "**/*"
            connection_type = "local"
        else:
            # Glob or unknown – best-effort
            root_path = str(p.parent) if p.parent.exists() else str(p)
            file_pattern = "**/*.zarr" if ".zarr" in data_path else "**/*.nc"
            connection_type = "local"

        # Build the model source entry matching the existing YAML schema
        eval_vars = self._detect_eval_variables()
        model_source = {
            "dataset": self.model_name,
            "observation_dataset": False,  # this is a prediction
            "config": "local",
            "connection_type": connection_type,
            "root_path": root_path,
            "file_pattern": file_pattern,
            "metrics": ["rmsd"],
            "full_day_data": True,
            "ignore_geometry": True,
            "keep_variables": eval_vars,
            "eval_variables": eval_vars,
        }

        overlay = {
            "submission": {
                "model_name": self.model_name,
                "model_description": self.model_description,
                "team_name": self.team_name,
                "contact_email": self.contact_email,
                "model_url": self.model_url,
                "data_path": self.data_path,
            },
            "model_source": model_source,
        }
        return overlay

    def _detect_eval_variables(self) -> List[str]:
        """Attempt to detect evaluation variables from the dataset.

        For a directory of files the first file is opened as a sample.
        """
        try:
            import xarray as xr
            from dctools.submission.validator import SubmissionValidator

            p = Path(self.data_path)

            # Determine the sample path to open
            if p.is_file():
                sample = str(p)
            elif p.is_dir() and SubmissionValidator._is_zarr_store(p):
                sample = str(p)
            elif p.is_dir():
                # Directory of files — pick first zarr or nc
                children = sorted(p.iterdir())
                zarr_stores = [
                    d for d in children
                    if d.is_dir() and SubmissionValidator._is_zarr_store(d)
                ]
                nc_files = [
                    f for f in p.rglob("*")
                    if f.is_file() and f.suffix in (".nc", ".nc4", ".netcdf")
                ]
                if zarr_stores:
                    sample = str(zarr_stores[0])
                elif nc_files:
                    sample = str(sorted(nc_files)[0])
                else:
                    return []
            else:
                sample = self.data_path

            # Open sample
            if sample.endswith(".zarr") or SubmissionValidator._is_zarr_store(Path(sample)):
                ds = xr.open_zarr(sample, consolidated=True)
            else:
                ds = xr.open_dataset(sample, engine="h5netcdf", lock=False)

            coord_names = {
                "time", "lat", "lon", "depth", "latitude", "longitude",
                "nav_lat", "nav_lon", "lev", "level",
            }
            eval_vars = [
                str(v) for v in ds.data_vars
                if str(v).lower() not in coord_names
            ]
            ds.close()
            return eval_vars
        except Exception:
            return []

    def _run_evaluation(self, config_path: str, data_directory: str) -> int:
        """Launch the evaluation pipeline."""
        try:
            from dctools.processing.runner import run_from_config

            # Build minimal CLI args namespace
            from argparse import Namespace
            cli_args = Namespace(
                data_directory=data_directory,
                config_name=self.dc_config,
                logfile=os.path.join(data_directory, "logs", f"{self.model_name}.log"),
                jsonfile=None,
                metric=None,
            )
            os.makedirs(os.path.dirname(cli_args.logfile), exist_ok=True)

            return run_from_config(
                Path(config_path),
                cli_args=cli_args,
            )

        except Exception as exc:
            logger.error(f"Evaluation failed: {exc}")
            return 1


# --------------------------------------------------------------------
# Utility functions for quick scripting use
# --------------------------------------------------------------------


def quick_validate(
    data_path: str | Path,
    *,
    model_name: str = "unnamed_model",
    dc_config: str = "dc2",
    quick: bool = True,
) -> ValidationReport:
    """One-liner validation for interactive use.

    >>> from dctools.submission.submission import quick_validate
    >>> report = quick_validate("/data/my_model.zarr")
    >>> print(report.pretty())
    """
    sub = ModelSubmission(
        model_name=model_name,
        data_path=str(data_path),
        dc_config=dc_config,
    )
    return sub.validate(quick=quick)


def quick_submit(
    data_path: str | Path,
    *,
    model_name: str = "unnamed_model",
    data_directory: Optional[str] = None,
    dc_config: str = "dc2",
    force: bool = False,
) -> int:
    """One-liner submission for quick evaluation.

    >>> from dctools.submission.submission import quick_submit
    >>> quick_submit("/data/my_model.zarr", model_name="MyModel")
    """
    sub = ModelSubmission(
        model_name=model_name,
        data_path=str(data_path),
        dc_config=dc_config,
    )
    return sub.submit(data_directory=data_directory, force=force)
