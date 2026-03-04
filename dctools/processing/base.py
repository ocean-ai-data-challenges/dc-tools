"""Base evaluation building blocks shared across DC evaluations.

The goal of this module is to keep per-DC evaluation classes (DC1Evaluation,
DC2Evaluation, DC3Evaluation, ...) focused on challenge-specific wiring, while
mutualizing generic helpers:
- Dask cluster initialisation
- Dask sizing extraction from YAML sources
- common init (target grid/time + dask logging + safe dask memory defaults)
- catalog fetching helper
- dataset manager setup
- transform setup
- coordinate conformance validation
- dataloader sanity checks
- common filtering utilities
- full run_eval loop
"""

from __future__ import annotations

import json
import os
import time as _time
import warnings
from argparse import Namespace
from datetime import timedelta
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional

import dask
import geopandas as gpd
import numpy as np
import pandas as pd
from dask.distributed import get_client
from loguru import logger
from oceanbench.core.distributed import DatasetProcessor
from shapely import geometry

from dctools.data.coordinates import (
    get_standardized_var_name,
    get_target_depth_values,
    get_target_dimensions,
    get_target_time_values,
)
from dctools.data.datasets.dataloader import EvaluationDataloader
from dctools.data.datasets.dataset import get_dataset_from_config
from dctools.data.datasets.dataset_manager import MultiSourceDatasetManager
from dctools.metrics.evaluator import Evaluator, _worker_full_cleanup
from dctools.metrics.metrics import MetricComputer
from dctools.metrics.oceanbench_metrics import get_variable_alias
from dctools.utilities.file_utils import empty_folder
from dctools.utilities.init_dask import configure_dask_logging, configure_dask_workers_env
from dctools.utilities.misc_utils import make_serializable, nan_to_none, transform_in_place

warnings.simplefilter("ignore", UserWarning)


class BaseDCEvaluation:
    """Base class for evaluation orchestration.

    Subclasses are expected to:
    - define `self.dataset_references` (pred -> list[ref])
    - create `self.dataset_processor` as needed
    - implement a `run_eval()` method
    """

    def __init__(self, arguments: Namespace) -> None:
        self.args = arguments
        self.results_directory = os.path.join(self.args.data_directory, "results")
        os.makedirs(self.results_directory, exist_ok=True)
        self.target_dimensions = get_target_dimensions(self.args)
        self.target_time_values = get_target_time_values(self.args)
        # Subclasses can set this to a dict before run_eval() is called to
        # customise the leaderboard (metric/variable/model names, page texts).
        # It is passed directly to render_site_from_results_dir as custom_config.
        self.leaderboard_custom_config: Optional[Dict[str, Any]] = None
        # Populated during run_eval(); non-empty means the leaderboard was
        # generated but incomplete (e.g. maps.html skipped) or failed entirely.
        self._leaderboard_warnings: List[str] = []

        configure_dask_logging()

        # Safe defaults for large datasets: start spilling earlier to reduce
        # OOM/pause cascades. Subclasses can override if needed.
        dask.config.set(
            {
                "distributed.worker.memory.target": 0.60,
                "distributed.worker.memory.spill": 0.70,
                "distributed.worker.memory.pause": 0.90,
                "distributed.worker.memory.terminate": 0.95,
            }
        )

    # ---------------------------------------------------------------------
    # Dask sizing helpers (agnostic)
    # ---------------------------------------------------------------------
    def _extract_dask_cfg_from_source(self, source: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract per-dataset Dask sizing from a source config.

        Supports both a nested `dask:` block and flat keys:
        - n_parallel_workers / nthreads_per_worker / memory_limit_per_worker
        - n_workers / threads_per_worker / memory_limit
        """
        if not isinstance(source, dict):
            return None

        dask_cfg: Dict[str, Any] = {}
        nested = source.get("dask")
        if isinstance(nested, dict):
            dask_cfg.update(nested)

        for k in ("n_parallel_workers", "nthreads_per_worker", "memory_limit_per_worker"):
            if k in source:
                dask_cfg[k] = source.get(k)

        for k in ("n_workers", "threads_per_worker", "memory_limit"):
            if k in source:
                dask_cfg[k] = source.get(k)

        n_workers = dask_cfg.get("n_parallel_workers", dask_cfg.get("n_workers"))
        threads_per_worker = dask_cfg.get("nthreads_per_worker", dask_cfg.get("threads_per_worker"))
        memory_limit = dask_cfg.get("memory_limit_per_worker", dask_cfg.get("memory_limit"))

        if n_workers is None and threads_per_worker is None and memory_limit is None:
            return None

        cfg: Dict[str, Any] = {}
        if n_workers is not None:
            cfg["n_workers"] = int(n_workers)
        if threads_per_worker is not None:
            cfg["threads_per_worker"] = int(threads_per_worker)
        if memory_limit is not None:
            cfg["memory_limit"] = memory_limit
        return cfg

    def _build_dask_cfgs_by_dataset(self) -> Dict[str, Dict[str, Any]]:
        cfgs: Dict[str, Dict[str, Any]] = {}
        for source in getattr(self.args, "sources", []) or []:
            if not isinstance(source, dict):
                continue
            dataset = source.get("dataset")
            if not dataset:
                continue
            cfg = self._extract_dask_cfg_from_source(source)
            if cfg:
                cfgs[str(dataset)] = cfg
        return cfgs

    def _global_dask_cfg_fallback(self) -> Dict[str, Any]:
        cfg: Dict[str, Any] = {}
        if hasattr(self.args, "n_parallel_workers"):
            cfg["n_workers"] = int(self.args.n_parallel_workers)
        if hasattr(self.args, "nthreads_per_worker"):
            cfg["threads_per_worker"] = int(self.args.nthreads_per_worker)
        if hasattr(self.args, "memory_limit_per_worker"):
            cfg["memory_limit"] = self.args.memory_limit_per_worker
        return cfg

    def _pick_initial_dask_cfg(self) -> Dict[str, Any]:
        """Pick an initial Dask config used for setup.

        Priority order:
        1. First prediction dataset in ``dataset_references`` that has a config.
        2. First *reference* (observation) dataset that has a config — this
           is the common case because per-dataset Dask sizing is typically
           set on observation datasets (saral, swot, argo, …).
        3. First value in ``dask_cfgs_by_dataset`` (insertion order from YAML).
        4. Global fallback from CLI / top-level YAML keys.
        5. Hard-coded safe default.

        Subclasses typically define ``self.dataset_references`` before calling this.
        """
        preferred: list[str] = []
        dataset_references = getattr(self, "dataset_references", None)
        if isinstance(dataset_references, dict):
            preferred = list(dataset_references.keys())

        dask_cfgs_by_dataset = getattr(self, "dask_cfgs_by_dataset", None) or {}

        # 1. Check prediction datasets.
        for ds in preferred:
            cfg = dask_cfgs_by_dataset.get(ds)
            if cfg:
                return dict(cfg)

        # 2. Check reference (observation) datasets — pick the first ref of
        #    the first prediction dataset so that the initial cluster matches
        #    the first evaluation that will actually run.
        if isinstance(dataset_references, dict):
            for ref_list in dataset_references.values():
                if not isinstance(ref_list, (list, tuple)):
                    continue
                for ref_alias in ref_list:
                    cfg = dask_cfgs_by_dataset.get(ref_alias)
                    if cfg:
                        return dict(cfg)

        # 3. Fallback to first available per-dataset config.
        if dask_cfgs_by_dataset:
            return dict(next(iter(dask_cfgs_by_dataset.values())))

        cfg = self._global_dask_cfg_fallback()
        if cfg:
            return cfg

        return {"n_workers": 1, "threads_per_worker": 1, "memory_limit": "4GB"}

    def _configure_thread_caps_env(self, *, threads: str = "1") -> None:
        """Cap C-library threads before creating a Dask cluster."""
        caps = {
            "OMP_NUM_THREADS": threads,
            "OPENBLAS_NUM_THREADS": threads,
            "MKL_NUM_THREADS": threads,
            "VECLIB_MAXIMUM_THREADS": threads,
            "NUMEXPR_NUM_THREADS": threads,
            "PYINTERP_NUM_THREADS": threads,
            "GOTO_NUM_THREADS": threads,
            "BLOSC_NTHREADS": threads,
        }
        for k, v in caps.items():
            os.environ[k] = v

    def _configure_dataset_processor_workers(self) -> None:
        """Propagate required HDF5/NetCDF env vars to workers when possible."""
        dataset_processor = getattr(self, "dataset_processor", None)
        if not dataset_processor or not getattr(dataset_processor, "client", None):
            return
        configure_dask_workers_env(dataset_processor.client)

    # ---------------------------------------------------------------------
    # Generic reusable evaluation helpers
    # ---------------------------------------------------------------------
    def filter_data(self, manager: MultiSourceDatasetManager, filter_region: Any):
        """Filter data by time and region."""
        manager.filter_all_by_date(
            start=pd.to_datetime(self.args.start_time),
            end=pd.to_datetime(self.args.end_time),
        )
        manager.filter_all_by_region(region=filter_region)
        return manager

    def check_dataloader(self, dataloader: EvaluationDataloader) -> None:
        """Basic integrity checks on batches."""
        for batch in dataloader:
            logger.debug(f"Batch: {batch}")
            assert "pred_data" in batch[0]
            assert "ref_data" in batch[0]
            assert isinstance(batch[0]["pred_data"], str)
            if batch[0]["ref_data"]:
                assert isinstance(batch[0]["ref_data"], str)

    def get_catalog(self, dataset_name: str, local_catalog_dir: str, catalog_cfg: dict) -> None:
        """Ensure a dataset catalog exists locally, downloading if necessary."""
        import fsspec

        def create_fs(cfg: dict):
            key = cfg.get("s3_key")
            secret_key = cfg.get("s3_secret_key")
            endpoint_url = cfg.get("url")
            client_kwargs = {"endpoint_url": endpoint_url}
            if key is None or secret_key is None:
                return fsspec.filesystem("s3", anon=True, client_kwargs=client_kwargs)
            return fsspec.filesystem("s3", key=key, secret=secret_key, client_kwargs=client_kwargs)

        def download_catalog_file(remote_path: str, local_path: str) -> bool:
            fs = create_fs(catalog_cfg)
            if not fs.exists(remote_path):
                logger.warning(f"Remote catalog file not found: {remote_path}")
                return False
            with fs.open(remote_path, "rb") as remote_file, open(local_path, "wb") as local_file:
                while True:
                    chunk = remote_file.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    local_file.write(chunk)
            return True

        local_catalog_path = os.path.join(local_catalog_dir, f"{dataset_name}.json")

        # Special case: ARGO uses a directory master index.
        if dataset_name == "argo_profiles":
            argo_index_path = os.path.join(local_catalog_dir, "argo_index")
            if os.path.isdir(argo_index_path) and os.path.exists(
                os.path.join(argo_index_path, "master_index.json")
            ):
                logger.info(f"Local ARGO catalog directory found at {argo_index_path}")
                return

        if os.path.isfile(local_catalog_path) and os.path.getsize(local_catalog_path) > 0:
            return

        remote_catalog_path = (
            f"s3://{catalog_cfg['s3_bucket']}/{catalog_cfg['s3_folder']}/{dataset_name}.json"
        )
        download_catalog_file(remote_catalog_path, local_catalog_path)

    def close(self) -> None:
        """Release resources held by this evaluation run."""
        dataset_processor = getattr(self, "dataset_processor", None)
        if dataset_processor is None:
            return
        try:
            dataset_processor.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Dask cluster initialisation (call from subclass __init__ after
    # setting self.dataset_references and self.all_datasets)
    # ------------------------------------------------------------------
    def _init_cluster(self) -> None:
        """Spin up the Dask DatasetProcessor for this evaluation run."""
        self.dask_cfgs_by_dataset = self._build_dask_cfgs_by_dataset()
        _initial_cfg = self._pick_initial_dask_cfg()
        memory_limit_per_worker = _initial_cfg.get("memory_limit", "4GB")
        n_parallel_workers = int(_initial_cfg.get("n_workers", 1))
        nthreads_per_worker = int(_initial_cfg.get("threads_per_worker", 1))

        # Cap worker count to batch_size to avoid idle workers.
        _batch_size = getattr(self.args, "batch_size", None)
        if isinstance(_batch_size, int) and _batch_size > 0:
            if n_parallel_workers > _batch_size:
                logger.info(
                    "Capping n_parallel_workers from "
                    f"{n_parallel_workers} to {_batch_size} "
                    f"(batch_size={_batch_size}) to avoid idle workers."
                )
                n_parallel_workers = _batch_size

        logger.info(
            f"Init DatasetProcessor with: Workers={n_parallel_workers}, "
            f"Threads={nthreads_per_worker}, MemLimit={memory_limit_per_worker}"
        )

        self._configure_thread_caps_env(threads="1")

        self.dataset_processor = DatasetProcessor(
            distributed=True,
            n_workers=n_parallel_workers,
            threads_per_worker=nthreads_per_worker,
            memory_limit=memory_limit_per_worker,
        )

        self._configure_dataset_processor_workers()

    # ------------------------------------------------------------------
    # Transform setup
    # ------------------------------------------------------------------
    def setup_transforms(
        self,
        dataset_manager: MultiSourceDatasetManager,
        aliases: List[str],
    ) -> Dict[str, Any]:
        """Configure and return the transform dict for all *aliases*."""
        transforms_dict = {}
        for alias in aliases:
            kwargs: Dict[str, Any] = {"reduce_precision": self.args.reduce_precision}
            # Some datasets need regridder weights (e.g. glorys interpolation).
            regridder_weights = getattr(self.args, "regridder_weights", None)
            if regridder_weights is not None and alias == "glorys_cmems":
                kwargs["regridder_weights"] = regridder_weights

            transforms_dict[alias] = dataset_manager.get_transform(
                dataset_alias=alias,
                **kwargs,
            )
        return transforms_dict

    # ------------------------------------------------------------------
    # Dataset manager setup
    # ------------------------------------------------------------------
    def setup_dataset_manager(self, list_all_references: List[str]) -> MultiSourceDatasetManager:
        """Build and return a fully configured :class:`MultiSourceDatasetManager`."""
        manager = MultiSourceDatasetManager(
            dataset_processor=self.dataset_processor,
            target_dimensions=self.target_dimensions,
            time_tolerance=pd.Timedelta(hours=self.args.delta_time),
            list_references=list_all_references,
            max_cache_files=self.args.max_cache_files,
        )

        all_datasets: List[str] = getattr(self, "all_datasets", [])

        raw_sources = getattr(self.args, "sources", []) or []
        valid_sources: List[Dict[str, Any]] = []
        for idx, source in enumerate(raw_sources):
            if not isinstance(source, dict):
                logger.warning(
                    f"Skipping sources[{idx}] because it is not a mapping: {type(source)}"
                )
                continue
            if "dataset" not in source:
                logger.warning(
                    "Skipping a source entry without 'dataset' key. "
                    f"Keys={sorted(list(source.keys()))}"
                )
                continue
            valid_sources.append(source)

        datasets: Dict[str, Any] = {}
        for source in sorted(valid_sources, key=lambda x: str(x.get("dataset", ""))):
            source_name: str = source["dataset"]
            if source_name not in all_datasets:
                logger.warning(f"Dataset {source_name} is not supported, skipping.")
                continue

            # Memory cleanup on Dask workers between datasets.
            try:
                client = get_client()
                client.run(_worker_full_cleanup)
                logger.debug("Memory cleanup (gc.collect + trim) executed on all Dask workers.")
            except Exception as exc:
                logger.warning(f"Could not execute memory cleanup on Dask workers: {exc}")

            self.get_catalog(
                source_name,
                self.args.catalog_dir,
                self.args.catalog_connection,
            )

            kwargs: Dict[str, Any] = {
                "source": source,
                "root_data_folder": self.args.data_directory,
                "root_catalog_folder": self.args.catalog_dir,
                "dataset_processor": self.dataset_processor,
                "max_samples": self.args.max_samples,
                "file_cache": manager.file_cache,
                "target_depth_values": get_target_depth_values(self.args),
                "filter_values": {
                    "start_time": self.args.start_time,
                    "end_time": self.args.end_time,
                    "min_lon": self.args.min_lon,
                    "max_lon": self.args.max_lon,
                    "min_lat": self.args.min_lat,
                    "max_lat": self.args.max_lat,
                },
            }

            datasets[source_name] = get_dataset_from_config(**kwargs)
            manager.add_dataset(source_name, datasets[source_name])

        filter_region = geometry.Polygon(
            [
                (self.args.min_lon, self.args.min_lat),
                (self.args.min_lon, self.args.max_lat),
                (self.args.max_lon, self.args.max_lat),
                (self.args.max_lon, self.args.min_lat),
            ]
        )
        filter_region_gs = gpd.GeoSeries([filter_region], crs="EPSG:4326")

        manager = self.filter_data(manager, filter_region_gs)
        return manager  # type: ignore[no-any-return]

    # ------------------------------------------------------------------
    # Coordinate conformance validation
    # ------------------------------------------------------------------
    def _validate_pred_datasets_coordinates(
        self,
        dataset_manager: MultiSourceDatasetManager,
        transforms_dict: Dict[str, Any],
    ) -> None:
        """Validate prediction dataset coordinates against the configured target grid.

        Writes a JSON report to ``coordinate_conformance_report.json`` and raises
        :class:`RuntimeError` when mismatches are found.
        """
        expected_dims = self.target_dimensions or {}
        expected_time_vals = self.target_time_values
        if not expected_dims:
            logger.warning("No target_dimensions configured; skipping coord validation.")
            return

        def _round_array(vals: Any, ndigits: int = 6) -> np.ndarray:
            arr = np.asarray(vals)
            if arr.size == 0:
                return arr
            if np.issubdtype(arr.dtype, np.floating):
                return np.round(arr.astype(float), ndigits)
            return arr

        report: Dict[str, Any] = {
            "target_dimensions": {
                k: (
                    v.tolist()
                    if hasattr(v, "tolist")
                    else list(v)
                    if isinstance(v, (list, tuple, range))
                    else v
                )
                for k, v in expected_dims.items()
            },
            "target_time_values": expected_time_vals,
            "datasets": {},
        }

        mismatches_found = 0
        dataset_references: Dict[str, Any] = getattr(self, "dataset_references", {})

        for pred_alias in list(dataset_references.keys()):
            if pred_alias not in dataset_manager.datasets:
                logger.warning(f"Pred dataset '{pred_alias}' not found; skipping.")
                continue

            ds_obj = dataset_manager.datasets[pred_alias]
            cat = ds_obj.get_catalog()
            gdf = cat.get_dataframe() if cat is not None else None
            if gdf is None or gdf.empty or "path" not in gdf.columns:
                logger.warning(
                    f"Catalog for '{pred_alias}' is empty or missing 'path';"
                    " skipping coord validation."
                )
                continue

            sample_path = str(gdf.iloc[0]["path"])
            try:
                sample_ds = ds_obj.get_connection_manager().open(sample_path, mode="rb")
            except Exception as exc:
                logger.warning(f"Could not open sample for '{pred_alias}': {exc}")
                continue

            if sample_ds is None:
                logger.warning(f"Sample dataset is None for '{pred_alias}'; skipping.")
                continue

            transform = transforms_dict.get(pred_alias)
            try:
                if transform is not None:
                    sample_ds = transform(sample_ds)
            except Exception as exc:
                logger.warning(f"Transform failed for '{pred_alias}': {exc}")

            if sample_ds is None:
                logger.warning(
                    f"Sample dataset became None after transform for '{pred_alias}'; skipping."
                )
                continue

            def _values_missing_extra_close(
                expected: Any,
                actual: Any,
                *,
                atol: float,
            ) -> tuple:
                exp_arr = np.asarray(expected)
                act_arr = np.asarray(actual)

                if not (
                    np.issubdtype(exp_arr.dtype, np.number)
                    and np.issubdtype(act_arr.dtype, np.number)
                ):
                    missing = exp_arr[~np.isin(exp_arr, act_arr)]
                    extra = act_arr[~np.isin(act_arr, exp_arr)]
                    return missing, extra

                exp_f = np.asarray(exp_arr, dtype=float)
                act_f = np.asarray(act_arr, dtype=float)
                exp_f = exp_f[np.isfinite(exp_f)]
                act_f = act_f[np.isfinite(act_f)]
                exp_sorted = np.sort(exp_f)
                act_sorted = np.sort(act_f)

                def _has_close(val: float, arr_sorted: np.ndarray) -> bool:
                    idx = int(np.searchsorted(arr_sorted, val))
                    if idx < arr_sorted.size and abs(arr_sorted[idx] - val) <= atol:
                        return True
                    if idx > 0 and abs(arr_sorted[idx - 1] - val) <= atol:
                        return True
                    return False

                missing_vals = [
                    float(v) for v in exp_sorted if not _has_close(float(v), act_sorted)
                ]
                extra_vals = [float(v) for v in act_sorted if not _has_close(float(v), exp_sorted)]
                return np.asarray(missing_vals, dtype=float), np.asarray(extra_vals, dtype=float)

            ds_report: Dict[str, Any] = {
                "sample_path": sample_path,
                "coords_present": sorted(list(sample_ds.coords)),
                "dims_present": dict(sample_ds.sizes),
                "missing": {},
                "extra": {},
            }

            for axis in ("lat", "lon", "depth"):
                exp = expected_dims.get(axis)
                if exp is None:
                    continue
                if axis not in sample_ds.coords and axis not in sample_ds.dims:
                    ds_report["missing"][axis] = {
                        "reason": "axis not found",
                        "expected_count": int(len(exp)) if hasattr(exp, "__len__") else None,
                    }
                    mismatches_found += 1
                    continue
                if axis not in sample_ds.coords:
                    ds_report["missing"][axis] = {
                        "reason": "axis has no coordinate values (only a dimension)",
                        "expected_count": int(len(exp)) if hasattr(exp, "__len__") else None,
                    }
                    mismatches_found += 1
                    continue

                try:
                    actual = sample_ds[axis].values
                except Exception:
                    actual = np.asarray(sample_ds.coords.get(axis))
                exp_arr = _round_array(exp)
                act_arr = _round_array(actual)
                atol = 1e-6 if axis in ("lat", "lon") else 1e-3
                missing, extra = _values_missing_extra_close(exp_arr, act_arr, atol=atol)

                if missing.size:
                    ds_report["missing"][axis] = {
                        "count": int(missing.size),
                        "values": missing.tolist(),
                    }
                    mismatches_found += 1
                if extra.size:
                    ds_report["extra"][axis] = {"count": int(extra.size), "values": extra.tolist()}
                    mismatches_found += 1

            if expected_time_vals is not None:
                expected_time_arr = np.asarray(list(expected_time_vals))
                expected_is_numeric = np.issubdtype(expected_time_arr.dtype, np.number)

                candidate_axes = (
                    ("lead_time", "forecast_time", "time")
                    if expected_is_numeric
                    else ("time", "lead_time", "forecast_time")
                )
                time_axis_name: Optional[str] = None
                for candidate in candidate_axes:
                    if candidate in sample_ds.coords or candidate in sample_ds.dims:
                        time_axis_name = candidate
                        break

                if time_axis_name is None:
                    ds_report["missing"]["time"] = {
                        "reason": "no time/lead_time axis found",
                        "expected_values": expected_time_vals,
                    }
                    mismatches_found += 1
                else:
                    try:
                        actual_time = sample_ds[time_axis_name].values
                    except Exception:
                        actual_time = np.asarray(sample_ds.coords.get(time_axis_name))

                    exp_time = expected_time_arr
                    act_time = np.asarray(actual_time)

                    if expected_is_numeric and not np.issubdtype(act_time.dtype, np.number):
                        if act_time.size != exp_time.size:
                            ds_report["missing"][time_axis_name] = {
                                "reason": (
                                    "time axis is datetime-like but target_time_values"
                                    " is numeric; horizon length mismatch"
                                ),
                                "expected_count": int(exp_time.size),
                                "actual_count": int(act_time.size),
                            }
                            mismatches_found += 1
                        else:
                            ds_report.setdefault("info", {})[time_axis_name] = {
                                "reason": (
                                    "time axis is datetime-like;"
                                    " validated horizon length only"
                                ),
                                "count": int(act_time.size),
                            }
                        report["datasets"][pred_alias] = ds_report
                        try:
                            sample_ds.close()
                        except Exception:
                            pass
                        continue

                    if (
                        expected_is_numeric
                        and time_axis_name == "time"
                        and np.issubdtype(act_time.dtype, np.number)
                    ):
                        try:
                            act_max = (
                                float(np.nanmax(np.abs(act_time.astype(float))))
                                if act_time.size
                                else 0.0
                            )
                            exp_max = (
                                float(np.nanmax(np.abs(exp_time.astype(float))))
                                if exp_time.size
                                else 0.0
                            )
                        except Exception:
                            act_max, exp_max = 0.0, 0.0

                        if act_max > 1e6 and exp_max <= 1e6:
                            if act_time.size != exp_time.size:
                                ds_report["missing"][time_axis_name] = {
                                    "reason": (
                                        "time axis appears to be epoch timestamps;"
                                        " validated horizon length only, but length mismatched"
                                    ),
                                    "expected_count": int(exp_time.size),
                                    "actual_count": int(act_time.size),
                                }
                                mismatches_found += 1
                            else:
                                ds_report.setdefault("info", {})[time_axis_name] = {
                                    "reason": (
                                        "time axis appears to be epoch timestamps;"
                                        " validated horizon length only"
                                    ),
                                    "count": int(act_time.size),
                                }
                            report["datasets"][pred_alias] = ds_report
                            try:
                                sample_ds.close()
                            except Exception:
                                pass
                            continue

                    if np.issubdtype(act_time.dtype, np.number) and np.issubdtype(
                        exp_time.dtype, np.number
                    ):
                        missing_t, extra_t = _values_missing_extra_close(
                            exp_time, act_time, atol=1e-6
                        )
                    else:
                        exp_s = np.asarray([str(x) for x in exp_time.tolist()])
                        act_s = np.asarray([str(x) for x in act_time.tolist()])
                        missing_t = exp_s[~np.isin(exp_s, act_s)]
                        extra_t = act_s[~np.isin(act_s, exp_s)]

                    if missing_t.size:
                        ds_report["missing"][time_axis_name] = {
                            "count": int(missing_t.size),
                            "values": missing_t.tolist(),
                        }
                        mismatches_found += 1
                    if extra_t.size:
                        ds_report["extra"][time_axis_name] = {
                            "count": int(extra_t.size),
                            "values": extra_t.tolist(),
                        }
                        mismatches_found += 1

            report["datasets"][pred_alias] = ds_report
            try:
                sample_ds.close()
            except Exception:
                pass

        out_path = os.path.join(self.results_directory, "coordinate_conformance_report.json")
        try:
            with open(out_path, "w") as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Coordinate conformance report written to: {out_path}")
        except Exception as exc:
            logger.warning(f"Failed to write coordinate conformance report: {exc}")

        if mismatches_found:
            bad = [
                name
                for name, dsrep in (report.get("datasets") or {}).items()
                if (dsrep.get("missing") or {}) or (dsrep.get("extra") or {})
            ]
            logger.error(
                "Prediction dataset coordinates do not match configured target grid. "
                f"Datasets with mismatches: {bad}. "
                f"See report: {out_path}"
            )
            raise RuntimeError(
                "Coordinate conformance check failed for prediction datasets. "
                f"See report: {out_path}"
            )

    # ------------------------------------------------------------------
    # Full evaluation loop
    # ------------------------------------------------------------------
    def run_eval(self) -> None:
        """Run the full evaluation pipeline."""
        all_datasets: List[str] = getattr(self, "all_datasets", [])
        dataset_references: Dict[str, Any] = getattr(self, "dataset_references", {})

        dataset_manager = self.setup_dataset_manager(all_datasets)
        aliases = dataset_manager.datasets.keys()

        transforms_dict = self.setup_transforms(dataset_manager, list(aliases))

        self._validate_pred_datasets_coordinates(dataset_manager, transforms_dict)

        dataloaders: Dict[str, Any] = {}
        metrics_names: Dict[str, Any] = {}
        metrics: Dict[str, Any] = {}
        metrics_kwargs: Dict[str, Any] = {}
        evaluators: Dict[str, Any] = {}
        models_results: Dict[str, Any] = {}

        # Affiche le lien dashboard Dask une seule fois, avant le début des traitements.
        try:
            dashboard_link = getattr(self.dataset_processor.client, "dashboard_link", None)
            if dashboard_link:
                logger.info("")
                logger.info(
                    f"============= Link to Dask dashboard : {dashboard_link} ============="
                )
                logger.info("")
        except Exception:
            pass

        for alias in dataset_references.keys():
            dataset_json_path = os.path.join(self.results_directory, f"results_{alias}.json")
            results_files_dir = os.path.join(self.args.data_directory, "results_batches")

            if os.path.isdir(results_files_dir):
                if os.listdir(results_files_dir):
                    logger.info("Results dir exists. Removing old results files.")
                    empty_folder(results_files_dir, extension=".json")
            else:
                os.makedirs(results_files_dir, exist_ok=True)

            dataset_manager.build_forecast_index(
                alias,
                init_date=self.args.start_time,
                end_date=self.args.end_time,
                n_days_forecast=int(self.args.n_days_forecast),
                n_days_interval=int(self.args.n_days_interval),
            )
            list_references = [
                ref for ref in dataset_references[alias] if ref in dataset_manager.datasets
            ]
            pred_source_dict: Dict[str, Any] = next(
                (s for s in self.args.sources if s.get("dataset") == alias), {}
            )
            metrics_names[alias] = pred_source_dict.get("metrics", ["rmsd"])

            metrics_kwargs[alias] = {}
            ref_transforms: Dict[str, Any] = {}
            metrics[alias] = {}
            pred_transform = transforms_dict.get(alias)

            _n_pred_total = len(dataset_references)
            _n_pred_current = list(dataset_references.keys()).index(alias) + 1

            _sep_pred = "▰" * 68
            print("")
            print(f"┌{_sep_pred}┐")
            print(
                f"│    ▶  Model to evaluate ({_n_pred_current}/{_n_pred_total}) :  {str(alias).upper():<34}│"  # noqa: E501
            )
            print(f"└{_sep_pred}┘")
            print("")

            for ref_alias in list_references:
                if ref_alias not in dataset_manager.datasets:
                    logger.warning(
                        f"Reference dataset '{ref_alias}' not found in dataset manager. Skipping."
                    )
                    continue

                ref_source_dict: Dict[str, Any] = next(
                    (s for s in self.args.sources if s.get("dataset") == ref_alias), {}
                )
                ref_transforms[ref_alias] = transforms_dict.get(ref_alias)
                metrics_names[ref_alias] = ref_source_dict.get("metrics", ["rmsd"])
                ref_is_observation = dataset_manager.datasets[ref_alias].get_global_metadata()[
                    "is_observation"
                ]
                pred_eval_vars = dataset_manager.datasets[alias].get_eval_variables()
                ref_eval_vars = dataset_manager.datasets[ref_alias].get_eval_variables()

                common_vars = [
                    get_standardized_var_name(var) for var in pred_eval_vars if var in ref_eval_vars
                ]
                if not common_vars:
                    logger.warning(
                        "No common variables found between pred_data and ref_data for evaluation."
                    )
                    continue

                oceanbench_eval_variables = (
                    [get_variable_alias(var) for var in common_vars] if common_vars else None
                )

                common_metrics = [
                    metric for metric in metrics_names[alias] if metric in metrics_names[ref_alias]
                ]
                metrics_kwargs[alias][ref_alias] = {"add_noise": False}

                # Forward per-bins spatial resolution from YAML config
                _pbr = getattr(self.args, "per_bins_resolution", None)
                if _pbr is not None:
                    metrics_kwargs[alias][ref_alias]["bin_resolution"] = int(_pbr)

                if not ref_is_observation:
                    metrics[alias][ref_alias] = [
                        MetricComputer(
                            common_vars,
                            oceanbench_eval_variables,  # type: ignore[arg-type]
                            metric_name=metric,
                            **metrics_kwargs[alias][ref_alias],
                        )
                        for metric in common_metrics
                    ]
                else:
                    interpolation_method = ref_source_dict.get("interpolation_method", "pyinterp")
                    time_tolerance_hours = ref_source_dict.get("time_tolerance", None)
                    class4_kwargs = {
                        "interpolation_method": interpolation_method,
                        "list_scores": common_metrics,
                        "time_tolerance": timedelta(hours=float(time_tolerance_hours or 0)),
                    }
                    metrics[alias][ref_alias] = [
                        MetricComputer(
                            common_vars,
                            oceanbench_eval_variables,  # type: ignore[arg-type]
                            metric_name=metric,
                            is_class4=True,
                            class4_kwargs=class4_kwargs,
                            **metrics_kwargs[alias][ref_alias],
                        )
                        for metric in common_metrics
                    ]

            forecast_mode = self.args.n_days_forecast > 1

            effective_references = [
                ref_alias
                for ref_alias in list_references
                if ref_alias in metrics[alias] and metrics[alias][ref_alias]
            ]
            if not effective_references:
                logger.warning(
                    f"No compatible references with common variables/metrics for "
                    f"candidate '{alias}'. Skipping evaluation for this candidate."
                )
                continue

            ref_transforms = {
                ref_alias: ref_transforms[ref_alias]
                for ref_alias in effective_references
                if ref_alias in ref_transforms
            }

            # ── Determine obs_batch_size ───────────────────────────────
            # Look for per-dataset obs_batch_size first, then global, then default.
            _obs_batch_size = None
            for _ref_alias in effective_references:
                _ref_src: Dict[str, Any] = next(
                    (s for s in self.args.sources if s.get("dataset") == _ref_alias), {}
                )
                if _ref_src.get("obs_batch_size") is not None:
                    _obs_batch_size = int(_ref_src["obs_batch_size"])
                    break
            if _obs_batch_size is None:
                _obs_batch_size = getattr(self.args, "obs_batch_size", None)
                if _obs_batch_size is not None:
                    _obs_batch_size = int(_obs_batch_size)

            # ── Determine gridded_batch_size ───────────────────────────
            # Per-reference gridded_batch_size overrides the global batch_size
            # for non-observation (gridded) reference datasets such as GLORYS.
            # A small value (e.g. 6) limits per-batch I/O + RAM pressure.
            _gridded_batch_size = None
            for _ref_alias in effective_references:
                _ref_src = next(
                    (s for s in self.args.sources if s.get("dataset") == _ref_alias), {}
                )
                if _ref_src.get("gridded_batch_size") is not None:
                    _gridded_batch_size = int(_ref_src["gridded_batch_size"])
                    break
            if _gridded_batch_size is None:
                _gridded_batch_size = getattr(self.args, "gridded_batch_size", None)
                if _gridded_batch_size is not None:
                    _gridded_batch_size = int(_gridded_batch_size)

            dataloaders[alias] = dataset_manager.get_dataloader(
                pred_alias=alias,
                ref_aliases=effective_references,
                batch_size=self.args.batch_size,
                obs_batch_size=_obs_batch_size,
                gridded_batch_size=_gridded_batch_size,
                pred_transform=pred_transform,
                ref_transforms=ref_transforms,  # type: ignore[arg-type]
                forecast_mode=forecast_mode,
                n_days_forecast=self.args.n_days_forecast,
                lead_time_unit="days",
            )

            evaluators[alias] = Evaluator(
                dataset_manager=dataset_manager,
                metrics=metrics[alias],
                dataloader=dataloaders[alias],
                ref_aliases=effective_references,
                dataset_processor=self.dataset_processor,
                dask_cfgs_by_dataset=self.dask_cfgs_by_dataset,
                results_dir=results_files_dir,
                reduce_precision=getattr(self.args, "reduce_precision", False),
                restart_workers_per_batch=getattr(self.args, "restart_workers_per_batch", False),
                restart_frequency=getattr(self.args, "restart_frequency", 1),
                max_worker_memory_fraction=getattr(self.args, "max_worker_memory_fraction", 0.85),
            )
            _n_pred_total = len(dataset_references)
            _n_pred_current = list(dataset_references.keys()).index(alias) + 1
            '''_sep_pred = "▰" * 68
            logger.info("")
            logger.info(f"┌{_sep_pred}┐")
            logger.info(
                f"│    ▶  Model to evaluate ({_n_pred_current}/{_n_pred_total})"
                f" :  {str(alias).upper():<38}│"
            )
            logger.info(f"└{_sep_pred}┘")
            logger.info("")'''
            models_results[alias] = evaluators[alias].evaluate()

            # Keep the processor reference in sync with the last active cluster.
            try:
                self.dataset_processor = evaluators[alias].dataset_processor
            except Exception:
                pass

            # ── Separator: evaluation done -> post-processing ──────────────
            _sep_post = "─" * 68
            print("")
            print(f"┌{_sep_post}┐")
            print(f"│{'  📦  POST-PROCESSING RESULTS  —  ' + alias.upper():^67}│")
            print(f"└{_sep_post}┘")
            print("")

            # Aggregate batch results and write final JSON.
            try:
                batch_files = glob(os.path.join(results_files_dir, "results_*_batch_*.json"))
                results_dict: Dict[str, Any] = {}
                n_errors = 0

                logger.info(
                    f"  ┌ Consolidating {len(batch_files)} batch file(s) "
                    f"for '{alias}' ..."
                )

                # Per-bins are written incrementally as a JSONL file (one
                # compact JSON object per line) so that the full list is never
                # held in RAM.  This prevents OOM crashes on large datasets
                # where an indented JSON dump can reach multiple gigabytes.
                per_bins_path = os.path.join(
                    self.results_directory,
                    f"results_{alias}_per_bins.jsonl",
                )
                _pb_count = 0
                with open(per_bins_path, "w", encoding="utf-8") as pb_file:
                    for batch_file in batch_files:
                        with open(batch_file, "r") as f:
                            batch_results = json.load(f)
                        for item in batch_results:
                            if isinstance(item, dict) and item.get("error"):
                                n_errors += 1
                            # Pop per_bins before serialization: keeps the main
                            # results file flat and leaderboard-compatible.
                            # Write each entry immediately – no in-RAM list.
                            if isinstance(item, dict) and "per_bins" in item:
                                _is_obs = item.get(
                                    "ref_is_observation",
                                    item.get("is_class4", False),
                                )
                                _pb_entry = {
                                    "ref_alias": item.get("ref_alias", alias),
                                    "valid_time": item.get("valid_time"),
                                    "forecast_reference_time": item.get(
                                        "forecast_reference_time"
                                    ),
                                    "lead_time": item.get("lead_time"),
                                    "ref_type": "observation" if _is_obs else "gridded",
                                    "per_bins": item.pop("per_bins"),
                                }
                                # Compact JSON (no indent) -> ~4× smaller than
                                # the previous indented format.
                                pb_file.write(
                                    json.dumps(
                                        nan_to_none(make_serializable(_pb_entry)),
                                        separators=(",", ":"),
                                        ensure_ascii=False,
                                    )
                                    + "\n"
                                )
                                _pb_count += 1
                        transform_in_place(batch_results, make_serializable)
                        serializable_result = nan_to_none(batch_results)
                        results_dict.setdefault(alias, []).extend(serializable_result)

                _total_entries = sum(len(v) for v in results_dict.values())
                logger.info(
                    f"  │  {_total_entries} result entr{'y' if _total_entries == 1 else 'ies'} "
                    f"consolidated from {len(batch_files)} batch(es)"
                )

                if _pb_count == 0:
                    # Nothing was written – remove the empty placeholder file.
                    try:
                        os.remove(per_bins_path)
                    except OSError:
                        pass
                    per_bins_path = None  # type: ignore[assignment]
                    logger.info("  │  No per-bins spatial data produced for this dataset")
                else:
                    logger.info(
                        f"  │  Per-bins spatial data  ->  {_pb_count} entr"
                        f"{'y' if _pb_count == 1 else 'ies'} written "
                        f"(JSONL, compact format)"
                    )

                # ── Write results JSON before checking for errors ──────────
                # Writing first ensures partial results are never discarded
                # even when a handful of tasks were cancelled by the watchdog.
                with open(dataset_json_path, "w") as json_file:
                    json_file.write("")
                    json.dump(
                        {
                            "dataset": alias,
                            "results": results_dict,
                            "metadata": {
                                "evaluation_date": pd.Timestamp.now().isoformat(),
                                "total_entries": _total_entries,
                                "n_errors": n_errors,
                                "config": {
                                    "start_time": self.args.start_time,
                                    "end_time": self.args.end_time,
                                    "n_days_forecast": self.args.n_days_forecast,
                                    "n_days_interval": self.args.n_days_interval,
                                },
                            },
                        },
                        json_file,
                        indent=2,
                        ensure_ascii=False,
                    )
                logger.info(
                    f"  └  Results saved  ->  {os.path.basename(dataset_json_path)}"
                )

                # ── Error threshold check ──────────────────────────────────
                # max_task_errors (int, default 0): tolerated number of
                # individual task failures before the run is considered failed.
                # Set a non-zero value in the YAML config to tolerate occasional
                # network timeouts / watchdog cancellations without losing all
                # consolidated results.
                _max_errors = int(getattr(self.args, "max_task_errors", 0))
                if n_errors > _max_errors:
                    raise RuntimeError(
                        f"Evaluation completed with {n_errors} computation error(s) "
                        f"for dataset '{alias}' "
                        f"(tolerance: max_task_errors={_max_errors}). "
                        f"Results were saved to {os.path.basename(dataset_json_path)}."
                    )
                if n_errors > 0:
                    logger.warning(
                        f"  └  {n_errors} task error(s) tolerated "
                        f"(max_task_errors={_max_errors}) — results saved."
                    )

            except Exception as exc:
                logger.error(f"  └  [ERROR] Failed to write JSON results: {exc}")
                raise

        dataset_manager.file_cache.clear()

        # ══════════════════════════════════════════════════════════════════
        # LEADERBOARD GENERATION
        # ══════════════════════════════════════════════════════════════════
        # waiting 1s for the final log messages to flush before printing the leaderboard header
        _time.sleep(1)
        _sep_lb = "═" * 68
        print("")
        print(f"╔{_sep_lb}╗")
        print(f"║{'  🏆  LEADERBOARD GENERATION':^67}║")
        print(f"╚{_sep_lb}╝")
        print("")

        if not models_results:
            logger.warning(
                "  Leaderboard generation skipped — no evaluation results were produced.\n"
                "  (All candidate datasets were skipped: missing references or "
                "incompatible variables/metrics.)\n"
                "  Check that the reference datasets listed in dataset_references "
                "are properly loaded."
            )
            return

        try:
            import shutil as _shutil

            from dcleaderboard.build import (
                render_site_from_results_dir as _render_leaderboard,
            )

            _leaderboard_dir = os.path.join(self.results_directory, "leaderboard")
            _leaderboard_input_dir = os.path.join(self.results_directory, "leaderboard_input")
            os.makedirs(_leaderboard_dir, exist_ok=True)
            os.makedirs(_leaderboard_input_dir, exist_ok=True)

            # Copy reference baseline JSONs.
            # Primary source: dc/leaderboard_results/ (sibling of evaluate.py).
            # __file__ = dctools/processing/base.py -> parents[2] = project root -> / "dc"
            # Fallback: results/ bundled inside the dcleaderboard package.
            _dc_dir = Path(__file__).resolve().parents[2] / "dc"
            _local_lb_dir = _dc_dir / "leaderboard_results"
            if _local_lb_dir.is_dir():
                _ref_results_src = str(_local_lb_dir)
                logger.info(f"  ┌ Using dc/leaderboard_results/  ->  {_local_lb_dir}")
            else:
                import dcleaderboard as _dcleaderboard
                _ref_results_src = os.path.join(
                    os.path.dirname(_dcleaderboard.__file__), "results"
                )
                logger.info(f"  ┌ Using dcleaderboard package results/  ->  {_ref_results_src}")
            _ref_jsons = glob(os.path.join(_ref_results_src, "results_*.json"))
            for _ref_json in _ref_jsons:
                _shutil.copy2(
                    _ref_json,
                    os.path.join(_leaderboard_input_dir, os.path.basename(_ref_json)),
                )
            logger.info(
                f"  │  Reference baselines copied  ({len(_ref_jsons)} file(s))"
            )
            # Also copy leaderboard_config.yaml from the reference source if present.
            _lb_config_src = os.path.join(_ref_results_src, "leaderboard_config.yaml")
            if os.path.isfile(_lb_config_src):
                _shutil.copy2(
                    _lb_config_src,
                    os.path.join(_leaderboard_input_dir, "leaderboard_config.yaml"),
                )
                logger.info("  │  leaderboard_config.yaml copied")

            # Copy current evaluation results into leaderboard input dir.
            # Use direct file lookup by alias for results JSON, and a robust
            # glob for per-bins JSONL files so we never miss them.
            _copied = []
            for _alias in dataset_references:
                _src = os.path.join(self.results_directory, f"results_{_alias}.json")
                if os.path.isfile(_src):
                    _shutil.copy2(
                        _src,
                        os.path.join(_leaderboard_input_dir, f"results_{_alias}.json"),
                    )
                    _copied.append(f"results_{_alias}.json")

            # Copy ALL per-bins files (both .jsonl and legacy .json) via glob
            # to avoid any alias-name mismatch issues.
            for _pb_src in glob(os.path.join(self.results_directory, "*_per_bins.jsonl")):
                _pb_dst = os.path.join(_leaderboard_input_dir, os.path.basename(_pb_src))
                _shutil.copy2(_pb_src, _pb_dst)
                _copied.append(os.path.basename(_pb_src))
            for _pb_src in glob(os.path.join(self.results_directory, "*_per_bins.json")):
                _pb_dst = os.path.join(_leaderboard_input_dir, os.path.basename(_pb_src))
                _shutil.copy2(_pb_src, _pb_dst)
                _copied.append(os.path.basename(_pb_src))

            for _fname in _copied:
                logger.info(f"  │  Staged for leaderboard  ->  {_fname}")
            if not any("_per_bins" in f for f in _copied):
                logger.warning(
                    "  │  [WARNING] No per-bins file found in results directory "
                    f"({self.results_directory}) — maps.html will be skipped."
                )

            logger.info("  │  Rendering leaderboard site ...")
            _render_leaderboard(
                results_dir=_leaderboard_input_dir,
                output_site_dir=_leaderboard_dir,
                custom_config=self.leaderboard_custom_config,
            )
            # Clean up the temporary input dir
            _shutil.rmtree(_leaderboard_input_dir, ignore_errors=True)

            # Verify completeness: maps.html requires per-bins data.
            _maps_html = os.path.join(_leaderboard_dir, "maps.html")
            if not os.path.isfile(_maps_html):
                _msg = (
                    "maps.html was not generated (no per-bins spatial data found "
                    "in the results directory — check that per_bins metrics are "
                    "enabled and that at least one batch produced spatial data)."
                )
                self._leaderboard_warnings.append(_msg)
                logger.warning(f"  └  [LEADERBOARD INCOMPLETE] {_msg}")
            else:
                logger.info(f"  └  Leaderboard ready  ->  {_leaderboard_dir}")
            print("")
        except Exception as _lb_exc:
            _msg = f"Leaderboard generation failed: {_lb_exc!r}"
            self._leaderboard_warnings.append(_msg)
            logger.warning(
                f"  └  [WARNING] Leaderboard generation failed (non-blocking):\n"
                f"              {_lb_exc!r}"
            )
