"""Core worker-side metric computation task.

This module contains the ``compute_metric`` function — the single entry point
submitted to each Dask worker to evaluate metrics for one prediction–reference
pair.  It is the largest standalone function in the evaluation pipeline (~870
lines).
"""

import gc
import os
import shutil
import time
import traceback
from argparse import Namespace
from typing import Any, Callable, Dict, List, Optional, Tuple

import dask
import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger

from dctools.data.connection.connection_manager import create_worker_connect_config
from dctools.data.datasets.dataloader import ObservationDataViewer, filter_by_time
from dctools.metrics.metrics import MetricComputer
from dctools.metrics.worker_cleanup import (
    _cap_worker_threads,
    _clear_stale_s3_state,
    _clear_xarray_file_cache,
    worker_memory_cleanup,
)
from dctools.metrics.worker_compute import (
    _compute_with_timeout,
    _open_dataset_worker_cached,
)
from dctools.utilities.format_converter import convert_format1_to_format2
from dctools.utilities.misc_utils import to_float32

try:
    from torchvision import transforms as _output_transforms
except ImportError:
    _output_transforms = None


def _open_prefetched_obs_data(
    shared_obs_path: str,
    time_bounds: Tuple[pd.Timestamp, pd.Timestamp],
    n_points_dim: str,
    ref_alias: str,
    valid_time: Any,
) -> Optional[xr.Dataset]:
    """Open prefetched shared observation data for one worker time window."""
    import json as _json_obs_manifest
    from pathlib import Path as _OPath

    import xarray as _xr_obs

    _shared_path = _OPath(str(shared_obs_path))
    _manifest_path = (
        _shared_path / "manifest.json"
        if _shared_path.is_dir()
        else _shared_path.parent / "manifest.json"
    )

    if _manifest_path.is_file():
        with open(_manifest_path) as _mf:
            _manifest = _json_obs_manifest.load(_mf)

        _t0 = np.datetime64(pd.Timestamp(time_bounds[0]))
        _t1 = np.datetime64(pd.Timestamp(time_bounds[1]))
        _t0_ns = float(np.datetime64(_t0, "ns").view(np.int64))
        _t1_ns = float(np.datetime64(_t1, "ns").view(np.int64))
        _manifest_dim = _manifest.get("n_points_dim", n_points_dim)

        _relevant = [
            _file_meta for _file_meta in _manifest["files"]
            if _file_meta["t_min_ns"] <= _t1_ns
            and _file_meta["t_max_ns"] >= _t0_ns
            and _file_meta["n_points"] > 0
            and os.path.exists(_file_meta["path"])
        ]

        if not _relevant:
            logger.debug(
                f"[{ref_alias}] valid_time={valid_time}: "
                f"0 mini-zarrs overlap time window "
                f"{time_bounds[0]}–{time_bounds[1]}"
            )
            return None

        _paths = [f["path"] for f in _relevant]
        try:
            if len(_paths) == 1:
                # Single mini-zarr: open directly (no concat overhead).
                try:
                    ref_data = _xr_obs.open_zarr(_paths[0], consolidated=True)
                except Exception:
                    ref_data = _xr_obs.open_zarr(_paths[0], consolidated=False)
            else:
                # Multiple mini-zarrs: use open_mfdataset for efficient
                # batch opening instead of N individual open_zarr + concat.
                # parallel=False avoids nested Dask-in-Dask scheduling.
                try:
                    ref_data = _xr_obs.open_mfdataset(
                        _paths,
                        engine="zarr",
                        concat_dim=_manifest_dim,
                        combine="nested",
                        consolidated=True,
                        parallel=False,
                    )
                except Exception:
                    ref_data = _xr_obs.open_mfdataset(
                        _paths,
                        engine="zarr",
                        concat_dim=_manifest_dim,
                        combine="nested",
                        consolidated=False,
                        parallel=False,
                    )

            if _manifest_dim != n_points_dim and _manifest_dim in ref_data.dims:
                ref_data = ref_data.rename({_manifest_dim: n_points_dim})

            _time_key = _manifest.get("time_key", "time")
            if _time_key in ref_data.coords or _time_key in ref_data.data_vars:
                _tv = ref_data[_time_key]
                ref_data = ref_data.where(
                    (_tv >= _t0) & (_tv <= _t1),
                    drop=True,
                )

            _obs_compute_timeout = int(
                os.environ.get("DCTOOLS_OBS_COMPUTE_TIMEOUT", "120")
            )
            ref_data = _compute_with_timeout(
                ref_data,
                timeout_s=_obs_compute_timeout,
                scheduler="synchronous",
            )
        finally:
            # open_mfdataset manages its own file handles; explicit close
            # is a safety measure for the single-file path.
            try:
                ref_data.close()  # type: ignore[possibly-undefined]
            except Exception:
                pass

        if ref_data.sizes.get(n_points_dim, 0) == 0:
            logger.debug(
                f"[{ref_alias}] valid_time={valid_time}: "
                f"0 points after time filter {time_bounds[0]}–{time_bounds[1]}"
            )
            return None

        return ref_data

    _zarr_target = str(shared_obs_path)
    _batch_store = _shared_path / "batch_shared.zarr"
    if _shared_path.is_dir() and _batch_store.is_dir():
        _zarr_target = str(_batch_store)

    try:
        ref_data = _xr_obs.open_zarr(_zarr_target, consolidated=True)
    except Exception:
        ref_data = _xr_obs.open_zarr(_zarr_target, consolidated=False)

    _t0 = np.datetime64(pd.Timestamp(time_bounds[0]))
    _t1 = np.datetime64(pd.Timestamp(time_bounds[1]))
    _zarr_parent = _OPath(_zarr_target).parent
    _time_npy = str(_zarr_parent / "time_index.npy")
    _time_vals = None
    if os.path.exists(_time_npy):
        _time_vals = np.load(_time_npy, mmap_mode="r")
        if np.issubdtype(_time_vals.dtype, np.integer):
            _time_vals = np.array(_time_vals).astype("datetime64[ns]")

    if _time_vals is None:
        _time_var = ref_data.coords.get("time")
        if _time_var is not None:
            if hasattr(_time_var, "compute"):
                _time_vals = _time_var.compute(scheduler="synchronous").values
            else:
                _time_vals = _time_var.values
            _time_vals = np.asarray(_time_vals)
            if np.issubdtype(_time_vals.dtype, np.integer):
                _time_vals = _time_vals.astype("datetime64[ns]")
            elif not np.issubdtype(_time_vals.dtype, np.datetime64):
                _time_vals = pd.to_datetime(_time_vals).values

    if _time_vals is not None:
        _obs_time_sorted = True
        try:
            _meta_json = str(_zarr_parent / "batch_metadata.json")
            if os.path.exists(_meta_json):
                with open(_meta_json) as _mj:
                    _batch_meta = _json_obs_manifest.load(_mj)
                _obs_time_sorted = bool(_batch_meta.get("time_sorted", True))
        except Exception:
            pass

        if _obs_time_sorted:
            _obs_time_sorted = (
                len(_time_vals) <= 1
                or bool(np.all(_time_vals[:-1] <= _time_vals[1:]))
            )

        if _obs_time_sorted:
            _i0 = int(np.searchsorted(_time_vals, _t0, side="left"))
            _i1 = int(np.searchsorted(_time_vals, _t1, side="right"))
            ref_data = ref_data.isel({n_points_dim: slice(_i0, _i1)})
        else:
            _time_mask = (_time_vals >= _t0) & (_time_vals <= _t1)
            ref_data = ref_data.isel({n_points_dim: _time_mask})

    _obs_compute_timeout = int(
        os.environ.get("DCTOOLS_OBS_COMPUTE_TIMEOUT", "120")
    )
    ref_data = _compute_with_timeout(
        ref_data,
        timeout_s=_obs_compute_timeout,
        scheduler="synchronous",
    )

    try:
        if hasattr(ref_data, "_file_obj") and ref_data._file_obj is not None:
            ref_data._file_obj.close()
    except Exception:
        pass

    if ref_data.sizes.get(n_points_dim, 0) == 0:
        logger.debug(
            f"[{ref_alias}] valid_time={valid_time}: "
            f"0 observation points in the shared zarr for time window "
            f"{time_bounds[0]}–{time_bounds[1]}"
        )
        return None

    return ref_data

def compute_metric(
    entry: Dict[str, Any],
    pred_source_config: Namespace,
    ref_source_config: Namespace,
    model: str,
    list_metrics: List[MetricComputer],
    pred_transform: Callable,
    ref_transform: Callable,
    argo_index: Optional[Optional[Any]] = None,
    reduce_precision: bool = False,
    results_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compute metrics for a single prediction-reference pair entry.

    Args:
        entry (Dict[str, Any]): Dictionary containing data and metadata for the evaluation.
        pred_source_config (Namespace): Configuration for the prediction source.
        ref_source_config (Namespace): Configuration for the reference source.
        model (str): Name of the model being evaluated.
        list_metrics (List[MetricComputer]): List of metric computers to apply.
        pred_transform (Callable): Tranformation function for prediction data.
        ref_transform (Callable): Transformation function for reference data.
        argo_index (Optional[Any], optional): Index for Argo data, if applicable. Defaults to None.
        reduce_precision (bool, optional): Whether to reduce floating point precision to float32.
            Defaults to False.

    Returns:
        Dict[str, Any]: Dictionary containing the evaluation results.
    """
    try:
        # -- Prevent CPU oversubscription --
        # Observation datasets (SWOT, saral, …) trigger pyinterp
        # bilinear interpolation which is CPU-bound C++ code that
        # releases the GIL.  With 8 workers × 2 threads × 4 pyinterp
        # threads the machine (16 physical cores) gets massively
        # oversubscribed -> 100 % CPU thrashing.
        # Cap library-level threads to 1 so only the Dask thread
        # concurrency controls parallelism.
        _cap_worker_threads(1)
        _t0_total = time.perf_counter()
        forecast_reference_time = entry.get("forecast_reference_time")
        lead_time = entry.get("lead_time")
        valid_time = entry.get("valid_time")
        pred_coords = entry.get("pred_coords")
        ref_coords = entry.get("ref_coords")
        ref_alias = entry.get("ref_alias")
        ref_is_observation = entry.get("ref_is_observation")

        # Logging debug: Confirm task start on worker
        # logger.debug(
        #     f"Start processing: {forecast_reference_time} "
        #     f"(Valid: {valid_time}) on worker"
        # )

        pred_protocol = pred_source_config.protocol
        ref_protocol = ref_source_config.protocol

        pred_source = entry["pred_data"]
        ref_source = entry["ref_data"]

        open_pred_func = create_worker_connect_config(
            pred_source_config,
            argo_index,
        )
        open_ref_func = create_worker_connect_config(
            ref_source_config,
            argo_index,
        )

        pred_data_from_cache = False
        pred_data_base = None
        if isinstance(pred_source, str):
            pred_data_base, pred_data_from_cache = _open_dataset_worker_cached(
                open_pred_func,
                pred_source,
            )
            pred_data = pred_data_base
        else:
            pred_data = pred_source

        if pred_data is None:
            raise RuntimeError(
                f"Failed to open prediction dataset: {pred_source!r}"
            )

        # Simple nearest-neighbor time selection (original, fast approach)
        pred_data_selected = pred_data.sel(time=valid_time, method="nearest")  # type: ignore[union-attr]

        # Ensure dimension "time" is preserved or restored (needed for concatenation later)
        if "time" not in pred_data_selected.dims:
            pred_data = pred_data_selected.expand_dims("time")
            pred_data = pred_data.assign_coords(time=[valid_time])
        else:
            pred_data = pred_data_selected.assign_coords(time=[valid_time])

        # -- Prediction: keep lazy until after transforms ---------------
        # For gridded datasets, transforms often subset/regrid the data.
        # Computing *before* transforms can load far more data than needed
        # and can blow worker memory (leading to worker restarts / scheduler
        # errors). We keep it lazy here, apply transforms, then compute once.

        # Drop unused variables early when possible (reduces IO + memory).
        _pred_keep = getattr(pred_source_config, "keep_variables", None)
        if not _pred_keep:
            _pred_keep = getattr(pred_source_config, "eval_variables", None)
        if _pred_keep and hasattr(pred_data, "data_vars"):
            _pred_keep_set = [v for v in list(_pred_keep) if v in pred_data.data_vars]
            if _pred_keep_set:
                pred_data = pred_data[_pred_keep_set]

        if reduce_precision:
            pred_data = to_float32(pred_data)

        if ref_source is not None:
            if ref_is_observation:
                # Observation entry is a dict with connection metadata;
                # we must reconstruct the actual xr.Dataset on the worker.
                raw_ref_df = ref_source["source"]
                keep_vars = ref_source["keep_vars"]
                target_dimensions = ref_source["target_dimensions"]
                time_bounds = ref_source["time_bounds"]
                metadata = ref_source["metadata"]

                ref_df = raw_ref_df.get_dataframe()
                t0, t1 = time_bounds
                ref_df = filter_by_time(ref_df, t0, t1)

                # -- Substitute remote paths with prefetched local copies --
                # When the driver has pre-downloaded observation files to
                # local disk (see prefetch_obs_files_to_local), replace
                # the S3 paths in the dataframe so the worker opens local
                # files instead of issuing remote requests.
                _prefetched_map = (
                    ref_source.get("prefetched_local_paths")
                    if isinstance(ref_source, dict) else None
                )
                if _prefetched_map and "path" in ref_df.columns:
                    ref_df = ref_df.copy()
                    ref_df["path"] = ref_df["path"].map(
                        lambda p: _prefetched_map.get(p, p)
                    )

                if ref_df.empty:
                    logger.warning(
                        f"No {ref_alias} data for time interval: {t0}/{t1}"
                    )
                    return {
                        "ref_alias": ref_alias,
                        "result": None,
                    }

                n_points_dim = "n_points"
                if ref_coords is not None and hasattr(ref_coords.coordinates, "n_points"):
                    n_points_dim = ref_coords.coordinates["n_points"]

                # -- ARGO shared-Zarr fast path ----------------------------
                # The driver has merged ALL batch time-windows into a
                # single shared, time-sorted Zarr (see
                # ArgoManager.prefetch_batch_shared_zarr).  Each worker
                # opens this Zarr and filters by its specific time_bounds
                # using np.searchsorted -> reads only contiguous chunks.
                # This replaces the old per-window Zarr approach.
                _argo_shared_zarr = (
                    ref_source.get("prefetched_argo_shared_zarr")
                    if isinstance(ref_source, dict) else None
                )
                # Legacy per-window path (kept for backward compat)
                _argo_legacy_zarr = (
                    ref_source.get("prefetched_zarr_path")
                    if isinstance(ref_source, dict) else None
                )
                _used_prefetch = False

                if (
                    ref_alias == "argo_profiles"
                    and _argo_shared_zarr
                ):
                    try:
                        import xarray as _xr_argo
                        # Support either a single shared store (str/path)
                        # or multiple stores (list[str]) for month-boundary
                        # windows.
                        if isinstance(_argo_shared_zarr, (list, tuple)):
                            _zarr_paths = [str(p) for p in _argo_shared_zarr]
                        else:
                            _zarr_paths = [str(_argo_shared_zarr)]

                        _zarr_paths = [p for p in _zarr_paths if p]
                        if not _zarr_paths or not all(os.path.exists(p) for p in _zarr_paths):
                            raise FileNotFoundError(
                                "One or more shared ARGO Zarr paths are missing"
                            )

                        _ds_parts = [
                            _xr_argo.open_zarr(
                                p,
                                consolidated=True,
                                chunks=None,
                            )
                            for p in _zarr_paths
                        ]

                        if len(_ds_parts) == 1:
                            ref_data = _ds_parts[0]
                        else:
                            # Concatenate in time order (paths injected already sorted)
                            ref_data = _xr_argo.concat(_ds_parts, dim="obs")

                        # Normalise dimension name
                        _obs_dim = "obs"
                        if _obs_dim in ref_data.dims and n_points_dim not in ref_data.dims:
                            ref_data = ref_data.rename({_obs_dim: n_points_dim})
                        elif _obs_dim not in ref_data.dims and n_points_dim in ref_data.dims:
                            pass  # already named correctly
                        else:
                            _obs_dim = n_points_dim  # no rename needed

                        # Detect time coordinate name (ARGO uses uppercase TIME)
                        _time_name = None
                        for _tc in ("TIME", "time", "JULD"):
                            if _tc in ref_data.coords or _tc in ref_data.data_vars:
                                _time_name = _tc
                                break

                        if _time_name is not None:
                            if _time_name in ref_data.coords:
                                _t_vals = np.asarray(ref_data.coords[_time_name].values)
                            else:
                                _t_vals = np.asarray(ref_data[_time_name].values)
                            if not np.issubdtype(_t_vals.dtype, np.datetime64):
                                try:
                                    _t_vals = pd.to_datetime(_t_vals).values
                                except Exception:
                                    _t_vals = _t_vals.astype("datetime64[ns]")

                            _t0_np = np.datetime64(pd.Timestamp(time_bounds[0]))
                            _t1_np = np.datetime64(pd.Timestamp(time_bounds[1]))

                            # Fast path: data is sorted -> contiguous slice
                            _is_sorted = (
                                len(_t_vals) <= 1
                                or bool(np.all(_t_vals[:-1] <= _t_vals[1:]))
                            )
                            if _is_sorted:
                                _i0 = int(np.searchsorted(_t_vals, _t0_np, side="left"))
                                _i1 = int(np.searchsorted(_t_vals, _t1_np, side="right"))
                                ref_data = ref_data.isel({n_points_dim: slice(_i0, _i1)})
                            else:
                                _mask = (_t_vals >= _t0_np) & (_t_vals <= _t1_np)
                                ref_data = ref_data.isel({n_points_dim: _mask})

                        # With chunks=None the data is already NumPy;
                        # .compute() is only needed if still dask-backed.
                        if any(
                            hasattr(ref_data[v].data, "dask")
                            for v in ref_data.variables
                        ):
                            ref_data = ref_data.compute(
                                scheduler="synchronous"
                            )

                        if ref_data.sizes.get(n_points_dim, 0) == 0:
                            ref_data = None

                        _used_prefetch = True
                    except Exception as exc:
                        logger.warning(
                            f"Cannot use shared ARGO Zarr "
                            f"'{_argo_shared_zarr}': {exc!r} "
                            "— falling back"
                        )
                        import traceback as _tb_argo
                        _tb_argo.print_exc()

                # -- Legacy per-window ARGO fast path (backward compat) ----
                if (
                    not _used_prefetch
                    and ref_alias == "argo_profiles"
                    and _argo_legacy_zarr
                    and os.path.exists(str(_argo_legacy_zarr))
                ):
                    try:
                        import xarray as _xr_local
                        ref_data = _xr_local.open_zarr(
                            str(_argo_legacy_zarr), consolidated=True
                        )
                        if "obs" in ref_data.dims and n_points_dim not in ref_data.dims:
                            ref_data = ref_data.rename({"obs": n_points_dim})
                        _used_prefetch = True
                    except Exception as exc:
                        logger.warning(
                            f"Cannot open prefetched ARGO Zarr "
                            f"'{_argo_legacy_zarr}': {exc!r} "
                            "— falling back to live download"
                        )

                # -- Shared obs fast path (SWOT, saral, …) -----------------
                # The driver has already preprocessed ALL unique observation
                # files into per-file mini-zarrs and written a manifest.json
                # listing paths + time ranges.  Workers open only the
                # mini-zarrs that overlap their time window → no serial
                # merge bottleneck on the driver, all CPUs stay busy.
                #
                # Legacy mode: if the path points to a single shared zarr
                # (no manifest.json), fall back to the slice-based approach.
                _shared_obs_zarr = (
                    ref_source.get("prefetched_obs_zarr_path")
                    if isinstance(ref_source, dict)
                    else None
                )
                if (
                    not _used_prefetch
                    and _shared_obs_zarr
                    and os.path.exists(str(_shared_obs_zarr))
                ):
                    _zarr_last_exc = None
                    for _zarr_attempt in range(3):
                        try:
                            ref_data = _open_prefetched_obs_data(
                                str(_shared_obs_zarr),
                                time_bounds,
                                n_points_dim,
                                str(ref_alias or ""),
                                valid_time,
                            )
                            _used_prefetch = True
                            _zarr_last_exc = None
                            break

                        except Exception as exc:
                            _zarr_last_exc = exc
                            ref_data = None
                            if _zarr_attempt < 2:
                                import time as _time_retry
                                _time_retry.sleep(
                                    1.0 * (1 + _zarr_attempt)
                                )
                                continue

                    if _zarr_last_exc is not None:
                        import traceback as _tb_zarr
                        logger.warning(
                            f"Cannot use shared obs data "
                            f"'{_shared_obs_zarr}' after 3 "
                            f"attempts: {_zarr_last_exc!r}\n"
                            f"{_tb_zarr.format_exc()}"
                        )

                if not _used_prefetch:
                    _MAX_FILES_PER_WORKER = int(
                        os.environ.get("DCTOOLS_MAX_OBS_FILES_PER_WORKER", "30")
                    )
                    _n_ref_files = len(ref_df)
                    logger.debug(
                        f"Worker fallback ({ref_alias}): processing "
                        f"{_n_ref_files} obs files for time window "
                        f"{time_bounds[0]}–{time_bounds[1]}"
                    )
                    _worker_shared_dir = None
                    if (
                        _MAX_FILES_PER_WORKER > 0
                        and _n_ref_files > _MAX_FILES_PER_WORKER
                    ):
                        _unique_local_paths = []
                        if "path" in ref_df.columns:
                            _unique_local_paths = list(
                                dict.fromkeys(str(_path) for _path in ref_df["path"].tolist())
                            )

                        _all_local_paths_available = bool(_unique_local_paths) and all(
                            os.path.exists(_path) for _path in _unique_local_paths
                        )

                        if _all_local_paths_available:
                            logger.info(
                                f"Worker fallback ({ref_alias}): {_n_ref_files} files exceed "
                                f"limit {_MAX_FILES_PER_WORKER}; building a local shared obs "
                                "store instead of dropping files"
                            )
                            from tempfile import mkdtemp

                            from dctools.data.datasets.batch_preprocessing import (
                                preprocess_batch_obs_files,
                            )

                            _coord_system = (
                                metadata.get("coord_system")
                                if isinstance(metadata, dict)
                                else getattr(metadata, "coord_system", None)
                            )
                            _coord_map = getattr(_coord_system, "coordinates", None)
                            if isinstance(_coord_map, dict):
                                _coord_map = dict(_coord_map)
                            elif hasattr(_coord_map, "items"):
                                _coord_map = dict(_coord_map.items())
                            else:
                                _coord_map = {"time": "time"}

                            _worker_shared_dir = mkdtemp(
                                prefix=f"dctools_{ref_alias}_worker_obs_"
                            )
                            try:
                                _worker_shared_path = preprocess_batch_obs_files(
                                    local_paths=_unique_local_paths,
                                    alias=str(ref_alias or ""),
                                    keep_vars=keep_vars,
                                    coordinates=_coord_map,
                                    n_points_dim=n_points_dim,
                                    output_zarr_dir=_worker_shared_dir,
                                    max_shared_obs_files=len(_unique_local_paths),
                                    prep_workers=1,
                                    prep_use_processes=False,
                                )
                            except Exception as _shared_exc:
                                logger.warning(
                                    f"Worker fallback ({ref_alias}): local shared obs build failed; "
                                    f"continuing with direct local preprocessing ({_shared_exc!r})"
                                )
                                _worker_shared_path = None

                            if _worker_shared_path:
                                ref_data = _open_prefetched_obs_data(
                                    _worker_shared_path,
                                    time_bounds,
                                    n_points_dim,
                                    str(ref_alias or ""),
                                    valid_time,
                                )
                                _used_prefetch = True
                            else:
                                logger.warning(
                                    f"Worker fallback ({ref_alias}): local shared obs store unavailable; "
                                    "continuing with direct local preprocessing."
                                )

                        if not _used_prefetch and not _all_local_paths_available:
                            raise RuntimeError(
                                f"Worker fallback ({ref_alias}) would require processing "
                                f"{_n_ref_files} files without shared preprocessing; "
                                "aborting instead of silently dropping files."
                            )

                    if not _used_prefetch:
                        ref_raw_data = ObservationDataViewer(
                            ref_df,
                            open_ref_func, str(ref_alias or ""),
                            keep_vars, target_dimensions, metadata,
                            time_bounds,
                            n_points_dim=n_points_dim,
                            dataset_processor=None,
                        )
                        ref_data = ref_raw_data.preprocess_datasets(ref_df)
            else:
                # Non-observation references are passed as a single source
                # (typically a path string or datetime for CMEMS). Do not pass
                # `ref_alias` as a positional arg: BaseConnectionManager.open
                # expects (path, mode="rb").
                ref_data_from_cache = False
                ref_data_base = None
                if ref_protocol == "cmems":
                    with dask.config.set(scheduler="synchronous"):
                        ref_data = open_ref_func(ref_source)
                else:
                    if isinstance(ref_source, str):
                        ref_data_base, ref_data_from_cache = _open_dataset_worker_cached(
                            open_ref_func,
                            ref_source,
                        )
                        ref_data = ref_data_base
                    else:
                        ref_data = open_ref_func(ref_source)

                # -- Ensure reference is sliced by time (like prediction) --
                # For large gridded reference datasets (GLORYS, DUACS), failing
                # to slice by time loads the entire multi-year dataset into memory
                # during .compute(), causing immediate worker kills.
                if ref_data is not None:
                    _time_dim = None
                    for _td in ("time", "TIME", "JULD"):
                        if _td in ref_data.coords or _td in ref_data.dims:
                            _time_dim = _td
                            break

                    if _time_dim:
                        try:
                            # Use nearest neighbor to match valid_time
                            _ref_sel = ref_data.sel({_time_dim: valid_time}, method="nearest")

                            # Preserve time dimension (needed for concatenation)
                            if _time_dim not in _ref_sel.dims:
                                ref_data = _ref_sel.expand_dims(_time_dim).assign_coords({_time_dim: [valid_time]})  # noqa: E501
                            else:
                                ref_data = _ref_sel.assign_coords({_time_dim: [valid_time]})
                        except Exception as _exc_slice:
                            logger.warning(
                                f"Could not slice reference {ref_alias} by time: {_exc_slice!r}"
                            )
        else:
            ref_data = None

        # Grid-to-Track Logic Handling
        run_grid_to_track = False

        # Determine if we should optimize for Grid-to-Track
        if _output_transforms is not None and ref_is_observation and pred_transform is not None:
            # modified_transforms = []

            # Helper to inspect and filter transforms
            def inspect_transform(t):
                # If it's a Compose, recurse
                if isinstance(t, _output_transforms.Compose):
                    sub_list: List[Any] = []
                    for sub_t in t.transforms:
                        res = inspect_transform(sub_t)
                        if res:
                            sub_list.append(res)
                    return _output_transforms.Compose(sub_list) if sub_list else None

                # Check for interpolation transform
                name = getattr(t, "transform_name", "")
                if name == "glorys_to_glonet":
                    # This is the Grid-to-Grid interpolation we want to avoid!
                    return None
                return t

            new_transform_structure = inspect_transform(pred_transform)

            # If structure changed, it means we removed the interpolation
            # -> we must add Grid-to-Track
            if new_transform_structure != pred_transform:
                pred_transform = new_transform_structure
                run_grid_to_track = True


        if pred_transform is not None:
            pred_data = pred_transform(pred_data)
        if pred_data is None:
            logger.debug(
                f"[{ref_alias}] valid_time={valid_time}: no prediction data "
                f"for time window {time_bounds[0]}–{time_bounds[1]} "
                f"— skipping metric computation (result=null)."
            )
            return {
                "ref_alias": ref_alias,
                "result": None,
                "n_points": 0,
                "duration_s": 0.0,
                "preprocess_s": 0.0,
                "forecast_reference_time": forecast_reference_time,
                "lead_time": lead_time,
                "valid_time": valid_time,
            }
        # Compute once after transforms (safe & memory-friendlier).
        if hasattr(pred_data, 'chunks') and pred_data.chunks:
            # Use a hard Python-level timeout instead of bare .compute().
            # With scheduler='synchronous', aiobotocore's asyncio read_timeout
            # never fires (event loop blocked by the synchronous scheduler).
            # _compute_with_timeout() runs the compute in a daemon thread and
            # raises after DCTOOLS_S3_COMPUTE_TIMEOUT seconds (default 90 s).
            _s3_timeout = int(os.environ.get("DCTOOLS_S3_COMPUTE_TIMEOUT", "90"))
            try:
                pred_data = _compute_with_timeout(
                    pred_data, timeout_s=_s3_timeout, scheduler="synchronous"
                )
            except RuntimeError as _timeout_err:
                if "timed out" not in str(_timeout_err):
                    raise
                # Stale S3 connection — clear caches and retry once with
                # a fresh filesystem so the new dask graph gets live
                # aiobotocore sessions.
                logger.warning(
                    f"[{ref_alias}] pred_data.compute() timed out — "
                    f"clearing S3 caches and retrying with fresh connection"
                )
                _clear_stale_s3_state()
                # Restore protocol (deleted by create_worker_connect_config)
                pred_source_config.protocol = pred_protocol
                _retry_open = create_worker_connect_config(
                    pred_source_config, argo_index
                )
                if isinstance(pred_source, str):
                    pred_data = _retry_open(pred_source)
                else:
                    pred_data = pred_source
                if pred_data is None:
                    raise RuntimeError(
                        f"Retry failed: cannot re-open prediction "
                        f"dataset: {pred_source!r}"
                    ) from _timeout_err
                # Re-apply time selection
                pred_data = pred_data.sel(time=valid_time, method="nearest")
                if "time" not in pred_data.dims:
                    pred_data = pred_data.expand_dims("time")
                    pred_data = pred_data.assign_coords(time=[valid_time])
                else:
                    pred_data = pred_data.assign_coords(time=[valid_time])
                if _pred_keep and hasattr(pred_data, "data_vars"):
                    _pk = [v for v in list(_pred_keep) if v in pred_data.data_vars]
                    if _pk:
                        pred_data = pred_data[_pk]
                if reduce_precision:
                    pred_data = to_float32(pred_data)
                if pred_transform is not None:
                    pred_data = pred_transform(pred_data)
                if (
                    pred_data is not None
                    and hasattr(pred_data, 'chunks')
                    and pred_data.chunks
                ):
                    pred_data = _compute_with_timeout(
                        pred_data, timeout_s=_s3_timeout,
                        scheduler="synchronous",
                    )

        # Grid-to-Track interpolation is now handled internally by Class4 metrics
        # if run_grid_to_track and ref_data is not None:
        #    pass

        if ref_data is not None and ref_transform is not None:
            _ref_data_before_transform = ref_data
            ref_data = ref_transform(ref_data)
            # Re-materialize if transform produced new lazy arrays
            if hasattr(ref_data, 'chunks') and ref_data.chunks:
                _s3_timeout = int(os.environ.get("DCTOOLS_S3_COMPUTE_TIMEOUT", "90"))
                try:
                    ref_data = _compute_with_timeout(
                        ref_data, timeout_s=_s3_timeout, scheduler="synchronous"
                    )
                except RuntimeError as _timeout_err:
                    if "timed out" not in str(_timeout_err):
                        raise
                    logger.warning(
                        f"[{ref_alias}] ref_data.compute() timed out — "
                        f"clearing S3 caches and retrying"
                    )
                    _clear_stale_s3_state()
                    ref_data = ref_transform(_ref_data_before_transform)
                    if hasattr(ref_data, 'chunks') and ref_data.chunks:
                        ref_data = _compute_with_timeout(
                            ref_data, timeout_s=_s3_timeout,
                            scheduler="synchronous",
                        )

        if reduce_precision:
            # pred_data is already float32.
            if ref_data is not None:
                ref_data = to_float32(ref_data)

        # Force reloading if memory becomes an issue? No, trust Dask, but do explicit GC
        _clear_xarray_file_cache()
        gc.collect()

        _preprocess_time = time.perf_counter() - _t0_total
        t_start = time.perf_counter()

        results: Any = None
        _per_bins_data = None  # populated by the class4 observation branch

        if ref_is_observation:
            if ref_data is None:
                logger.debug(
                    f"[{ref_alias}] valid_time={valid_time}: no observation data "
                    f"for time window {time_bounds[0]}–{time_bounds[1]} "
                    f"— skipping metric computation (result=null)."
                )
                return {
                    "ref_alias": ref_alias,
                    "result": None,
                    "n_points": 0,
                    "duration_s": 0.0,
                    "preprocess_s": _preprocess_time,
                    "forecast_reference_time": forecast_reference_time,
                    "lead_time": lead_time,
                    "valid_time": valid_time,
                }

            n_points_raw = 0
            _npt_candidates = ('n_points', 'N_POINTS', 'obs')
            if isinstance(ref_data, list):
                # Count points (metadata only, fast)
                for ds in ref_data:
                    if hasattr(ds, 'sizes'):
                        for _npt in _npt_candidates:
                            if _npt in ds.sizes:
                                n_points_raw += ds.sizes[_npt]
                                break
            elif hasattr(ref_data, 'sizes'):
                for _npt in _npt_candidates:
                    if _npt in ref_data.sizes:
                        n_points_raw = ref_data.sizes[_npt]
                        break

            '''logger.debug(
                f"[{ref_alias}] valid_time={valid_time} — preprocessing done, "
                f"{n_points_raw} obs points — starting metric computation…"
            )'''

            _cap_worker_threads(1)

            with dask.config.set(scheduler='synchronous'):
                results = list_metrics[0].compute(
                    pred_data, ref_data,
                    pred_coords, ref_coords,
                )

            # Free SWOT/obs numpy slice immediately after metric computation —
            # ref_data can be several GB and is no longer needed past this point.
            try:
                if hasattr(ref_data, 'close'):
                    ref_data.close()
                del ref_data
                ref_data = None  # type: ignore[assignment]
                gc.collect()
            except Exception:
                pass

            t_end = time.perf_counter()
            duration = t_end - t_start
            logger.debug(
                f"[{ref_alias}] valid_time={valid_time}: "
                f"metrics done in {duration:.1f}s ({n_points_raw} obs pts)"
            )

            # Class4Evaluator.run() now returns {"results": DataFrame, "per_bins": {...}}
            # instead of a bare DataFrame.  Unpack before converting to records.
            if isinstance(results, dict) and "results" in results and "per_bins" in results:
                _per_bins_data = results["per_bins"]
                results = results["results"]
            if isinstance(results, pd.DataFrame):
                results = results.to_dict('records')
        else:
            # results = {}
            results = {}
            # -- Re-cap thread pools before metric computation (grid path) --
            _cap_worker_threads(1)

            # Context manager for the loop
            with dask.config.set(scheduler='synchronous'):
                for metric in list_metrics:
                    return_res = metric.compute(
                        pred_data, ref_data,
                        pred_coords, ref_coords,
                    )

                    # Handle per_bins when rmsd() returns {"results": DataFrame, "per_bins": dict}
                    if isinstance(return_res, dict) and "results" in return_res and "per_bins" in return_res:  # noqa: E501
                        _metric_per_bins = return_res.get("per_bins") or {}
                        if _metric_per_bins:
                            if _per_bins_data is None:
                                _per_bins_data = {}
                            _per_bins_data.update(_metric_per_bins)
                        return_res = return_res["results"]

                    if len(return_res) == 0:
                        return {
                            "ref_alias": ref_alias,
                            "result": None,
                            "n_points": 0,
                            "duration_s": 0.0,
                            "preprocess_s": _preprocess_time,
                            "forecast_reference_time": forecast_reference_time,
                            "lead_time": lead_time,
                            "valid_time": valid_time,
                        }

                    # Convert each DataFrame row to dictionary
                    res_dict: Dict[Any, Any] = {}
                    _lead_day_label = None
                    if lead_time is not None:
                        try:
                            _lead_day_label = f"Lead day {int(lead_time) + 1}"
                        except Exception:
                            _lead_day_label = None
                    for var_depth_label in return_res.index:
                        # Extract metric values for all lead days
                        metric_values = return_res.loc[var_depth_label].to_dict()
                        # Structure : {variable: metric_value}
                        if _lead_day_label and _lead_day_label in metric_values:
                            res_dict[var_depth_label] = metric_values[_lead_day_label]
                        elif 'Lead day 1' in metric_values:
                            res_dict[var_depth_label] = metric_values['Lead day 1']
                        else:
                            # Fallback: take the first value (stable order not guaranteed,
                            # but avoids KeyError and keeps a signal).
                            try:
                                res_dict[var_depth_label] = next(iter(metric_values.values()))
                            except StopIteration:
                                res_dict[var_depth_label] = None

                    results[metric.get_metric_name()] = res_dict

            # Convert from nested Format1 to Format2
            results = convert_format1_to_format2(results)
            t_end = time.perf_counter()
            duration = t_end - t_start
            # logger.debug(f"Fcast {forecast_reference_time} (LT {lead_time}): {duration:.2f}s")

            # Count grid points for reporting (was missing -> always showed 0)
            n_points_raw = 0
            if ref_data is not None and hasattr(ref_data, 'sizes'):
                _skip = {'time', 'forecast_reference_time', 'lead_time'}
                _grid_dims = [d for d in ref_data.sizes if d not in _skip]
                if _grid_dims:
                    n_points_raw = 1
                    for _gd in _grid_dims:
                        n_points_raw *= ref_data.sizes[_gd]

        res = {
            "ref_alias": ref_alias,
            "result": results,
            "duration_s": duration,
            "preprocess_s": _preprocess_time,
            "n_points": n_points_raw if 'n_points_raw' in locals() else 0
        }
        # Add forecast fields if present
        if forecast_reference_time is not None:
            res["forecast_reference_time"] = forecast_reference_time
        if lead_time is not None:
            res["lead_time"] = lead_time
        if valid_time is not None:
            res["valid_time"] = valid_time
        # Carry the observation flag so downstream writers can tag per_bins.
        res["ref_is_observation"] = bool(ref_is_observation)
        # Attach per_bins from the class4 observation branch (if present).
        if _per_bins_data is not None:
            res["per_bins"] = _per_bins_data

        return res

    except Exception as exc:  # noqa: E722
        logger.error(
            f"Error computing metrics for dataset {ref_alias} and date {forecast_reference_time}: "
            f"{repr(exc)}"
        )
        traceback.print_exc()
        return {
            "ref_alias": ref_alias,
            "result": None,
            "error": repr(exc),
        }
    finally:
        # R9: Aggressive memory cleanup at the end of each task.
        # This is critical to prevent memory leaks from accumulating across
        # a batch, especially with large observation datasets.
        # Explicitly delete large objects and trigger garbage collection.
        for var_name in ['pred_data', 'ref_data', 'pred_data_selected', 'pred_data_base', 'ref_data_base']:
            if var_name in locals():
                try:
                    del locals()[var_name]
                except Exception:
                    pass
        
        # Call the dedicated cleanup utility.
        worker_memory_cleanup()

