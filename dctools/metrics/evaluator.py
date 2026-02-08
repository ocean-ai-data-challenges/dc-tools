"""Metrics evaluator module for distributed evaluation."""

import gc
import json
import os
import time
import traceback
import ctypes
from argparse import Namespace
from typing import Any, Callable, Dict, List, Optional

import dask
import numpy as np
import pandas as pd
from dask.distributed import as_completed, wait
from loguru import logger
from oceanbench.core.distributed import DatasetProcessor
from tqdm import tqdm

from dctools.data.connection.connection_manager import (
    clean_for_serialization,
    create_worker_connect_config,
)
from dctools.data.datasets.dataloader import (
    EvaluationDataloader,
    ObservationDataViewer,
    filter_by_time,
)
from dctools.data.datasets.dataset_manager import MultiSourceDatasetManager
from dctools.metrics.metrics import MetricComputer
from dctools.utilities.format_converter import convert_format1_to_format2
from dctools.utilities.misc_utils import (
    deep_copy_object,
    serialize_structure,
    to_float32,
)


def worker_memory_cleanup():
    """
    Manual memory cleanup to be run on workers.

    Performs aggressive garbage collection and memory trimming.
    """
    # Collect multiple times to handle reference cycles
    for _ in range(3):
        gc.collect()

    # Linux-specific memory trimming (release to OS)
    try:
        ctypes.CDLL('libc.so.6').malloc_trim(0)
    except Exception:
        pass


def _clear_xarray_file_cache() -> bool:
    """Best-effort clearing of xarray's global file cache on the current process."""
    try:
        import xarray as xr

        # CRITICAL: Ensure file cache is minimal to prevent accumulation
        # Note: Some xarray versions don't allow 0, so we use 1 (minimal)
        xr.set_options(file_cache_maxsize=1)

        try:
            # Clear any existing cached file handles
            # Not part of xarray's public API, but widely used and necessary
            xr.backends.file_manager.FILE_CACHE.clear()
        except Exception:
            pass
        return True
    except Exception:
        return False


def _worker_full_cleanup() -> bool:
    """Full cleanup routine to run on workers via client.run()."""
    _clear_xarray_file_cache()
    worker_memory_cleanup()
    return True


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
        forecast_reference_time = entry.get("forecast_reference_time")
        lead_time = entry.get("lead_time")
        valid_time = entry.get("valid_time")
        pred_coords = entry.get("pred_coords")
        ref_coords = entry.get("ref_coords")
        ref_alias = entry.get("ref_alias")
        ref_is_observation = entry.get("ref_is_observation")

        # LOGGING DEBUG: Confirm task start on worker
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

        if isinstance(pred_source, str):
            if pred_protocol == "cmems":
                # cmems not compatible with Dask workers (pickling errors)
                with dask.config.set(scheduler='synchronous'):
                    pred_data = open_pred_func(pred_source)
            else:
                pred_data = open_pred_func(pred_source)
        else:
            pred_data = pred_source


        # SCIENTIFIC IMPROVEMENT: Use linear interpolation in time when possible
        # Standard benchmarking practice is to reconstruct the field at the
        # exact observation time (4D interpolation) rather than snapping to the
        # nearest model output step (Nearest Neighbor).
        if "time" in pred_data.dims and pred_data.sizes["time"] > 1:
            # Prepare valid_time as numpy datetime if needed
            vt = np.datetime64(valid_time)
            # Use .compute() only on min/max scalars to avoid loading full array
            # .values keeps numpy dtype (datetime64), unlike .item() which might convert to int(ns)
            t_min = pred_data.time.min().compute().values
            t_max = pred_data.time.max().compute().values

            # Check bounds to avoid extrapolation errors
            if t_min <= vt <= t_max:
                # Interpolation preserves the schema and interpolates variables
                pred_data = pred_data.interp(time=[valid_time], method="linear", assume_sorted=True)
            else:
                # Fallback to nearest if out of bounds (avoids NaNs of extrapolation)
                pred_data = pred_data.sel(time=[valid_time], method="nearest")
        else:
            # Fallback for snapshots / single records or huge datasets where interp is too costly
            pred_data = pred_data.sel(time=[valid_time], method="nearest")

        # Ensure dimension "time" is preserved or restored (needed for concatenation later)
        if "time" not in pred_data.dims:
            pred_data = pred_data.expand_dims("time")
            # Force the coordinate to be exactly the valid_time
            pred_data = pred_data.assign_coords(time=[valid_time])

        # Apply precision reduction as early as possible (before computing/loading)
        if reduce_precision:
            pred_data = to_float32(pred_data)
            # logger.debug("Converted model data to float32")

        # OPTIMIZATION (Reverted):
        # Do NOT load full data with .compute() for Grid-to-Track case
        # This forces download of full grid (global) when we only need a track (sparse).
        # We let Dask optimize I/O access.
        # try:
        #    pred_data = pred_data.compute()
        # except Exception as e:
        #    logger.warning(f"Could not preload model data: {e}")

        if ref_source is not None:
            if ref_is_observation:
                ref_source = entry["ref_data"]
                raw_ref_df = ref_source["source"]
                keep_vars = ref_source["keep_vars"]
                target_dimensions = ref_source["target_dimensions"]
                time_bounds = ref_source["time_bounds"]
                metadata = ref_source["metadata"]

                ref_df = raw_ref_df.get_dataframe()
                t0, t1 = time_bounds
                ref_df = filter_by_time(ref_df, t0, t1)

                if ref_df.empty:
                    logger.warning(f"No {ref_alias} Data for time interval: {t0}/{t1}]")
                    return {
                        "ref_alias": ref_alias,
                        "result": None,
                    }
                n_points_dim = "n_points"   # default
                if ref_coords is not None and hasattr(ref_coords.coordinates, "n_points"):
                        n_points_dim=ref_coords.coordinates["n_points"]
                ref_raw_data = ObservationDataViewer(
                    ref_df,
                    open_ref_func, str(ref_alias or ""),
                    keep_vars, target_dimensions, metadata,
                    time_bounds,
                    n_points_dim = n_points_dim,
                    dataset_processor=None,
                )
                # load immediately before increasing Dask graph size
                # MODIF: load_to_memory=False to avoid RAM overhead on workers
                ref_data = ref_raw_data.preprocess_datasets(
                    ref_df,
                    load_to_memory=False,
                )
                # logger.debug("END PREPROCESS REF DATA")
            else:
                if ref_protocol == "cmems":
                    with dask.config.set(scheduler='synchronous'):
                        ref_data = open_ref_func(ref_source, ref_alias)
                else:
                    ref_data = open_ref_func(ref_source, ref_alias)
        else:
            ref_data = None

        # Grid-to-Track Logic Handling
        run_grid_to_track = False
        from torchvision import transforms as output_transforms

        # Determine if we should optimize for Grid-to-Track
        if ref_is_observation and pred_transform is not None:
            # modified_transforms = []

            # Helper to inspect and filter transforms
            def inspect_transform(t):
                # If it's a Compose, recurse
                if isinstance(t, output_transforms.Compose):
                    sub_list: List[Any] = []
                    for sub_t in t.transforms:
                        res = inspect_transform(sub_t)
                        if res:
                            sub_list.append(res)
                    return output_transforms.Compose(sub_list) if sub_list else None

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
            if ref_protocol == "cmems":
                with dask.config.set(scheduler='synchronous'):
                    pred_data = pred_transform(pred_data)
            else:
                pred_data = pred_transform(pred_data)

        # Grid-to-Track interpolation is now handled internally by Class4 metrics
        # if run_grid_to_track and ref_data is not None:
        #    pass

        if ref_data is not None and ref_transform is not None:
            if ref_protocol == "cmems":
                with dask.config.set(scheduler='synchronous'):
                    ref_data = ref_transform(ref_data)
            else:
                ref_data = ref_transform(ref_data)

        if reduce_precision:
            # pred_data is already float32.
            if ref_data is not None:
                ref_data = to_float32(ref_data)

        # Force reloading if memory becomes an issue? No, trust Dask, but do explicit GC
        # import gc
        # gc.collect()

        t_start = time.perf_counter()

        results: Any = None

        if ref_is_observation:
            if ref_data is None:
                return {
                    "ref_alias": ref_alias,
                    "result": None,
                }

            n_points_raw = 0
            if isinstance(ref_data, list):
                # Count points (metadata only, fast)
                for ds in ref_data:
                    if hasattr(ds, 'sizes') and 'n_points' in ds.sizes:
                        n_points_raw += ds.sizes['n_points']
            elif hasattr(ref_data, 'sizes') and 'n_points' in ref_data.sizes:
                n_points_raw = ref_data.sizes['n_points']

            # Use synchronous scheduler to prevent deadlock (Workers waiting for sub-tasks)
            # This forces the interpolation and metric computation to run locally on the worker
            # instead of submitting new tasks to the already-saturated cluster.
            with dask.config.set(scheduler='synchronous'):
                results = list_metrics[0].compute(
                    pred_data, ref_data,
                    pred_coords, ref_coords,
                )

            t_end = time.perf_counter()
            duration = t_end - t_start
            # logger.info(
            #     f"Fcast {forecast_reference_time} (LT {lead_time}): "
            #     f"{n_points_raw} pts (metadata), {duration:.2f}s"
            # )

            if isinstance(results, pd.DataFrame):
                results = results.to_dict('records')
        else:
            # results = {}
            results = {}
            # Context manager for the loop
            # with dask.config.set(scheduler='synchronous'):
            for metric in list_metrics:
                return_res = metric.compute(
                    pred_data, ref_data,
                    pred_coords, ref_coords,
                )

                if len(return_res) == 0:
                    return {
                        "ref_alias": ref_alias,
                        "result": None,
                        "duration_s": 0.0,
                    }

                # Convert each DataFrame row to dictionary
                res_dict: Dict[Any, Any] = {}
                for var_depth_label in return_res.index:
                    # Extract metric values for all lead days
                    metric_values = return_res.loc[var_depth_label].to_dict()
                    # Structure : {variable: metric_value}
                    res_dict[var_depth_label] = metric_values['Lead day 1']

                results[metric.get_metric_name()] = res_dict

            # Convert from nested Format1 to Format2
            results = convert_format1_to_format2(results)
            t_end = time.perf_counter()
            duration = t_end - t_start
            logger.info(f"Fcast {forecast_reference_time} (LT {lead_time}): {duration:.2f}s")

        res = {
            "ref_alias": ref_alias,
            "result": results,
            "duration_s": duration,
            "n_points": n_points_raw if 'n_points_raw' in locals() else 0
        }
        # Add forecast fields if present
        if forecast_reference_time is not None:
            res["forecast_reference_time"] = forecast_reference_time
        if lead_time is not None:
            res["lead_time"] = lead_time
        if valid_time is not None:
            res["valid_time"] = valid_time

        return res

    except Exception as exc:
        logger.error(
            f"Error computing metrics for dataset {ref_alias} and date {forecast_reference_time}: "
            f"{repr(exc)}"
        )
        traceback.print_exc()
        return {
            "ref_alias": ref_alias,
            "result": None,
        }

    finally:
        # Aggressive memory release at end of task
        if 'pred_data' in locals():
            try:
                if hasattr(pred_data, 'close'):
                    pred_data.close()
                del pred_data
            except Exception:
                pass

        if 'ref_data' in locals():
            try:
                # Handle list of datasets (e.g. from observation data viewer)
                if isinstance(ref_data, list):
                    for ds in ref_data:
                        try:
                            if hasattr(ds, 'close'):
                                ds.close()
                        except Exception:
                            pass
                elif hasattr(ref_data, 'close'):
                    ref_data.close()  # type: ignore[union-attr]
                del ref_data
            except Exception:
                pass
        # pred_data_selected variable does not exist in the scope, removing it
        # if 'pred_data_selected' in locals():
        #     try:
        #         del pred_data_selected
        #     except Exception: pass

        worker_memory_cleanup()


class Evaluator:
    """Class to evaluate metrics on datasets."""

    def __init__(
        self,
        dataset_manager: MultiSourceDatasetManager,
        metrics: Dict[str, List[MetricComputer]],
        dataloader: EvaluationDataloader,
        ref_aliases: List[str],
        dataset_processor: DatasetProcessor,
        results_dir: Optional[str] = None,
        reduce_precision: bool = False,
        restart_workers_per_batch: bool = False,
        restart_frequency: int = 1,
        max_p_memory_increase: float = 0.2, # 20% increase default
        max_worker_memory_fraction: float = 0.85,
    ):
        """
        Initializes the evaluator.

        Args:
            dataset_manager (MultiSourceDatasetManager): Multi-source dataset manager.
            metrics (Dict[str, List[MetricComputer]]):
                Dictionary {ref_alias: [MetricComputer, ...]}.
            dataloader (EvaluationDataloader):
                Dataloader for evaluation.
            ref_aliases (List[str]): List of reference aliases.
            dataset_processor (DatasetProcessor): Dataset processor for distribution.
            results_dir (str, optional): Folder to save results. Defaults to None.
            reduce_precision (bool, optional): Reduce float precision (float32).
                Defaults to False.
            restart_workers_per_batch (bool, optional): Restart workers after each batch.
                Defaults to False.
            restart_frequency (int, optional): Frequency (nb of batches) cleanup/restart.
                Defaults to 1.
            max_p_memory_increase (float, optional): RAM increase threshold before
                restart. Defaults to 0.5 (50%).
            max_worker_memory_fraction (float, optional): Absolute threshold (fraction of
                Dask memory_limit) beyond which restart is triggered.
                Defaults to 0.85 (85%).
        """
        self.dataset_manager = dataset_manager
        self.dataset_processor = dataset_processor
        self.metrics = metrics
        self.dataloader = dataloader
        self.reduce_precision = reduce_precision
        self.restart_workers_per_batch = restart_workers_per_batch
        self.restart_frequency = restart_frequency
        self.max_p_memory_increase = max_p_memory_increase
        self.max_worker_memory_fraction = max_worker_memory_fraction
        # self.results = []
        self.ref_aliases = ref_aliases
        self.results_dir = results_dir

        (
            self.ref_managers,
            self.ref_catalogs,
            self.ref_connection_params,
        ) = dataset_manager.get_config()

    def log_cluster_memory_usage(self, batch_idx: int):
        """Log memory usage of each Dask worker."""
        if not hasattr(self.dataset_processor, "client") or self.dataset_processor.client is None:
            return

        try:
            info = self.dataset_processor.client.scheduler_info()
            workers = info.get('workers', {})

            logger.info(f"=== Memory Usage Start Batch {batch_idx} ===")
            for w_addr, w_info in workers.items():
                # Some versions of dask put 'metrics' in the info
                mem_used = w_info.get('metrics', {}).get('memory', w_info.get('memory', 0))
                mem_limit = w_info.get('memory_limit', 0)

                if mem_limit > 0:
                    percent = (mem_used / mem_limit) * 100
                    logger.info(
                        f"Worker {w_info.get('name', w_addr)}: "
                        f"{percent:.1f}% ({mem_used / 1024**3:.2f}GB / {mem_limit / 1024**3:.2f}GB)"
                    )
                else:
                    logger.info(
                        f"Worker {w_info.get('name', w_addr)}: "
                        f"{mem_used / 1024**3:.2f}GB used (no limit)"
                    )

        except Exception as e:
            logger.warning(f"Could not log cluster memory usage: {e}")

    def get_max_memory_usage(self) -> float:
        """Get the maximum memory usage across all workers (in bytes)."""
        if not hasattr(self.dataset_processor, "client") or self.dataset_processor.client is None:
            return 0.0

        try:
            info = self.dataset_processor.client.scheduler_info()
            workers = info.get('workers', {})
            max_mem = 0.0
            for w_info in workers.values():
                mem_used = w_info.get('metrics', {}).get('memory', w_info.get('memory', 0))
                if mem_used > max_mem:
                    max_mem = mem_used
            return max_mem
        except Exception:
            return 0.0

    def get_max_memory_fraction(self) -> float:
        """Get max(memory_used / memory_limit) across workers.

        Returns:
            float: Fraction in [0, +inf). Returns 0.0 if unavailable.
        """
        if not hasattr(self.dataset_processor, "client") or self.dataset_processor.client is None:
            return 0.0

        try:
            info = self.dataset_processor.client.scheduler_info()
            workers = info.get("workers", {})
            max_frac = 0.0
            for w_info in workers.values():
                mem_used = w_info.get("metrics", {}).get("memory", w_info.get("memory", 0))
                mem_limit = w_info.get("memory_limit", 0) or 0
                if mem_limit and mem_limit > 0:
                    max_frac = max(max_frac, float(mem_used) / float(mem_limit))
            return max_frac
        except Exception:
            return 0.0

    def evaluate(self) -> List[Dict[str, Any]]:
        """
        Evaluates metrics on dataloader data for each reference.

        Returns:
            List[Dict[str, Any]]: Metric results for each batch and each reference.
        """
        self.scattered_argo_indexes: Dict[str, Any] = {}
        self.scattered_ref_catalogs: Dict[str, Any] = {}

        # Baseline memory usage (will be set at first batch or after restart)
        self.baseline_memory = None

        try:
            for batch_idx, batch in enumerate(self.dataloader):

                # Check memory before deciding to restart
                current_max_memory = self.get_max_memory_usage()
                current_max_fraction = self.get_max_memory_fraction()

                if self.baseline_memory is None:
                    # Initialize baseline if first batch or just restarted
                    # If memory is currently very low (start of process), use it as baseline
                    # But if we just restarted, it should be low.
                    self.baseline_memory = current_max_memory
                    if self.baseline_memory == 0:
                         # Fallback to avoid division by zero (e.g. 100MB)
                         self.baseline_memory = 100 * 1024 * 1024

                # Calculate increase
                mem_increase_ratio = (
                    current_max_memory - self.baseline_memory
                ) / self.baseline_memory

                # Cleanup logic
                should_restart = False

                # Restart if memory increased significantly compared to baseline
                if self.restart_workers_per_batch and batch_idx > 0:
                    if (
                        mem_increase_ratio > self.max_p_memory_increase
                        or current_max_fraction > self.max_worker_memory_fraction
                    ):
                        logger.warning(
                            f"Worker memory increased by {mem_increase_ratio*100:.1f}% "
                            f"(Baseline: {self.baseline_memory/1024**3:.2f}GB -> "
                            f"Current: {current_max_memory/1024**3:.2f}GB). "
                            f"Max worker fraction: {current_max_fraction*100:.1f}% "
                            f"(threshold: {self.max_worker_memory_fraction*100:.1f}%). "
                            "Triggering restart."
                        )
                        should_restart = True
                    # Keep periodic restart as a fallback safety net (e.g. every 10 batches)
                    # if user wants strictly memory based, they can set frequency to very high.
                    # But user implied conditionning restart to memory.
                    # Let's use memory trigger primarily.

                if should_restart:
                    try:
                        logger.info(
                            f"[Batch {batch_idx}] Restarting workers to "
                            "free memory before processing..."
                        )

                        # Clear local references to scattered data.
                        # We DO NOT explicitly call client.cancel() here because client.restart()
                        # will already clear all data on the cluster, and sending cancel messages
                        # concurrently can cause 'CommClosedError' race conditions.
                        self.scattered_argo_indexes.clear()
                        self.scattered_ref_catalogs.clear()

                        # Force restart of all workers to ensure complete memory release
                        # This handles memory fragmentation and unmanaged memory leaks (NetCDF/HDF5)
                        n_workers_before = len(
                            self.dataset_processor.client.scheduler_info().get('workers', {})
                        )
                        logger.info(
                            f"Restarting {n_workers_before} workers to clear memory..."
                        )

                        # ROBUST METHOD: Restart with wait_for_workers=False
                        # + Active Waiting
                        # We avoid Dask native TimeoutError, while
                        # ensuring workers are ready before
                        # sending the next batch.
                        try:
                            # 1. Close cluster gracefully if possible? No, restart is better.
                            # Use restart
                            try:
                                self.dataset_processor.client.restart(
                                    wait_for_workers=False, timeout=5
                                )
                            except Exception as e:
                                logger.warning(
                                    f"Client restart command encountered error: {e}. "
                                    "Proceeding to validation/recovery."
                                )

                            # Active Waiting (Polling) for workers return
                            # We allow up to 120 seconds for the cluster
                            # to return to initial state
                            # Sometimes workers take time to stop
                            # cleanly (NetCDF locks etc)
                            import time
                            timeout = 120
                            deadline = time.time() + timeout

                            pbar_wait = tqdm(
                                total=n_workers_before,
                                desc="Restarting Workers",
                                leave=False
                            )
                            current_workers = 0

                            while time.time() < deadline:
                                try:
                                    s_info = self.dataset_processor.client.scheduler_info()
                                    if s_info:
                                        workers_dict = s_info.get('workers', {})
                                        current_workers = len(workers_dict)
                                        # Update progress bar
                                        pbar_wait.n = current_workers
                                        pbar_wait.refresh()

                                        # If we recovered all our workers, it's good
                                        if current_workers >= n_workers_before:
                                            break
                                except Exception:
                                    # Ignore transient connection errors during restart
                                    pass
                                time.sleep(1.0)

                            pbar_wait.close()

                            scheduler_info = self.dataset_processor.client.scheduler_info()
                            final_workers = len(scheduler_info.get('workers', {}))

                            if final_workers < n_workers_before:
                                logger.warning(
                                    f"Cluster restart incomplete after {timeout}s : "
                                    f"{final_workers}/{n_workers_before} workers ready. "
                                    "Attempting emergency recovery."
                                )
                                try:
                                    has_cluster = (
                                        hasattr(self.dataset_processor, 'cluster')
                                        and self.dataset_processor.cluster
                                    )
                                    if has_cluster:
                                        # If 0 workers, attempt full reset (0 -> N)
                                        # If incomplete (1..N-1), just request scale up
                                        # (force Dask to relaunch missing ones)
                                        if final_workers == 0:
                                            logger.critical(
                                                "No workers alive. "
                                                "Performing hard reset (scale 0->N)."
                                            )
                                            self.dataset_processor.cluster.scale(0)
                                            time.sleep(2)

                                        logger.info(
                                            f"Scaling cluster to target "
                                            f"{n_workers_before} workers..."
                                        )
                                        self.dataset_processor.cluster.scale(n_workers_before)

                                        # New short active wait (30s max)
                                        # for recovery
                                        deadline_recover = time.time() + 30
                                        while time.time() < deadline_recover:
                                            s_info = self.dataset_processor.client.scheduler_info()
                                            fw_count = len(s_info.get('workers', {}))
                                            if fw_count >= n_workers_before:
                                                final_workers = fw_count
                                                break
                                            time.sleep(1)

                                        if final_workers < n_workers_before:
                                             logger.error(
                                                 f"Recovery failed. Only "
                                                 f"{final_workers}/{n_workers_before} "
                                                 f"workers available."
                                             )
                                        else:
                                             logger.info(
                                                 f"Recovery successful! "
                                                 f"{final_workers} workers active."
                                             )

                                except Exception as e_scale:
                                    logger.error(f"Emergency scaling failed: {e_scale}")

                            else:
                                logger.info(
                                    f"Cluster successfully restarted with {final_workers} workers."
                                )

                        except Exception as e:
                             logger.warning(f"Restart procedure warning: {e}")

                        # Allow strict stabilization time
                        time.sleep(2.0)

                        # Reset baseline after restart
                        # We must wait a bit or just set it to None to be re-evaluated next
                        # iteration?
                        # It is better to re-evaluate immediately to set correct baseline for
                        # *next* check.
                        # However, right now memory is effectively 0 or very low.
                        # Wait, we are at start of batch loop.
                        # We can reset baseline memory to current (low) value
                        self.baseline_memory = self.get_max_memory_usage()
                        if self.baseline_memory == 0:
                            self.baseline_memory = 100 * 1024 * 1024

                    except Exception as e:
                        logger.error(f"Worker memory cleanup failed: {e}")
                        # Non-fatal error, we can continue
                        pass

                self.log_cluster_memory_usage(batch_idx)

                pred_alias =self.dataloader.pred_alias
                ref_alias = batch[0].get("ref_alias")
                # Extract necessary information
                pred_connection_params = self.dataloader.pred_connection_params
                ref_connection_params = self.dataloader.ref_connection_params[ref_alias]
                pred_transform = self.dataloader.pred_transform
                if self.dataloader.ref_transforms is not None:
                    ref_transform = self.dataloader.ref_transforms[ref_alias]

                argo_index = None
                if hasattr(self.dataloader.ref_managers[ref_alias], 'argo_index'):
                    # Disable scattering for Argo Index to improve stability
                    argo_index = self.dataloader.ref_managers[ref_alias].get_argo_index()

                batch_results = self._evaluate_batch(
                    batch, pred_alias, ref_alias,
                    pred_connection_params, ref_connection_params,
                    pred_transform, ref_transform,
                    argo_index=argo_index,
                )
                if batch_results is None:
                    continue
                serial_results = [
                    serialize_structure(res)
                    for res in batch_results
                    if res is not None
                ]

                # Save batch by batch
                batch_file = os.path.join(
                    self.results_dir or ".", f"results_{pred_alias}_batch_{batch_idx}.json"
                )
                with open(batch_file, "w") as f:
                    json.dump(serial_results, f, indent=2, ensure_ascii=False)

                # CRITICAL: Explicit cleanup of batch variables to avoid hidden references
                # Pattern: process → delete results → delete intermediate vars → gc.collect()
                del batch_results
                del serial_results
                # Also clear the batch data itself
                del batch

                # Force garbage collection on driver to release any remaining references
                gc.collect()

            # Cleanup scattered data
            self.scattered_argo_indexes.clear()
            self.scattered_ref_catalogs.clear()

        except Exception as exc:
            logger.error(f"Evaluation failed: {repr(exc)}")
            raise

        return []

    def clean_namespace(self, namespace: Namespace) -> Namespace:
        """Clean namespace by removing unpicklable objects."""
        ns = Namespace(**vars(namespace))
        # Removes unpicklable attributes
        for key in ['dask_cluster', 'fs', 'dataset_processor', 'client', 'session']:
            if hasattr(ns, key):
                delattr(ns, key)
        # Also cleans objects in ns.params if present
        if hasattr(ns, "params"):
            for key in ['fs', 'client', 'session', 'dataset_processor']:
                if hasattr(ns.params, key):
                    delattr(ns.params, key)
        return ns

    def _evaluate_batch(
        self, batch: List[Dict[str, Any]],
        pred_alias: str, ref_alias: str,
        pred_connection_params: Dict[str, Any], ref_connection_params: Dict[str, Any],
        pred_transform: Any, ref_transform: Any,
        argo_index: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        delayed_tasks: List[Any] = []

        if batch:
            raw_dates = [e.get("forecast_reference_time") for e in batch]
            dates = [d for d in raw_dates if d is not None]
            if dates:
                logger.info(f"Process batch forecasts: {min(dates)} to {max(dates)}")

        ref_alias = batch[0].get("ref_alias") or ref_alias

        pred_connection_params = deep_copy_object(
            pred_connection_params, skip_list=['dataset_processor', 'fs']
        )
        pred_connection_params = clean_for_serialization(pred_connection_params)
        pred_connection_params = self.clean_namespace(pred_connection_params)

        if hasattr(pred_transform, 'dataset_processor'):
            delattr(pred_transform, 'dataset_processor')
        if hasattr(ref_transform, 'dataset_processor'):
            delattr(ref_transform, 'dataset_processor')

        ref_connection_params = deep_copy_object(
            ref_connection_params, skip_list=['dataset_processor', 'fs']
        )
        ref_connection_params = clean_for_serialization(ref_connection_params)
        ref_connection_params = self.clean_namespace(ref_connection_params)

        # argo_index is now passed as a Future (already scattered) or None.
        # No need to scatter it again per batch.
        scattered_argo_index = argo_index

        try:
            for entry in batch:
                delayed_tasks.append(dask.delayed(compute_metric)(
                    entry,
                    pred_connection_params,
                    ref_connection_params,
                    pred_alias,
                    self.metrics[ref_alias],
                    pred_transform=pred_transform,
                    ref_transform=ref_transform,
                    argo_index=scattered_argo_index,
                    reduce_precision=self.reduce_precision,
                ))

            # Optimization: execute tasks manually to release memory aggressively
            # calling dataset_processor.compute_delayed_tasks(delayed_tasks) waits for all results

            # Retry policy for KilledWorker situations
            # max_retries = 3
            batch_results: List[Any] = []

            # We process the batch by chunks to avoid overwhelming the scheduler/workers
            # if batch is large. But 'batch' here is already a subdivision from dataloader.

            batch_t0 = time.time()
            futures = self.dataset_processor.client.compute(delayed_tasks, retries=1)

            # CRITICAL: Save the count before deleting to avoid UnboundLocalError
            num_tasks = len(delayed_tasks)

            # CRITICAL: Clear delayed_tasks immediately to avoid keeping references
            # to intermediate objects (closures, captured variables, etc.)
            del delayed_tasks
            gc.collect()

            # Use as_completed to process results as they arrive
            # Note: as_completed yields futures that are done
            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Processing batch metrics"
            ):
                try:
                    result = future.result()
                    batch_results.append(result)
                    del result # Free memory immediately
                except Exception as e:
                    # Log specific error for the failed future
                    logger.error(f"Task failed in batch: {e}")
                    # Don't fail the whole batch immediately?
                    # If a task fails (e.g. KilledWorker), Dask already
                    # retried it per 'retries=1'
                    # If it still fails, we might just append None or raise
                    # Currently, let's append None to survive partial failures
                    batch_results.append(None)

            # CRITICAL: wait() ensures all transitions are complete
            # This prevents intermediate states from lingering in memory
            wait(futures)

            # CRITICAL: cancel(force=True) MUST be called BEFORE deleting references
            # This tells the scheduler these results are not needed anymore
            # Without this, Dask keeps data in memory even if Python doesn't reference it
            self.dataset_processor.client.cancel(futures, force=True)

            # Now it's safe to drop client-side references
            del futures

            # Force garbage collection on driver
            gc.collect()

            # Aggressive cleanup on workers: xarray file cache + gc + malloc_trim
            try:
                self.dataset_processor.client.run(_worker_full_cleanup)
            except Exception as e:
                logger.warning(f"Worker-side cleanup failed (non-fatal): {e}")

            batch_duration = time.time() - batch_t0

            # Analyze task timings
            times = [r.get('duration_s', 0) for r in batch_results if r and isinstance(r, dict)]
            points = [r.get('n_points', 0) for r in batch_results if r and isinstance(r, dict)]

            if times:
                min_t, max_t, avg_t = min(times), max(times), sum(times)/len(times)
                total_pts = sum(points)
                logger.info(
                    f"Batch processed in {batch_duration:.2f}s "
                    f"({len(batch_results)}/{num_tasks} tasks)"
                )
                logger.info(
                    f"Batch Stats: Tasks={len(times)} | "
                    f"Time: Min={min_t:.1f}s Max={max_t:.1f}s Avg={avg_t:.1f}s | "
                    f"Total Points={total_pts}"
                )

                # Check for stragglers or degradation
                n_workers = len(
                    self.dataset_processor.client.scheduler_info().get('workers', {})
                )
                logger.info(f"Cluster State at Batch End: {n_workers} active workers")

                if batch_duration > (sum(times) / max(1, n_workers)) * 2:
                    logger.warning(
                        "Batch duration significantly exceeds theoretical parallel time. "
                        "Potential resource contention or worker death."
                    )
            else:
                logger.info(f"Batch processed in {batch_duration:.2f}s (No valid results)")

            return batch_results
        except Exception as exc:
            logger.error(f"Error processing batch: {repr(exc)}")
            traceback.print_exc()
            return []
