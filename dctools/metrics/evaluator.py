"""Metrics evaluator module for distributed evaluation.

The ``Evaluator`` class orchestrates distributed metric computation over
Dask clusters.  Helper functions that run inside workers have been moved
to dedicated sub-modules:

* :mod:`dctools.metrics.worker_cleanup` — memory, thread and cache management
* :mod:`dctools.metrics.worker_compute` — dataset caching, timeout wrapper
* :mod:`dctools.metrics.compute_task`   — the ``compute_metric`` worker task

All previously-public symbols are re-exported here for backward compatibility.
"""

import gc
import json
import os
import time
import traceback
from argparse import Namespace
from typing import Any, Callable, Dict, List, Optional

import dask
import numpy as np
import pandas as pd
import xarray as xr
from dask.distributed import as_completed, wait
from loguru import logger
from oceanbench.core.distributed import DatasetProcessor
from tqdm import tqdm

from dctools.data.connection.connection_manager import clean_for_serialization
from dctools.data.datasets.dataloader import EvaluationDataloader, filter_by_time
from dctools.data.datasets.dataset_manager import MultiSourceDatasetManager
from dctools.metrics.compute_task import compute_metric
from dctools.metrics.metrics import MetricComputer
from dctools.metrics.worker_cleanup import _worker_full_cleanup
from dctools.metrics.worker_compute import _parse_memory_limit
from dctools.utilities.parallelism import ParallelismConfig
from dctools.utilities.misc_utils import deep_copy_object, serialize_structure

# ---------------------------------------------------------------------------
# Backward-compatible re-exports
# (so that ``from dctools.metrics.evaluator import X`` keeps working)
# ---------------------------------------------------------------------------
from dctools.metrics.worker_cleanup import (  # noqa: F401
    _WORKER_DATASET_CACHE,
    _WORKER_DATASET_CACHE_LOCK,
    _cap_worker_threads,
    _clear_xarray_file_cache,
    worker_memory_cleanup,
)
from dctools.metrics.worker_compute import (  # noqa: F401
    _compute_with_timeout,
    _open_dataset_worker_cached,
)
from dctools.metrics.compute_task import compute_metric as compute_metric  # noqa: F401


class Evaluator:
    """Class to evaluate metrics on datasets."""

    def __init__(
        self,
        dataset_manager: MultiSourceDatasetManager,
        metrics: Dict[str, List[MetricComputer]],
        dataloader: EvaluationDataloader,
        ref_aliases: List[str],
        dataset_processor: DatasetProcessor,
        dask_cfgs_by_dataset: Optional[Dict[str, Dict[str, Any]]] = None,
        results_dir: Optional[str] = None,
        parallelism: Optional[ParallelismConfig] = None,
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
            dask_cfgs_by_dataset (Dict[str, Dict[str, Any]], optional):
                Per-dataset Dask configuration (n_workers, threads_per_worker,
                memory_limit) extracted from the YAML config sources.
                Defaults to None.
            results_dir (str, optional): Folder to save results. Defaults to None.
            parallelism (ParallelismConfig, optional): Centralised parallelism
                and resource config.  When *None*, uses defaults.
        """
        self.dataset_manager = dataset_manager
        self.dataset_processor = dataset_processor
        self.metrics = metrics
        self.dataloader = dataloader
        self.dask_cfgs_by_dataset = dask_cfgs_by_dataset or {}
        self.pcfg = parallelism or ParallelismConfig()
        self.ref_aliases = ref_aliases
        self.results_dir = results_dir
        # Track the current cluster sizing so we know when to reconfigure.
        self._current_cluster_ref: Optional[str] = None

        (
            self.ref_managers,
            self.ref_catalogs,
            self.ref_connection_params,
        ) = dataset_manager.get_config()

    # ------------------------------------------------------------------
    # Cluster reconfiguration when switching observation datasets
    # ------------------------------------------------------------------
    def _reconfigure_cluster_for_ref(self, ref_alias: str) -> None:
        """Resize the Dask cluster if *ref_alias* needs a different config.

        Each observation dataset can declare its own
        ``n_parallel_workers / nthreads_per_worker / memory_limit_per_worker``
        in the YAML config.  When the evaluator switches from one obs
        dataset to another, this method tears down the current
        ``DatasetProcessor`` and spins up a new one matching the target
        config.  If no per-dataset config exists for *ref_alias*, or the
        existing cluster already matches, this is a no-op.
        """
        if self._current_cluster_ref == ref_alias:
            return  # already configured for this dataset

        desired = self.dask_cfgs_by_dataset.get(ref_alias)
        if not desired:
            # No per-dataset override -> keep current cluster as-is.
            self._current_cluster_ref = ref_alias
            return

        # For observation datasets (SWOT, saral, argo, …), force
        # threads_per_worker=1 to prevent CPU oversubscription from
        # C-level libraries (pyinterp, BLAS, OpenMP) that release the
        # GIL and spawn their own threads.  This is the critical fix
        # for the 0% progress stall on SWOT swath data.
        from dctools.data.datasets.dataset import is_observation_alias
        if is_observation_alias(str(ref_alias)):
            desired = self.pcfg._adapt_obs_dask_cfg(desired)
            logger.debug(
                f"Observation dataset '{ref_alias}': forced "
                f"threads_per_worker={desired.get('threads_per_worker', '?')}, "
                f"n_workers={desired.get('n_workers', '?')}"
            )

        # Read desired sizing.
        d_workers = int(desired.get("n_workers", 1))
        d_threads = int(desired.get("threads_per_worker", 1))
        d_memory = desired.get("memory_limit", "4GB")

        # Read current cluster sizing for comparison.
        _client = getattr(self.dataset_processor, "client", None)
        if _client is not None:
            try:
                _info = _client.scheduler_info()
                _ws = _info.get("workers", {})
                _cur_n = len(_ws)
                # threads & memory from first worker
                _any_w: dict = next(iter(_ws.values()), {})
                _cur_threads = _any_w.get("nthreads", 1)
                _cur_mem = _any_w.get("memory_limit", 0)
                # Parse desired memory into bytes for comparison.
                _d_mem_bytes = _parse_memory_limit(d_memory)
                if (
                    _cur_n == d_workers
                    and _cur_threads == d_threads
                    and _cur_mem == _d_mem_bytes
                ):
                    # Already matches -> nothing to do.
                    self._current_cluster_ref = ref_alias
                    return
            except Exception:
                pass  # cannot query -> proceed with reconfiguration

        logger.info(
            f"Reconfiguring Dask cluster for '{ref_alias}': "
            f"Workers={d_workers}, Threads={d_threads}, MemLimit={d_memory}"
        )

        # Tear down existing cluster.
        # Silence distributed.* loggers during teardown: workers that still
        # have an in-flight heartbeat will log a CommClosedError when the
        # scheduler stream is closed.  This is expected and non-fatal.
        import logging as _logging
        import time as _time

        class _SuppressHeartbeatNoise(_logging.Filter):
            _NOISE = ("CommClosedError", "heartbeat", "Stream is closed", "comm closed")
            def filter(self, record):
                if record.levelno >= _logging.CRITICAL:
                    return True
                if record.name.startswith(("distributed", "tornado")):
                    msg = record.getMessage()
                    if any(kw.lower() in msg.lower() for kw in self._NOISE):
                        return False
                return True

        _suppress_filter = _SuppressHeartbeatNoise()
        _noisy_loggers = [
            _logging.getLogger(n) for n in (
                "distributed", "distributed.worker", "distributed.comm",
                "distributed.comm.tcp", "distributed.core",
                "tornado", "tornado.application",
            )
        ]
        _saved_levels = [(lg, lg.level) for lg in _noisy_loggers]
        for _lg in _noisy_loggers:
            _lg.setLevel(_logging.CRITICAL)
            _lg.addFilter(_suppress_filter)
        _root_logger = _logging.getLogger()
        _root_logger.addFilter(_suppress_filter)
        try:
            self.dataset_processor.close()
            # Give in-flight heartbeat RPCs time to fail silently.
            _time.sleep(2.0)
        except Exception:
            pass
        finally:
            for _lg, _lvl in _saved_levels:
                _lg.setLevel(_lvl)
                _lg.removeFilter(_suppress_filter)
            _root_logger.removeFilter(_suppress_filter)

        # Create a fresh DatasetProcessor.
        _proc_kwargs: Dict[str, Any] = {
            "distributed": True,
            "n_workers": d_workers,
            "threads_per_worker": d_threads,
            "memory_limit": d_memory,
        }
        import tempfile as _tempfile
        import os as _os
        _old_tempdir = _tempfile.tempdir
        if self.pcfg.dask_tmp_dir:
            _os.makedirs(self.pcfg.dask_tmp_dir, exist_ok=True)
            _tempfile.tempdir = self.pcfg.dask_tmp_dir
        try:
            self.dataset_processor = DatasetProcessor(**_proc_kwargs)
        finally:
            _tempfile.tempdir = _old_tempdir

        # Propagate HDF5/NetCDF env vars to new workers.
        from dctools.utilities.init_dask import configure_dask_workers_env
        try:
            configure_dask_workers_env(
                self.dataset_processor.client,
                self.pcfg,
            )
        except Exception:
            pass

        self._current_cluster_ref = ref_alias
        # Reset baseline memory after cluster rebuild.
        self.baseline_memory = None
        self._first_batch_for_workers = True
        logger.debug(
            f"Dask cluster reconfigured for '{ref_alias}': "
            f"dashboard={getattr(self.dataset_processor.client, 'dashboard_link', 'N/A')}"
        )

    def log_cluster_memory_usage(self, batch_idx: int):
        """Log memory usage of each Dask worker."""
        if not hasattr(self.dataset_processor, "client") or self.dataset_processor.client is None:
            return

        try:
            info = self.dataset_processor.client.scheduler_info()
            workers = info.get('workers', {})

            logger.debug(f"=== Memory Usage Start Batch {batch_idx} ===")
            for w_addr, w_info in workers.items():
                # Some versions of dask put 'metrics' in the info
                mem_used = w_info.get('metrics', {}).get('memory', w_info.get('memory', 0))
                mem_limit = w_info.get('memory_limit', 0)

                if mem_limit > 0:
                    percent = (mem_used / mem_limit) * 100
                    logger.debug(
                        f"Worker {w_info.get('name', w_addr)}: "
                        f"{percent:.1f}% ({mem_used / 1024**3:.2f}GB / {mem_limit / 1024**3:.2f}GB)"
                    )
                else:
                    logger.debug(
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

    def get_max_worker_rss(self) -> float:
        """Get the maximum RSS (Resident Set Size) across all workers (bytes).

        Unlike ``get_max_memory_usage`` which only reports Dask-managed
        memory, this measures the actual process RSS — capturing unmanaged
        memory from C libraries (HDF5, Blosc, PyTorch), heap fragmentation,
        and cached module imports that ``gc.collect()`` cannot reclaim.
        """
        if not hasattr(self.dataset_processor, "client") or self.dataset_processor.client is None:
            return 0.0

        def _worker_rss():
            import os as _os_rss
            try:
                # Linux: read /proc/self/statm (faster than psutil)
                with open("/proc/self/statm", "r") as _f:
                    _pages = int(_f.read().split()[1])  # resident pages
                return _pages * _os_rss.sysconf("SC_PAGE_SIZE")
            except Exception:
                pass
            try:
                import psutil
                return psutil.Process().memory_info().rss
            except Exception:
                return 0

        try:
            rss_map = self.dataset_processor.client.run(_worker_rss)
            return float(max(rss_map.values())) if rss_map else 0.0
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
        # First batch on the current set of workers: the per-batch RSS
        # increase check is skipped because library imports and cache
        # warm-up cause a one-time growth that is not a memory leak.
        self._first_batch_for_workers = True

        # -- ARGO pre-fetch cache dir --------------------------------------
        # Zarr files created by prefetch_batch_shared_zarr persist across
        # batches (a window fetched for batch N is reused by batch N+1 if
        # the same time window appears again).  The cache is stored under
        # data_directory (parent of results_dir) so it persists across runs
        # and avoids re-downloading the same ARGO months every evaluation.
        from pathlib import Path as _PfPath
        _results_path = _PfPath(self.results_dir) if self.results_dir else _PfPath("/tmp")
        # results_dir == data_directory/results_batches  -->  parent == data_directory
        _data_dir = _results_path.parent if _results_path.name == "results_batches" else _results_path  # noqa: E501
        self._argo_zarr_cache_dir = str(_data_dir / "argo_batch_cache")

        # -- Purge stale obs batch zarr from any previous run -------------
        # The shared obs zarr (obs_batch_shared/{alias}/batch_shared.zarr) is
        # written to a fixed path that persists across runs.  If a previous
        # run was aborted after only 1-2 batches, the zarr covers only a
        # fraction of the year.  Subsequent runs would silently reuse it,
        # causing all tasks outside that time range to return 0 pts with no
        # error.  Deleting the directory at the start of each run forces a
        # clean rebuild from the full set of observation files.
        _obs_shared_root = _results_path / "obs_batch_shared"
        if _obs_shared_root.exists():
            import shutil as _shutil_obs
            try:
                _shutil_obs.rmtree(_obs_shared_root, ignore_errors=True)
                logger.debug(f"Purged stale obs_batch_shared cache: {_obs_shared_root}")
            except Exception:
                pass

        try:
            # -- Pre-materialise batches to know total count -----------
            # Batches are lightweight metadata dicts (no actual data),
            # so materialising upfront is cheap and lets us display
            # clear "Batch X/N" progress throughout the run.
            self._lookahead_cache = getattr(self, '_lookahead_cache', {})

            _all_batches: List[List[Dict[str, Any]]] = list(self.dataloader)
            _total_batches = len(_all_batches)
            logger.debug(
                f"Evaluation plan: {_total_batches} batch(es), "
                f"{sum(len(b) for b in _all_batches)} total tasks"
            )

            _prev_ref_alias: Optional[str] = None
            _ref_aliases_ordered: List[str] = list(
                dict.fromkeys(
                    b[0].get("ref_alias") for b in _all_batches if b and b[0].get("ref_alias")  # type: ignore[misc]
                )
            )

            for batch_idx, batch in enumerate(_all_batches):
                _next_raw = _all_batches[batch_idx + 1] if batch_idx + 1 < _total_batches else None

                pred_alias = self.dataloader.pred_alias
                ref_alias = batch[0].get("ref_alias")

                # Print the reference banner the first time a new reference is encountered.
                if ref_alias != _prev_ref_alias:
                    _n_ref_total = len(_ref_aliases_ordered)
                    _n_ref_current = (
                        _ref_aliases_ordered.index(ref_alias) + 1
                        if ref_alias in _ref_aliases_ordered
                        else "?"
                    )
                    _c = "\033[1;96m"   # bright cyan, bold
                    _y = "\033[1;93m"   # bright yellow, bold
                    _r = "\033[0m"      # reset
                    _sep_ref = "─" * 60
                    print(f"\n    {_c}┌{_sep_ref}┐{_r}")
                    print(
                        f"    {_c}│{_r}  {_y}📡 Reference dataset ({_n_ref_current}/{_n_ref_total}){_r} ›  {_c}{str(ref_alias).upper():<27}{_c}│{_r}"  # noqa: E501
                    )
                    print(f"    {_c}└{_sep_ref}┘{_r}\n")
                    _prev_ref_alias = ref_alias

                # -- Reconfigure cluster if this ref dataset needs
                #    different sizing (workers / threads / memory) --
                self._reconfigure_cluster_for_ref(ref_alias)  # type: ignore[arg-type]

                # Extract necessary information
                pred_connection_params = self.dataloader.pred_connection_params
                ref_connection_params = self.dataloader.ref_connection_params[ref_alias]  # type: ignore[index]
                pred_transform = self.dataloader.pred_transform
                if self.dataloader.ref_transforms is not None:
                    ref_transform = self.dataloader.ref_transforms[ref_alias]  # type: ignore[index]

                argo_index = None
                if hasattr(self.dataloader.ref_managers[ref_alias], 'argo_index'):  # type: ignore[index]
                    argo_index = self.dataloader.ref_managers[ref_alias].get_argo_index()  # type: ignore[index]

                # Build look-ahead context for the NEXT batch (if any).
                # _evaluate_batch will launch the background download during
                # its as_completed loop (workers busy -> driver has spare CPU).
                _la_next = None
                if _next_raw is not None:
                    _la_next = {
                        'batch': _next_raw,
                        'ref_alias': _next_raw[0].get("ref_alias") if _next_raw else None,
                    }

                batch_results = self._evaluate_batch(
                    batch, pred_alias, ref_alias,  # type: ignore[arg-type]
                    pred_connection_params, ref_connection_params,
                    pred_transform, ref_transform,
                    argo_index=argo_index,
                    _lookahead_next=_la_next,
                    _batch_idx=batch_idx,
                    _total_batches=_total_batches,
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

                # CRITICAL: Explicit cleanup
                del batch_results
                del serial_results
                gc.collect()

                # -- Cleanup previous batch shared obs zarr ----------------
                # Each batch writes its own shared zarr under
                # obs_batch_shared/{alias}/batch_{idx}/.  Once the batch is
                # saved we no longer need it, and leaving it around wastes
                # disk space (multi-GB per dataset per batch).
                if batch_idx >= 0:
                    from pathlib import Path as _PfPathClean
                    _obs_shared_root = _PfPathClean(
                        self.results_dir or "."
                    ) / "obs_batch_shared"
                    _batch_shared_dir = _obs_shared_root / str(ref_alias) / f"batch_{batch_idx}"
                    if _batch_shared_dir.exists():
                        try:
                            import shutil as _shutil_batch
                            _shutil_batch.rmtree(_batch_shared_dir, ignore_errors=True)
                        except Exception:
                            pass

                # -- Inter-batch worker restart: condition-based -----------
                # Restart is gated on restart_workers_per_batch=True.
                # A restart is triggered when the absolute Dask managed
                # memory fraction exceeds the configured threshold.
                if self.pcfg.restart_workers_per_batch:
                    _should_restart = False
                    _restart_reasons: List[str] = []
                    # Absolute memory fraction (Dask managed memory)
                    try:
                        _mem_frac = self.get_max_memory_fraction()
                        if _mem_frac >= self.pcfg.max_memory_fraction:
                            _should_restart = True
                            _restart_reasons.append(
                                f"memory fraction {_mem_frac:.0%} ≥ "
                                f"{self.pcfg.max_memory_fraction:.0%}"
                            )
                    except Exception:
                        pass
                if self.pcfg.restart_workers_per_batch and _should_restart:
                    try:
                        _restart_client = self.dataset_processor.client
                        logger.info(
                            "Restarting Dask workers to reclaim unmanaged "
                            "memory between batches "
                            f"(reason: {'; '.join(_restart_reasons)})"
                        )
                        # Suppress CommClosedError / heartbeat noise during
                        # restart().  Workers with an in-flight heartbeat will
                        # log an ERROR when the scheduler stream closes — this
                        # is expected and non-fatal.
                        #
                        # Strategy: (1) raise the level to CRITICAL on every
                        # relevant stdlib logger so that ERROR records are
                        # dropped early; (2) also attach a keyword-based filter
                        # to the root logger as a backstop for records that
                        # propagate through loggers we didn't enumerate (e.g.
                        # distributed.comm.tcp in newer Dask releases).
                        import logging as _logging
                        import time as _time

                        class _SuppressHeartbeatNoise(_logging.Filter):
                            _NOISE = (
                                "CommClosedError",
                                "heartbeat",
                                "Stream is closed",
                                "comm closed",
                            )
                            def filter(self, record):  # noqa: D401
                                if record.levelno >= _logging.CRITICAL:
                                    return True
                                if record.name.startswith("distributed") or \
                                        record.name.startswith("tornado"):
                                    msg = record.getMessage()
                                    if any(kw.lower() in msg.lower()
                                           for kw in self._NOISE):
                                        return False
                                return True

                        _suppress_filter = _SuppressHeartbeatNoise()
                        _noisy_loggers = [
                            _logging.getLogger("distributed"),
                            _logging.getLogger("distributed.worker"),
                            _logging.getLogger("distributed.comm"),
                            _logging.getLogger("distributed.comm.tcp"),
                            _logging.getLogger("distributed.core"),
                            _logging.getLogger("tornado"),
                            _logging.getLogger("tornado.application"),
                        ]
                        _saved_levels = [(lg, lg.level) for lg in _noisy_loggers]
                        for _lg in _noisy_loggers:
                            _lg.setLevel(_logging.CRITICAL)
                            _lg.addFilter(_suppress_filter)
                        _root_logger = _logging.getLogger()
                        _root_logger.addFilter(_suppress_filter)
                        try:
                            _restart_client.restart()
                            # Let in-flight heartbeat RPCs drain silently.
                            _time.sleep(2.0)
                        finally:
                            for _lg, _lvl in _saved_levels:
                                _lg.setLevel(_lvl)
                                _lg.removeFilter(_suppress_filter)
                            _root_logger.removeFilter(_suppress_filter)
                        # Re-init env vars + cleanup on fresh workers.
                        _restart_client.run(_worker_full_cleanup)
                        from dctools.utilities.init_dask import (
                            configure_dask_workers_env,
                        )
                        try:
                            configure_dask_workers_env(
                                _restart_client, self.pcfg
                            )
                        except Exception:
                            pass
                        # Next batch is the first on fresh workers →
                        # skip the per-batch RSS increase check.
                        self._first_batch_for_workers = True
                    except Exception as _exc_restart:
                        logger.warning(
                            f"Inter-batch worker restart failed: "
                            f"{_exc_restart!r}"
                        )
                gc.collect()

            # Cleanup scattered data
            self.scattered_argo_indexes.clear()
            self.scattered_ref_catalogs.clear()

        except Exception as exc:
            logger.error(f"Evaluation failed: {repr(exc)}")
            raise

        finally:
            # -- ARGO pre-fetch Zarr cache: kept on disk for reuse ---------
            # The cache lives under data_directory/argo_batch_cache and is
            # intentionally NOT deleted so subsequent runs can reuse already-
            # downloaded months.  Each month's zarr file has a deterministic
            # name (argo_full_month_YYYY-MM.zarr) so there is no staleness risk.
            _cache = getattr(self, "_argo_zarr_cache_dir", None)
            if _cache:
                logger.debug(f"ARGO Zarr cache preserved for future runs: {_cache}")

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
        _lookahead_next: Optional[Dict[str, Any]] = None,
        _batch_idx: int = 0,
        _total_batches: int = 1,
    ) -> List[Dict[str, Any]]:
        _phase_t0 = time.time()
        if batch:
            raw_dates = [e.get("forecast_reference_time") for e in batch]
            dates = [d for d in raw_dates if d is not None]
            if dates:
                logger.debug(f"Process batch forecasts: {min(dates)} to {max(dates)}")

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
        # Track scattered Futures for cleanup at end of batch.
        _scattered_futs: list = []

        # Scatter argo_index once for the whole batch so it is not
        # re-serialized inside the partial for every task.
        if argo_index is not None:
            try:
                _client = self.dataset_processor.client
                scattered_argo_index = _client.scatter(
                    argo_index, broadcast=True
                )
                _scattered_futs.append(scattered_argo_index)
            except Exception:
                scattered_argo_index = argo_index

        metric_list = self.metrics.get(ref_alias)
        if not metric_list:
            err = (
                f"No metric configuration found for reference alias '{ref_alias}' "
                f"(available: {list(self.metrics.keys())})."
            )
            logger.error(err)
            return [{
                "forecast_reference_time": batch[0].get("forecast_reference_time") if batch else None,  # noqa: E501
                "model": pred_alias,
                "reference": ref_alias,
                "result": None,
                "n_points": 0,
                "duration_s": 0.0,
                "error": err,
            }]

        try:
            # Use map_tasks for direct task submission (no delayed graph overhead)
            from functools import partial

            # Retrieve look-ahead data downloaded during the previous batch.
            _la_data = getattr(self, '_lookahead_cache', {}).pop(id(batch), None)

            fn = partial(
                compute_metric,
                pred_source_config=pred_connection_params,
                ref_source_config=ref_connection_params,
                model=pred_alias,
                list_metrics=metric_list,
                pred_transform=pred_transform,
                ref_transform=ref_transform,
                argo_index=scattered_argo_index,
                reduce_precision=self.pcfg.reduce_precision,
                results_dir=self.results_dir,
            )
            fn.__name__ = "compute_metric"  # type: ignore[attr-defined]  # prevent full repr in tqdm progress bar

            batch_t0 = time.time()
            num_tasks = len(batch)

            # -- Throttle observation batches to prevent CPU oversubscription --
            # Observation datasets (satellite) trigger heavy
            # CPU-bound interpolation (pyinterp) on each worker.  Submitting
            # all tasks at once lets Dask schedule them across all workers
            # simultaneously, each of which may spawn internal C++ threads
            # -> total thread count far exceeds physical cores -> 100 % CPU
            # thrashing.  We split large observation batches into smaller
            # sub-batches so that at most *max_concurrent_obs* tasks run in
            # parallel, leaving headroom for the OS and driver.
            is_obs_batch = batch and batch[0].get("ref_is_observation", False)
            # All tasks run with _cap_worker_threads(1) so internal C++
            # libraries (pyinterp, BLAS, Blosc) create only 1 thread each.
            # No need to throttle concurrency below n_workers.
            # max_concurrent_obs = num_tasks

            # with threads_per_worker set to 1, C libraries are capped to 1.
            # Dask limits concurrency to n_workers × threads_per_worker.
            # However, submitting all tasks upfront can overwhelm the scheduler
            # task queue or cause memory fragmentation if tasks are large.
            # Limit concurrent tasks to avoid scheduler overhead/pauses.
            _client = self.dataset_processor.client
            _ncores = _client.ncores()  # {worker_addr: nthreads}
            _N = len(_ncores)
            _total_slots = int(sum(_ncores.values())) if _ncores else _N
            # Cap in-flight futures to the actual execution capacity.
            # This keeps a small queue (Dask will queue the rest anyway),
            # avoids scheduler overload, and keeps behaviour consistent
            # with dc2.yaml (n_parallel_workers × nthreads_per_worker).
            # For observation batches without a shared Zarr (i.e. when the
            # driver-side preprocessing was skipped because the file count
            # exceeded the limit), each task independently processes all
            # matching files on its worker — extremely memory-intensive.
            # In that case, limit concurrency to n_workers (not total_slots)
            # so that at most one heavy task runs per physical worker,
            # preventing concurrent threads from doubling memory pressure.
            # R9: For observation batches, cap concurrency to the number of
            # available workers, not total slots (threads_per_worker is forced
            # to 1 anyway). This prevents memory exhaustion when multiple
            # heavy data loading tasks run on the same physical worker.
            if is_obs_batch:
                max_concurrent_tasks = max(_N, 1)
                logger.debug(
                    f"Observation batch: limiting concurrency to {max_concurrent_tasks} "
                    f"tasks (1 per worker) to manage memory pressure."
                )
            else:
                max_concurrent_tasks = max(_total_slots, 4)

            # -- Single clean progress bar ---------------------------------
            # One overall bar on the driver. No per-worker bars, no
            # worker-side tqdm, no monkey-patched metrics bar.  Each
            # completed task prints a one-line summary via tqdm.write()
            # which is designed to coexist with the progress bar.
            import sys as _sys_bars

            # Clean duplicate definition
            # _client = self.dataset_processor.client
            # _N = len(_client.ncores())

            # -- ARGO pipeline: shared batch Zarr prefetch -------------
            # Instead of per-window download (N separate HTTP sessions,
            # heavy profile overlap, N separate Zarr writes), we now:
            #   1. Merge ALL time windows into one global bounding interval
            #   2. Download ALL profiles in a single pass (one HTTP session
            #      with connection pooling -> connection reuse)
            #   3. Write ONE shared time-sorted Zarr for the entire batch
            #   4. Workers read the shared Zarr + filter by their specific
            #      time_bounds via searchsorted (contiguous chunk reads)
            #
            # Typical savings for a 10-entry batch with time_tolerance=12h:
            #   Downloads:   10 × ~1 day -> 1 × ~11 days (overlap removed)
            #   Zarr writes: 10 -> 1
            #   Worker I/O:  each reads only its slice (searchsorted)
            _shared_argo_zarr: Optional[str] = None
            _argo_pipeline = (
                is_obs_batch
                and ref_alias == "argo_profiles"
                and hasattr(
                    self.dataloader.ref_managers.get(ref_alias, None),
                    "prefetch_batch_shared_zarr",
                )
            )

            if _argo_pipeline:
                # --- Collect all time windows from the batch ---------------
                _all_time_bounds: List[tuple] = []
                for _entry in batch:
                    _ref_d = _entry.get("ref_data")
                    if isinstance(_ref_d, dict) and "time_bounds" in _ref_d:
                        _tb = _ref_d["time_bounds"]
                        _all_time_bounds.append(
                            (pd.Timestamp(_tb[0]), pd.Timestamp(_tb[1]))
                        )

                if _all_time_bounds:
                    _mgr = self.dataloader.ref_managers.get(ref_alias)
                    if _mgr is not None:
                        from pathlib import Path as _PfPath
                        _cache_dir = _PfPath(
                            getattr(
                                self,
                                "_argo_zarr_cache_dir",
                                str(
                                    _PfPath(
                                        getattr(self, "results_dir", None)
                                        or "/tmp"
                                    ) / "argo_batch_cache"
                                ),
                            )
                        )
                        # Prefer partitioned monthly prefetch when available.
                        _partitions = None
                        if hasattr(_mgr, "prefetch_batch_shared_zarr_partitioned"):
                            try:
                                _partitions = _mgr.prefetch_batch_shared_zarr_partitioned(
                                    time_bounds_list=_all_time_bounds,
                                    cache_dir=_cache_dir,
                                )
                            except Exception as exc:
                                logger.warning(
                                    f"ARGO partitioned shared prefetch failed: {exc!r} — falling back"  # noqa: E501
                                )

                        if _partitions:
                            # Inject per-entry path(s). Each entry gets either
                            # a single path (same-month) or a list of paths
                            # (month boundary).
                            for _entry in batch:
                                _ref_d = _entry.get("ref_data")
                                if not isinstance(_ref_d, dict) or "time_bounds" not in _ref_d:
                                    continue
                                _tb = _ref_d["time_bounds"]
                                _t0_e = pd.Timestamp(_tb[0])
                                _t1_e = pd.Timestamp(_tb[1])
                                if _t1_e < _t0_e:
                                    _t0_e, _t1_e = _t1_e, _t0_e
                                _paths = []
                                for _p in _partitions:
                                    try:
                                        _p0 = pd.Timestamp(_p.get("t0"))
                                        _p1 = pd.Timestamp(_p.get("t1"))
                                        _pp = _p.get("zarr_path")
                                    except Exception:
                                        continue
                                    if not _pp:
                                        continue
                                    if not (_p1 < _t0_e or _p0 > _t1_e):
                                        _paths.append((_p0, str(_pp)))
                                _paths_sorted = [p for _, p in sorted(_paths, key=lambda x: x[0])]
                                if len(_paths_sorted) == 1:
                                    _ref_d["prefetched_argo_shared_zarr"] = _paths_sorted[0]
                                elif len(_paths_sorted) > 1:
                                    _ref_d["prefetched_argo_shared_zarr"] = _paths_sorted
                        else:
                            _shared_argo_zarr = _mgr.prefetch_batch_shared_zarr(
                                time_bounds_list=_all_time_bounds,
                                cache_dir=_cache_dir,
                            )

                            # --- Inject shared Zarr path into every batch entry --------
                            if _shared_argo_zarr:
                                for _entry in batch:
                                    _ref_d = _entry.get("ref_data")
                                    if isinstance(_ref_d, dict):
                                        _ref_d["prefetched_argo_shared_zarr"] = _shared_argo_zarr

            # -- Observation data: download + shared Zarr preprocess pipeline --
            # For all observation datasets (SWOT, nadir, etc.), we now use a
            # unified pipeline that:
            #   1. Gathers all unique remote file paths for the batch.
            #   2. Downloads them to a local cache directory.
            #   3. Preprocesses them into a shared Zarr store (manifest-based).
            #   4. Injects the path to this shared store into each task.
            # This is now the default, non-optional path.
            _obs_path_map: Dict[str, str] = {}
            _shared_obs_manifest_dir: Optional[str] = None
            _t_obs_dl = 0.0

            if is_obs_batch and ref_alias != "argo_profiles":
                _obs_pipeline_t0 = time.time()
                _unique_remote: List[str] = []
                _obs_cache_dir: str = ""
                _obs_fs = None
                _ref_mgr = self.dataloader.ref_managers.get(ref_alias)
                _has_fs = (
                    _ref_mgr is not None
                    and hasattr(_ref_mgr, "params")
                    and hasattr(_ref_mgr.params, "fs")
                    and _ref_mgr.params.fs is not None
                )

                if _has_fs:
                    # Collect all unique remote paths across the batch
                    _all_remote_paths: List[str] = []
                    for _entry in batch:
                        _ref_d = _entry.get("ref_data")
                        if isinstance(_ref_d, dict) and "source" in _ref_d:
                            _cat = _ref_d["source"]
                            _tb = _ref_d.get("time_bounds")
                            if _tb is not None and hasattr(_cat, "get_dataframe"):
                                _cat_df = _cat.get_dataframe()
                                _filt = filter_by_time(
                                    _cat_df,
                                    pd.Timestamp(_tb[0]),
                                    pd.Timestamp(_tb[1]),
                                )
                                _paths = _filt["path"].tolist()
                                _all_remote_paths.extend(_paths)
                    _unique_remote = list(dict.fromkeys(_all_remote_paths))

                if _unique_remote:
                    from pathlib import Path as _PfPath
                    from dctools.data.datasets.batch_preprocessing import (
                        download_and_preprocess_obs_pipeline,
                    )
                    _obs_fs = _ref_mgr.params.fs
                    _obs_cache_dir = str(
                        _PfPath(os.path.abspath(self.results_dir or "/tmp"))
                        / "obs_prefetch_cache" / str(ref_alias)
                    )
                    # The shared zarr is written to a batch-specific directory
                    # to prevent conflicts and allow per-batch cleanup.
                    _batch_shared_dir = (
                        _PfPath(os.path.abspath(self.results_dir or "."))
                        / "obs_batch_shared"
                        / str(ref_alias)
                        / f"batch_{_batch_idx}"
                    )

                    logger.info(
                        f"Obs pipeline ({ref_alias}, batch {_batch_idx}): "
                        f"{len(_unique_remote)} files queued for "
                        f"concurrent download+preprocess"
                    )

                    _obs_coords = getattr(_ref_mgr.params, 'coordinates', None)
                    if _obs_coords is None:
                        _gm = getattr(_ref_mgr, '_global_metadata', None) or {}
                        _cs = _gm.get('coord_system')
                        _obs_coords = getattr(_cs, 'coordinates', None) or {"time": "time"}
                    if not isinstance(_obs_coords, dict):
                        _obs_coords = dict(_obs_coords)
                    _obs_n_pts_dim = getattr(_ref_mgr.params, 'n_points_dim', None)
                    if _obs_n_pts_dim is None:
                        _obs_n_pts_dim = _obs_coords.get("n_points", "n_points")

                    _obs_path_map, _shared_obs_manifest_dir = download_and_preprocess_obs_pipeline(
                        remote_paths=_unique_remote,
                        cache_dir=_obs_cache_dir,
                        fs=_obs_fs,
                        alias=ref_alias,
                        keep_vars=getattr(_ref_mgr.params, 'keep_variables', None),
                        coordinates=_obs_coords,
                        n_points_dim=_obs_n_pts_dim,
                        output_zarr_dir=str(_batch_shared_dir),
                        download_workers=self.pcfg.prefetch_obs_workers,
                        max_shared_obs_files=self.pcfg.max_shared_obs_files,
                        prep_workers=self.pcfg.prep_workers,
                        prep_use_processes=self.pcfg.prep_use_processes,
                        dask_client=self.dataset_processor.client,
                    )
                    _t_obs_dl = time.time() - _obs_pipeline_t0

                    # Inject path map and shared obs zarr into batch entries.
                    # Use the same keys as the late pipeline so that workers
                    # find data through _open_prefetched_obs_data().
                    for _entry in batch:
                        _ref_d = _entry.get("ref_data")
                        if isinstance(_ref_d, dict):
                            if _obs_path_map:
                                _ref_d["prefetched_local_paths"] = _obs_path_map
                            if _shared_obs_manifest_dir:
                                _ref_d["prefetched_obs_zarr_path"] = _shared_obs_manifest_dir
                    if not _shared_obs_manifest_dir:
                        logger.warning(
                            f"Obs pipeline ({ref_alias}, batch {_batch_idx}): "
                            f"Failed to create shared obs manifest. "
                            f"Workers will fall back to individual file processing."
                        )

            # -- Prediction data prefetch — define + launch in background --
            # Runs concurrently with obs preprocessing below (different
            # S3 endpoints -> no contention).  Joined before task dispatch.
            import threading as _pred_thr

            # -- Reference grid data prefetch (Wasabi/S3 Zarr) -------------
            # For non-observation gridded references stored as Zarr on S3/Wasabi
            # workers can deadlock/hang due
            # to many concurrent small S3 requests. We prefetch the required Zarr
            # stores to local disk on the driver and remap paths in the batch so
            # workers open local files only.
            _ref_prefetched = False
            _ref_result: Dict[str, str] = {}

            def _do_ref_prefetch():
                nonlocal _ref_prefetched
                if not batch:
                    return
                if is_obs_batch:
                    return

                if not self.pcfg.enable_ref_prefetch:
                    return

                _ref_protocol = getattr(ref_connection_params, "protocol", None)
                if _ref_protocol not in ("wasabi", "s3"):
                    return

                _sample_ref = batch[0].get("ref_data")
                _ref_is_remote = (
                    isinstance(_sample_ref, str)
                    and _sample_ref.endswith(".zarr")
                    and _sample_ref.startswith(("https://", "http://", "s3://"))
                )
                if not _ref_is_remote:
                    return

                _ref_mgr = self.dataloader.ref_managers.get(ref_alias)
                _ref_fs_params = getattr(_ref_mgr, "params", None) if _ref_mgr is not None else None
                _ref_s3fs = getattr(_ref_fs_params, "fs", None)
                if _ref_s3fs is None:
                    return

                from pathlib import Path as _PfRef
                import shutil as _sh_ref
                from concurrent.futures import ThreadPoolExecutor as _RefPool
                import threading as _ref_dl_threading

                _ref_cache_dir = str(
                    _PfRef(os.path.abspath(getattr(self, "results_dir", None) or "/tmp"))
                    / "ref_prefetch_cache"
                    / str(ref_alias)
                )
                os.makedirs(_ref_cache_dir, exist_ok=True)

                _ref_endpoint = getattr(_ref_fs_params, "endpoint_url", "") or ""
                _unique_ref_paths = list(
                    dict.fromkeys(
                        e["ref_data"]
                        for e in batch
                        if isinstance(e.get("ref_data"), str)
                        and str(e.get("ref_data")).endswith(".zarr")
                    )
                )
                if not _unique_ref_paths:
                    return

                _counters = {"dl": 0, "hit": 0}
                _ref_lock = _ref_dl_threading.Lock()

                def _dl_one_ref(_rp: str) -> None:
                    _fname = _PfRef(_rp).name
                    _local_zarr = os.path.join(_ref_cache_dir, _fname)
                    if os.path.isdir(_local_zarr) and os.listdir(_local_zarr):
                        with _ref_lock:
                            _ref_result[_rp] = _local_zarr
                            _counters["hit"] += 1
                        return
                    try:
                        # logger.debug(f"Prefetching reference: {_fname}")
                        _tid = _ref_dl_threading.current_thread().ident
                        _tmp_zarr = _local_zarr + f".downloading.{_tid}"
                        if os.path.isdir(_tmp_zarr):
                            _sh_ref.rmtree(_tmp_zarr, ignore_errors=True)

                        _s3_key = _rp
                        if _ref_endpoint and _s3_key.startswith(_ref_endpoint):
                            _s3_key = _s3_key[len(_ref_endpoint):].lstrip("/")
                        elif _s3_key.startswith("s3://"):
                            _s3_key = _s3_key[len("s3://"):]

                        _ref_s3fs.get(_s3_key, _tmp_zarr, recursive=True)

                        if os.path.isdir(_local_zarr):
                            _sh_ref.rmtree(_local_zarr, ignore_errors=True)
                        os.rename(_tmp_zarr, _local_zarr)
                        with _ref_lock:
                            _ref_result[_rp] = _local_zarr
                            _counters["dl"] += 1
                    except Exception as _exc_rf:
                        logger.warning(
                            f"Reference prefetch failed for {_fname}: {_exc_rf!r}"
                        )

                # Be conservative: reference stores can be large.
                _N_REF_DL = min(self.pcfg.prefetch_workers, len(_unique_ref_paths))
                with _RefPool(max_workers=_N_REF_DL) as _rp:
                    list(_rp.map(_dl_one_ref, _unique_ref_paths))

                if _ref_result:
                    logger.debug(
                        f"Reference prefetch ({ref_alias}): "
                        f"{_counters['dl']} downloaded, "
                        f"{_counters['hit']} cached "
                        f"({len(_unique_ref_paths)} unique files)"
                    )
                    _ref_prefetched = True

            _pred_prefetched = False
            _pred_result: Dict[str, str] = {}

            def _do_pred_prefetch():
                """Download prediction zarr stores in parallel (background thread)."""
                nonlocal _pred_prefetched
                if not batch:
                    return
                _sample_pred = batch[0].get("pred_data")
                _pred_is_remote = (
                    isinstance(_sample_pred, str)
                    and (
                        _sample_pred.startswith("https://")
                        or _sample_pred.startswith("http://")
                        or _sample_pred.startswith("s3://")
                    )
                )
                if not _pred_is_remote:
                    return

                from pathlib import Path as _PfPred
                import shutil as _sh_pred
                from concurrent.futures import ThreadPoolExecutor as _PredPool
                import threading as _pred_dl_threading

                _pred_cache_dir = str(
                    _PfPred(
                        os.path.abspath(getattr(self, "results_dir", None) or "/tmp")
                    )
                    / "pred_prefetch_cache"
                    / str(pred_alias)
                )
                os.makedirs(_pred_cache_dir, exist_ok=True)

                _pred_fs = getattr(
                    getattr(
                        self.dataloader, "pred_manager", None
                    ),
                    "params", None,
                )
                _pred_s3fs = getattr(_pred_fs, "fs", None)
                _pred_endpoint = getattr(
                    _pred_fs, "endpoint_url", ""
                ) or ""

                _unique_pred_paths = list(dict.fromkeys(
                    e["pred_data"]
                    for e in batch
                    if isinstance(e.get("pred_data"), str)
                ))

                _counters = {"dl": 0, "hit": 0}
                _pred_lock = _pred_dl_threading.Lock()

                def _dl_one_pred(_rp):
                    _fname = _PfPred(_rp).name
                    _local_zarr = os.path.join(
                        _pred_cache_dir, _fname
                    )
                    if os.path.isdir(_local_zarr) and os.listdir(
                        _local_zarr
                    ):
                        with _pred_lock:
                            _pred_result[_rp] = _local_zarr
                            _counters["hit"] += 1
                        return
                    try:
                        logger.debug(
                            f"Prefetching prediction: {_fname}"
                        )
                        _tid = _pred_dl_threading.current_thread().ident
                        _tmp_zarr = _local_zarr + f".downloading.{_tid}"
                        if os.path.isdir(_tmp_zarr):
                            _sh_pred.rmtree(
                                _tmp_zarr, ignore_errors=True
                            )

                        _s3_key = _rp
                        if _pred_endpoint and _s3_key.startswith(
                            _pred_endpoint
                        ):
                            _s3_key = _s3_key[
                                len(_pred_endpoint):
                            ].lstrip("/")
                        elif _s3_key.startswith("s3://"):
                            _s3_key = _s3_key[len("s3://"):]

                        if _pred_s3fs is not None:
                            _pred_s3fs.get(
                                _s3_key,
                                _tmp_zarr,
                                recursive=True,
                            )
                        else:
                            import xarray as _xr_prefetch
                            with dask.config.set(
                                scheduler="synchronous"
                            ):
                                _ds_pf = _xr_prefetch.open_zarr(
                                    _rp, chunks={}
                                )
                                _ds_pf = _ds_pf.compute()
                            _ds_pf.to_zarr(
                                _tmp_zarr, mode="w",
                                consolidated=True,
                            )
                            _ds_pf.close()
                            del _ds_pf

                        if os.path.isdir(_local_zarr):
                            _sh_pred.rmtree(
                                _local_zarr, ignore_errors=True
                            )
                        os.rename(_tmp_zarr, _local_zarr)
                        with _pred_lock:
                            _pred_result[_rp] = _local_zarr
                            _counters["dl"] += 1
                    except Exception as _exc_pf:
                        logger.warning(
                            f"Prediction prefetch failed for "
                            f"{_fname}: {_exc_pf!r}"
                        )

                _N_PRED_DL = min(self.pcfg.prefetch_workers, len(_unique_pred_paths))
                with _PredPool(max_workers=_N_PRED_DL) as _pp:
                    list(_pp.map(_dl_one_pred, _unique_pred_paths))

                if _pred_result:
                    logger.debug(
                        f"Prediction prefetch ({pred_alias}): "
                        f"{_counters['dl']} downloaded, "
                        f"{_counters['hit']} cached "
                        f"({len(_unique_pred_paths)} unique files)"
                    )
                    _pred_prefetched = True

            # -- Start prediction prefetch in background --------------
            _t_pred_dl = time.time()

            _t_ref_dl = 0.0
            _ref_thread = None
            _ref_protocol = getattr(ref_connection_params, "protocol", None)
            _sample_ref = batch[0].get("ref_data") if batch else None
            _need_ref_prefetch = (
                (not is_obs_batch)
                and self.pcfg.enable_ref_prefetch
                and (_ref_protocol in ("wasabi", "s3"))
                and isinstance(_sample_ref, str)
                and _sample_ref.endswith(".zarr")
                and _sample_ref.startswith(("https://", "http://", "s3://"))
            )
            if _need_ref_prefetch:
                _t_ref_dl = time.time()
                _ref_thread = _pred_thr.Thread(
                    target=_do_ref_prefetch, daemon=True, name="ref-prefetch"
                )
                _ref_thread.start()

            _pred_thread = _pred_thr.Thread(
                target=_do_pred_prefetch, daemon=True, name="pred-prefetch"
            )
            _pred_thread.start()

            # -- Download + preprocess pipeline for swath/track obs ----------
            # pred/ref background threads are now running (started above).
            # Launch the obs pipeline HERE so that all three operations proceed
            # simultaneously: obs download, obs preprocessing, pred/ref downloads.
            #
            # Cases:
            #   • Look-ahead cache hit (_obs_path_map non-empty): files are
            #     already local; skip download and preprocess directly.
            #   • Normal case (_unique_remote + _obs_fs): pipeline downloads
            #     files with _DL_W threads; each file is immediately submitted
            #     to the preprocessing pool (ProcessPool / ThreadPool);
            #     download and preprocessing run concurrently.
            _shared_obs_zarr: Optional[str] = None
            _t_obs_pipeline = time.time()
            # Run the concurrent obs pipeline only if the early pipeline block
            # (above, before pred/ref threads) did not already handle it.
            _obs_prefetch = (
                is_obs_batch
                and ref_alias != "argo_profiles"
                and not _obs_path_map
                and not _shared_obs_manifest_dir
            )
            if _obs_prefetch:
                _n_remote = len(_unique_remote) if _unique_remote else 0
                if _obs_path_map:
                    logger.info(
                        f"📦 Batch N°{_batch_idx+1}/{_total_batches} ▸ {ref_alias.upper()}: "
                        f"preprocessing {len(_obs_path_map)} cached obs files…"
                    )
                elif _n_remote > 0:
                    logger.info(
                        f"📦 Batch N°{_batch_idx+1}/{_total_batches} ▸ {ref_alias.upper()}: "
                        f"downloading + preprocessing {_n_remote} obs files…"
                    )
                _ref_d0 = batch[0].get("ref_data") if batch else None
                if isinstance(_ref_d0, dict):
                    try:
                        _md = _ref_d0.get("metadata", {})
                        _coord_sys = (
                            _md.get("coord_system")
                            if isinstance(_md, dict)
                            else getattr(_md, "coord_system", None)
                        )
                        _coords = (
                            getattr(_coord_sys, "coordinates", None)
                            if _coord_sys is not None
                            else None
                        )
                        if _coords is None:
                            _coords = {"time": "time"}

                        _kv = _ref_d0.get("keep_vars")
                        _n_pts_dim = "n_points"
                        _rc0 = batch[0].get("ref_coords")
                        if (
                            _rc0 is not None
                            and hasattr(_rc0, "coordinates")
                            and isinstance(
                                getattr(_rc0.coordinates, "get", None),
                                type(dict.get),
                            )
                        ):
                            _n_pts_dim = _rc0.coordinates.get(
                                "n_points", "n_points"
                            )

                        from pathlib import Path as _PfPath2
                        _shared_zarr_dir = str(
                            _PfPath2(
                                os.path.abspath(
                                    getattr(self, "results_dir", None)
                                    or "/tmp"
                                )
                            )
                            / "obs_batch_shared"
                            / str(ref_alias)
                            / f"batch_{_batch_idx}"
                        )
                        _coords_dict = (
                            dict(_coords)
                            if not isinstance(_coords, dict)
                            else _coords
                        )

                        if _obs_path_map:
                            # Look-ahead: files already local — inject paths
                            # and run preprocessing only (no download needed).
                            logger.debug("Obs: using look-ahead cache, preprocessing only")
                            for _entry in batch:
                                _ref_d = _entry.get("ref_data")
                                if isinstance(_ref_d, dict):
                                    _ref_d["prefetched_local_paths"] = _obs_path_map
                            _local_unique = list(dict.fromkeys(_obs_path_map.values()))
                            from dctools.data.datasets.dataloader import (
                                preprocess_batch_obs_files,
                            )
                            # When use_distributed_prep is enabled, pass the Dask client
                            # so preprocessing runs on the cluster instead of a local
                            # ProcessPoolExecutor (avoids RAM competition on the driver).
                            _prep_dask_client = (
                                self.dataset_processor.client
                                if self.pcfg.use_distributed_prep
                                else None
                            )
                            _shared_obs_zarr = preprocess_batch_obs_files(
                                local_paths=_local_unique,
                                alias=str(ref_alias),
                                keep_vars=_kv,
                                coordinates=_coords_dict,
                                n_points_dim=_n_pts_dim,
                                output_zarr_dir=_shared_zarr_dir,
                                max_shared_obs_files=self.pcfg.max_shared_obs_files,
                                prep_workers=self.pcfg.prep_workers,
                                prep_use_processes=self.pcfg.prep_use_processes,
                                dask_client=_prep_dask_client,
                            )
                            if _shared_obs_zarr:
                                for _entry in batch:
                                    _rd = _entry.get("ref_data")
                                    if isinstance(_rd, dict):
                                        _rd["prefetched_obs_zarr_path"] = _shared_obs_zarr

                        elif _unique_remote and _obs_fs is not None:
                            # Pipeline: download + preprocess simultaneously.
                            # pred/ref downloads are already running in background
                            # threads → all three sets of I/O overlap.
                            from dctools.data.datasets.batch_preprocessing import (
                                download_and_preprocess_obs_pipeline,
                            )
                            _t_pipeline_start = time.time()
                            _prep_dask_client2 = (
                                self.dataset_processor.client
                                if self.pcfg.use_distributed_prep
                                else None
                            )
                            _obs_path_map, _shared_obs_zarr = (
                                download_and_preprocess_obs_pipeline(
                                    remote_paths=_unique_remote,
                                    cache_dir=_obs_cache_dir,
                                    fs=_obs_fs,
                                    alias=str(ref_alias),
                                    keep_vars=_kv,
                                    coordinates=_coords_dict,
                                    n_points_dim=_n_pts_dim,
                                    output_zarr_dir=_shared_zarr_dir,
                                    download_workers=self.pcfg.prefetch_obs_workers,
                                    max_shared_obs_files=self.pcfg.max_shared_obs_files,
                                    prep_workers=self.pcfg.prep_workers,
                                    prep_use_processes=self.pcfg.prep_use_processes,
                                    dask_client=_prep_dask_client2,
                                )
                            )
                            _t_obs_dl = time.time() - _t_pipeline_start
                            # Inject path map and optional shared zarr into batch.
                            for _entry in batch:
                                _rd = _entry.get("ref_data")
                                if isinstance(_rd, dict):
                                    if _obs_path_map:
                                        _rd["prefetched_local_paths"] = _obs_path_map
                                    if _shared_obs_zarr:
                                        _rd["prefetched_obs_zarr_path"] = _shared_obs_zarr

                    except Exception as _exc_pipeline:
                        logger.warning(
                            f"Obs pipeline ({ref_alias}): {_exc_pipeline!r}"
                        )
                        import traceback as _tb_pipeline
                        _tb_pipeline.print_exc()

            _t_obs_pipeline = time.time() - _t_obs_pipeline
            _t_obs_prep = 0.0  # included in pipeline timing

            # -- Wait for prediction prefetch thread ------------------
            if _ref_thread is not None:
                _ref_thread.join()
                _t_ref_dl = time.time() - _t_ref_dl
            _pred_thread.join()
            _t_pred_dl = time.time() - _t_pred_dl

            _t_prefetch_total = time.time() - _phase_t0
            logger.debug(
                f"Prefetch done in {_t_prefetch_total:.1f}s "
                f"(obs_dl={_t_obs_dl:.1f}s  obs_prep={_t_obs_prep:.1f}s  "
                f"ref_dl={_t_ref_dl:.1f}s  pred_dl={_t_pred_dl:.1f}s) — dispatching {num_tasks} tasks"  # noqa: E501
            )

            # -- Cleanup workers after distributed preprocessing -----------
            # When use_distributed_prep is true, preprocessing tasks run on
            # cluster workers and leave behind unmanaged memory (NumPy
            # arrays, mini-zarr write buffers, etc.).  Without cleanup,
            # workers start metric tasks with ~3-4 GiB already consumed,
            # leaving almost no headroom → tasks OOM → 0% progress.
            if self.pcfg.use_distributed_prep and is_obs_batch:
                try:
                    _client.run(_worker_full_cleanup)
                    time.sleep(0.5)  # let OS reclaim pages
                except Exception:
                    pass

            # Apply reference path remapping (from current batch prefetch)
            if _ref_result:
                for _entry in batch:
                    _rd = _entry.get("ref_data")
                    if isinstance(_rd, str) and _rd in _ref_result:
                        _entry["ref_data"] = _ref_result[_rd]

            # Apply prediction path remapping (from current batch prefetch)
            if _pred_result:
                for _entry in batch:
                    _pd = _entry.get("pred_data")
                    if (
                        isinstance(_pd, str)
                        and _pd in _pred_result
                    ):
                        _entry["pred_data"] = _pred_result[_pd]

            # Apply look-ahead prediction paths (downloaded during previous batch)
            _la_pred_map = _la_data.get('pred_map', {}) if _la_data else {}
            if _la_pred_map:
                for _entry in batch:
                    _pd = _entry.get("pred_data")
                    if (
                        isinstance(_pd, str)
                        and _pd in _la_pred_map
                    ):
                        _entry["pred_data"] = _la_pred_map[_pd]

            # Always create the progress bar, even if no lookahead
            _bar_bold   = "\033[1m"
            _bar_cyan   = "\033[1;96m"   # bright cyan, bold
            _bar_yellow = "\033[1;93m"   # bright yellow, bold
            _bar_reset  = "\033[0m"
            _overall_bar = tqdm(
                total=num_tasks,
                desc=(
                    f"{_bar_cyan}📦 Batch N°{_batch_idx+1}/{_total_batches}{_bar_reset}"
                    f" {_bar_bold}│{_bar_reset}"
                    f" {_bar_yellow}🔗 {str(ref_alias).upper()}{_bar_reset}"
                ),
                leave=True,
                unit="task",
                dynamic_ncols=True,
                file=_sys_bars.stderr,
                bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
            )

            logger.debug(
                f"{ref_alias}: {num_tasks} tasks on {_N} workers"
            )

            # -- Observation scheduling: submit heavy windows first --
            # We only keep ~slots tasks in-flight. If the batch list is in
            # chronological order and cost varies a lot, the heaviest tasks
            # can end up being submitted last, creating a long end-of-batch
            # tail. Sorting by a simple cost proxy reduces the tail (LPT).
            _task_order: List[int] = list(range(num_tasks))
            if is_obs_batch:
                _task_order.sort(
                    key=lambda i: int(batch[i].get("_obs_cost", 0) or 0),
                    reverse=True,
                )
                _max_cost = max((int(e.get("_obs_cost", 0) or 0) for e in batch), default=0)
                _min_cost = min((int(e.get("_obs_cost", 0) or 0) for e in batch), default=0)
                if _max_cost != _min_cost:
                    logger.debug(
                        f"{ref_alias}: scheduling obs tasks by descending cost "
                        f"(_obs_cost rows): min={_min_cost}, max={_max_cost}"
                    )

            # NOTE: If the batch has <= workers tasks, Dask cannot keep all
            # workers busy for the whole batch. As soon as faster tasks finish,
            # their workers go idle while a few slow/IO-bound tasks ("stragglers")
            # continue running. This typically shows up as a CPU usage drop around
            # the middle/last third of the batch even though tasks remain.
            if num_tasks <= _N:
                logger.debug(
                    f"{ref_alias}: batch has {num_tasks} tasks for {_N} workers; "
                    "CPU may drop mid-batch due to stragglers. "
                    "If this is undesirable, increase batch_size (more tasks than workers) "
                    "or reduce n_workers."
                )

            # -- CRITICAL: force synchronous dask scheduler --------
            # With processes=False the cluster uses threads that all
            # share the same process.  An active distributed Client
            # makes .compute() submit sub-tasks to the cluster.
            # If all worker slots are busy -> deadlock.
            # Set once here, never restore — this is intentional.
            # _client.submit() is a Client method and is not affected
            # by this config setting, so task dispatch still works.
            # dask.config.set(scheduler="synchronous")

            _active: Dict[Any, int] = {}
            _results: Dict[int, Any] = {}
            _all_futures: List[Any] = []
            # Driver-observed (submit -> result) wall times help diagnose
            # end-of-batch slowdowns (queueing, pauses, IO waits) vs pure
            # compute time reported by workers.
            _submitted_at: Dict[Any, float] = {}
            _wall_times_s: List[float] = []
            _wall_tail_s: List[float] = []
            _wall_by_idx: Dict[int, float] = {}
            _wall_tail_by_idx: Dict[int, float] = {}
            _next = 0
            _n_collected = 0
            _ac = as_completed([])

            # -- Recalculate concurrency now that preprocessing is done --
            # The initial max_concurrent_obs was set before the obs
            # preprocessing pipeline ran, so prefetched_obs_zarr_path
            # wasn't known yet.  Re-evaluate now.
            if is_obs_batch:
                _has_shared_zarr = (
                    batch
                    and isinstance(batch[0].get("ref_data"), dict)
                    and batch[0]["ref_data"].get("prefetched_obs_zarr_path") is not None
                )
                if _has_shared_zarr:
                    max_concurrent_obs = max(_total_slots, 2)
                else:
                    max_concurrent_obs = max(_N, 1)

            # -- Scatter heavy shared objects once -------------------------
            # Without scattering, Dask serializes every `entry` dict
            # independently per task.  The observation catalog
            # (DatasetCatalog / GeoDataFrame) is the *same* Python object
            # across all entries in the batch — scattering it once
            # eliminates N-1 redundant pickle round-trips through the
            # scheduler, cutting submission overhead from ~100 MB to ~1 MB
            # for a typical 30-task SWOT batch.
            # When a shared zarr was built, workers never access the
            # catalog at all — skip the scatter to save memory.
            if is_obs_batch and batch and not _has_shared_zarr:
                _ref_d0 = batch[0].get("ref_data")
                if isinstance(_ref_d0, dict) and "source" in _ref_d0:
                    try:
                        _catalog_obj = _ref_d0["source"]
                        _scattered_cat = _client.scatter(
                            _catalog_obj, broadcast=True
                        )
                        _scattered_futs.append(_scattered_cat)
                        for _entry in batch:
                            _rd = _entry.get("ref_data")
                            if isinstance(_rd, dict):
                                _rd["source"] = _scattered_cat
                        logger.debug(
                            f"{ref_alias}: scattered obs catalog to "
                            f"{_N} workers (avoiding {num_tasks}× "
                            f"re-serialization)"
                        )
                    except Exception as _exc_scatter:
                        logger.debug(
                            f"{ref_alias}: scatter failed "
                            f"({_exc_scatter!r}), proceeding without"
                        )

            # Submit all tasks at once — ARGO data is already prefetched
            # to local Zarr, so workers won't block on HTTP.
            # For non-ARGO batches, same behaviour as before.
            _n_seed = min(max_concurrent_obs, num_tasks)
            while _next < _n_seed:
                _task_i = _task_order[_next]
                _f = _client.submit(fn, batch[_task_i], retries=1, pure=False)
                _all_futures.append(_f)
                _active[_f] = _task_i
                _submitted_at[_f] = time.monotonic()
                _ac.add(_f)
                _next += 1

            logger.debug(
                f"{ref_alias}: {_n_seed}/{num_tasks} tasks submitted to "
                f"{_N} workers — waiting for first result…"
            )
            if _overall_bar is not None:
                _overall_bar.set_postfix_str("workers busy…")

            # -- Look-ahead: download next batch's data during as_completed --
            # While workers are busy computing, the driver has spare CPU/IO.
            # Use it to prefetch obs+pred files for the next batch.
            import threading as _la_thr

            _la_thread = None
            if _lookahead_next is not None:
                _la_batch = _lookahead_next.get('batch')
                _la_ref_alias = _lookahead_next.get('ref_alias')
                _la_is_obs = (
                    _la_batch
                    and _la_batch[0].get("ref_is_observation", False)
                )

                def _do_lookahead():
                    """Download obs+pred files for the next batch."""
                    _result: Dict[str, Any] = {}
                    try:
                        # -- Obs download ------------------------------
                        if (
                            _la_is_obs
                            and _la_ref_alias != "argo_profiles"
                        ):
                            _la_mgr = self.dataloader.ref_managers.get(
                                _la_ref_alias  # type: ignore[arg-type]
                            )
                            _la_has_fs = (
                                _la_mgr is not None
                                and hasattr(_la_mgr, "params")
                                and hasattr(_la_mgr.params, "fs")
                                and _la_mgr.params.fs is not None
                            )
                            if _la_has_fs:
                                _la_paths: List[str] = []
                                for _e in _la_batch:  # type: ignore[union-attr]
                                    _rd = _e.get("ref_data")
                                    if (
                                        isinstance(_rd, dict)
                                        and "source" in _rd
                                    ):
                                        _cat = _rd["source"]
                                        _tb = _rd.get("time_bounds")
                                        if (
                                            _tb is not None
                                            and hasattr(
                                                _cat, "get_dataframe"
                                            )
                                        ):
                                            _cdf = _cat.get_dataframe()
                                            _ff = filter_by_time(
                                                _cdf,
                                                pd.Timestamp(_tb[0]),
                                                pd.Timestamp(_tb[1]),
                                            )
                                            _la_paths.extend(
                                                _ff["path"].tolist()
                                            )
                                _la_uniq = list(
                                    dict.fromkeys(_la_paths)
                                )
                                if _la_uniq:
                                    from pathlib import Path as _PfLA
                                    from dctools.data.connection.connection_manager import (
                                        prefetch_obs_files_to_local,
                                    )
                                    _la_cache = str(
                                        _PfLA(
                                            os.path.abspath(
                                                getattr(
                                                    self, "results_dir",
                                                    None,
                                                )
                                                or "/tmp"
                                            )
                                        )
                                        / "obs_prefetch_cache"
                                        / str(_la_ref_alias)
                                    )
                                    logger.debug(
                                        f"Look-ahead: downloading "
                                        f"{len(_la_uniq)} obs files "
                                        f"for {_la_ref_alias}"
                                    )
                                    _la_obs_map = (
                                        prefetch_obs_files_to_local(
                                            remote_paths=_la_uniq,
                                            cache_dir=_la_cache,
                                            fs=_la_mgr.params.fs,  # type: ignore[union-attr]
                                            ref_alias=(
                                                f"LA:{_la_ref_alias}"
                                            ),
                                            max_workers=self.pcfg.prefetch_obs_workers,
                                        )
                                    )
                                    if _la_obs_map:
                                        _result['obs_map'] = _la_obs_map

                        # -- Pred download -----------------------------
                        if _la_batch:
                            _sample = _la_batch[0].get("pred_data")
                            _is_remote = isinstance(_sample, str) and (
                                _sample.startswith("https://")
                                or _sample.startswith("http://")
                                or _sample.startswith("s3://")
                            )
                            if _is_remote:
                                from pathlib import Path as _PfLA2
                                import shutil as _sh_la
                                _la_pred_cache = str(
                                    _PfLA2(
                                        os.path.abspath(
                                            getattr(
                                                self, "results_dir",
                                                None,
                                            )
                                            or "/tmp"
                                        )
                                    )
                                    / "pred_prefetch_cache"
                                    / str(
                                        self.dataloader.pred_alias
                                    )
                                )
                                os.makedirs(
                                    _la_pred_cache, exist_ok=True
                                )
                                _la_pfs = getattr(
                                    getattr(
                                        self.dataloader,
                                        "pred_manager",
                                        None,
                                    ),
                                    "params",
                                    None,
                                )
                                _la_s3fs = getattr(
                                    _la_pfs, "fs", None
                                )
                                _la_ep = getattr(
                                    _la_pfs, "endpoint_url", ""
                                ) or ""
                                _la_upreds = list(
                                    dict.fromkeys(
                                        e["pred_data"]
                                        for e in _la_batch
                                        if isinstance(
                                            e.get("pred_data"), str
                                        )
                                    )
                                )
                                _la_pred_map: Dict[str, str] = {}
                                for _rp in _la_upreds:
                                    _fn = _PfLA2(_rp).name
                                    _lz = os.path.join(
                                        _la_pred_cache, _fn
                                    )
                                    if (
                                        os.path.isdir(_lz)
                                        and os.listdir(_lz)
                                    ):
                                        _la_pred_map[_rp] = _lz
                                        continue
                                    try:
                                        _s3k = _rp
                                        if (
                                            _la_ep
                                            and _s3k.startswith(
                                                _la_ep
                                            )
                                        ):
                                            _s3k = _s3k[
                                                len(_la_ep):
                                            ].lstrip("/")
                                        elif _s3k.startswith(
                                            "s3://"
                                        ):
                                            _s3k = _s3k[5:]
                                        _tmpz = (
                                            _lz + ".downloading.la"
                                        )
                                        if os.path.isdir(_tmpz):
                                            _sh_la.rmtree(
                                                _tmpz,
                                                ignore_errors=True,
                                            )
                                        if _la_s3fs is not None:
                                            _la_s3fs.get(
                                                _s3k,
                                                _tmpz,
                                                recursive=True,
                                            )
                                        if os.path.isdir(_lz):
                                            _sh_la.rmtree(
                                                _lz,
                                                ignore_errors=True,
                                            )
                                        os.rename(_tmpz, _lz)
                                        _la_pred_map[_rp] = _lz
                                    except Exception:
                                        pass
                                if _la_pred_map:
                                    _result['pred_map'] = _la_pred_map

                    except Exception as _exc_la:
                        logger.debug(
                            f"Look-ahead prefetch error: {_exc_la!r}"
                        )

                    # Store results for the next _evaluate_batch call
                    if _result:
                        _la_cache_dict = getattr(
                            self, '_lookahead_cache', {}
                        )
                        _la_cache_dict[id(_la_batch)] = _result
                        self._lookahead_cache = _la_cache_dict
                        logger.debug(
                            "Look-ahead: pre-downloaded "
                            f"{len(_result.get('obs_map', {}))} obs + "
                            f"{len(_result.get('pred_map', {}))} pred "
                            f"files for next batch"
                        )

                _la_thread = _la_thr.Thread(
                    target=_do_lookahead,
                    daemon=True,
                    name="lookahead-dl",
                )
                _la_thread.start()

            # -- Background heartbeat + stall-watchdog --------------------
            # The heartbeat updates the progress bar every 30 s.
            # The watchdog detects when NO task has completed for more than
            # STALL_TIMEOUT seconds (e.g. all workers blocked on a stalled
            # S3 / HTTP connection) and cancels the stuck futures to unblock
            # the as_completed loop.  Cancelled futures are yielded as errors.
            # Default: 300 s (5 min). Override: DCTOOLS_EVAL_STALL_TIMEOUT env var.
            import threading as _threading_hb

            _hb_stop = _threading_hb.Event()
            _hb_t0 = time.time()
            # Track the last time a task completed (initialised to start).
            _last_progress: list = [time.time()]
            # After this many seconds with zero new completions -> cancel all.
            # Default: 5 minutes. Override with DCTOOLS_EVAL_STALL_TIMEOUT (seconds).
            _STALL_TIMEOUT = self.pcfg.stall_timeout

            # Log the inevitable "tail" once: when pending tasks <= execution slots,
            # the cluster cannot keep all slots busy (under-subscription).
            _tail_logged: list = [False]
            _last_state_log_s: list = [0.0]

            def _maybe_log_cluster_state(elapsed_s: float, pending: int):
                if elapsed_s - _last_state_log_s[0] < 60.0:
                    return
                _last_state_log_s[0] = elapsed_s
                try:
                    info = _client.scheduler_info()
                    workers = info.get("workers", {})
                    paused = 0
                    max_frac = 0.0
                    for w_info in workers.values():
                        if w_info.get("status") == "paused":
                            paused += 1
                        mem_used = w_info.get("metrics", {}).get(
                            "memory", w_info.get("memory", 0)
                        )
                        mem_limit = w_info.get("memory_limit", 0) or 0
                        if mem_limit and mem_limit > 0:
                            max_frac = max(max_frac, float(mem_used) / float(mem_limit))

                    # Only speak up when it is truly actionable (workers
                    # paused means tasks cannot progress).  High-but-not-
                    # paused memory is expected during busy batches and is
                    # handled at inter-batch restart time; log it at DEBUG
                    # to avoid flooding the console every 60 s.
                    if paused:
                        logger.warning(
                            f"{ref_alias}: state pending={pending}, active={len(_active)}, "
                            f"paused_workers={paused}/{len(workers)}, "
                            f"max_mem_frac={max_frac:.2f}"
                        )
                    elif max_frac >= self.pcfg.max_memory_fraction:
                        logger.debug(
                            f"{ref_alias}: state pending={pending}, active={len(_active)}, "
                            f"paused_workers=0/{len(workers)}, "
                            f"max_mem_frac={max_frac:.2f} "
                            f"(≥ {self.pcfg.max_memory_fraction:.0%} but no workers paused)"
                        )
                except Exception as _exc_state:
                    logger.debug(f"{ref_alias}: cannot query scheduler state: {_exc_state!r}")

            def _heartbeat_fn():
                while not _hb_stop.is_set():
                    _hb_stop.wait(30)
                    if _hb_stop.is_set():
                        break
                    _elapsed = time.time() - _hb_t0
                    _pending = max(num_tasks - _n_collected, 0)
                    pct = 100.0 * (_n_collected / num_tasks) if num_tasks else 0.0
                    _overall_bar.set_postfix_str(
                        f"{pct:.1f}% done, {_elapsed:.0f}s elapsed"
                    )

                    if (not _tail_logged[0]) and _pending <= _total_slots:
                        _tail_logged[0] = True
                        logger.debug(
                            f"{ref_alias}: entering end-of-batch tail: "
                            f"pending={_pending} <= slots={_total_slots}. "
                            "CPU drop is expected here; remaining time is dominated by the slowest tasks."  # noqa: E501
                        )

                    _maybe_log_cluster_state(_elapsed, _pending)

                    # -- Watchdog: cancel + resubmit stuck futures --------
                    _stall_s = time.time() - _last_progress[0]
                    # In the "tail" phase (fewer pending tasks than worker slots)
                    # the remaining tasks are often the slowest ones and run on
                    # already-loaded workers.  Give them 2× the normal budget
                    # to avoid false-positive cancellations.
                    _pending_now = max(num_tasks - _n_collected, 0)
                    _effective_stall = (
                        _STALL_TIMEOUT * 2
                        if _pending_now <= _total_slots
                        else _STALL_TIMEOUT
                    )
                    if _stall_s >= _effective_stall and _active:
                        _n_stuck = len(_active)
                        # Diagnose: are workers actually paused (OOM)?
                        _paused_workers = []
                        try:
                            _sched_info = _client.scheduler_info()
                            for _waddr, _winfo in _sched_info.get("workers", {}).items():
                                if _winfo.get("status") == "paused":
                                    _paused_workers.append(_waddr)
                        except Exception:
                            pass
                        _cause = (
                            f"{len(_paused_workers)} worker(s) paused (OOM)"
                            if _paused_workers
                            else "likely S3 timeout or slow I/O"
                        )
                        logger.warning(
                            f"{ref_alias}: NO task completed in the last "
                            f"{_stall_s:.0f}s (timeout={_effective_stall:.0f}s, "
                            f"tail={'yes' if _pending_now <= _total_slots else 'no'}) "
                            f"— {_cause}.  "
                            f"Cancelling {_n_stuck} future(s) and resubmitting."
                        )
                        # Collect stuck (future, idx) pairs.
                        _stuck_pairs = []
                        for _stuck_f in list(_active.keys()):
                            _stuck_pairs.append((_stuck_f, _active.pop(_stuck_f)))
                        # Submit replacements and register them with _ac
                        # BEFORE cancelling the stuck futures.  This ensures
                        # as_completed knows there are still futures to yield
                        # and the main collection loop does not exit early
                        # (race condition: cancel() makes the future "done"
                        # immediately, which could drain _ac before we add
                        # the replacement).
                        for _stuck_f, _resubmit_idx in _stuck_pairs:
                            _f_re = _client.submit(
                                fn, batch[_resubmit_idx],
                                retries=1, pure=False,
                            )
                            _all_futures.append(_f_re)
                            _active[_f_re] = _resubmit_idx
                            _submitted_at[_f_re] = time.monotonic()
                            _ac.add(_f_re)
                        # NOW cancel the stuck originals (replacements are
                        # already registered so _ac won't exit early).
                        for _stuck_f, _ in _stuck_pairs:
                            try:
                                _stuck_f.cancel()
                            except Exception:
                                pass
                        # If workers are paused (OOM), restart them so the
                        # resubmitted task can actually run.  Use targeted
                        # per-worker restart to avoid disturbing healthy workers.
                        if _paused_workers:
                            logger.warning(
                                f"{ref_alias}: restarting {len(_paused_workers)} "
                                f"paused worker(s) to reclaim unmanaged memory: "
                                f"{_paused_workers}"
                            )
                            import logging as _logging
                            import time as _time_wd
                            _dist_loggers_wd = [
                                _logging.getLogger("distributed"),
                                _logging.getLogger("distributed.worker"),
                                _logging.getLogger("distributed.comm"),
                                _logging.getLogger("distributed.comm.tcp"),
                            ]
                            _saved_wd = [(lg, lg.level) for lg in _dist_loggers_wd]
                            for _lg in _dist_loggers_wd:
                                _lg.setLevel(_logging.CRITICAL)
                            try:
                                try:
                                    _client.restart_workers(_paused_workers)
                                    _time_wd.sleep(2.0)
                                except Exception as _exc_rw:
                                    # restart_workers not available on older
                                    # dask versions — fall back to run-cleanup.
                                    logger.debug(
                                        f"restart_workers failed ({_exc_rw!r}), "
                                        f"falling back to worker cleanup"
                                    )
                                    _client.run(_worker_full_cleanup)
                            finally:
                                for _lg, _lvl in _saved_wd:
                                    _lg.setLevel(_lvl)
                        else:
                            # No paused workers: soft cleanup is enough.
                            try:
                                _client.run(_worker_full_cleanup)
                            except Exception:
                                pass
                        # Reset timer so we don't cancel again immediately.
                        _last_progress[0] = time.time()

            _hb_thread = _threading_hb.Thread(
                target=_heartbeat_fn, daemon=True
            )
            _hb_thread.start()

            try:
                for _done in _ac:
                    _idx = _active.pop(_done, None)
                    if _idx is None:
                        # Future was already removed from _active by the watchdog
                        # (cancelled + resubmitted).  Skip it — the resubmitted
                        # future will be collected separately.
                        _submitted_at.pop(_done, None)
                        continue
                    _t_submit = _submitted_at.pop(_done, None)
                    if _t_submit is not None:
                        _wall = time.monotonic() - _t_submit
                        _wall_times_s.append(_wall)
                        _wall_by_idx[int(_idx)] = float(_wall)
                        _tail_remaining = max(num_tasks - _n_collected, 0)
                        if _tail_remaining <= _total_slots:
                            _wall_tail_s.append(_wall)
                            _wall_tail_by_idx[int(_idx)] = float(_wall)
                    try:
                        _res = _done.result()
                    except Exception as _exc:
                        logger.warning(
                            f"Task {_idx} ({ref_alias}) raised: {_exc!r}"
                        )
                        _res = {
                            "ref_alias": ref_alias,
                            "result": None,
                            "n_points": 0,
                            "duration_s": 0.0,
                            "error": repr(_exc),
                        }
                    _results[_idx] = _res
                    _n_collected += 1
                    _last_progress[0] = time.time()  # reset stall watchdog

                    # Log timing for the very first completed task
                    if _n_collected == 1:
                        _first_elapsed = time.time() - _hb_t0
                        logger.debug(
                            f"{ref_alias}: first result received "
                            f"after {_first_elapsed:.1f}s"
                        )
                        _overall_bar.set_postfix_str("")

                    # Only log errors; normal completions are silent
                    # (the tqdm bar + batch summary are enough).
                    _err = _res.get("error") if isinstance(_res, dict) else None
                    if _err:
                        _vt = _res.get("valid_time", "") if isinstance(_res, dict) else ""
                        _overall_bar.write(f"  \u2717 {_vt}  ERROR: {_err}")

                    _overall_bar.update(1)

                    # Submit remaining tasks as slots become available
                    if _next < num_tasks:
                        _task_i = _task_order[_next]
                        _f_new = _client.submit(
                            fn, batch[_task_i], retries=1, pure=False
                        )
                        _all_futures.append(_f_new)
                        _active[_f_new] = _task_i
                        _submitted_at[_f_new] = time.monotonic()
                        _ac.add(_f_new)
                        _next += 1

                    if _n_collected >= num_tasks:
                        break
            finally:
                _hb_stop.set()
                _hb_thread.join(timeout=2)
                try:
                    _overall_bar.close()
                except Exception:
                    pass

            # -- Wait for look-ahead thread (non-blocking if already done) --
            if _la_thread is not None:
                _la_thread.join(timeout=5)

            # -- Explicit batch cleanup on client/workers -------------------
            # Step 1: release futures so the Dask scheduler instructs workers
            # to drop their in-memory task results.  wait() confirms the
            # driver-side state transition but the worker-side deletion is
            # asynchronous; a brief pause lets those messages propagate
            # before we ask workers to run gc.collect() + malloc_trim().
            try:
                if _all_futures:
                    _client.cancel(_all_futures, force=True)
                    wait(_all_futures)
                    import time as _t_cleanup
                    _t_cleanup.sleep(1.0)  # let scheduler → worker "delete" msgs arrive
            except Exception:
                pass
            # Release scattered shared data from workers' memory.
            for _sf in _scattered_futs:
                try:
                    _client.cancel([_sf], force=True)
                except Exception:
                    pass
            _scattered_futs.clear()
            # Step 2: GC + malloc_trim on every worker (called AFTER the task
            # store has had a chance to clear so RSS is as low as possible).
            try:
                _client.run(_worker_full_cleanup)
            except Exception:
                pass

            # Restore original batch order
            batch_results: List[Any] = [_results[i] for i in range(num_tasks)]

            batch_duration = time.time() - batch_t0

            def _pct(values: List[float], q: float) -> float:
                if not values:
                    return 0.0
                if q <= 0:
                    return float(min(values))
                if q >= 1:
                    return float(max(values))
                xs = sorted(values)
                pos = q * (len(xs) - 1)
                lo = int(pos)
                hi = min(lo + 1, len(xs) - 1)
                if hi == lo:
                    return float(xs[lo])
                w = pos - lo
                return float(xs[lo] * (1.0 - w) + xs[hi] * w)

            if _wall_times_s:
                logger.debug(
                    f"{ref_alias}: wall-times submit->result: "
                    f"p50={_pct(_wall_times_s, 0.50):.1f}s "
                    f"p90={_pct(_wall_times_s, 0.90):.1f}s "
                    f"p99={_pct(_wall_times_s, 0.99):.1f}s "
                    f"max={max(_wall_times_s):.1f}s"
                )
                if _wall_tail_s:
                    logger.debug(
                        f"{ref_alias}: tail wall-times (when pending<=slots): "
                        f"p50={_pct(_wall_tail_s, 0.50):.1f}s "
                        f"p90={_pct(_wall_tail_s, 0.90):.1f}s "
                        f"max={max(_wall_tail_s):.1f}s"
                    )

                # Pin down extreme stragglers (often the reason the tail feels like a stall).
                _slow_idxs = sorted(
                    _wall_by_idx.keys(),
                    key=lambda i: _wall_by_idx.get(i, 0.0),
                    reverse=True,
                )[: min(5, len(_wall_by_idx))]
                if _slow_idxs:
                    _lines: List[str] = []
                    for _i in _slow_idxs:
                        _entry = batch[_i] if 0 <= _i < len(batch) else {}
                        _vt = _entry.get("valid_time")
                        _frt = _entry.get("forecast_reference_time")
                        _cost = _entry.get("_obs_cost")
                        _tb = None
                        _rd = _entry.get("ref_data")
                        if isinstance(_rd, dict) and "time_bounds" in _rd:
                            _tb = _rd.get("time_bounds")

                        _res_i = batch_results[_i] if 0 <= _i < len(batch_results) else None
                        _np = _res_i.get("n_points") if isinstance(_res_i, dict) else None
                        _pp = _res_i.get("preprocess_s") if isinstance(_res_i, dict) else None
                        _mt = _res_i.get("duration_s") if isinstance(_res_i, dict) else None

                        _w = _wall_by_idx.get(_i, 0.0)
                        _is_tail = _i in _wall_tail_by_idx
                        _tail_tag = " tail" if _is_tail else ""
                        _lines.append(
                            f"idx={_i} wall={_w:.1f}s{_tail_tag} "
                            f"preproc={_pp!s} metrics={_mt!s} pts={_np!s} "
                            f"_obs_cost={_cost!s} valid_time={_vt} forecast_ref={_frt} time_bounds={_tb}"  # noqa: E501
                        )

                    '''logger.debug(
                        f"{ref_alias}: slowest tasks (driver wall-time):\n  "
                        + "\n  ".join(_lines)
                    )'''

            # Analyze task timings
            _valid = [r for r in batch_results if r and isinstance(r, dict)]
            times = [r.get('duration_s', 0) for r in _valid]
            preprocs = [r.get('preprocess_s', 0) for r in _valid]
            points = [r.get('n_points', 0) for r in _valid]

            if times:
                total_pts = sum(points)
                avg_pp = sum(preprocs) / len(preprocs)
                avg_mt = sum(times) / len(times)
                logger.info(
                    f"Batch done: {len(batch_results)}/{num_tasks} tasks "
                    f"in {batch_duration:.1f}s | "
                    f"Avg preproc={avg_pp:.1f}s  metrics={avg_mt:.1f}s | "
                    f"{total_pts:,} total pts"
                )
            else:
                logger.info(f"Batch done in {batch_duration:.1f}s (no valid results)")

            return batch_results
        except Exception as exc:
            logger.error(f"Error processing batch: {repr(exc)}")
            traceback.print_exc()
            return [{
                "forecast_reference_time": batch[0].get("forecast_reference_time") if batch else None,  # noqa: E501
                "model": pred_alias,
                "reference": ref_alias,
                "result": None,
                "n_points": 0,
                "duration_s": 0.0,
                "error": repr(exc),
            }]
