"""Centralized parallelism and resource management configuration.

All parallelism parameters — batch sizes, timeouts, preprocessing workers,
prefetch threads, Dask memory thresholds — are defined here in a single
frozen dataclass.  This replaces the previous pattern of scattered env vars,
hardcoded constants, and disconnected YAML keys.

Usage::

    from dctools.utilities.parallelism import ParallelismConfig

    # Load from the ``parallelism:`` YAML section (with env-var overrides).
    pcfg = ParallelismConfig.from_dict(yaml_config.get("parallelism", {}))

    # Pass *pcfg* to every component that needs it.
    evaluator = Evaluator(..., parallelism=pcfg)

Env vars (``DCTOOLS_*``) still work as runtime overrides — they take
precedence over YAML values when ``env_override=True`` (the default).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# Backward-compat: old top-level YAML keys --> new field names
# ---------------------------------------------------------------------------
_COMPAT_RENAMES: Dict[str, str] = {
    "max_p_memory_increase": "max_memory_increase",
    "max_worker_memory_fraction": "max_memory_fraction",
}

# Old top-level YAML keys that may appear outside the ``parallelism:``
# section in legacy configs.  Used by :meth:`from_args` for fallback.
_TOP_LEVEL_FALLBACK_KEYS: list[tuple[str, str]] = [
    # (old YAML key, field name)
    ("obs_batch_size", "obs_batch_size"),
    ("reduce_precision", "reduce_precision"),
    ("restart_workers_per_batch", "restart_workers_per_batch"),
    ("max_p_memory_increase", "max_memory_increase"),
    ("max_worker_memory_fraction", "max_memory_fraction"),
]

# ---------------------------------------------------------------------------
# Env-var overrides (read only when env_override=True in from_dict)
# ---------------------------------------------------------------------------
_ENV_OVERRIDES: Dict[str, tuple[str, type]] = {
    "compute_timeout": ("DCTOOLS_S3_COMPUTE_TIMEOUT", int),
    "obs_compute_timeout": ("DCTOOLS_OBS_COMPUTE_TIMEOUT", int),
    "stall_timeout": ("DCTOOLS_EVAL_STALL_TIMEOUT", int),
    "prep_workers": ("DCTOOLS_PREP_WORKERS", int),
    "max_shared_obs_files": ("DCTOOLS_SHARED_OBS_MAX_FILES", int),
    "max_obs_files_per_batch": ("DCTOOLS_MAX_OBS_FILES_PER_BATCH", int),
    "obs_viewer_threads": ("DCTOOLS_OBS_VIEWER_THREADS", int),
    "worker_cache_size": ("DCTOOLS_WORKER_DATASET_CACHE_SIZE", int),
    "worker_scale": ("DCTOOLS_WORKER_SCALE", float),
}

_ENV_BOOL_OVERRIDES: Dict[str, str] = {
    "auto_adapt": "DCTOOLS_AUTO_ADAPT",
}


@dataclass(frozen=True)
class ParallelismConfig:
    """Immutable container for all parallelism & resource parameters.

    Loaded once from ``parallelism:`` in the YAML config.  Individual fields
    can still be overridden at runtime via ``DCTOOLS_*`` env vars.
    """

    # -- Batch sizing -------------------------------------------------
    obs_batch_size: int = 30
    gridded_batch_size: int | None = None   # tasks per batch for gridded refs (e.g. GLORYS)

    # -- Timeouts (seconds) -------------------------------------------
    compute_timeout: int = 90           # S3 .compute() for pred/ref on workers
    obs_compute_timeout: int = 120      # shared obs zarr .compute() on workers
    stall_timeout: int = 1200           # seconds without progress --> cancel+retry

    # -- Driver-side preprocessing ------------------------------------
    prep_workers: int = 0               # 0 = auto
    prep_use_processes: bool = True     # ProcessPool (True) vs ThreadPool
    max_shared_obs_files: int = 5000    # skip shared zarr above this count
    max_obs_files_per_batch: int = 150  # volume-aware batch split threshold
    use_distributed_prep: bool = True   # R5: use Dask cluster for preprocessing (avoids ProcessPool/Dask RAM competition)
    obs_viewer_threads: int = 2         # R7: parallel threads in ObservationDataViewer (reduced to avoid contention)

    # -- Data prefetch (S3 --> local) -----------------------------------
    enable_ref_prefetch: bool = True
    prefetch_workers: int = 4           # threads for ref + pred S3 downloads
    prefetch_obs_workers: int = 8       # threads for obs file downloads

    # -- Worker behaviour ---------------------------------------------
    worker_cache_size: int = 4          # LRU dataset cache per worker
    blosc_threads: int = 4              # Blosc decompression threads (I/O-bound, benefits from >1)
    #: Local directory for Dask workers' scratch space.
    #: Set to a fast local disk (e.g. /tmp/dask-scratch) to avoid the
    #: "Creating scratch directories is taking a surprisingly long time"
    #: warning that appears when the default tempdir lives on a network FS.
    #: ``None`` lets Dask choose (usually /tmp, which may be NFS-mounted).
    dask_tmp_dir: str | None = None

    # -- Memory management --------------------------------------------
    reduce_precision: bool = True       # float64 --> float32 where safe (R4)
    restart_workers_per_batch: bool = False
    max_memory_increase: float = 0.50   # relative RAM increase --> restart
    max_memory_fraction: float = 0.80   # absolute fraction --> restart

    # -- Dask memory thresholds ---------------------------------------
    memory_target: float = 0.60         # start spilling to disk
    memory_spill: float = 0.70          # actively spill
    memory_pause: float = 0.90          # pause worker
    memory_terminate: float = 0.95      # kill worker

    # -- Per-dataset worker scaling -----------------------------------
    #: Multiplicative factor applied to each dataset's ``n_parallel_workers``
    #: and ``nthreads_per_worker`` at runtime.  Corresponds to the chosen
    #: parallelism profile:  LOW=0.5 · AVERAGE=1.0 · HIGH=1.5.
    #: Always clipped to a minimum of 1 worker / 1 thread.
    #: Can be overridden via DCTOOLS_WORKER_SCALE env var.
    worker_scale: float = 1.0

    # -- Adaptive resource detection ----------------------------------
    #: When True, detect machine resources (CPU, RAM) and adapt all
    #: per-dataset Dask sizing and parallelism profile parameters
    #: proportionally.  YAML values are treated as indicative (calibrated
    #: for the ``reference_machine`` declared in the config).
    #: Can be overridden via DCTOOLS_AUTO_ADAPT env var.
    auto_adapt: bool = True

    # -----------------------------------------------------------------
    # Factory methods
    # -----------------------------------------------------------------
    @classmethod
    def from_dict(
        cls,
        d: Dict[str, Any] | None = None,
        *,
        env_override: bool = True,
    ) -> ParallelismConfig:
        """Build from a YAML dict with optional env-var overrides.

        Parameters
        ----------
        d : dict, optional
            The ``parallelism:`` section from the YAML config.
        env_override : bool
            When *True* (default), ``DCTOOLS_*`` env vars take precedence
            over the values in *d*.
        """
        raw: Dict[str, Any] = dict(d or {})

        # Backward-compat renames
        for old_key, new_key in _COMPAT_RENAMES.items():
            if old_key in raw and new_key not in raw:
                raw[new_key] = raw.pop(old_key)

        # Env-var overrides (int/float-valued)
        if env_override:
            for field_name, (env_var, converter) in _ENV_OVERRIDES.items():
                val = os.environ.get(env_var)
                if val is not None:
                    try:
                        raw[field_name] = converter(val)
                    except (ValueError, TypeError):
                        pass

            # DCTOOLS_PREP_THREADS_ONLY --> invert to prep_use_processes
            _thr = os.environ.get("DCTOOLS_PREP_THREADS_ONLY", "")
            if _thr.lower() in ("1", "true", "yes"):
                raw["prep_use_processes"] = False

            # DCTOOLS_ENABLE_REF_PREFETCH
            _rp = os.environ.get("DCTOOLS_ENABLE_REF_PREFETCH")
            if _rp is not None:
                raw["enable_ref_prefetch"] = _rp in ("1", "true", "True")

            # DCTOOLS_AUTO_ADAPT --> enable/disable adaptive resource detection
            for field_name, env_var in _ENV_BOOL_OVERRIDES.items():
                _val = os.environ.get(env_var)
                if _val is not None:
                    raw[field_name] = _val.lower() in ("1", "true", "yes")

        # Keep only known fields + coerce types
        kwargs: Dict[str, Any] = {}
        for f in fields(cls):
            if f.name not in raw:
                continue
            val = raw[f.name]
            try:
                if f.type == "int":
                    val = int(val)
                elif f.type == "float":
                    val = float(val)
                elif f.type == "bool" and not isinstance(val, bool):
                    val = str(val).lower() in ("true", "1", "yes")
            except (ValueError, TypeError):
                continue
            kwargs[f.name] = val

        return cls(**kwargs)

    @classmethod
    def from_args(cls, args: Any) -> ParallelismConfig:
        """Build from an argparse Namespace (backward compatible).

        If ``args.parallelism`` exists (new YAML format), it takes priority.
        Otherwise, scattered top-level keys are collected.
        """
        p = getattr(args, "parallelism", None)
        if isinstance(p, cls):
            return p
        if not isinstance(p, dict):
            p = {}
        p = dict(p)  # copy

        # Collect scattered top-level keys as fallback
        for old_key, field_name in _TOP_LEVEL_FALLBACK_KEYS:
            if field_name not in p and hasattr(args, old_key):
                p[field_name] = getattr(args, old_key)

        return cls.from_dict(p)

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------
    def worker_env_vars(self) -> Dict[str, str]:
        """Env vars to propagate to Dask workers.

        Called by ``configure_dask_workers_env()`` so that worker-side code
        (which reads ``os.environ``) picks up the centralised values.
        """
        return {
            "DCTOOLS_S3_COMPUTE_TIMEOUT": str(self.compute_timeout),
            "DCTOOLS_OBS_COMPUTE_TIMEOUT": str(self.obs_compute_timeout),
            "DCTOOLS_OBS_VIEWER_THREADS": str(self.obs_viewer_threads),
            "DCTOOLS_WORKER_DATASET_CACHE_SIZE": str(self.worker_cache_size),
            "BLOSC_NTHREADS": str(self.blosc_threads),
        }

    def dask_memory_config(self) -> Dict[str, float]:
        """Dask config dict for ``dask.config.set(...)``."""
        return {
            "distributed.worker.memory.target": self.memory_target,
            "distributed.worker.memory.spill": self.memory_spill,
            "distributed.worker.memory.pause": self.memory_pause,
            "distributed.worker.memory.terminate": self.memory_terminate,
        }

    # -----------------------------------------------------------------
    # Adaptive resource helpers
    # -----------------------------------------------------------------
    def get_machine_resources(self) -> Optional["MachineResources"]:
        """Detect current machine resources (lazy, cached-like via detect).

        Returns None when ``auto_adapt`` is False.
        """
        if not self.auto_adapt:
            return None
        from dctools.utilities.adaptive_resources import MachineResources
        return MachineResources.detect()

    def adapt_dask_cfg_for_dataset(
        self,
        yaml_cfg: Dict[str, Any],
        machine: Optional[Any] = None,
        reference_machine_dict: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Adapt a per-dataset Dask config to the current machine.

        When ``auto_adapt`` is False, falls back to the legacy behaviour
        (just apply ``worker_scale`` multiplicatively to workers/threads).

        Parameters
        ----------
        yaml_cfg : dict
            Raw per-dataset config (n_workers, threads_per_worker, memory_limit).
        machine : MachineResources, optional
            Pre-detected resources.  If None and auto_adapt is True, will detect.
        reference_machine_dict : dict, optional
            ``reference_machine:`` section from the YAML config.
        """
        if not self.auto_adapt:
            # Legacy: just scale workers and threads by worker_scale.
            cfg: Dict[str, Any] = {}
            scale = self.worker_scale
            if "n_workers" in yaml_cfg:
                cfg["n_workers"] = max(1, round(int(yaml_cfg["n_workers"]) * scale))
            if "threads_per_worker" in yaml_cfg:
                cfg["threads_per_worker"] = max(1, round(int(yaml_cfg["threads_per_worker"]) * scale))
            if "memory_limit" in yaml_cfg:
                cfg["memory_limit"] = yaml_cfg["memory_limit"]
            return cfg

        from dctools.utilities.adaptive_resources import (
            MachineResources,
            adapt_dask_cfg,
            reference_machine_from_dict,
        )

        if machine is None:
            machine = MachineResources.detect()
        ref = (
            reference_machine_from_dict(reference_machine_dict)
            if reference_machine_dict
            else None
        )
        return adapt_dask_cfg(
            yaml_cfg=yaml_cfg,
            machine=machine,
            reference_machine=ref,
            worker_scale=self.worker_scale,
        )

    def adapt_profile(
        self,
        machine: Optional[Any] = None,
        reference_machine_dict: Optional[Dict[str, Any]] = None,
    ) -> "ParallelismConfig":
        """Return a new :class:`ParallelismConfig` with batch sizes, prefetch
        workers, timeouts, etc. adapted to the current machine.

        When ``auto_adapt`` is False, returns *self* unchanged.
        """
        if not self.auto_adapt:
            return self

        from dctools.utilities.adaptive_resources import (
            MachineResources,
            adapt_parallelism_profile,
            reference_machine_from_dict,
        )

        if machine is None:
            machine = MachineResources.detect()
        ref = (
            reference_machine_from_dict(reference_machine_dict)
            if reference_machine_dict
            else None
        )

        # Build a dict from our current frozen fields, adapt, then rebuild.
        from dataclasses import asdict
        current = asdict(self)
        adapted = adapt_parallelism_profile(current, machine, ref)
        # Preserve frozen semantics.
        return ParallelismConfig(**adapted)

    def _adapt_obs_dask_cfg(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        """Specially adapt Dask config for observation datasets.

        Forces threads_per_worker=1 to prevent CPU oversubscription from
        C-level libraries (pyinterp, BLAS) and compensates by increasing
        n_workers to leverage all available cores.
        """
        adapted_cfg = cfg.copy()

        # Force single-threaded workers for obs tasks
        adapted_cfg["threads_per_worker"] = 1

        # Compensate by increasing worker count if machine info is available
        machine = self.get_machine_resources()
        if machine and machine.cpu_count > 0:
            # Use all available cores as workers
            adapted_cfg["n_workers"] = machine.cpu_count

        return adapted_cfg

    # R9: Force threads_per_worker=1 for observation datasets
    # This is a critical fix to prevent CPU oversubscription.
    from dctools.data.datasets.dataset import is_observation_alias

    def is_observation_alias(self, alias: str) -> bool:
        """Check if the dataset alias is an observation dataset."""
        return is_observation_alias(alias)
