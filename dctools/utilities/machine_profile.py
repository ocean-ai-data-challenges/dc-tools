"""Runtime auto-tuning of Dask parallelism parameters based on hardware.

Called once at config-load time by :func:`dctools.utilities.args_config.load_args_and_config`.

A parameter is filled in when **all** of the following hold:

* ``auto_tune: true`` (or key absent) in the root config  *or*  the YAML value
  is the literal string ``"auto"``
* The value is not already an explicit number (integers/floats are always kept)

The YAML can therefore be used in three modes:

1. **Fully automatic** (recommended) — set ``auto_tune: true`` at the root and
   omit per-source parallelism keys (or set them to ``null`` / ``"auto"``)::

       auto_tune: true
       sources:
         - dataset: swot
           observation_dataset: true
           # n_parallel_workers, nthreads_per_worker, memory_limit_per_worker
           # are all filled automatically

2. **Selective override** — keep ``auto_tune: true`` but pin specific params::

       sources:
         - dataset: swot
           n_parallel_workers: 3   # fixed; everything else still auto-tuned

3. **Fully manual** — set ``auto_tune: false``; only params set to the string
   ``"auto"`` are filled, everything else is kept as-is::

       auto_tune: false
       sources:
         - dataset: swot
           n_parallel_workers: 5          # kept as-is
           memory_limit_per_worker: "auto"  # ← filled from hardware
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from loguru import logger

# ---------------------------------------------------------------------------
# File-size probe constants
# ---------------------------------------------------------------------------

# Peak-RAM safety multiplier applied to uncompressed file size.
# Accounts for: data array + xarray overhead + prediction lookup copy +
# intermediate metric arrays.  Obs workers carry more overhead because they
# also hold the interpolated prediction slice.
_MEM_SAFETY_OBS: float = 3.0
_MEM_SAFETY_GRIDDED: float = 2.5

# Maximum wall-clock seconds allowed for a single S3 probe attempt.
_PROBE_TIMEOUT_S: float = 7.0

# Minimum memory floor per worker (GB) regardless of file-size estimate.
_PROBE_MIN_MEM_GB: float = 0.2


# ---------------------------------------------------------------------------
# Hardware detection
# ---------------------------------------------------------------------------

def _detect_hardware() -> Dict[str, Any]:
    """Return a dict with physical-core count and total RAM (GB).

    Falls back to conservative estimates when *psutil* is unavailable.
    """
    n_logical: int = os.cpu_count() or 4
    try:
        import psutil
        n_physical: int = psutil.cpu_count(logical=False) or n_logical
        total_ram_gb: float = psutil.virtual_memory().total / 1e9
    except ImportError:
        n_physical = n_logical
        total_ram_gb = 16.0
    return {
        "n_physical_cores": n_physical,
        "n_logical_cores": n_logical,
        "total_ram_gb": total_ram_gb,
    }


# ---------------------------------------------------------------------------
# Helper predicates
# ---------------------------------------------------------------------------

def _is_explicit(value: Any) -> bool:
    """Return True when *value* is an explicitly set number (not auto-tunable)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_auto_sentinel(value: Any) -> bool:
    """Return True when *value* requests auto-tuning (None, absent, or 'auto')."""
    return value is None or (isinstance(value, str) and value.strip().lower() == "auto")


def _should_fill(key: str, src: dict, global_auto: bool) -> bool:
    """Return True when the parameter should be overwritten by the auto-tuner."""
    if key not in src:
        # Key absent: fill only when the global flag is on
        return global_auto
    value = src[key]
    if _is_explicit(value):
        return False  # user pinned an explicit number — never override
    # None / "auto" → always fill
    return _is_auto_sentinel(value)


# ---------------------------------------------------------------------------
# File-size probe helpers
# ---------------------------------------------------------------------------

def _build_s3_fs(src: dict):
    """Return an fsspec S3 filesystem from source config, or None on error."""
    try:
        import fsspec
        url = src.get("url") or src.get("endpoint_url") or ""
        key = src.get("s3_key") or src.get("key") or None
        secret = src.get("s3_secret_key") or src.get("secret_key") or None
        bucket = src.get("s3_bucket") or ""
        if not bucket:
            return None, ""
        ck = {"endpoint_url": url} if url else {}
        cks = {"connect_timeout": 5, "read_timeout": 10}
        if key and secret:
            fs = fsspec.filesystem("s3", key=key, secret=secret,
                                   client_kwargs=ck, config_kwargs=cks)
        else:
            fs = fsspec.filesystem("s3", anon=True,
                                   client_kwargs=ck, config_kwargs=cks)
        root = f"{bucket}/{src.get('s3_folder', '')}"
        return fs, root
    except Exception:
        return None, ""


def _find_sample_zarr_store(fs, root: str) -> Optional[str]:
    """Return path to the first zarr store found under *root* (≤ 2 levels)."""
    try:
        entries = fs.ls(root, detail=False)
    except Exception:
        return None
    for e in entries:
        if e.rstrip("/").endswith(".zarr"):
            return e.rstrip("/")
    # One level deeper
    for entry in entries[:6]:
        try:
            sub = fs.ls(entry, detail=False)
            for e in sub:
                if e.rstrip("/").endswith(".zarr"):
                    return e.rstrip("/")
        except Exception:
            continue
    return None


def _zarr_uncompressed_per_timestep_mb(
    fs,
    store: str,
    eval_vars: list,
    surface_only: bool,
    is_obs: bool = False,
) -> float:
    """Compute uncompressed MB per timestep from zarr consolidated metadata.

    Reads a single ``.zmetadata`` file (a few KB) — no actual data is
    downloaded.  Filters to *eval_vars* when provided and applies a
    ``1/n_depths`` discount when *surface_only* is True.

    For **observation** datasets (*is_obs=True*) each zarr store is one
    evaluation event (a satellite pass / profile set) so no time-axis
    normalisation is applied.  For **gridded** datasets a store may contain
    multiple forecast lead-times and is divided by ``n_time``.
    """
    try:
        import json
        import math
        import numpy as np

        with fs.open(store + "/.zmetadata", "r") as f:
            meta = json.load(f)
        metadata = meta.get("metadata", {})

        var_info: Dict[str, Any] = {}
        for key, val in metadata.items():
            if not key.endswith("/.zarray"):
                continue
            # e.g. "ssha_filtered/.zarray" → var_name = "ssha_filtered"
            var_name = key[:-8].lstrip("/").split("/")[-1]
            shape = val.get("shape", [])
            try:
                itemsize = np.dtype(val.get("dtype", "<f4")).itemsize
            except Exception:
                itemsize = 4
            var_info[var_name] = {"shape": shape, "itemsize": itemsize}

        if not var_info:
            return 0.0

        # Number of timesteps in this store (from the 'time' variable shape).
        time_shape = var_info.get("time", {}).get("shape", [1])
        n_time = max(1, time_shape[0]) if time_shape else 1

        _COORD_VARS = frozenset({
            "time", "lat", "lon", "latitude", "longitude", "depth",
            "num_nadir", "num_pixels", "num_lines", "n_points",
        })

        total_bytes = 0
        for var_name, info in var_info.items():
            if var_name in _COORD_VARS:
                continue
            if eval_vars and var_name not in eval_vars:
                continue
            shape = info["shape"]
            if not shape:
                continue

            sz = math.prod(shape) * info["itemsize"]

            # For gridded datasets, normalise by the number of timesteps in
            # the store (Glonet stores 10 lead-times per file).  For obs
            # datasets the entire file is one evaluation event — do not
            # divide (the time axis is really along-track position, not
            # separate calendar timesteps).
            if not is_obs and n_time > 1 and len(shape) >= 1 and shape[0] == n_time:
                sz //= n_time
                spatial_shape = shape[1:]
            else:
                spatial_shape = shape

            # Surface-only discount: divide by n_depths when depth dim is
            # present (3-D: depth, lat, lon).
            if surface_only and len(spatial_shape) >= 3:
                n_depths = spatial_shape[0]
                if n_depths > 1:
                    sz //= n_depths

            total_bytes += sz

        return max(0.0, total_bytes / 1e6)

    except Exception:
        return 0.0


def _probe_source_mem_gb(
    src: dict,
    surface_only: bool = False,
    catalog_path: Optional[str] = None,
    delta_time_hours: float = 12.0,
) -> Optional[float]:
    """Estimate expected peak RAM per worker (GB) for *src*.

    For **observation** datasets the estimate accounts for the fact that a
    single evaluation window spans ``2 × delta_time_hours`` and may load
    many files simultaneously.  The number of files per window is derived
    from the local catalog JSON (``catalog_path``) using ``date_start`` /
    ``date_end`` — the same mechanism the pipeline uses to select files.
    Falling back to ``n_files=1`` when the catalog is unavailable.

    For **gridded** datasets each worker processes exactly one timestep
    (``n_files=1`` always applies).

    The returned value already includes the safety multiplier so it can be
    used directly as the required memory per worker.
    The probe reads **only** the ``.zmetadata`` file — no ocean data is
    downloaded.
    """
    # Only zarr sources accessible via S3/Wasabi are supported.
    conn = (src.get("connection_type") or "").lower()
    if conn not in ("wasabi", "s3", ""):
        return None
    pattern = (src.get("file_pattern") or "").lower()
    if not pattern.endswith(".zarr"):
        return None

    is_obs: bool = bool(src.get("observation_dataset", False))
    safety = _MEM_SAFETY_OBS if is_obs else _MEM_SAFETY_GRIDDED

    # The user may hard-code a reference size to skip network probing.
    ref_mb = _explicit_ref_file_size_mb(src)
    if ref_mb is not None:
        n_files = 1
        if is_obs and catalog_path:
            n_files = _count_files_per_window(catalog_path, delta_time_hours) or 1
        return max(_PROBE_MIN_MEM_GB, n_files * ref_mb / 1000 * safety)

    from concurrent.futures import ThreadPoolExecutor

    def _do() -> Optional[float]:
        try:
            fs, root = _build_s3_fs(src)
            if fs is None:
                return None
            store = _find_sample_zarr_store(fs, root)
            if store is None:
                return None
            eval_vars = list(src.get("eval_variables") or [])
            mb = _zarr_uncompressed_per_timestep_mb(
                fs, store, eval_vars, surface_only, is_obs=is_obs,
            )
            if mb <= 0:
                return None
            n_files = 1
            if is_obs and catalog_path:
                n_files = _count_files_per_window(catalog_path, delta_time_hours) or 1
            return max(_PROBE_MIN_MEM_GB, n_files * mb / 1000 * safety)
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=1) as pool:
        fut = pool.submit(_do)
        try:
            return fut.result(timeout=_PROBE_TIMEOUT_S)
        except Exception:
            return None


def _explicit_ref_file_size_mb(src: dict) -> Optional[float]:
    """Return the manually-set ``ref_file_size_mb`` for *src*, or ``None``.

    Allows users to pin a known uncompressed file size (in MB) per source
    in the YAML, bypassing the S3 probe entirely::

        - dataset: glonet
          ref_file_size_mb: 658   # uncompressed MB per timestep (per eval_variable)
    """
    val = src.get("ref_file_size_mb")
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _count_files_per_window(catalog_path: str, delta_time_hours: float) -> Optional[int]:
    """Count the 75th-percentile number of obs files that overlap a time window.

    Reads the local GeoJSON catalog (already downloaded to
    ``{data_directory}/catalogs/{dataset}.json``) and, for each unique
    calendar day present in the catalog, counts how many features have a
    ``[date_start, date_end]`` interval that intersects the
    ``[day - delta_time_hours, day + delta_time_hours]`` window.

    Returns the 75th percentile count across all sampled days — robust
    against sparse days at the edges of the dataset.  Returns ``None`` if
    the catalog file does not exist or cannot be parsed.
    """
    import json as _json

    if not os.path.isfile(catalog_path):
        return None

    try:
        with open(catalog_path, "r") as fh:
            cat = _json.load(fh)
    except Exception:
        return None

    features = cat.get("features", [])
    if not features:
        return None

    # Parse date_start / date_end from each feature's properties.
    import pandas as pd

    rows = []
    for feat in features:
        props = feat.get("properties", {}) if isinstance(feat, dict) else {}
        ds = props.get("date_start")
        de = props.get("date_end")
        if not ds or not de:
            continue
        try:
            rows.append((
                pd.Timestamp(ds).tz_localize(None),
                pd.Timestamp(de).tz_localize(None),
            ))
        except Exception:
            continue

    if not rows:
        return None

    delta = pd.Timedelta(hours=delta_time_hours)

    # Sample representative anchor points: one per unique UTC day from the
    # catalogue (up to 30 days spread uniformly to keep startup fast).
    all_days = sorted({r[0].normalize() for r in rows})
    step = max(1, len(all_days) // 30)
    sample_days = all_days[::step]

    counts = []
    for anchor in sample_days:
        w0 = anchor - delta
        w1 = anchor + delta
        n = sum(1 for (ds, de) in rows if ds <= w1 and de >= w0)
        if n > 0:
            counts.append(n)

    if not counts:
        return None

    counts.sort()
    p75_idx = int(len(counts) * 0.75)
    return counts[p75_idx]


# ---------------------------------------------------------------------------
# Budget helpers
# ---------------------------------------------------------------------------

def _compute_budgets(ram: float) -> tuple[float, float]:
    """Return *(driver_reserve_gb, worker_pool_gb)* from total RAM."""
    # Keep at least 4 GB free for the driver process (scheduler, prefetch
    # cache, Python interpreter, OS), and up to 15 % of total RAM.
    driver_reserve_gb = max(4.0, ram * 0.15)
    # Apply an 85 % safety margin so workers don't fill RAM completely
    # and trigger Linux OOM killer.
    worker_pool_gb = max(4.0, (ram - driver_reserve_gb) * 0.85)
    return driver_reserve_gb, worker_pool_gb


# ---------------------------------------------------------------------------
# Per-dataset sizing prescriptions
# ---------------------------------------------------------------------------

def _obs_prescription(
    n_physical: int,
    worker_pool_gb: float,
    file_mem_gb: Optional[float] = None,
) -> Dict[str, Any]:
    """Sizing for I/O-bound observation datasets (SWOT, SARAL, Jason-3, etc.).

    When *file_mem_gb* is provided (from the S3 probe or a manual reference),
    the worker count is capped so that ``n_workers × file_mem_gb`` fits within
    the worker memory pool.  The Dask ``memory_limit_per_worker`` is set to
    the equitable share ``pool / n_workers``, which is always at least
    *file_mem_gb* by construction.
    """
    max_workers = min(n_physical, 8)
    if file_mem_gb and file_mem_gb > 0:
        workers_by_mem = max(1, int(worker_pool_gb / file_mem_gb))
        n_workers = min(max_workers, workers_by_mem)
    else:
        n_workers = max_workers
    # Give each worker its equitable share; respect a sensible ceiling.
    mem_gb = min(max(file_mem_gb or 2.0, worker_pool_gb / n_workers), 16.0)
    return {
        "n_parallel_workers": n_workers,
        "nthreads_per_worker": 2,
        "memory_limit_per_worker": f"{mem_gb:.1f}GB",
    }


def _gridded_prescription(
    n_physical: int,
    worker_pool_gb: float,
    file_mem_gb: Optional[float] = None,
) -> Dict[str, Any]:
    """Sizing for memory-hungry gridded reference datasets (glorys, glonet).

    Each worker processes one full timestep of the 3-D prediction vs. the
    reference field.  When *file_mem_gb* is derived from the S3 probe, the
    worker count is reduced on memory-constrained machines so that the total
    memory usage stays within the worker pool budget.
    """
    max_workers = max(2, min(n_physical // 2, 6))
    if file_mem_gb and file_mem_gb > 0:
        workers_by_mem = max(1, int(worker_pool_gb / file_mem_gb))
        n_workers = min(max_workers, max(1, workers_by_mem))
    else:
        n_workers = max_workers
    mem_gb = min(max(file_mem_gb or 4.0, worker_pool_gb / n_workers), 20.0)
    batch_size = max(3, min(n_workers, 10))
    return {
        "n_parallel_workers": n_workers,
        "nthreads_per_worker": 2,
        "memory_limit_per_worker": f"{mem_gb:.1f}GB",
        "gridded_batch_size": batch_size,
    }


def _obs_batch_size(driver_reserve_gb: float, *, is_swath: bool) -> int:
    """Compute obs_batch_size for one dataset type.

    *obs_batch_size* controls how many evaluation time-windows are batched
    together before the driver preprocesses their observation files.  More
    windows → more unique files → more driver RAM consumed.

    * Swath datasets (SWOT): each file is large (~200 MB uncompressed) → a
      budget of 0.5 GB per window gives conservative batches.
    * Nadir / profile datasets (SARAL, Jason-3, …): files are small (~5 MB)
      → a budget of 0.15 GB per window allows larger batches.
    """
    if is_swath:
        return max(5, min(15, int(driver_reserve_gb / 0.5) + 1))
    else:
        return max(10, min(30, int(driver_reserve_gb / 0.15)))


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def auto_tune_config(
    config: Dict[str, Any],
    data_directory: Optional[str] = None,
) -> Dict[str, Any]:
    """Fill auto-tuned parallelism parameters into *config* in-place.

    Call this directly after loading the raw YAML dict, before converting it
    to an ``argparse.Namespace``.

    Parameters
    ----------
    config:
        Raw dict loaded from the YAML file.  Modified in-place and returned.
    data_directory:
        Path to the runtime output directory (e.g. ``dc1_output/``).  When
        provided, the local catalog JSON files stored at
        ``{data_directory}/catalogs/{dataset_name}.json`` are used to count
        how many observation files fall inside a ``2 × delta_time`` window.
        This is the same catalog the pipeline uses for file selection, so the
        estimate is always consistent.  Falls back to ``n_files=1`` if the
        catalog file does not exist yet.

    Returns
    -------
    Dict
        Same dict, with parallelism parameters filled in where appropriate.
    """
    global_auto: bool = config.get("auto_tune", True)  # default: enabled
    if global_auto is False:
        # Disabled globally: only honour explicit "auto" strings.
        _apply_explicit_autos(config)
        return config

    # ── Hardware detection ────────────────────────────────────────────────
    hw = _detect_hardware()
    n_physical: int = int(hw["n_physical_cores"])
    ram: float = hw["total_ram_gb"]
    driver_reserve_gb, worker_pool_gb = _compute_budgets(ram)

    logger.info(
        f"[auto_tune] Hardware: {n_physical} physical cores, {ram:.0f} GB RAM — "
        f"worker pool budget: {worker_pool_gb:.0f} GB "
        f"(driver reserve: {driver_reserve_gb:.0f} GB)"
    )

    # ── Per-source parameters ─────────────────────────────────────────────
    should_probe: bool = bool(config.get("probe_file_sizes", True))
    surface_only: bool = bool(config.get("surface_only", False))
    # Window width: delta_time is the half-width (hours), so the full
    # window loaded per evaluation anchor is 2 × delta_time.
    delta_time_hours: float = float(config.get("delta_time", 12))

    for src in config.get("sources", []):
        if not isinstance(src, dict):
            continue

        is_obs: bool = bool(src.get("observation_dataset", False))
        alias: str = str(src.get("dataset", "?"))

        # --- File-size-aware memory estimate --------------------------------
        file_mem_gb: Optional[float] = None
        if should_probe:
            # For obs datasets: resolve the local catalog JSON path so the
            # probe can count files per window without any S3 listing.
            catalog_path: Optional[str] = None
            if is_obs and data_directory:
                catalog_path = os.path.join(
                    data_directory, "catalogs", f"{alias}.json"
                )

            file_mem_gb = _probe_source_mem_gb(
                src,
                surface_only=surface_only,
                catalog_path=catalog_path,
                delta_time_hours=delta_time_hours,
            )
            if file_mem_gb is not None:
                logger.info(
                    f"[auto_tune] {alias}: probed ~{file_mem_gb:.1f} GB RAM/worker "
                    f"(safety×{_MEM_SAFETY_OBS if is_obs else _MEM_SAFETY_GRIDDED})"
                )
            else:
                logger.debug(
                    f"[auto_tune] {alias}: file-size probe unavailable, "
                    "using hardware-budget formula"
                )

        presc = (
            _obs_prescription(n_physical, worker_pool_gb, file_mem_gb)
            if is_obs
            else _gridded_prescription(n_physical, worker_pool_gb, file_mem_gb)
        )

        filled: list[str] = []

        # Per-worker Dask sizing
        for key in ("n_parallel_workers", "nthreads_per_worker", "memory_limit_per_worker"):
            if _should_fill(key, src, global_auto) and key in presc:
                src[key] = presc[key]
                filled.append(f"{key}={presc[key]}")

        # gridded_batch_size (only for gridded reference datasets)
        if not is_obs:
            key = "gridded_batch_size"
            if _should_fill(key, src, global_auto):
                src[key] = presc[key]
                filled.append(f"{key}={presc[key]}")

        # Per-source obs_batch_size (only for observation datasets)
        if is_obs:
            key = "obs_batch_size"
            if _should_fill(key, src, global_auto):
                # Detect swath datasets: they list num_lines / num_pixels as
                # keep_variables (2-D swath grid flattened to n_points).
                kv = src.get("keep_variables") or []
                is_swath = any(v in kv for v in ("num_lines", "num_pixels"))
                bs = _obs_batch_size(driver_reserve_gb, is_swath=is_swath)
                src[key] = bs
                filled.append(f"{key}={bs}")

        if filled:
            logger.debug(f"[auto_tune] {alias}: {', '.join(filled)}")

    # ── Global obs_batch_size (used as fallback for sources without a
    #    per-source value, and as the default for non-observation batches) ──
    key = "obs_batch_size"
    if _should_fill(key, config, global_auto):
        bs = _obs_batch_size(driver_reserve_gb, is_swath=False)
        config[key] = bs
        logger.debug(f"[auto_tune] global obs_batch_size={bs}")

    return config


# ---------------------------------------------------------------------------
# Internal: handle auto_tune: false (only explicit "auto" strings)
# ---------------------------------------------------------------------------

def _apply_explicit_autos(config: Dict[str, Any]) -> None:
    """Fill only params explicitly set to the string ``'auto'``.

    Called when ``auto_tune: false`` in the YAML; respects the user's wish
    to manage most parameters manually while still allowing selective opt-in
    via ``"auto"`` values.
    """
    hw = _detect_hardware()
    n_physical: int = int(hw["n_physical_cores"])
    ram: float = hw["total_ram_gb"]
    driver_reserve_gb, worker_pool_gb = _compute_budgets(ram)

    obs_presc = _obs_prescription(n_physical, worker_pool_gb)
    grid_presc = _gridded_prescription(n_physical, worker_pool_gb)

    for src in config.get("sources", []):
        if not isinstance(src, dict):
            continue
        is_obs = bool(src.get("observation_dataset", False))
        presc = obs_presc if is_obs else grid_presc

        for key in ("n_parallel_workers", "nthreads_per_worker", "memory_limit_per_worker"):
            if _is_auto_sentinel(src.get(key)) and key in src and key in presc:
                src[key] = presc[key]

        if not is_obs and _is_auto_sentinel(src.get("gridded_batch_size")) and "gridded_batch_size" in src:
            src["gridded_batch_size"] = presc["gridded_batch_size"]

        if is_obs and _is_auto_sentinel(src.get("obs_batch_size")) and "obs_batch_size" in src:
            kv = src.get("keep_variables") or []
            is_swath = any(v in kv for v in ("num_lines", "num_pixels"))
            src["obs_batch_size"] = _obs_batch_size(driver_reserve_gb, is_swath=is_swath)

    if _is_auto_sentinel(config.get("obs_batch_size")) and "obs_batch_size" in config:
        config["obs_batch_size"] = _obs_batch_size(driver_reserve_gb, is_swath=False)
