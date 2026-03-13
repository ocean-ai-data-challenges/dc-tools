"""Worker-side compute helpers: dataset caching, timeout wrapper, memory parsing.

These functions are called from within Dask worker tasks (primarily
``compute_metric``) to safely open datasets, enforce timeouts, and parse
human-readable memory limits.
"""

import os
import re
import threading
from typing import Any, Callable, Optional

import xarray as xr

from dctools.metrics.worker_cleanup import (
    _WORKER_DATASET_CACHE,
    _WORKER_DATASET_CACHE_LOCK,
)


# ---------------------------------------------------------------------------
# Memory-limit parsing
# ---------------------------------------------------------------------------
def _parse_memory_limit(value: Any) -> int:
    """Parse a human-readable memory string (e.g. ``"6GB"``) into bytes.

    Supports units: B, KB, MB, GB, TB (case-insensitive).
    If *value* is already numeric it is returned as-is.
    """
    if isinstance(value, (int, float)):
        return int(value)
    _s = str(value).strip().upper()
    _m = re.match(r"^([\d.]+)\s*(TB|GB|MB|KB|B)?$", _s)
    if not _m:
        raise ValueError(f"Cannot parse memory limit: {value!r}")
    _num = float(_m.group(1))
    _unit = (_m.group(2) or "B")
    _multipliers = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3, "TB": 1024**4}
    return int(_num * _multipliers[_unit])


# ---------------------------------------------------------------------------
# Per-worker LRU dataset opener
# ---------------------------------------------------------------------------
def _open_dataset_worker_cached(
    open_func: Callable[[str], Optional[xr.Dataset]],
    source: str,
) -> tuple[Optional[xr.Dataset], bool]:
    """Open a dataset with a small per-worker LRU cache.

    This primarily targets remote Zarr datasets (S3/Wasabi) where repeated
    open calls are expensive (metadata reads). Cache size can be tuned via
    `DCTOOLS_WORKER_DATASET_CACHE_SIZE`.
    """
    try:
        cache_size = int(os.environ.get("DCTOOLS_WORKER_DATASET_CACHE_SIZE", "4"))
    except Exception:
        cache_size = 4

    if cache_size <= 0:
        return open_func(source), False

    with _WORKER_DATASET_CACHE_LOCK:
        cached = _WORKER_DATASET_CACHE.get(source)
        if cached is not None:
            _WORKER_DATASET_CACHE.move_to_end(source)
            return cached, True

    ds = open_func(source)
    if ds is None:
        return None, False

    with _WORKER_DATASET_CACHE_LOCK:
        existing = _WORKER_DATASET_CACHE.get(source)
        if existing is not None:
            _WORKER_DATASET_CACHE.move_to_end(source)
            return existing, True

        _WORKER_DATASET_CACHE[source] = ds
        _WORKER_DATASET_CACHE.move_to_end(source)
        while len(_WORKER_DATASET_CACHE) > cache_size:
            _, evicted = _WORKER_DATASET_CACHE.popitem(last=False)
            try:
                if hasattr(evicted, "close"):
                    evicted.close()
            except Exception:
                pass
    return ds, False


# ---------------------------------------------------------------------------
# Timeout-guarded compute
# ---------------------------------------------------------------------------
def _compute_with_timeout(
    arr: "xr.Dataset",
    timeout_s: int = 90,
    **kwargs: Any,
) -> "xr.Dataset":
    """Run ``arr.compute()`` in a daemon thread with a hard Python timeout.

    When the Dask synchronous scheduler is used from inside a Dask worker,
    the underlying ``aiobotocore`` (asyncio) ``read_timeout`` may never fire:
    its cancellation coroutine is registered in an event loop that is not
    progressing while the calling thread is blocked.  This wrapper guarantees
    the call cannot hang indefinitely regardless of the S3 client's internal
    timeout settings.
    """
    _result: list = [None]
    _exc: list = [None]
    _done = threading.Event()

    def _run() -> None:
        try:
            _result[0] = arr.compute(**kwargs)
        except Exception as _e:  # noqa: BLE001
            _exc[0] = _e
        finally:
            _done.set()

    _t = threading.Thread(target=_run, daemon=True)
    _t.start()
    if not _done.wait(timeout=timeout_s):
        # The daemon thread is still blocking on S3 — the OS will kill it at
        # process exit.  Raise so the task is marked as failed quickly.
        raise RuntimeError(
            f"arr.compute() timed out after {timeout_s}s. "
            "Likely cause: aiobotocore async read_timeout not firing in "
            "synchronous Dask-scheduler context (stale S3 connection). "
            "The task will be skipped by the watchdog and retried next run."
        )
    if _exc[0] is not None:
        raise _exc[0]
    return _result[0]  # type: ignore[no-any-return]
