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

import gc
import gzip
import json
import math
import os
import re
import time as _time
import warnings
from argparse import Namespace
from datetime import timedelta
from glob import glob
from pathlib import Path
from textwrap import wrap
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
from dctools.utilities.misc_utils import make_serializable, transform_in_place

import dcleaderboard as _dcleaderboard
from rich import progress as _rich_progress
warnings.simplefilter("ignore", UserWarning)

# Fast JSON backend: orjson is 5–10× faster than stdlib json for large
# files. Used in post-processing (batch consolidation + per-bins JSONL).
try:
    import orjson as _orjson
    def _json_loads(s: Any) -> Any:
        return _orjson.loads(s)
    def _json_load(fp: Any) -> Any:
        return _orjson.loads(fp.read())
    def _json_dumps_compact(obj: Any) -> str:
        return _orjson.dumps(obj).decode()
except ImportError:  # graceful degradation if orjson is unavailable
    _orjson = None  # type: ignore[assignment]
    def _json_loads(s: Any) -> Any:  # type: ignore[misc]
        return json.loads(s)
    def _json_load(fp: Any) -> Any:  # type: ignore[misc]
        return json.load(fp)
    def _json_dumps_compact(obj: Any) -> str:  # type: ignore[misc]
        return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


# Stable short-alias dictionaries for the compact .jsonl.gz format (v2).
# These are intentionally module-level constants so both the writer (here)
# and the reader (dcleaderboard.map_processing) share the same definition.
#   _PB_FIELD_ALIASES  : short --> full name for top-level entry fields
#   _PB_COORD_ALIASES  : short --> full name for columnar coordinate keys
_PB_FIELD_ALIASES: Dict[str, str] = {
    "ra": "ref_alias",
    "rt": "ref_type",
    "lt": "lead_time",
    "ft": "forecast_reference_time",
    "pb": "per_bins",
}
_PB_COORD_ALIASES: Dict[str, str] = {
    "yl": "lat_l",
    "yr": "lat_r",
    "xl": "lon_l",
    "xr": "lon_r",
    "dl": "depth_l",
    "dr": "depth_r",
}
_PB_FIELD_SHORT: Dict[str, str] = {v: k for k, v in _PB_FIELD_ALIASES.items()}
_PB_COORD_SHORT: Dict[str, str] = {v: k for k, v in _PB_COORD_ALIASES.items()}
_PB_COORD_DP: int = 2   # decimal places for bin boundary coordinates
_PB_METRIC_DP: int = 6  # decimal places for metric values
_PB_SKIP: frozenset = frozenset({
    "lat_bin", "lon_bin", "depth_bin",
    "count", "n", "n_points", "time_bin",
})

_GEO_LABEL_PAT = re.compile(r'^(\d+(?:\.\d+)?[SNWEsnwe]|0)-(\d+(?:\.\d+)?[SNWEsnwe]|0)$')


def _fmt_elapsed(t0: float) -> str:
    """Return a human-readable elapsed-time string since *t0* (monotonic)."""
    s = int(_time.monotonic() - t0)
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m {s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m {s:02d}s"


def _parse_geo_label(label: str) -> Dict[str, float]:
    """Parse a human-readable geographic bin label back to a {left, right} dict.

    Handles labels produced by oceanbench's ``_lat_bin_label`` / ``_lon_bin_label``
    helpers such as ``"78S-74S"`` (--> ``{"left": -78.0, "right": -74.0}``) or
    ``"180W-176W"`` (--> ``{"left": -180.0, "right": -176.0}``).  The special
    label ``"global"`` is treated as ``{"left": -180.0, "right": 180.0}`` for
    geographic bins; it is handled gracefully but will not produce a meaningful
    bin boundary.
    """
    if label == "global":
        return {"left": -180.0, "right": 180.0}

    def _side(s: str) -> float:
        if s == "0":
            return 0.0
        if s[-1].upper() in ("S", "W"):
            return -float(s[:-1])
        return float(s[:-1])  # N or E

    m = _GEO_LABEL_PAT.match(label)
    if m:
        return {"left": _side(m.group(1)), "right": _side(m.group(2))}
    # Fallback: return zeros rather than raising so the JSONL aggregation
    # can continue; affected bins will aggregate to (0, 0) boundaries.
    logger.warning(f"_parse_geo_label: unrecognised label {label!r}, defaulting to 0")
    return {"left": 0.0, "right": 0.0}


def _pb_raw_to_v2_row(
    ref_alias: str,
    ref_type: str,
    lead_time: Any,
    forecast_reference_time: str,
    per_bins_raw: Dict,
) -> Optional[Dict]:
    """Convert a single raw per-bins item to a v2 columnar row dict.

    Input *per_bins_raw* format (row-based, one dict per bin)::

        {"var": [{"lat_bin": {"left": -90, "right": -86}, "lon_bin": {…},
                  "rmse": 0.42, …}, …], …}

    Output per_bins format (columnar, short-aliased keys)::

        {"var": {"yl": [-90, …], "yr": [-86, …], "xl": […], "xr": […],
                 "rmse": [0.42, …], …}, …}

    Bin boundaries are rounded to ``_PB_COORD_DP`` decimal places.
    Metric values are rounded to ``_PB_METRIC_DP`` decimal places.
    ``None`` metric values (after :func:`nan_to_none`) are preserved as ``None``.
    Both the ``{"left", "right"}`` dict format and the oceanbench string-label
    format (e.g. ``"78S-74S"``) are accepted for coordinate bins.

    Returns ``None`` when *per_bins_raw* is empty or all variables have no bins,
    so callers can skip writing that row.  Agnostic to which reference datasets
    are present — works for any combination of gridded and observation refs.
    """
    per_bins_out: Dict[str, Dict] = {}
    for var, bins in per_bins_raw.items():
        if not bins:
            continue
        yl: List = []
        yr: List = []
        xl: List = []
        xr: List = []
        dl: List = []
        dr: List = []
        has_depth = False
        for b in bins:
            lb = b["lat_bin"]
            lob = b["lon_bin"]
            if isinstance(lb, str):
                lb = _parse_geo_label(lb)
            if isinstance(lob, str):
                lob = _parse_geo_label(lob)
            yl.append(round(lb["left"], _PB_COORD_DP))
            yr.append(round(lb["right"], _PB_COORD_DP))
            xl.append(round(lob["left"], _PB_COORD_DP))
            xr.append(round(lob["right"], _PB_COORD_DP))
            if "depth_bin" in b:
                has_depth = True
                db = b["depth_bin"]
                dl.append(round(db["left"], _PB_COORD_DP))
                dr.append(round(db["right"], _PB_COORD_DP))
        col: Dict[str, List] = {"yl": yl, "yr": yr, "xl": xl, "xr": xr}
        if has_depth:
            col["dl"] = dl
            col["dr"] = dr
        all_metrics: set = {k for b in bins for k in b if k not in _PB_SKIP}
        for metric in sorted(all_metrics):
            col[metric] = [
                # Use ``m == m`` to detect NaN (NaN != NaN) so that NaN metric
                # values are stored as JSON null rather than the non-standard
                # "NaN" literal that would cause orjson to raise TypeError.
                None if (m := b.get(metric)) is None or m != m
                else round(float(m), _PB_METRIC_DP)
                for b in bins
            ]
        per_bins_out[var] = col
    if not per_bins_out:
        return None
    return {
        "ra": ref_alias,
        "rt": ref_type,
        "lt": lead_time,
        "ft": forecast_reference_time,
        "pb": per_bins_out,
    }


def _read_batch_file(path: str) -> Any:
    """Read and JSON-parse a batch result file (gzip or plain)."""
    if path.endswith(".gz"):
        with gzip.open(path, "rb") as f:
            return _json_load(f)
    with open(path, "rb") as f:
        return _json_load(f)


def _prefetch_iter(paths, max_ahead: int = 2):
    """Yield ``(path, parsed_data)`` with look-ahead parsing in background threads.

    .. deprecated::
        Replaced by :func:`_iter_batch_items` for the consolidation loop.
        Kept for backward compatibility.

    Peak extra memory: *max_ahead* × one parsed batch (~500 MB each).
    """
    from collections import deque as _deque
    from concurrent.futures import ThreadPoolExecutor as _TPE

    if not paths:
        return
    n_ahead = min(max_ahead, len(paths))
    with _TPE(max_workers=n_ahead) as pool:
        buf: _deque = _deque()
        it = iter(paths)
        for _ in range(min(n_ahead + 1, len(paths))):
            p = next(it, None)
            if p is not None:
                buf.append((p, pool.submit(_read_batch_file, p)))
        while buf:
            p, fut = buf.popleft()
            nxt = next(it, None)
            if nxt is not None:
                buf.append((nxt, pool.submit(_read_batch_file, nxt)))
            yield p, fut.result()


def _iter_batch_items(path: str):
    """Yield individual items from a batch file (JSON array), one at a time.

    Uses ``orjson`` (Rust-based, iterative parser) when available to avoid the
    C stack overflow that CPython's ``json.raw_decode`` triggers on large
    per_bins objects (~164 MB each, ~892 MB decompressed per glorys batch).
    CPython's recursive C scanner overflows its stack on deeply nested 160+ MB
    JSON objects, causing a SIGSEGV (exit 245).

    Peak memory: raw bytes buffer + all parsed items ≈ 1.8–2.5 GB for the
    largest glorys batches.  Acceptable given the cluster has been shut down
    before post-processing runs.
    """
    try:
        import orjson as _json_lib  # type: ignore[import-not-found]
        _use_orjson = True
    except ImportError:
        _json_lib = None  # type: ignore[assignment]
        _use_orjson = False

    if _use_orjson:
        # orjson operates on bytes — read raw, skip text decode overhead.
        if path.endswith(".gz"):
            with gzip.open(path, "rb") as f:
                raw = f.read()
        else:
            with open(path, "rb") as f:
                raw = f.read()
        data = _json_lib.loads(raw)  # type: ignore[union-attr]
        del raw
        if not isinstance(data, list):
            return
        # Yield items and release the backing list as we go to cap peak RAM.
        for _i in range(len(data)):
            yield data[_i]
            data[_i] = None  # type: ignore[index]  # release parsed item
        return

    # ── Fallback: CPython json with incremental raw_decode ────────────────
    # Only used when orjson is not installed.  May SIGSEGV on per_bins items
    # > ~160 MB due to C stack overflow in the recursive scanner.
    if path.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            text = f.read()
    else:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

    decoder = json.JSONDecoder()
    idx = 0
    n = len(text)

    # Skip to opening '['
    while idx < n and text[idx] != "[":
        idx += 1
    if idx >= n:
        return
    idx += 1  # skip '['

    while idx < n:
        # Skip whitespace and commas between items
        while idx < n and text[idx] in " \t\n\r,":
            idx += 1
        if idx >= n or text[idx] == "]":
            break
        item, end = decoder.raw_decode(text, idx)
        idx = end
        yield item


def _aggregate_per_bins_jsonl(raw_path: str) -> Optional[str]:
    """Convert a legacy raw per-bins JSONL file to compact gzip JSONL (format v2).

    **Legacy fallback only.**  New evaluations write directly to ``.jsonl.gz`` v2
    via :func:`_pb_raw_to_v2_row` during consolidation and never produce a raw
    ``.jsonl`` intermediate file.  This function exists to migrate older ``.jsonl``
    files produced by previous versions of the pipeline.

    Processes one line at a time — O(1) RAM regardless of file size or temporal
    extent of the evaluation.

    Returns the path to the new ``.jsonl.gz`` file, or ``None`` if the raw file
    did not exist or contained no per-bins entries.
    """
    if not os.path.isfile(raw_path):
        return None

    out_path = raw_path + ".gz"
    _schema = json.dumps(
        {"_v": 2, "f": _PB_FIELD_ALIASES, "c": _PB_COORD_ALIASES},
        separators=(",", ":"), ensure_ascii=False,
    )
    n_written = 0
    with open(raw_path, encoding="utf-8") as fh, \
         gzip.open(out_path, "wt", encoding="utf-8", compresslevel=1) as gz:
        gz.write(_schema + "\n")
        for raw_line in fh:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            entry = _json_loads(raw_line)
            per_bins_raw = entry.get("per_bins")
            if not per_bins_raw:
                continue
            row = _pb_raw_to_v2_row(
                ref_alias=entry.get("ref_alias", ""),
                ref_type=entry.get("ref_type", "gridded"),
                lead_time=entry.get("lead_time"),
                forecast_reference_time=entry.get("forecast_reference_time", ""),
                per_bins_raw=per_bins_raw,
            )
            if row is not None:
                gz.write(_json_dumps_compact(row) + "\n")
                n_written += 1
    if n_written == 0:
        logger.debug("  _aggregate_per_bins_jsonl: no entries found in {}", raw_path)
        try:
            os.remove(out_path)
        except OSError:
            pass
        return None
    logger.opt(colors=True).info(
        f"  Converted per-bins: <b>{n_written}</b> entr"
        f"{'y' if n_written == 1 else 'ies'}  <dim>→</dim>  <cyan>{out_path}</cyan>"
    )
    return out_path


class BaseDCEvaluation:
    """Base class for evaluation orchestration.

    Subclasses are expected to:
    - define `self.dataset_references` (pred -> list[ref])
    - create `self.dataset_processor` as needed
    - implement a `run_eval()` method
    """

    CHALLENGE_NAME = "DC"

    def __init__(self, arguments: Namespace) -> None:
        self.args = arguments
        self.results_directory = os.path.join(self.args.data_directory, "results")
        os.makedirs(self.results_directory, exist_ok=True)
        self._startup_summary_logged = False
        self._startup_dask_cfg: Dict[str, Any] = {}
        self.surface_only: bool = getattr(arguments, "surface_only", False)
        self.target_dimensions = get_target_dimensions(
            self.args, surface=self.surface_only,
        )
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
        # Propagate optional C-library thread cap (pyinterp, BLAS, OpenMP).
        if "c_lib_threads" in source:
            cfg["c_lib_threads"] = int(source["c_lib_threads"])
        # Propagate optional download concurrency cap.
        if "download_workers" in source:
            cfg["download_workers"] = int(source["download_workers"])
        # Propagate optional look-ahead controls.
        if "lookahead_depth" in dask_cfg or "lookahead_depth" in source:
            cfg["lookahead_depth"] = max(
                int(dask_cfg.get("lookahead_depth", source.get("lookahead_depth", 0))),
                0,
            )
        if "lookahead_pred_workers" in dask_cfg or "lookahead_pred_workers" in source:
            cfg["lookahead_pred_workers"] = max(
                int(
                    dask_cfg.get(
                        "lookahead_pred_workers",
                        source.get("lookahead_pred_workers", 1),
                    )
                ),
                1,
            )
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

    def _box_line(self, text: str, width: int, center: bool = False) -> str:
        """Format *text* inside a fixed-width Unicode box line."""
        if center:
            clipped = text[:width]
            return f"║{clipped.center(width)}║"

        usable = max(width - 2, 0)
        clipped = text[:usable]
        return f"║  {clipped.ljust(usable)}║"

    def _append_wrapped_box_lines(
        self,
        rows: List[str],
        label: str,
        value: Any,
        width: int,
    ) -> None:
        """Append wrapped ``label: value`` lines to the startup summary box."""
        value_text = "?" if value in (None, "") else str(value)
        prefix = f"{label:<14} "
        inner_width = max(width - 2, 10)
        wrapped_lines = wrap(
            f"{prefix}{value_text}",
            width=inner_width,
            subsequent_indent=" " * len(prefix),
            break_long_words=False,
            break_on_hyphens=False,
        ) or [prefix.rstrip()]
        rows.extend(self._box_line(line, width) for line in wrapped_lines)

    def _build_eval_summary_banner(self) -> str:
        """Build a one-shot startup summary banner for the evaluation run."""
        width = 78
        top = f"╔{'═' * width}╗"
        bottom = f"╚{'═' * width}╝"
        sep = f"╟{'─' * width}╢"
        blank = f"║{' ' * width}║"

        challenge_name = getattr(self, "CHALLENGE_NAME", self.__class__.__name__) or "DC"
        args = getattr(self, "args", Namespace())
        dataset_references = getattr(self, "dataset_references", {}) or {}
        all_datasets = sorted(set(getattr(self, "all_datasets", []) or []))
        dask_cfg = getattr(self, "_startup_dask_cfg", {}) or {}

        data_directory = getattr(args, "data_directory", "?")
        results_directory = getattr(
            self,
            "results_directory",
            os.path.join(data_directory, "results") if data_directory != "?" else "?",
        )

        dask_parts = [
            f"{dask_cfg.get('n_workers', '?')} worker(s)",
            f"{dask_cfg.get('threads_per_worker', '?')} thread(s)/worker",
        ]
        if dask_cfg.get("memory_limit"):
            dask_parts.append(f"{dask_cfg['memory_limit']}/worker")

        dataset_processor = getattr(self, "dataset_processor", None)
        client = getattr(dataset_processor, "client", None)
        dashboard_link = getattr(client, "dashboard_link", None)

        rows: List[str] = [top, blank]
        rows.append(self._box_line(f"{challenge_name} Evaluation Summary", width, center=True))
        rows.append(blank)
        rows.append(sep)
        self._append_wrapped_box_lines(
            rows,
            "Period",
            f"{getattr(args, 'start_time', '?')} -> {getattr(args, 'end_time', '?')}",
            width,
        )
        self._append_wrapped_box_lines(
            rows,
            "Forecast",
            f"{getattr(args, 'n_days_forecast', '?')} day(s)",
            width,
        )
        self._append_wrapped_box_lines(
            rows,
            "Interval",
            f"{getattr(args, 'n_days_interval', '?')} day(s)",
            width,
        )
        self._append_wrapped_box_lines(rows, "Data dir", data_directory, width)
        self._append_wrapped_box_lines(rows, "Results dir", results_directory, width)
        self._append_wrapped_box_lines(rows, "Dask", ", ".join(dask_parts), width)
        if dashboard_link:
            self._append_wrapped_box_lines(rows, "Dashboard", dashboard_link, width)

        rows.append(sep)
        models = list(dataset_references.keys())
        rows.append(self._box_line(f"Models to evaluate ({len(models)})", width))
        if models:
            for model in models:
                refs = dataset_references.get(model) or []
                refs_text = ", ".join(refs) if refs else "(no references)"
                self._append_wrapped_box_lines(rows, f"- {model}", refs_text, width)
        else:
            rows.append(self._box_line("No prediction dataset configured.", width))

        rows.append(sep)
        self._append_wrapped_box_lines(
            rows,
            "Datasets",
            ", ".join(all_datasets) if all_datasets else "(none)",
            width,
        )

        max_cache_files = getattr(args, "max_cache_files", None)
        if max_cache_files not in (None, ""):
            rows.append(sep)
            self._append_wrapped_box_lines(rows, "Max cache", max_cache_files, width)
        else:
            rows.append(blank)
        rows.append(bottom)
        return "\n".join(rows)

    def _log_startup_summary_once(self) -> None:
        """Log the startup summary exactly once for the whole evaluation."""
        if getattr(self, "_startup_summary_logged", False):
            return

        logger.opt(colors=True).info(
            f"\n<bold><magenta>{self._build_eval_summary_banner()}</magenta></bold>"
        )
        self._startup_summary_logged = True

    def _log_dask_dashboard_once(self) -> None:
        """Backward-compatible wrapper for the startup summary banner."""
        self._log_startup_summary_once()

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

        logger.info(
            f"Init DatasetProcessor with: Workers={n_parallel_workers}, "
            f"Threads={nthreads_per_worker}, MemLimit={memory_limit_per_worker}"
        )

        self._startup_dask_cfg = {
            "n_workers": n_parallel_workers,
            "threads_per_worker": nthreads_per_worker,
            "memory_limit": memory_limit_per_worker,
        }

        self._configure_thread_caps_env(threads="1")

        self.dataset_processor = DatasetProcessor(
            distributed=True,
            n_workers=n_parallel_workers,
            threads_per_worker=nthreads_per_worker,
            memory_limit=memory_limit_per_worker,
        )

        self._configure_dataset_processor_workers()
        self._log_startup_summary_once()

    # ------------------------------------------------------------------
    # Transform setup
    # ------------------------------------------------------------------
    def setup_transforms(
        self,
        dataset_manager: MultiSourceDatasetManager,
        aliases: List[str],
    ) -> Dict[str, Any]:
        """Configure and return the transform dict for all *aliases*.

        When ``self.args.surface_only`` is *True*, every non-GLORYS dataset
        uses the ``standardize_to_surface`` transform (selects the surface
        depth level only).  This is the standard mode for 2-D challenges
        such as DC1.
        """
        surface_only: bool = getattr(self, "surface_only", False)
        transforms_dict = {}
        for alias in aliases:
            kwargs: Dict[str, Any] = {"reduce_precision": self.args.reduce_precision}
            # Some datasets need regridder weights (e.g. glorys interpolation).
            regridder_weights = getattr(self.args, "regridder_weights", None)
            if regridder_weights is not None and alias == "glorys_cmems":
                kwargs["regridder_weights"] = regridder_weights

            if surface_only and alias != "glorys_cmems":
                kwargs["transform_name"] = "standardize_to_surface"

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
        _t0 = _time.monotonic()  # wall-clock start for elapsed-time display
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

        _resume = getattr(self.args, "resume", False)
        _skip_existing_post = bool(getattr(self.args, "skip_existing_postprocessing", False))

        for alias in dataset_references.keys():
            dataset_json_path = os.path.join(self.results_directory, f"results_{alias}.json")
            results_files_dir = os.path.join(self.args.data_directory, "results_batches")

            if _resume:
                # Resume mode: keep existing batch files, only remove stale
                # final results (they will be re-consolidated at the end).
                os.makedirs(results_files_dir, exist_ok=True)
                logger.info("Resume mode: keeping existing batch result files.")
            elif os.path.isdir(results_files_dir):
                if os.listdir(results_files_dir):
                    logger.debug("Results dir exists. Removing old results files.")
                    empty_folder(results_files_dir, extension=".json")
                    # Also remove compressed batch files (.json.gz) left by
                    # previous runs — empty_folder only matches .json suffix.
                    for _stale_gz in Path(results_files_dir).glob("*.json.gz"):
                        _stale_gz.unlink(missing_ok=True)
            else:
                os.makedirs(results_files_dir, exist_ok=True)

            # Remove stale final results unless explicit skip-existing mode is
            # enabled.  In skip-existing mode we preserve prior post-processing
            # artefacts so a resumed run can bypass expensive consolidation.
            if _skip_existing_post:
                logger.debug(
                    "skip_existing_postprocessing=true: preserving existing post-processing files."
                )
            else:
                _stale_final = os.path.join(self.results_directory, f"results_{alias}.json")
                _stale_pb = os.path.join(self.results_directory, f"results_{alias}_per_bins.jsonl.gz")
                for _sf in (_stale_final, _stale_pb):
                    if os.path.isfile(_sf):
                        logger.debug(f"Removing stale result file: {os.path.basename(_sf)}")
                        os.remove(_sf)

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

            _inner_model = f"  ▶  Model {_n_pred_current}/{_n_pred_total}  —  {alias.upper()}"
            _inner_model = f"{_inner_model:<68}"
            logger.opt(colors=True).info(
                f"\n<yellow>┌{'─' * 68}┐\n"
                f"│<bold>{_inner_model}</bold>│\n"
                f"└{'─' * 68}┘</yellow>"
            )

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
                logger.info(
                    f"[Metrics] {alias} vs {ref_alias}: "
                    f"pred_metrics={metrics_names[alias]}, "
                    f"ref_metrics={metrics_names[ref_alias]}, "
                    f"common={common_metrics}"
                )
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
                max_p_memory_increase=getattr(self.args, "max_p_memory_increase", 0.2),
                max_worker_memory_fraction=getattr(self.args, "max_worker_memory_fraction", 0.85),
                resume=_resume,
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

            # ── Release Dask cluster before post-processing ────────────────
            # The cluster workers (e.g. 6 × 3 GB = 18 GB) are no longer needed
            # after all batches are processed.  Shutting down now frees that RAM
            # for the consolidation and leaderboard pipelines, which run on the
            # driver only.  Without this, the combined memory pressure (idle
            # workers + driver post-processing) exceeds available RAM and
            # triggers SIGKILL (exit code 247).
            try:
                self.close()
                gc.collect()
                logger.debug("Dask cluster shut down — RAM freed for post-processing.")
            except Exception as _close_exc:
                logger.debug(f"Cluster shutdown before post-processing: {_close_exc!r}")

            # ── Separator: evaluation done -> post-processing ──────────────
            _inner_post = f"  📦  POST-PROCESSING  —  {alias.upper()}"
            _inner_post = f"{_inner_post:<68}"  # pre-pad before markup
            logger.opt(colors=True).info(
                f"\n<magenta>┌{'─' * 68}┐\n"
                f"│<bold>{_inner_post}</bold>│\n"
                f"└{'─' * 68}┘</magenta>  <dim>[+{_fmt_elapsed(_t0)}]</dim>"
            )

            # Optional fast-path for resumed runs: if the post-processing
            # outputs already exist, skip consolidation for this dataset.
            _existing_final = os.path.join(self.results_directory, f"results_{alias}.json")
            _existing_pb_candidates = (
                os.path.join(self.results_directory, f"results_{alias}_per_bins.jsonl.gz"),
                os.path.join(self.results_directory, f"results_{alias}_per_bins.jsonl"),
                os.path.join(self.results_directory, f"results_{alias}_per_bins.json"),
            )
            _has_final = os.path.isfile(_existing_final) and os.path.getsize(_existing_final) > 0
            _has_per_bins = any(os.path.isfile(_p) and os.path.getsize(_p) > 0 for _p in _existing_pb_candidates)
            if _skip_existing_post and _has_final and _has_per_bins:
                logger.opt(colors=True).info(
                    "  <dim>└</dim>  <yellow>↷  Skipping post-processing:</yellow>"
                    f" existing artefacts found for <cyan>{alias}</cyan>"
                )
                continue

            # Aggregate batch results and write final JSON.
            try:
                batch_files = sorted(
                    glob(os.path.join(results_files_dir, "results_*_batch_*.json"))
                    + glob(os.path.join(results_files_dir, "results_*_batch_*.json.gz"))
                )
                n_errors = 0
                _total_entries = 0

                logger.opt(colors=True).info(
                    f"  <dim>┌</dim> Consolidating <b>{len(batch_files)}</b> batch file(s)"
                    f" for <cyan>'{alias}'</cyan> …"
                )

                # Per-bins are written directly to gzip JSONL (format v2) via
                # _pb_raw_to_v2_row().  No intermediate raw .jsonl is ever
                # created: peak RAM is bounded to one batch item regardless of
                # temporal extent or which reference datasets are evaluated.
                per_bins_gz_path = os.path.join(
                    self.results_directory,
                    f"results_{alias}_per_bins.jsonl.gz",
                )
                _pb_count = 0
                _pb_schema = json.dumps(
                    {"_v": 2, "f": _PB_FIELD_ALIASES, "c": _PB_COORD_ALIASES},
                    separators=(",", ":"), ensure_ascii=False,
                )

                # ── Streaming JSON writer ──────────────────────────────────
                # Items are parsed one at a time from each batch file via
                # _iter_batch_items() + raw_decode.  Only ONE parsed item
                # lives in memory at a time (~400 MB for large per_bins
                # items).  The decompressed text buffer (~1 GB) is the only
                # other significant allocation per batch file.
                # Previous approach: _prefetch_iter loaded up to 3 full
                # batches as Python objects — 9–12 GB peak.
                _first_item = True
                with open(dataset_json_path, "w", encoding="utf-8") as json_file, \
                     gzip.open(per_bins_gz_path, "wt", encoding="utf-8", compresslevel=1) as pb_gz_file:
                    pb_gz_file.write(_pb_schema + "\n")

                    # Write JSON header
                    json_file.write("{\n")
                    json_file.write(f'  "dataset": {json.dumps(alias)},\n')
                    json_file.write(f'  "results": {{\n')
                    json_file.write(f'    {json.dumps(alias)}: [\n')

                    for batch_file in _rich_progress.track(
                        batch_files,
                        description=f"  [{alias}] consolidating …",
                        total=len(batch_files),
                        transient=True,
                    ):
                        for item in _iter_batch_items(batch_file):
                            if isinstance(item, dict) and item.get("error"):
                                n_errors += 1
                            # Pop per_bins and convert to v2 columnar format
                            # immediately.  The per_bins dict (~400 MB) is
                            # freed as soon as _pb_raw_to_v2_row returns.
                            if isinstance(item, dict) and "per_bins" in item:
                                _is_obs = item.get(
                                    "ref_is_observation",
                                    item.get("is_class4", False),
                                )
                                _v2_row = _pb_raw_to_v2_row(
                                    ref_alias=item.get("ref_alias", alias),
                                    ref_type="observation" if _is_obs else "gridded",
                                    lead_time=item.get("lead_time"),
                                    forecast_reference_time=item.get(
                                        "forecast_reference_time", ""
                                    ),
                                    # Batch files are JSON: all types are already basic
                                    # Python types (no pd.Interval / np.ndarray etc.).
                                    # make_serializable is a pure no-op here and would
                                    # waste ~5 s per item traversing 750 K bin entries.
                                    # NaN→None handling is done inline inside
                                    # _pb_raw_to_v2_row (metric column building).
                                    per_bins_raw=item.pop("per_bins"),
                                )
                                if _v2_row is not None:
                                    pb_gz_file.write(
                                        _json_dumps_compact(_v2_row) + "\n"
                                    )
                                    _pb_count += 1

                            # Serialize and write this item immediately.
                            # After per_bins pop the item is tiny (~a few KB).
                            transform_in_place(item, make_serializable)
                            transform_in_place(
                                item,
                                lambda x: None if isinstance(x, float) and x != x else x,
                            )
                            if not _first_item:
                                json_file.write(",\n")
                            _item_str = json.dumps(
                                item, indent=2, ensure_ascii=False,
                            )
                            json_file.write(
                                "      "
                                + _item_str.replace("\n", "\n      ")
                            )
                            _first_item = False
                            _total_entries += 1
                        # Batch file text buffer freed when generator exhausts.
                        gc.collect()

                    # Close the results array and object, write metadata.
                    json_file.write("\n    ]\n  },\n")
                    _metadata = {
                        "evaluation_date": pd.Timestamp.now().isoformat(),
                        "total_entries": _total_entries,
                        "n_errors": n_errors,
                        "config": {
                            "start_time": self.args.start_time,
                            "end_time": self.args.end_time,
                            "n_days_forecast": self.args.n_days_forecast,
                            "n_days_interval": self.args.n_days_interval,
                        },
                    }
                    _meta_str = json.dumps(
                        _metadata, indent=2, ensure_ascii=False,
                    )
                    json_file.write(
                        '  "metadata": '
                        + _meta_str.replace("\n", "\n  ")
                        + "\n}\n"
                    )

                logger.opt(colors=True).info(
                    f"  <dim>│</dim>  <b>{_total_entries}</b> result entr"
                    f"{'y' if _total_entries == 1 else 'ies'} from"
                    f" <b>{len(batch_files)}</b> batch(es)"
                )

                if _pb_count == 0:
                    # Nothing was written – remove the empty placeholder file.
                    try:
                        os.remove(per_bins_gz_path)
                    except OSError:
                        pass
                    per_bins_gz_path = None  # type: ignore[assignment]
                    logger.opt(colors=True).info("  <dim>│  No per-bins spatial data produced for this dataset</dim>")
                else:
                    gz_bytes = os.path.getsize(per_bins_gz_path)
                    logger.opt(colors=True).info(
                        f"  <dim>│</dim>  Per-bins  <dim>→</dim>  <b>{_pb_count}</b> entr"
                        f"{'y' if _pb_count == 1 else 'ies'} <dim>(v2 columnar, direct)</dim>"
                        f"  <dim>→</dim>  <b>{gz_bytes / 1e6:.1f} MB</b>"
                        f"  <cyan>({os.path.basename(per_bins_gz_path)})</cyan>"
                    )

                logger.opt(colors=True).success(
                    f"  └  <green>✓</green>  Results saved  <dim>→</dim>"
                    f"  <cyan><b>{os.path.basename(dataset_json_path)}</b></cyan>"
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
                    logger.opt(colors=True).warning(
                        f"  └  <yellow>⚠  {n_errors} task error(s) tolerated</yellow>"
                        f" (max_task_errors={_max_errors}) — results saved."
                    )

            except Exception as exc:
                logger.opt(colors=True).error(
                    f"  <red>└  ✗  Failed to write JSON results:</red>  {exc}"
                )
                raise

        dataset_manager.file_cache.clear()

        # ══════════════════════════════════════════════════════════════════
        # LEADERBOARD GENERATION
        # ══════════════════════════════════════════════════════════════════
        # waiting 1s for the final log messages to flush before printing the leaderboard header
        _time.sleep(1)
        _inner_lb = f"  🏆  LEADERBOARD GENERATION"
        _inner_lb = f"{_inner_lb:^68}"  # pre-pad before markup
        logger.opt(colors=True).info(
            f"\n<green>╔{'═' * 68}╗\n"
            f"║<bold>{_inner_lb}</bold>║\n"
            f"╚{'═' * 68}╝</green>  <dim>[+{_fmt_elapsed(_t0)}]</dim>"
        )

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

            # Output the leaderboard site directly into the Sphinx _extra/ tree
            # so that it is picked up by html_extra_path and published on RTD
            # without any manual sync step.
            # Derive the project root from self.results_directory (e.g.
            # dc2_output/results/) rather than __file__, because dctools may be
            # installed as a *separate* package (its own repo) so __file__ would
            # resolve to the dctools source tree instead of the user's DC project.
            # results_directory --> dc2_output/ --> project root (parents[1]).
            _repo_root = Path(self.results_directory).resolve().parents[1]
            _docs_leaderboard = _repo_root / "docs" / "source" / "_extra" / "leaderboard"
            if (_repo_root / "docs").is_dir():
                _leaderboard_dir = str(_docs_leaderboard)
                logger.debug(
                    f"  ┌ Leaderboard will be written to docs/  ->  {_leaderboard_dir}"
                )
            else:
                _leaderboard_dir = os.path.join(self.results_directory, "leaderboard")
                logger.debug(
                    f"  ┌ docs/ not found — writing leaderboard to results/  ->  {_leaderboard_dir}"
                )
            _leaderboard_input_dir = os.path.join(self.results_directory, "leaderboard_input")
            os.makedirs(_leaderboard_dir, exist_ok=True)
            os.makedirs(_leaderboard_input_dir, exist_ok=True)

            # Optional fast-path for resumed runs: skip leaderboard rebuild if
            # the pages already exist.
            _leaderboard_html = os.path.join(_leaderboard_dir, "leaderboard.html")
            _maps_html = os.path.join(_leaderboard_dir, "maps.html")
            if _skip_existing_post and os.path.isfile(_leaderboard_html) and os.path.isfile(_maps_html):
                logger.opt(colors=True).info(
                    "  <dim>└</dim>  <yellow>↷  Skipping leaderboard generation:</yellow>"
                    " existing site artefacts found"
                )
                return

            # Copy reference baseline JSONs.
            # Primary source: <challenge_pkg>/leaderboard_results/ (sibling of evaluate.py).
            # _repo_root was derived from self.results_directory above (not __file__).
            _challenge_pkg_name = getattr(self, "CHALLENGE_NAME", self.__class__.__name__).lower()
            _challenge_dir = _repo_root / _challenge_pkg_name
            _local_lb_dir = _challenge_dir / "leaderboard_results"
            if _local_lb_dir.is_dir():
                _ref_results_src = str(_local_lb_dir)
                logger.debug(
                    f" Using {_challenge_pkg_name}/leaderboard_results/  ->  {_local_lb_dir}"
                )
            else:
                _ref_results_src = os.path.join(
                    os.path.dirname(_dcleaderboard.__file__), "results"
                )
                logger.debug(f" Using dcleaderboard package results/  ->  {_ref_results_src}")
            _ref_jsons = glob(os.path.join(_ref_results_src, "results_*.json"))
            for _ref_json in _ref_jsons:
                _shutil.copy2(
                    _ref_json,
                    os.path.join(_leaderboard_input_dir, os.path.basename(_ref_json)),
                )
            logger.debug(
                f" Reference baselines copied  ({len(_ref_jsons)} file(s))"
            )
            # Also copy leaderboard_config.yaml from the reference source if present.
            _lb_config_src = os.path.join(_ref_results_src, "leaderboard_config.yaml")
            if os.path.isfile(_lb_config_src):
                _shutil.copy2(
                    _lb_config_src,
                    os.path.join(_leaderboard_input_dir, "leaderboard_config.yaml"),
                )
                logger.debug("  leaderboard_config.yaml copied")

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

            # Stage ALL per-bins files via glob (prefer .jsonl.gz, fall back to
            # .jsonl then legacy .json) to avoid any alias-name mismatch.
            # For each stem, only stage the most-compact format.
            # Large .jsonl.gz files are symlinked (not copied) so that:
            #   1. Staging is instant regardless of file size.
            #   2. map_processing._NpzBinStore resolves the symlink to the REAL
            #      path and places the NPZ cache next to the source file in
            #      results_directory, so the cache survives leaderboard_input
            #      cleanup and is reused on subsequent runs.
            _pb_stems_copied: set = set()
            for _pb_src in glob(os.path.join(self.results_directory, "*_per_bins.jsonl.gz")):
                _stem = os.path.basename(_pb_src).replace(".jsonl.gz", "")
                _pb_dst = os.path.join(_leaderboard_input_dir, os.path.basename(_pb_src))
                # Remove any stale symlink/file left by a previous interrupted run
                # to avoid FileExistsError → shutil.copy2 → SameFileError.
                if os.path.lexists(_pb_dst):
                    os.remove(_pb_dst)
                try:
                    os.symlink(os.path.abspath(_pb_src), _pb_dst)
                except (OSError, NotImplementedError):
                    try:
                        _shutil.copy2(_pb_src, _pb_dst)
                    except _shutil.SameFileError:
                        pass
                _copied.append(os.path.basename(_pb_src))
                _pb_stems_copied.add(_stem)
            for _pb_src in glob(os.path.join(self.results_directory, "*_per_bins.jsonl")):
                _stem = os.path.basename(_pb_src).replace(".jsonl", "")
                if _stem not in _pb_stems_copied:
                    _pb_dst = os.path.join(_leaderboard_input_dir, os.path.basename(_pb_src))
                    if os.path.lexists(_pb_dst):
                        os.remove(_pb_dst)
                    try:
                        os.symlink(os.path.abspath(_pb_src), _pb_dst)
                    except (OSError, NotImplementedError):
                        try:
                            _shutil.copy2(_pb_src, _pb_dst)
                        except _shutil.SameFileError:
                            pass
                    _copied.append(os.path.basename(_pb_src))
                    _pb_stems_copied.add(_stem)
            for _pb_src in glob(os.path.join(self.results_directory, "*_per_bins.json")):
                _stem = os.path.basename(_pb_src).replace("_per_bins.json", "")
                if _stem not in _pb_stems_copied:
                    _pb_dst = os.path.join(_leaderboard_input_dir, os.path.basename(_pb_src))
                    try:
                        _shutil.copy2(_pb_src, _pb_dst)
                    except _shutil.SameFileError:
                        pass
                    _copied.append(os.path.basename(_pb_src))

            for _fname in _copied:
                logger.debug(f" Staged for leaderboard  ->  {_fname}")
            if not any("_per_bins" in f for f in _copied):
                logger.warning(
                    "  No per-bins file found in results directory "
                    f"({self.results_directory}) — maps.html will be skipped."
                )

            # Persist current evaluation results into <dc_pkg>/leaderboard_results/
            # so they serve as baselines for future runs with other models.
            os.makedirs(str(_local_lb_dir), exist_ok=True)
            for _fname in _copied:
                _src_path = os.path.join(_leaderboard_input_dir, _fname)
                if os.path.isfile(_src_path):
                    _shutil.copy2(_src_path, os.path.join(str(_local_lb_dir), _fname))
            logger.debug(
                f"  Persisted {len(_copied)} file(s) to {_local_lb_dir} for future leaderboards"
            )

            logger.opt(colors=True).info("  <dim>◎</dim>  Rendering leaderboard site …")
            # Collect the union of metrics explicitly configured across all
            # sources in the pipeline YAML (e.g. dc2_wasabi.yaml).  This
            # becomes the allowed_metrics whitelist injected into the
            # leaderboard config so that only intentional metrics are shown,
            # regardless of any extra sub-statistics emitted by metric classes.
            _pipeline_metrics: set = set()
            for _src in getattr(self.args, "sources", []) or []:
                if isinstance(_src, dict):
                    for _m in _src.get("metrics", []) or []:
                        if isinstance(_m, str):
                            _pipeline_metrics.add(_m)
            # Normalise metric names: the pipeline YAML uses "rmsd" but
            # dctools per-bins data stores the key as "rmse" (Root Mean
            # Squared Error).  Map the YAML names to the actual data keys so
            # the allowed_metrics whitelist matches what is in the files.
            _METRIC_ALIASES = {"rmsd": "rmse"}
            _pipeline_metrics = {_METRIC_ALIASES.get(m, m) for m in _pipeline_metrics}
            _lb_cfg = dict(self.leaderboard_custom_config or {})
            if _pipeline_metrics and "allowed_metrics" not in _lb_cfg:
                _lb_cfg["allowed_metrics"] = sorted(_pipeline_metrics)
                logger.debug(
                    "  Leaderboard metric whitelist from pipeline config: {}",
                    sorted(_pipeline_metrics),
                )
            _skip_frt = bool(getattr(self.args, "skip_frt_snapshots", False))
            _render_leaderboard(
                results_dir=_leaderboard_input_dir,
                output_site_dir=_leaderboard_dir,
                custom_config=_lb_cfg if _lb_cfg else None,
                skip_frt_snapshots=_skip_frt,
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
                logger.opt(colors=True).warning(
                    f"  <yellow>⚠  [INCOMPLETE]</yellow>  maps.html not generated —"
                    " no per-bins spatial data found; check per_bins metrics are enabled."
                )
            else:
                logger.opt(colors=True).success(
                    f"  <green>✓</green>  Leaderboard ready  <dim>→</dim>"
                    f"  <cyan>{_leaderboard_dir}</cyan>  <dim>[+{_fmt_elapsed(_t0)}]</dim>"
                )
            print("")
        except Exception as _lb_exc:
            _msg = f"Leaderboard generation failed: {_lb_exc!r}"
            self._leaderboard_warnings.append(_msg)
            logger.opt(colors=True).warning(
                f"  <yellow>└  ⚠  Leaderboard generation failed (non-blocking):</yellow>\n"
                f"              {_lb_exc!r}"
            )
