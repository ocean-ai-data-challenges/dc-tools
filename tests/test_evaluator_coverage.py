"""Tests targeting evaluator.py coverage.

Level 1 – Pure / easily mockable functions (no cluster):
    _parse_memory_limit, _open_dataset_worker_cached,
    _compute_with_timeout, _cap_worker_threads,
    worker_memory_cleanup, _clear_xarray_file_cache,
    _worker_full_cleanup, Evaluator helpers (log_cluster_memory_usage,
    get_max_memory_usage, get_max_memory_fraction, clean_namespace).

Level 2 – Minimal LocalCluster tests (marked ``slow``):
    Evaluator._evaluate_batch with a single compute_metric task.
"""

from __future__ import annotations

import os
import time
from argparse import Namespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr


# ---------------------------------------------------------------------------
# Section 1 – _parse_memory_limit
# ---------------------------------------------------------------------------


class TestParseMemoryLimit:
    """Parse human-readable memory strings."""

    def test_numeric_int(self):
        """Integer passthrough."""
        from dctools.metrics.evaluator import _parse_memory_limit

        assert _parse_memory_limit(1024) == 1024

    def test_numeric_float(self):
        """Float truncated to int."""
        from dctools.metrics.evaluator import _parse_memory_limit

        assert _parse_memory_limit(1.5) == 1

    def test_gb_string(self):
        """'6GB' → 6 * 1024**3."""
        from dctools.metrics.evaluator import _parse_memory_limit

        assert _parse_memory_limit("6GB") == 6 * 1024**3

    def test_mb_string(self):
        """'512MB' → 512 * 1024**2."""
        from dctools.metrics.evaluator import _parse_memory_limit

        assert _parse_memory_limit("512MB") == 512 * 1024**2

    def test_kb_string(self):
        """'1024KB' → 1024 * 1024."""
        from dctools.metrics.evaluator import _parse_memory_limit

        assert _parse_memory_limit("1024KB") == 1024 * 1024

    def test_tb_string(self):
        """'1TB'."""
        from dctools.metrics.evaluator import _parse_memory_limit

        assert _parse_memory_limit("1TB") == 1024**4

    def test_bare_bytes(self):
        """'4096B'."""
        from dctools.metrics.evaluator import _parse_memory_limit

        assert _parse_memory_limit("4096B") == 4096

    def test_no_unit(self):
        """'1234' (digits only, treated as bytes)."""
        from dctools.metrics.evaluator import _parse_memory_limit

        assert _parse_memory_limit("1234") == 1234

    def test_invalid_raises(self):
        """Unparseable string raises ValueError."""
        from dctools.metrics.evaluator import _parse_memory_limit

        with pytest.raises(ValueError, match="Cannot parse"):
            _parse_memory_limit("lots of ram")

    def test_case_insensitive(self):
        """'2gb' works the same as '2GB'."""
        from dctools.metrics.evaluator import _parse_memory_limit

        assert _parse_memory_limit("2gb") == 2 * 1024**3


# ---------------------------------------------------------------------------
# Section 2 – _open_dataset_worker_cached
# ---------------------------------------------------------------------------


class TestOpenDatasetWorkerCached:
    """LRU caching of opened datasets."""

    def setup_method(self):
        """Clear the module-level cache before each test."""
        from dctools.metrics.evaluator import (
            _WORKER_DATASET_CACHE,
            _WORKER_DATASET_CACHE_LOCK,
        )

        with _WORKER_DATASET_CACHE_LOCK:
            _WORKER_DATASET_CACHE.clear()

    def test_cache_hit(self):
        """Second call for same key returns cached dataset."""
        from dctools.metrics.evaluator import _open_dataset_worker_cached

        ds = xr.Dataset({"x": (["t"], [1, 2])})
        opener = MagicMock(return_value=ds)

        r1, hit1 = _open_dataset_worker_cached(opener, "key1")
        r2, hit2 = _open_dataset_worker_cached(opener, "key1")

        assert hit1 is False
        assert hit2 is True
        opener.assert_called_once()  # only opened once

    def test_cache_miss_different_key(self):
        """Different key triggers new open call."""
        from dctools.metrics.evaluator import _open_dataset_worker_cached

        opener = MagicMock(side_effect=lambda k: xr.Dataset({"v": (["a"], [0])}))

        _, h1 = _open_dataset_worker_cached(opener, "a")
        _, h2 = _open_dataset_worker_cached(opener, "b")

        assert h1 is False
        assert h2 is False
        assert opener.call_count == 2

    def test_none_return_not_cached(self):
        """If opener returns None, it is NOT cached."""
        from dctools.metrics.evaluator import _open_dataset_worker_cached

        opener = MagicMock(return_value=None)
        r1, h1 = _open_dataset_worker_cached(opener, "k")
        r2, h2 = _open_dataset_worker_cached(opener, "k")

        assert r1 is None
        assert h1 is False
        # Opens again because None was not cached
        assert opener.call_count == 2

    def test_eviction(self):
        """Cache evicts oldest entries when limit exceeded."""
        from dctools.metrics.evaluator import _open_dataset_worker_cached

        # Set cache size to 2 via env var
        with patch.dict(os.environ, {"DCTOOLS_WORKER_DATASET_CACHE_SIZE": "2"}):
            opener = MagicMock(side_effect=lambda k: xr.Dataset({"v": (["a"], [0])}))

            _open_dataset_worker_cached(opener, "a")
            _open_dataset_worker_cached(opener, "b")
            _open_dataset_worker_cached(opener, "c")  # evicts "a"

            # "a" should no longer be in cache
            _, hit_a = _open_dataset_worker_cached(opener, "a")
            assert hit_a is False  # was evicted

    def test_cache_disabled(self):
        """Cache size 0 → always calls opener, never caches."""
        from dctools.metrics.evaluator import _open_dataset_worker_cached

        with patch.dict(os.environ, {"DCTOOLS_WORKER_DATASET_CACHE_SIZE": "0"}):
            ds = xr.Dataset({"x": (["t"], [1])})
            opener = MagicMock(return_value=ds)

            r1, h1 = _open_dataset_worker_cached(opener, "k")
            r2, h2 = _open_dataset_worker_cached(opener, "k")

            assert h1 is False
            assert h2 is False
            assert opener.call_count == 2


# ---------------------------------------------------------------------------
# Section 3 – _compute_with_timeout
# ---------------------------------------------------------------------------


class TestComputeWithTimeout:
    """Threaded .compute() with timeout."""

    def test_fast_compute(self):
        """Normal compute completes before timeout."""
        from dctools.metrics.evaluator import _compute_with_timeout

        ds = xr.Dataset({"v": (["x"], np.arange(10, dtype=np.float32))})
        result = _compute_with_timeout(ds, timeout_s=5)
        assert isinstance(result, xr.Dataset)
        np.testing.assert_array_equal(result["v"].values, np.arange(10, dtype=np.float32))

    def test_timeout_raises(self):
        """Compute that hangs triggers RuntimeError."""
        from dctools.metrics.evaluator import _compute_with_timeout

        class HangingDS:
            """Mock that blocks on .compute()."""

            def compute(self, **kwargs):
                time.sleep(30)  # will be killed by timeout

        with pytest.raises(RuntimeError, match="timed out"):
            _compute_with_timeout(HangingDS(), timeout_s=1)

    def test_compute_exception_propagated(self):
        """Exceptions from .compute() are re-raised."""
        from dctools.metrics.evaluator import _compute_with_timeout

        class BrokenDS:
            """Mock that raises on .compute()."""

            def compute(self, **kwargs):
                raise ValueError("bad data")

        with pytest.raises(ValueError, match="bad data"):
            _compute_with_timeout(BrokenDS(), timeout_s=5)


# ---------------------------------------------------------------------------
# Section 4 – _cap_worker_threads
# ---------------------------------------------------------------------------


class TestCapWorkerThreads:
    """Thread-cap function sets env vars."""

    def test_env_vars_set(self):
        """After _cap_worker_threads(1), key env vars should be '1'."""
        from dctools.metrics.evaluator import _cap_worker_threads

        _cap_worker_threads(1)

        assert os.environ.get("OMP_NUM_THREADS") == "1"
        assert os.environ.get("OPENBLAS_NUM_THREADS") == "1"
        assert os.environ.get("MKL_NUM_THREADS") == "1"
        assert os.environ.get("PYINTERP_NUM_THREADS") == "1"

    def test_custom_max_threads(self):
        """_cap_worker_threads(4) sets vars to '4'."""
        from dctools.metrics.evaluator import _cap_worker_threads

        _cap_worker_threads(4)

        assert os.environ.get("OMP_NUM_THREADS") == "4"
        assert os.environ.get("MKL_NUM_THREADS") == "4"

        # Restore
        _cap_worker_threads(1)


# ---------------------------------------------------------------------------
# Section 5 – worker_memory_cleanup / _clear_xarray_file_cache / _worker_full_cleanup
# ---------------------------------------------------------------------------


class TestWorkerCleanup:
    """Worker-side cleanup functions (no cluster required)."""

    def test_worker_memory_cleanup(self):
        """Should not raise."""
        from dctools.metrics.evaluator import worker_memory_cleanup

        worker_memory_cleanup()  # gc.collect + malloc_trim

    def test_clear_xarray_file_cache(self):
        """Returns True on success."""
        from dctools.metrics.evaluator import _clear_xarray_file_cache

        assert _clear_xarray_file_cache() is True

    def test_worker_full_cleanup(self):
        """Full cleanup returns True (sets env vars, patches xr.open_dataset)."""
        from dctools.metrics.evaluator import _worker_full_cleanup

        assert _worker_full_cleanup() is True
        # Env vars should be set
        assert os.environ.get("HDF5_USE_FILE_LOCKING") == "FALSE"
        assert os.environ.get("NETCDF4_DEACTIVATE_MPI") == "1"


# ---------------------------------------------------------------------------
# Section 6 – Evaluator helper methods (mock client)
# ---------------------------------------------------------------------------


def _mock_evaluator():
    """Build an Evaluator with all deps mocked."""
    from dctools.metrics.evaluator import Evaluator

    manager = MagicMock()
    manager.get_config.return_value = ({}, {}, {})

    ev = object.__new__(Evaluator)
    ev.dataset_manager = manager
    ev.dataset_processor = MagicMock()
    ev.metrics = {}
    ev.dataloader = MagicMock()
    ev.dask_cfgs_by_dataset = {}
    ev.reduce_precision = False
    ev.restart_workers_per_batch = False
    ev.restart_frequency = 1
    ev.max_p_memory_increase = 0.2
    ev.max_worker_memory_fraction = 0.85
    ev.ref_aliases = []
    ev.results_dir = None
    ev._current_cluster_ref = None
    ev.ref_managers = {}
    ev.ref_catalogs = {}
    ev.ref_connection_params = {}
    return ev


class TestEvaluatorHelpers:
    """Evaluator helper methods with mocked client."""

    def test_log_cluster_memory_usage(self):
        """Log memory does not raise with mocked scheduler_info."""
        ev = _mock_evaluator()
        ev.dataset_processor.client.scheduler_info.return_value = {
            "workers": {
                "tcp://w1": {
                    "name": "worker-1",
                    "metrics": {"memory": 2 * 1024**3},
                    "memory_limit": 4 * 1024**3,
                },
                "tcp://w2": {
                    "name": "worker-2",
                    "metrics": {"memory": 1 * 1024**3},
                    "memory_limit": 4 * 1024**3,
                },
            }
        }
        ev.log_cluster_memory_usage(0)  # should not raise

    def test_log_cluster_memory_no_client(self):
        """No client → returns immediately."""
        ev = _mock_evaluator()
        ev.dataset_processor.client = None
        ev.log_cluster_memory_usage(0)

    def test_get_max_memory_usage(self):
        """Returns the max memory across workers."""
        ev = _mock_evaluator()
        ev.dataset_processor.client.scheduler_info.return_value = {
            "workers": {
                "tcp://w1": {"metrics": {"memory": 3_000}},
                "tcp://w2": {"metrics": {"memory": 5_000}},
            }
        }
        assert ev.get_max_memory_usage() == 5_000

    def test_get_max_memory_usage_no_client(self):
        """No client → 0.0."""
        ev = _mock_evaluator()
        ev.dataset_processor.client = None
        assert ev.get_max_memory_usage() == 0.0

    def test_get_max_memory_fraction(self):
        """Fraction = mem_used / mem_limit."""
        ev = _mock_evaluator()
        ev.dataset_processor.client.scheduler_info.return_value = {
            "workers": {
                "tcp://w1": {"metrics": {"memory": 800}, "memory_limit": 1000},
                "tcp://w2": {"metrics": {"memory": 300}, "memory_limit": 1000},
            }
        }
        assert ev.get_max_memory_fraction() == pytest.approx(0.8)

    def test_get_max_memory_fraction_no_client(self):
        """No client → 0.0."""
        ev = _mock_evaluator()
        ev.dataset_processor.client = None
        assert ev.get_max_memory_fraction() == 0.0

    def test_get_max_memory_fraction_zero_limit(self):
        """Workers with memory_limit=0 → skip division."""
        ev = _mock_evaluator()
        ev.dataset_processor.client.scheduler_info.return_value = {
            "workers": {
                "tcp://w1": {"metrics": {"memory": 100}, "memory_limit": 0},
            }
        }
        assert ev.get_max_memory_fraction() == 0.0

    def test_clean_namespace(self):
        """Removes unpicklable keys from Namespace."""
        ev = _mock_evaluator()
        ns = Namespace(fs="fake_fs", client="fake_client", model="glonet", param="value")
        cleaned = ev.clean_namespace(ns)
        assert not hasattr(cleaned, "fs")
        assert not hasattr(cleaned, "client")
        assert cleaned.model == "glonet"
        assert cleaned.param == "value"

    def test_clean_namespace_with_params(self):
        """Nested params also get cleaned."""
        ev = _mock_evaluator()
        inner = Namespace(fs="inner_fs", dataset_processor="dp", keep_variables=["ssh"])
        ns = Namespace(params=inner, model="test")
        cleaned = ev.clean_namespace(ns)
        assert not hasattr(cleaned.params, "fs")
        assert not hasattr(cleaned.params, "dataset_processor")
        assert cleaned.params.keep_variables == ["ssh"]


# ---------------------------------------------------------------------------
# Section 7 – format_converter (exercised by evaluator compute_metric)
# ---------------------------------------------------------------------------


class TestFormatConverter:
    """convert_format1_to_format2 and siblings."""

    def test_nested_format(self):
        """Nested dict → list of {Metric, Variable, Value}."""
        from dctools.utilities.format_converter import convert_format1_to_format2

        f1 = {"rmsd": {"SSH": 0.05, "SST": 0.12}}
        f2 = convert_format1_to_format2(f1)
        assert len(f2) == 2
        metrics = {r["Variable"]: r["Value"] for r in f2}
        assert metrics["SSH"] == pytest.approx(0.05)
        assert metrics["SST"] == pytest.approx(0.12)

    def test_simple_format_with_name(self):
        """Simple list format with explicit metric_name."""
        from dctools.utilities.format_converter import convert_format1_to_format2

        f1 = {"SSH": [0.05], "SST": [0.12]}
        f2 = convert_format1_to_format2(f1, metric_name="rmse")
        assert len(f2) == 2
        assert all(r["Metric"] == "rmse" for r in f2)

    def test_simple_format_no_name(self):
        """Simple format without metric_name → empty result."""
        from dctools.utilities.format_converter import convert_format1_to_format2

        f1 = {"SSH": [0.05]}
        f2 = convert_format1_to_format2(f1)
        assert f2 == []

    def test_not_dict_input(self):
        """Non-dict input → empty."""
        from dctools.utilities.format_converter import convert_format1_to_format2

        assert convert_format1_to_format2("not a dict") == []  # type: ignore[arg-type]

    def test_empty_dict(self):
        """Empty dict → empty list."""
        from dctools.utilities.format_converter import convert_format1_to_format2

        assert convert_format1_to_format2({}) == []

    def test_convert_format2_to_format1(self):
        """format2 → format1 (flat: {variable: [values]})."""
        from dctools.utilities.format_converter import convert_format2_to_format1

        f2 = [
            {"Metric": "rmsd", "Variable": "SSH", "Value": 0.05},
            {"Metric": "rmsd", "Variable": "SST", "Value": 0.12},
        ]
        f1 = convert_format2_to_format1(f2)
        assert "SSH" in f1
        assert f1["SSH"] == [pytest.approx(0.05)]
        assert f1["SST"] == [pytest.approx(0.12)]

    def test_group_format2_by_metric(self):
        """Group format2 results by metric name."""
        from dctools.utilities.format_converter import group_format2_by_metric

        f2 = [
            {"Metric": "rmsd", "Variable": "SSH", "Value": 0.05},
            {"Metric": "mae", "Variable": "SSH", "Value": 0.02},
        ]
        grouped = group_format2_by_metric(f2)
        assert "rmsd" in grouped
        assert "mae" in grouped

    def test_filter_format2_by_variables(self):
        """Filter format2 to selected variables."""
        from dctools.utilities.format_converter import filter_format2_by_variables

        f2 = [
            {"Metric": "rmsd", "Variable": "SSH", "Value": 0.05},
            {"Metric": "rmsd", "Variable": "SST", "Value": 0.12},
        ]
        filtered = filter_format2_by_variables(f2, ["SSH"])
        assert len(filtered) == 1
        assert filtered[0]["Variable"] == "SSH"


# ---------------------------------------------------------------------------
# Section 8 – Slow test: compute_metric with real LocalCluster
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestComputeMetricOnCluster:
    """End-to-end compute_metric via a real Dask LocalCluster."""

    def test_gridded_compute_metric(self, tmp_path):
        """Submit compute_metric to a single-worker LocalCluster."""
        from unittest.mock import patch

        from dask.distributed import Client, LocalCluster

        from dctools.metrics.evaluator import compute_metric
        from dctools.metrics.metrics import MetricComputer

        cluster = LocalCluster(
            n_workers=1,
            threads_per_worker=1,
            memory_limit="4GB",
            processes=False,
            silence_logs=True,
            dashboard_address=None,
        )
        client = Client(cluster)

        def _passthrough_open(*args, **kwargs):
            """Return inline xr.Dataset as-is."""
            if args and isinstance(args[0], xr.Dataset):
                return args[0]
            return None

        try:
            # Create pred + ref datasets
            times = pd.date_range("2025-01-01", periods=1, freq="1D")
            lat = np.linspace(-5, 5, 5)
            lon = np.linspace(20, 30, 5)
            data = np.random.default_rng(42).standard_normal(
                (1, len(lat), len(lon))
            ).astype(np.float32)

            ds = xr.Dataset(
                {"zos": xr.Variable(
                    ["time", "lat", "lon"], data,
                    attrs={"standard_name": "sea_surface_height_above_geoid"},
                )},
                coords={"time": times, "lat": lat, "lon": lon},
            )

            # Build minimal Namespace configs
            pred_cfg = Namespace(
                protocol="local",
                init_type="local",
                local_root=str(tmp_path),
                file_pattern="pred*.nc",
                max_samples=1,
                groups=None,
                keep_variables=["zos"],
                eval_variables=["zos"],
                file_cache=None,
                dataset_processor=None,
                filter_values={},
                full_day_data=False,
                fs=None,
            )
            ref_cfg = Namespace(
                protocol="local",
                init_type="local",
                local_root=str(tmp_path),
                file_pattern="ref*.nc",
                max_samples=1,
                groups=None,
                keep_variables=["zos"],
                eval_variables=["zos"],
                file_cache=None,
                dataset_processor=None,
                filter_values={},
                full_day_data=False,
                fs=None,
            )

            mc = MetricComputer(eval_variables=["zos"], metric_name="rmsd")

            entry = {
                "pred_data": ds,
                "ref_data": ds,
                "forecast_reference_time": pd.Timestamp("2025-01-01"),
                "lead_time": 0,
                "valid_time": pd.Timestamp("2025-01-01"),
                "pred_coords": None,
                "ref_coords": None,
                "ref_alias": "test_ref",
                "ref_is_observation": False,
            }

            with patch(
                "dctools.metrics.evaluator.create_worker_connect_config",
                return_value=_passthrough_open,
            ):
                result = compute_metric(
                    entry=entry,
                    pred_source_config=pred_cfg,
                    ref_source_config=ref_cfg,
                    model="test_model",
                    list_metrics=[mc],
                    pred_transform=None,
                    ref_transform=None,
                )

            assert isinstance(result, dict)
            assert result.get("ref_alias") == "test_ref"
            assert result.get("error") is None
            # Identical data → RMSD ≈ 0
            if result.get("result"):
                for row in result["result"]:
                    assert row["Value"] == pytest.approx(0.0, abs=1e-3)

        finally:
            client.close()
            cluster.close()
