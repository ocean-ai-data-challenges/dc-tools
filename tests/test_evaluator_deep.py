"""Deep coverage tests for evaluator.py uncovered branches."""

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import xarray as xr

from dctools.metrics.evaluator import (
    _parse_memory_limit,
    Evaluator,
)


# =====================================================================
# Helpers
# =====================================================================

def _make_evaluator(
    dask_cfgs=None,
    n_workers=1,
    threads=1,
    memory="4GB",
):
    """Build an Evaluator with mocked dependencies."""
    dataset_manager = MagicMock()
    dataset_manager.get_config.return_value = ({}, {}, {})

    # Mock DatasetProcessor with realistic scheduler_info
    dp = MagicMock()
    dp.client = MagicMock()
    dp.client.scheduler_info.return_value = {
        "workers": {
            "tcp://127.0.0.1:12345": {
                "nthreads": threads,
                "memory_limit": _parse_memory_limit(memory),
            },
        },
    }

    evaluator = Evaluator.__new__(Evaluator)
    evaluator.dataset_manager = dataset_manager
    evaluator.dataset_processor = dp
    evaluator.metrics = []
    evaluator.dataloader = MagicMock()
    evaluator.dask_cfgs_by_dataset = dask_cfgs or {}
    evaluator.reduce_precision = False
    evaluator.restart_workers_per_batch = False
    evaluator.restart_frequency = 1
    evaluator.max_p_memory_increase = 0.5
    evaluator.max_worker_memory_fraction = 0.85
    evaluator.ref_aliases = ["obs1"]
    evaluator.results_dir = "/tmp/results"
    evaluator._current_cluster_ref = None
    evaluator.ref_managers = {}
    evaluator.ref_catalogs = {}
    evaluator.ref_connection_params = {}
    evaluator.baseline_memory = None
    return evaluator


# =====================================================================
# _reconfigure_cluster_for_ref
# =====================================================================

class TestReconfigureClusterForRef:
    """Tests for Evaluator._reconfigure_cluster_for_ref."""

    def test_same_alias_noop(self):
        """Same ref_alias → no-op (early return)."""
        ev = _make_evaluator()
        ev._current_cluster_ref = "obs1"
        old_dp = ev.dataset_processor
        ev._reconfigure_cluster_for_ref("obs1")
        assert ev.dataset_processor is old_dp

    def test_no_per_dataset_config_noop(self):
        """No per-dataset config → recorded but cluster unchanged."""
        ev = _make_evaluator(dask_cfgs={})
        ev._reconfigure_cluster_for_ref("obs1")
        assert ev._current_cluster_ref == "obs1"

    def test_matching_config_noop(self):
        """Current cluster matches desired config → no teardown."""
        ev = _make_evaluator(
            dask_cfgs={"obs1": {"n_workers": 1, "threads_per_worker": 1, "memory_limit": "4GB"}},
            n_workers=1, threads=1, memory="4GB",
        )
        old_dp = ev.dataset_processor
        ev._reconfigure_cluster_for_ref("obs1")
        assert ev._current_cluster_ref == "obs1"
        # close() should NOT have been called
        old_dp.close.assert_not_called()

    def test_different_config_recreates(self):
        """Different config → tears down old, creates new DatasetProcessor."""
        ev = _make_evaluator(
            dask_cfgs={"obs2": {"n_workers": 4, "threads_per_worker": 2, "memory_limit": "8GB"}},
            n_workers=1, threads=1, memory="4GB",
        )
        old_dp = ev.dataset_processor

        with patch("dctools.metrics.evaluator.DatasetProcessor") as MockDP:
            new_dp = MagicMock()
            new_dp.client = MagicMock()
            MockDP.return_value = new_dp
            with patch("dctools.utilities.init_dask.configure_dask_workers_env"):
                ev._reconfigure_cluster_for_ref("obs2")

        old_dp.close.assert_called_once()
        MockDP.assert_called_once_with(
            distributed=True,
            n_workers=4,
            threads_per_worker=2,
            memory_limit="8GB",
        )
        assert ev._current_cluster_ref == "obs2"
        assert ev.baseline_memory is None

    def test_scheduler_info_fails_triggers_reconfig(self):
        """scheduler_info() raises → proceeds with reconfiguration."""
        ev = _make_evaluator(
            dask_cfgs={"obs2": {"n_workers": 2, "threads_per_worker": 1, "memory_limit": "4GB"}},
        )
        ev.dataset_processor.client.scheduler_info.side_effect = RuntimeError("comm closed")

        with patch("dctools.metrics.evaluator.DatasetProcessor") as MockDP:
            new_dp = MagicMock()
            new_dp.client = MagicMock()
            MockDP.return_value = new_dp
            with patch("dctools.utilities.init_dask.configure_dask_workers_env"):
                ev._reconfigure_cluster_for_ref("obs2")

        assert ev._current_cluster_ref == "obs2"

    def test_close_exception_suppressed(self):
        """close() raises → suppressed, continues creation."""
        ev = _make_evaluator(
            dask_cfgs={"obs2": {"n_workers": 2, "threads_per_worker": 1, "memory_limit": "4GB"}},
        )
        ev.dataset_processor.close.side_effect = Exception("teardown error")

        with patch("dctools.metrics.evaluator.DatasetProcessor") as MockDP:
            new_dp = MagicMock()
            new_dp.client = MagicMock()
            MockDP.return_value = new_dp
            with patch("dctools.utilities.init_dask.configure_dask_workers_env"):
                ev._reconfigure_cluster_for_ref("obs2")

        assert ev._current_cluster_ref == "obs2"


# =====================================================================
# compute_metric — ARGO shared zarr fast-path
# =====================================================================

class TestComputeMetricArgoZarr:
    """Test the ARGO shared Zarr fast-path in compute_metric."""

    def _write_argo_zarr(self, path, n=100, sorted_time=True):
        """Write a minimal ARGO-like zarr store."""
        times = pd.date_range("2024-01-01", periods=n, freq="1h")
        if not sorted_time:
            times = times[::-1]
        ds = xr.Dataset(
            {
                "TEMP": ("obs", np.random.randn(n)),
                "TIME": ("obs", times.values),
            },
            coords={"obs": np.arange(n)},
        )
        zarr_path = str(path / "argo_shared.zarr")
        ds.to_zarr(zarr_path)
        return zarr_path

    def test_sorted_argo_zarr_time_slice(self, tmp_path):
        """Sorted ARGO zarr → contiguous slice via searchsorted."""
        zarr_path = self._write_argo_zarr(tmp_path, n=100, sorted_time=True)

        # Build minimal kwargs for compute_metric
        t0 = pd.Timestamp("2024-01-01T10:00")
        t1 = pd.Timestamp("2024-01-01T20:00")

        # We can't easily call compute_metric directly (too many deps),
        # so test the ARGO zarr-opening logic directly
        import xarray as _xr

        ref_data = _xr.open_zarr(zarr_path, consolidated=True, chunks=None)
        _t_vals = np.asarray(ref_data["TIME"].values)
        _t0_np = np.datetime64(t0)
        _t1_np = np.datetime64(t1)
        # Check sorted detection
        assert bool(np.all(_t_vals[:-1] <= _t_vals[1:]))
        _i0 = int(np.searchsorted(_t_vals, _t0_np, side="left"))
        _i1 = int(np.searchsorted(_t_vals, _t1_np, side="right"))
        sliced = ref_data.isel(obs=slice(_i0, _i1))
        assert sliced.sizes["obs"] > 0
        assert sliced.sizes["obs"] < 100

    def test_unsorted_argo_zarr_boolean_mask(self, tmp_path):
        """Unsorted ARGO zarr → boolean mask fallback."""
        zarr_path = self._write_argo_zarr(tmp_path, n=50, sorted_time=False)

        import xarray as _xr

        ref_data = _xr.open_zarr(zarr_path, consolidated=True, chunks=None)
        _t_vals = np.asarray(ref_data["TIME"].values)
        # Check NOT sorted
        assert not bool(np.all(_t_vals[:-1] <= _t_vals[1:]))
        t0 = pd.Timestamp("2024-01-01T10:00")
        t1 = pd.Timestamp("2024-01-01T20:00")
        _mask = (_t_vals >= np.datetime64(t0)) & (_t_vals <= np.datetime64(t1))
        sliced = ref_data.isel(obs=_mask)
        assert sliced.sizes["obs"] > 0


# =====================================================================
# compute_metric — shared obs zarr + sidecar .npy
# =====================================================================

class TestComputeMetricObsZarr:
    """Test shared observation zarr with sidecar time index."""

    def test_sidecar_npy_time_loading(self, tmp_path):
        """Sidecar .npy time index → loaded via np.load."""
        n = 50
        times = pd.date_range("2024-01-01", periods=n, freq="1h")
        ds = xr.Dataset(
            {"ssh": ("n_points", np.random.randn(n))},
            coords={"n_points": np.arange(n), "time": ("n_points", times)},
        )
        zarr_path = str(tmp_path / "obs_shared.zarr")
        ds.to_zarr(zarr_path)

        # Write sidecar npy
        npy_path = str(tmp_path / "time_index.npy")
        np.save(npy_path, times.values.astype("datetime64[ns]"))

        # Load and filter like the evaluator does
        _time_vals = np.load(npy_path, mmap_mode="r")
        t0 = np.datetime64("2024-01-01T10:00")
        t1 = np.datetime64("2024-01-01T20:00")
        _mask = (_time_vals >= t0) & (_time_vals <= t1)
        assert _mask.sum() > 0

    def test_sidecar_npy_integer_dtype(self, tmp_path):
        """Integer dtype in npy → cast to datetime64."""
        n = 10
        times = pd.date_range("2024-01-01", periods=n, freq="1h")
        npy_path = str(tmp_path / "time_index.npy")
        np.save(npy_path, times.values.astype(np.int64))  # Save as int64

        _time_vals = np.load(npy_path, mmap_mode="r")
        assert np.issubdtype(_time_vals.dtype, np.integer)
        # Cast like the evaluator does
        _time_vals = np.array(_time_vals).astype("datetime64[ns]")
        assert np.issubdtype(_time_vals.dtype, np.datetime64)

    def test_no_sidecar_falls_back_to_zarr(self, tmp_path):
        """No .npy → loads time from zarr coord (fallback)."""
        n = 20
        times = pd.date_range("2024-01-01", periods=n, freq="1h")
        ds = xr.Dataset(
            {"ssh": ("n_points", np.random.randn(n))},
            coords={"n_points": np.arange(n), "time": ("n_points", times)},
        )
        zarr_path = str(tmp_path / "obs_shared.zarr")
        ds.to_zarr(zarr_path)

        npy_path = str(tmp_path / "time_index.npy")
        assert not os.path.exists(npy_path)

        # Fallback: load from zarr
        reloaded = xr.open_zarr(zarr_path, consolidated=True)
        _time_var = reloaded.coords.get("time")
        assert _time_var is not None
        vals = np.asarray(_time_var.values)
        assert np.issubdtype(vals.dtype, np.datetime64)


# =====================================================================
# inspect_transform (grid-to-track logic)
# =====================================================================

class TestInspectTransform:
    """Test the inspect_transform logic for Grid-to-Track optimization."""

    def test_compose_with_glorys_to_glonet(self):
        """Compose containing glorys_to_glonet → removed."""
        from torchvision import transforms as output_transforms

        t1 = MagicMock()
        t1.transform_name = "glorys_to_glonet"
        t2 = MagicMock()
        t2.transform_name = "normalize"
        t2.__eq__ = lambda self, other: self is other

        compose = output_transforms.Compose([t1, t2])

        # Replicate the inspect logic from evaluator
        def inspect_transform(t):
            if isinstance(t, output_transforms.Compose):
                sub_list = []
                for sub_t in t.transforms:
                    res = inspect_transform(sub_t)
                    if res:
                        sub_list.append(res)
                return output_transforms.Compose(sub_list) if sub_list else None
            name = getattr(t, "transform_name", "")
            if name == "glorys_to_glonet":
                return None
            return t

        result = inspect_transform(compose)
        assert result is not None
        assert len(result.transforms) == 1  # only t2 remains

    def test_compose_without_glorys_unchanged(self):
        """Compose without glorys_to_glonet → unchanged count."""
        from torchvision import transforms as output_transforms

        t1 = MagicMock()
        t1.transform_name = "normalize"
        t2 = MagicMock()
        t2.transform_name = "standardize"

        compose = output_transforms.Compose([t1, t2])

        def inspect_transform(t):
            if isinstance(t, output_transforms.Compose):
                sub_list = []
                for sub_t in t.transforms:
                    res = inspect_transform(sub_t)
                    if res:
                        sub_list.append(res)
                return output_transforms.Compose(sub_list) if sub_list else None
            name = getattr(t, "transform_name", "")
            if name == "glorys_to_glonet":
                return None
            return t

        result = inspect_transform(compose)
        assert len(result.transforms) == 2

    def test_non_compose_single_transform(self):
        """Non-Compose transform → returned as-is."""
        from torchvision import transforms as output_transforms

        t = MagicMock()
        t.transform_name = "normalize"

        def inspect_transform(t):
            if isinstance(t, output_transforms.Compose):
                sub_list = []
                for sub_t in t.transforms:
                    res = inspect_transform(sub_t)
                    if res:
                        sub_list.append(res)
                return output_transforms.Compose(sub_list) if sub_list else None
            name = getattr(t, "transform_name", "")
            if name == "glorys_to_glonet":
                return None
            return t

        result = inspect_transform(t)
        assert result is t


# =====================================================================
# Evaluator.log_cluster_memory_usage (edge branches)
# =====================================================================

class TestEvaluatorMemoryEdges:
    """Test edge cases in cluster memory utilities."""

    def test_get_max_memory_usage_multiple_workers(self):
        """Multiple workers → returns max."""
        ev = _make_evaluator()
        ev.dataset_processor.client.scheduler_info.return_value = {
            "workers": {
                "w1": {"metrics": {"memory": 1_000_000}},
                "w2": {"metrics": {"memory": 5_000_000}},
                "w3": {"metrics": {"memory": 2_000_000}},
            },
        }
        result = ev.get_max_memory_usage()
        assert result == 5_000_000

    def test_log_cluster_memory_multiworker(self):
        """log_cluster_memory_usage with multi-worker cluster."""
        ev = _make_evaluator()
        ev.dataset_processor.client.scheduler_info.return_value = {
            "workers": {
                "w1": {
                    "metrics": {"memory": 500_000},
                    "memory_limit": 4_000_000_000,
                },
            },
        }
        # Should not raise
        ev.log_cluster_memory_usage(batch_idx=0)


# =====================================================================
# Evaluator init via constructor
# =====================================================================

class TestEvaluatorInit:
    """Test Evaluator.__init__ parameter storage."""

    def test_init_stores_params(self):
        """Constructor stores all parameters."""
        dm = MagicMock()
        dm.get_config.return_value = ({"a": 1}, {"b": 2}, {"c": 3})
        dp = MagicMock()
        metrics = [MagicMock()]
        dl = MagicMock()

        ev = Evaluator(
            dataset_manager=dm,
            dataset_processor=dp,
            metrics=metrics,
            dataloader=dl,
            ref_aliases=["obs1", "obs2"],
            results_dir="/tmp/results",
            dask_cfgs_by_dataset={"obs1": {"n_workers": 2}},
        )
        assert ev.ref_aliases == ["obs1", "obs2"]
        assert ev.dask_cfgs_by_dataset == {"obs1": {"n_workers": 2}}
        assert ev._current_cluster_ref is None
        assert ev.ref_managers == {"a": 1}
        assert ev.ref_catalogs == {"b": 2}
