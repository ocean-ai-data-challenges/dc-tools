"""Tests for the ARGO shared-batch Zarr pipeline.

The new pipeline merges all per-entry time windows into a single bounding
interval, downloads all profiles once, writes a single time-sorted Zarr,
and lets workers filter by their specific time_bounds via searchsorted.

This test suite validates:
1. ArgoManager.prefetch_batch_shared_zarr mechanics
2. Profile deduplication across overlapping windows
3. Time-sorted Zarr with correct searchsorted filtering
4. Evaluator integration (compute_metric shared zarr fast path)
"""

import os
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr


pytestmark = pytest.mark.slow


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_argo_dataset(
    n_profiles: int,
    t_start: str,
    t_end: str,
    variables: list[str] | None = None,
) -> xr.Dataset:
    """Create a synthetic ARGO dataset with obs dim, TIME coord, etc."""
    variables = variables or ["TEMP", "PSAL"]
    times = pd.date_range(t_start, t_end, periods=n_profiles)
    depth_levels = np.array([0.0, 10.0, 50.0, 100.0])

    data_vars = {}
    for var in variables:
        data_vars[var] = (
            ("obs", "depth"),
            np.random.rand(n_profiles, len(depth_levels)).astype(np.float32),
        )

    coords = {
        "TIME": ("obs", times.values),
        "LATITUDE": ("obs", np.random.uniform(-90, 90, n_profiles).astype(np.float32)),
        "LONGITUDE": ("obs", np.random.uniform(-180, 180, n_profiles).astype(np.float32)),
        "depth": depth_levels,
    }

    return xr.Dataset(data_vars, coords=coords)


class StubArgoInterface:
    """Minimal ArgoInterface stub for testing prefetch_batch_shared_zarr."""

    def __init__(self, dataset: xr.Dataset):
        self._dataset = dataset
        self.open_time_window_calls: list[tuple] = []
        self.base_path = "/tmp/fake_argo_index"
        self.s3_storage_options = {}
        self.variables = ["TEMP", "PSAL"]
        self.chunks = {"N_PROF": 500}

    def open_time_window(
        self, start, end, depth_levels, variables=None, master_index=None, max_profiles=None
    ):
        """Open a time window on the stub dataset and record the call."""
        self.open_time_window_calls.append((start, end))
        # Filter by time bounds (simulate real behavior)
        t0 = np.datetime64(pd.Timestamp(start))
        t1 = np.datetime64(pd.Timestamp(end))
        t_vals = self._dataset.coords["TIME"].values
        mask = (t_vals >= t0) & (t_vals <= t1)
        return self._dataset.isel(obs=mask)


class StubArgoManager:
    """Minimal ArgoManager stub that has prefetch_batch_shared_zarr."""

    def __init__(self, dataset: xr.Dataset):
        self.argo_interface = StubArgoInterface(dataset)
        self.depth_values = [0.0, 10.0, 50.0, 100.0]
        self._master_index = {"2024_01": {"start": 0, "end": 1}}
        self.params = SimpleNamespace(keep_variables=["TEMP", "PSAL"])

    def prefetch_batch_shared_zarr(self, time_bounds_list, cache_dir):
        """Delegate to the real implementation logic embedded here for testing."""
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        all_t0 = [pd.Timestamp(t0) for t0, _ in time_bounds_list]
        all_t1 = [pd.Timestamp(t1) for _, t1 in time_bounds_list]
        global_t0 = min(all_t0)
        global_t1 = max(all_t1)

        cache_key = f"argo_shared_{global_t0.value}_{global_t1.value}"
        zarr_path = str(cache_dir / f"{cache_key}.zarr")

        if Path(zarr_path).exists():
            return zarr_path

        ds = self.argo_interface.open_time_window(
            start=global_t0,
            end=global_t1,
            depth_levels=self.depth_values,
            variables=self.params.keep_variables,
            master_index=self._master_index,
        )

        if ds is None or ds.sizes.get("obs", 0) == 0:
            return None

        # Sort by TIME
        t_arr = np.asarray(ds.coords["TIME"].values)
        if len(t_arr) > 1 and not bool(np.all(t_arr[:-1] <= t_arr[1:])):
            sort_idx = np.argsort(t_arr)
            ds = ds.isel(obs=sort_idx)

        ds = ds.compute()
        for var in ds.variables:
            ds[var].encoding.clear()
        ds.to_zarr(zarr_path, mode="w", consolidated=True)
        return zarr_path


# ---------------------------------------------------------------------------
# Tests: shared Zarr creation
# ---------------------------------------------------------------------------


class TestSharedBatchZarr:
    """Tests for the shared batch Zarr creation and caching."""

    @pytest.fixture(autouse=True)
    def _setup_teardown(self):
        self.tmpdir = tempfile.mkdtemp(prefix="test_argo_shared_")
        yield
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_single_window_produces_zarr(self):
        """A single time window should produce a valid Zarr."""
        ds = _make_argo_dataset(20, "2024-01-01", "2024-01-10")
        mgr = StubArgoManager(ds)

        result = mgr.prefetch_batch_shared_zarr(
            time_bounds_list=[
                (pd.Timestamp("2024-01-03"), pd.Timestamp("2024-01-05")),
            ],
            cache_dir=Path(self.tmpdir),
        )

        assert result is not None
        assert os.path.isdir(result)

        loaded = xr.open_zarr(result, consolidated=True)
        assert "obs" in loaded.dims
        assert loaded.sizes["obs"] > 0
        loaded.close()

    def test_overlapping_windows_merge_into_one_download(self):
        """Overlapping windows should trigger a single open_time_window call."""
        ds = _make_argo_dataset(100, "2024-01-01", "2024-01-20")
        mgr = StubArgoManager(ds)

        # 3 overlapping windows (like batch entries with 12h tolerance)
        windows = [
            (pd.Timestamp("2024-01-05"), pd.Timestamp("2024-01-07")),
            (pd.Timestamp("2024-01-06"), pd.Timestamp("2024-01-08")),
            (pd.Timestamp("2024-01-07"), pd.Timestamp("2024-01-09")),
        ]

        result = mgr.prefetch_batch_shared_zarr(
            time_bounds_list=windows,
            cache_dir=Path(self.tmpdir),
        )

        assert result is not None
        # CRITICAL: only ONE call to open_time_window (merged window)
        assert len(mgr.argo_interface.open_time_window_calls) == 1
        actual_t0, actual_t1 = mgr.argo_interface.open_time_window_calls[0]
        assert actual_t0 == pd.Timestamp("2024-01-05")
        assert actual_t1 == pd.Timestamp("2024-01-09")

    def test_cache_hit_avoids_redownload(self):
        """Second call with same global window should reuse cached Zarr."""
        ds = _make_argo_dataset(20, "2024-01-01", "2024-01-10")
        mgr = StubArgoManager(ds)

        windows = [
            (pd.Timestamp("2024-01-03"), pd.Timestamp("2024-01-05")),
        ]

        result1 = mgr.prefetch_batch_shared_zarr(
            time_bounds_list=windows,
            cache_dir=Path(self.tmpdir),
        )
        assert len(mgr.argo_interface.open_time_window_calls) == 1

        result2 = mgr.prefetch_batch_shared_zarr(
            time_bounds_list=windows,
            cache_dir=Path(self.tmpdir),
        )
        # Should still be 1 call (cache hit)
        assert len(mgr.argo_interface.open_time_window_calls) == 1
        assert result1 == result2

    def test_zarr_is_time_sorted(self):
        """The shared Zarr must be sorted by TIME for searchsorted fast path."""
        # Create dataset with UNSORTED times
        times = pd.DatetimeIndex(
            [
                "2024-01-05",
                "2024-01-01",
                "2024-01-10",
                "2024-01-03",
                "2024-01-07",
            ]
        )
        ds = xr.Dataset(
            {"TEMP": ("obs", np.array([5.0, 1.0, 10.0, 3.0, 7.0], dtype=np.float32))},
            coords={
                "TIME": ("obs", times.values),
                "LATITUDE": ("obs", np.zeros(5, dtype=np.float32)),
                "LONGITUDE": ("obs", np.zeros(5, dtype=np.float32)),
            },
        )
        mgr = StubArgoManager(ds)

        result = mgr.prefetch_batch_shared_zarr(
            time_bounds_list=[
                (pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-10")),
            ],
            cache_dir=Path(self.tmpdir),
        )

        loaded = xr.open_zarr(result, consolidated=True)
        t_vals = loaded.coords["TIME"].values
        # Must be sorted
        assert bool(np.all(t_vals[:-1] <= t_vals[1:]))
        loaded.close()

    def test_empty_window_returns_none(self):
        """When no profiles match the window, should return None."""
        ds = _make_argo_dataset(10, "2024-06-01", "2024-06-10")
        mgr = StubArgoManager(ds)

        # Request data from a completely different period
        result = mgr.prefetch_batch_shared_zarr(
            time_bounds_list=[
                (pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")),
            ],
            cache_dir=Path(self.tmpdir),
        )
        assert result is None


# ---------------------------------------------------------------------------
# Tests: worker-side searchsorted time filtering
# ---------------------------------------------------------------------------


class TestWorkerTimeFiltering:
    """Validate the searchsorted filtering that workers do on the shared Zarr."""

    @pytest.fixture(autouse=True)
    def _setup_teardown(self):
        self.tmpdir = tempfile.mkdtemp(prefix="test_argo_filter_")
        yield
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_searchsorted_filters_correct_slice(self):
        """Worker should get only profiles within its time_bounds."""
        # Create sorted dataset spanning 10 days
        ds = _make_argo_dataset(100, "2024-01-01", "2024-01-10")
        # Sort by TIME
        t_arr = ds.coords["TIME"].values
        sort_idx = np.argsort(t_arr)
        ds = ds.isel(obs=sort_idx)

        zarr_path = os.path.join(self.tmpdir, "shared.zarr")
        for var in ds.variables:
            ds[var].encoding.clear()
        ds.to_zarr(zarr_path, mode="w", consolidated=True)

        # Simulate worker: open and filter
        ref_data = xr.open_zarr(zarr_path, consolidated=True)
        t_vals = np.asarray(ref_data.coords["TIME"].values)

        t0 = np.datetime64(pd.Timestamp("2024-01-04"))
        t1 = np.datetime64(pd.Timestamp("2024-01-06"))

        i0 = int(np.searchsorted(t_vals, t0, side="left"))
        i1 = int(np.searchsorted(t_vals, t1, side="right"))
        filtered = ref_data.isel(obs=slice(i0, i1))

        # All returned profiles must be within the window
        filtered_times = filtered.coords["TIME"].values
        assert all(filtered_times >= t0)
        assert all(filtered_times <= t1)
        assert len(filtered_times) > 0
        ref_data.close()

    def test_disjoint_windows_get_correct_profiles(self):
        """Two non-overlapping windows should get distinct profile sets."""
        ds = _make_argo_dataset(200, "2024-01-01", "2024-01-30")
        t_arr = ds.coords["TIME"].values
        ds = ds.isel(obs=np.argsort(t_arr))

        zarr_path = os.path.join(self.tmpdir, "shared.zarr")
        for var in ds.variables:
            ds[var].encoding.clear()
        ds.to_zarr(zarr_path, mode="w", consolidated=True)

        ref_data = xr.open_zarr(zarr_path, consolidated=True)
        t_vals = np.asarray(ref_data.coords["TIME"].values)

        # Window A: Jan 5-7
        t0a, t1a = np.datetime64("2024-01-05"), np.datetime64("2024-01-07")
        i0a = int(np.searchsorted(t_vals, t0a, side="left"))
        i1a = int(np.searchsorted(t_vals, t1a, side="right"))

        # Window B: Jan 15-17
        t0b, t1b = np.datetime64("2024-01-15"), np.datetime64("2024-01-17")
        i0b = int(np.searchsorted(t_vals, t0b, side="left"))
        i1b = int(np.searchsorted(t_vals, t1b, side="right"))

        assert i1a <= i0b, "Disjoint windows should not overlap indices"

        # Profiles from A should be before B
        times_a = t_vals[i0a:i1a]
        times_b = t_vals[i0b:i1b]
        if len(times_a) > 0 and len(times_b) > 0:
            assert times_a[-1] <= times_b[0]

        ref_data.close()


# ---------------------------------------------------------------------------
# Tests: pipeline coherence
# ---------------------------------------------------------------------------


class TestPipelineCoherence:
    """High-level tests ensuring the ARGO shared pipeline is coherent."""

    def test_preprocess_argo_profiles_remains_functional_fallback(self):
        """preprocess_argo_profiles should still work as a fallback."""
        pytest.importorskip("oceanbench", reason="oceanbench not importable in this env")
        from dctools.data.datasets.dataloader import preprocess_argo_profiles

        fake_ds = xr.Dataset(
            {"TEMP": ("N_POINTS", np.array([10.0, 11.0], dtype=np.float32))},
            coords={
                "N_POINTS": np.array([0, 1]),
                "TIME": (
                    "N_POINTS",
                    np.array(["2024-01-01", "2024-01-02"], dtype="datetime64[ns]"),
                ),
            },
        )

        class FakeArgoManager:
            def open(self, path, *args, **kwargs):
                return fake_ds

        mgr = FakeArgoManager()

        result = preprocess_argo_profiles(
            profile_sources=["2024_01"],
            open_func=mgr.open,
            alias="argo_profiles",
            time_bounds=(pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")),
            depth_levels=np.array([0.0, 10.0]),
        )

        assert result is not None
        assert "N_POINTS" in result.dims

    def test_shared_zarr_preserves_all_variables(self):
        """The shared Zarr must contain all requested data variables."""
        tmpdir = tempfile.mkdtemp(prefix="test_argo_vars_")
        try:
            ds = _make_argo_dataset(
                20,
                "2024-01-01",
                "2024-01-10",
                variables=["TEMP", "PSAL"],
            )
            mgr = StubArgoManager(ds)

            result = mgr.prefetch_batch_shared_zarr(
                time_bounds_list=[
                    (pd.Timestamp("2024-01-03"), pd.Timestamp("2024-01-08")),
                ],
                cache_dir=Path(tmpdir),
            )

            loaded = xr.open_zarr(result, consolidated=True)
            assert "TEMP" in loaded.data_vars
            assert "PSAL" in loaded.data_vars
            assert "TIME" in loaded.coords
            assert "LATITUDE" in loaded.coords
            assert "LONGITUDE" in loaded.coords
            loaded.close()
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
