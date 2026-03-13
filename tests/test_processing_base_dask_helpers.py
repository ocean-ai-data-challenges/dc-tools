"""Unit tests for BaseDCEvaluation Dask sizing helper methods."""

from __future__ import annotations

from argparse import Namespace
from types import SimpleNamespace

import pandas as pd

from dctools.processing.base import BaseDCEvaluation
from dctools.utilities.parallelism import ParallelismConfig


def _obj() -> BaseDCEvaluation:
    """Create an uninitialized BaseDCEvaluation instance for helper testing."""
    ev = object.__new__(BaseDCEvaluation)
    ev.pcfg = ParallelismConfig(auto_adapt=False)
    ev._machine_resources = None
    ev._reference_machine_dict = None
    return ev


def test_extract_dask_cfg_from_source_returns_none_for_non_dict():
    """Non-dict source should be ignored."""
    ev = _obj()
    assert ev._extract_dask_cfg_from_source(None) is None


def test_extract_dask_cfg_from_source_parses_nested_and_flat_keys():
    """Nested/flat keys should be normalized to scheduler kwargs."""
    ev = _obj()
    source = {
        "dataset": "obs",
        "dask": {"n_workers": 4},
        "threads_per_worker": "2",
        "memory_limit": "8GB",
    }

    cfg = ev._extract_dask_cfg_from_source(source)

    assert cfg == {
        "n_workers": 4,
        "threads_per_worker": 2,
        "memory_limit": "8GB",
    }


def test_build_dask_cfgs_by_dataset_collects_only_valid_sources():
    """Per-dataset map should only include datasets with effective Dask config."""
    ev = _obj()
    ev.args = Namespace(
        sources=[
            {"dataset": "pred", "n_parallel_workers": 3, "nthreads_per_worker": 1},
            {"dataset": "ref", "file_pattern": "*.nc"},
            "invalid",
        ]
    )

    out = ev._build_dask_cfgs_by_dataset()

    assert out == {"pred": {"n_workers": 3, "threads_per_worker": 1}}


def test_global_dask_cfg_fallback_uses_args():
    """Global fallback should map top-level args to Dask scheduler kwargs."""
    ev = _obj()
    ev.args = Namespace(
        n_parallel_workers=5,
        nthreads_per_worker=2,
        memory_limit_per_worker="6GB",
    )

    out = ev._global_dask_cfg_fallback()
    assert out == {"n_workers": 5, "threads_per_worker": 2, "memory_limit": "6GB"}


def test_pick_initial_dask_cfg_prefers_prediction_dataset():
    """Initial config should first try prediction datasets."""
    ev = _obj()
    ev.dataset_references = {"pred": ["ref"]}
    ev.dask_cfgs_by_dataset = {
        "pred": {"n_workers": 3, "threads_per_worker": 1},
        "ref": {"n_workers": 9, "threads_per_worker": 9},
    }
    ev.args = Namespace()

    out = ev._pick_initial_dask_cfg()

    assert out == {"n_workers": 3, "threads_per_worker": 1}


def test_pick_initial_dask_cfg_then_reference_then_first_available():
    """If pred has no config, use first reference then insertion-order fallback."""
    ev = _obj()
    ev.dataset_references = {"pred": ["refA", "refB"]}
    ev.dask_cfgs_by_dataset = {
        "refB": {"n_workers": 7, "threads_per_worker": 2},
    }
    ev.args = Namespace()

    out = ev._pick_initial_dask_cfg()
    assert out == {"n_workers": 7, "threads_per_worker": 2}


def test_pick_initial_dask_cfg_falls_back_to_global_then_default(monkeypatch):
    """Fallback order should be global args then hard-coded safe default."""
    ev = _obj()
    ev.dataset_references = {"pred": ["ref"]}
    ev.dask_cfgs_by_dataset = {}
    ev.args = Namespace(n_parallel_workers=2, nthreads_per_worker=1, memory_limit_per_worker="5GB")
    assert ev._pick_initial_dask_cfg() == {
        "n_workers": 2,
        "threads_per_worker": 1,
        "memory_limit": "5GB",
    }

    ev2 = _obj()
    ev2.dataset_references = None
    ev2.dask_cfgs_by_dataset = {}
    ev2.args = Namespace()
    monkeypatch.setattr(ev2, "_global_dask_cfg_fallback", lambda: {})
    assert ev2._pick_initial_dask_cfg() == {
        "n_workers": 1,
        "threads_per_worker": 1,
        "memory_limit": "4GB",
    }


def test_configure_thread_caps_env_sets_expected_vars(monkeypatch):
    """Thread cap helper should export all expected env vars."""
    ev = _obj()
    ev._configure_thread_caps_env(threads="3")

    import os

    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "PYINTERP_NUM_THREADS",
        "GOTO_NUM_THREADS",
        "BLOSC_NTHREADS",
    ):
        assert os.environ.get(key) == "3"


def test_configure_dataset_processor_workers_calls_hook(monkeypatch):
    """Worker env propagation should call configure_dask_workers_env when client exists."""
    ev = _obj()
    called: list[object] = []

    import dctools.processing.base as base_mod

    monkeypatch.setattr(
        base_mod,
        "configure_dask_workers_env",
        lambda client, pcfg=None: called.append(client),
    )

    ev.dataset_processor = SimpleNamespace(client="client-1")
    ev._configure_dataset_processor_workers()

    assert called == ["client-1"]


def test_configure_dataset_processor_workers_noop_without_client(monkeypatch):
    """Without dataset_processor/client the helper should no-op."""
    ev = _obj()
    ev.dataset_processor = None

    import dctools.processing.base as base_mod

    monkeypatch.setattr(
        base_mod,
        "configure_dask_workers_env",
        lambda client, pcfg=None: (_ for _ in ()).throw(AssertionError(client)),
    )
    ev._configure_dataset_processor_workers()


def test_filter_data_applies_time_and_region_filters():
    """filter_data should call manager date and region filters and return manager."""
    ev = _obj()
    ev.args = Namespace(start_time="2024-01-01", end_time="2024-01-02")

    calls: list[tuple[str, object, object]] = []

    class _Mgr:
        def filter_all_by_date(self, start, end):
            calls.append(("date", start, end))

        def filter_all_by_region(self, region):
            calls.append(("region", region, None))

    manager = _Mgr()
    region = object()

    out = ev.filter_data(manager, region)

    assert out is manager
    assert calls[0][0] == "date"
    assert calls[0][1] == pd.Timestamp("2024-01-01")
    assert calls[0][2] == pd.Timestamp("2024-01-02")
    assert calls[1] == ("region", region, None)


def test_check_dataloader_accepts_valid_batch():
    """check_dataloader should pass for valid batch shape/types."""
    ev = _obj()
    dataloader = [[{"pred_data": "pred.nc", "ref_data": "ref.nc"}]]
    ev.check_dataloader(dataloader)


def test_check_dataloader_rejects_invalid_batch():
    """check_dataloader should fail on invalid entry types."""
    ev = _obj()
    dataloader = [[{"pred_data": 123, "ref_data": "ref.nc"}]]
    try:
        ev.check_dataloader(dataloader)
    except AssertionError:
        pass
    else:
        raise AssertionError("Expected AssertionError for invalid pred_data type")


def test_close_calls_dataset_processor_close_and_swallows_errors():
    """Close should call processor.close when available and swallow exceptions."""
    ev = _obj()
    called: list[str] = []

    class _Proc:
        def close(self):
            called.append("ok")

    ev.dataset_processor = _Proc()
    ev.close()
    assert called == ["ok"]

    class _BadProc:
        def close(self):
            raise RuntimeError("boom")

    ev2 = _obj()
    ev2.dataset_processor = _BadProc()
    ev2.close()


def test_setup_transforms_injects_precision_and_optional_weights():
    """setup_transforms should pass reduce_precision and optional weights for glorys."""
    ev = _obj()
    ev.pcfg = ParallelismConfig(auto_adapt=False, reduce_precision=True)
    ev.args = Namespace(reduce_precision=True, regridder_weights="/tmp/w.nc")

    calls: list[tuple[str, dict]] = []

    class _Mgr:
        def get_transform(self, dataset_alias, **kwargs):
            calls.append((dataset_alias, kwargs))
            return f"T:{dataset_alias}"

    out = ev.setup_transforms(_Mgr(), ["glorys_cmems", "other"])

    assert out == {"glorys_cmems": "T:glorys_cmems", "other": "T:other"}
    assert calls[0][1]["reduce_precision"] is True
    assert calls[0][1]["regridder_weights"] == "/tmp/w.nc"
    assert "regridder_weights" not in calls[1][1]


def test_setup_dataset_manager_happy_path_with_skips(monkeypatch):
    """setup_dataset_manager should skip invalid/unsupported sources and add valid datasets."""
    ev = _obj()
    ev.dataset_processor = "proc"
    ev.target_dimensions = {"lat": [0, 1]}
    ev.all_datasets = ["supported"]
    ev.args = Namespace(
        delta_time=12,
        max_cache_files=3,
        sources=[
            "invalid",
            {"foo": "bar"},
            {"dataset": "unsupported"},
            {"dataset": "supported", "config": "local"},
        ],
        catalog_dir="/tmp/catalog",
        catalog_connection={"s3_bucket": "b", "s3_folder": "f"},
        data_directory="/tmp/data",
        max_samples=10,
        start_time="2024-01-01",
        end_time="2024-01-02",
        min_lon=-1,
        max_lon=1,
        min_lat=-2,
        max_lat=2,
    )

    class _FakeManager:
        def __init__(self, **kwargs):
            self.init_kwargs = kwargs
            self.added: list[tuple[str, object]] = []
            self.file_cache = "cache"

        def add_dataset(self, name, ds):
            self.added.append((name, ds))

    fake_manager_holder: dict[str, _FakeManager] = {}

    def _manager_ctor(**kwargs):
        fake_manager_holder["m"] = _FakeManager(**kwargs)
        return fake_manager_holder["m"]

    import dctools.processing.base as base_mod

    monkeypatch.setattr(base_mod, "MultiSourceDatasetManager", _manager_ctor)
    monkeypatch.setattr(base_mod, "get_client", lambda: SimpleNamespace(run=lambda fn: None))
    monkeypatch.setattr(base_mod, "get_target_depth_values", lambda args: [0.0, 10.0])

    created_ds: list[dict] = []

    def _fake_get_dataset_from_config(**kwargs):
        created_ds.append(kwargs)
        return {"dataset": kwargs["source"]["dataset"]}

    monkeypatch.setattr(base_mod, "get_dataset_from_config", _fake_get_dataset_from_config)

    catalog_calls: list[str] = []
    ev.get_catalog = (
        lambda dataset_name, local_catalog_dir, catalog_cfg: catalog_calls.append(dataset_name)
    )
    ev.filter_data = lambda manager, filter_region: manager

    out = ev.setup_dataset_manager(["refA"])
    manager = fake_manager_holder["m"]

    assert out is manager
    assert manager.init_kwargs["dataset_processor"] == "proc"
    assert manager.init_kwargs["list_references"] == ["refA"]
    assert len(manager.added) == 1
    assert manager.added[0][0] == "supported"
    assert created_ds and created_ds[0]["source"]["dataset"] == "supported"
    assert catalog_calls == ["supported"]


def test_get_catalog_returns_early_for_existing_local_files(tmp_path):
    """get_catalog should early-return for existing local JSON and ARGO master index directory."""
    ev = _obj()

    local_catalog_dir = tmp_path / "catalog"
    local_catalog_dir.mkdir(parents=True)

    existing = local_catalog_dir / "x.json"
    existing.write_text("{}", encoding="utf-8")

    # Existing local file path branch
    ev.get_catalog(
        dataset_name="x",
        local_catalog_dir=str(local_catalog_dir),
        catalog_cfg={"s3_bucket": "b", "s3_folder": "f", "url": "http://example"},
    )

    # ARGO local directory branch
    argo_dir = local_catalog_dir / "argo_index"
    argo_dir.mkdir()
    (argo_dir / "master_index.json").write_text("{}", encoding="utf-8")
    ev.get_catalog(
        dataset_name="argo_profiles",
        local_catalog_dir=str(local_catalog_dir),
        catalog_cfg={"s3_bucket": "b", "s3_folder": "f", "url": "http://example"},
    )
