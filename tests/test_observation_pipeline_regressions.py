"""Regression coverage for the observation preprocessing pipeline."""

from argparse import Namespace
from io import BytesIO
from types import SimpleNamespace

import numpy as np
import pandas as pd
import xarray as xr

from dctools.data.datasets.batch_preprocessing import (
    download_and_preprocess_obs_pipeline,
)
from dctools.data.datasets.observation_viewer import ObservationDataViewer
from dctools.metrics.compute_task import compute_metric


def test_observation_viewer_processes_files_sequentially(monkeypatch):
    """Worker-side obs preprocessing must run files one at a time.

    The ObservationDataViewer runs inside a Dask worker process, which is
    already an isolated OS process.  Adding threads for CPU-bound work
    (numpy, swath_to_points) provides no speedup under the GIL and adds
    oversubscription.  Verify that all files are processed and results
    are returned without using ThreadPoolExecutor.
    """
    processed_paths = []

    def fake_preprocess(path, is_swath, n_points_dim, df, idx,
                        alias, load_fn, keep_vars, target_dims,
                        coordinates, time_bounds, load_to_mem):
        processed_paths.append(path)
        return xr.Dataset(
            {"ssh": ("n_points", np.array([float(idx)], dtype=np.float32))},
            coords={
                "n_points": np.array([idx]),
                "time": np.array([np.datetime64("2024-01-01")]),
            },
        )

    monkeypatch.setattr(
        "dctools.data.datasets.observation_viewer.preprocess_one_npoints",
        fake_preprocess,
    )

    viewer = ObservationDataViewer(
        source=pd.DataFrame({"path": ["a.nc", "b.nc", "c.nc"]}),
        load_fn=lambda path, alias=None: xr.Dataset(
            {"ssh": ("n_points", np.array([1.0], dtype=np.float32))},
            coords={
                "n_points": np.array([0]),
                "time": np.array([np.datetime64("2024-01-01")]),
            },
        ),
        alias="swot",
        keep_vars=["ssh"],
        target_dimensions={},
        dataset_metadata={
            "coord_system": SimpleNamespace(coordinates={"time": "time"}),
        },
        time_bounds=(pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")),
        n_points_dim="n_points",
    )

    result = viewer.preprocess_datasets(viewer.meta_df)

    assert result is not None
    # All 3 files must be processed
    assert len(processed_paths) == 3
    assert set(processed_paths) == {"a.nc", "b.nc", "c.nc"}


def test_download_pipeline_does_not_publish_failed_downloads(tmp_path, monkeypatch):
    """A failed download must not look like a valid local-path success."""

    class FakeFS:
        def open(self, path, mode):
            if path.endswith("missing.nc"):
                raise FileNotFoundError(path)
            return BytesIO(b"dummy")

    def fake_process_file(args):
        output_dir = args[-1]
        return (str(output_dir / "mini_0.zarr"), 10, 1.0, 2.0)

    monkeypatch.setattr(
        "dctools.data.datasets.batch_preprocessing._process_file_to_zarr",
        fake_process_file,
    )

    cache_dir = tmp_path / "cache"
    output_dir = tmp_path / "shared"
    path_map, shared_path = download_and_preprocess_obs_pipeline(
        remote_paths=["good.nc", "missing.nc"],
        cache_dir=str(cache_dir),
        fs=FakeFS(),
        alias="swot",
        keep_vars=["ssh"],
        coordinates={"time": "time"},
        output_zarr_dir=str(output_dir),
        download_workers=2,
        prep_workers=1,
        prep_use_processes=False,
    )

    assert shared_path is None
    assert set(path_map) == {"good.nc"}
    assert path_map["good.nc"].endswith("good.nc")


def test_compute_metric_errors_instead_of_truncating_obs_files(monkeypatch):
    """Large worker fallback batches must fail loudly instead of dropping files."""

    class RawObsFrame:
        def get_dataframe(self):
            return pd.DataFrame(
                {
                    "path": ["s3://a.nc", "s3://b.nc", "s3://c.nc"],
                    "date_start": [pd.Timestamp("2024-01-01")] * 3,
                    "date_end": [pd.Timestamp("2024-01-02")] * 3,
                }
            )

    pred_data = xr.Dataset(
        {"ssh": (("time", "x"), np.array([[1.0]], dtype=np.float32))},
        coords={
            "time": np.array([np.datetime64("2024-01-01T00:00:00")]),
            "x": np.array([0]),
        },
    )

    entry = {
        "forecast_reference_time": pd.Timestamp("2024-01-01T00:00:00"),
        "lead_time": 0,
        "valid_time": pd.Timestamp("2024-01-01T00:00:00"),
        "pred_coords": None,
        "ref_coords": None,
        "ref_alias": "swot",
        "ref_is_observation": True,
        "pred_data": pred_data,
        "ref_data": {
            "source": RawObsFrame(),
            "keep_vars": ["ssh"],
            "target_dimensions": {},
            "time_bounds": (
                pd.Timestamp("2024-01-01T00:00:00"),
                pd.Timestamp("2024-01-02T00:00:00"),
            ),
            "metadata": {
                "coord_system": SimpleNamespace(coordinates={"time": "time"}),
            },
        },
    }

    monkeypatch.setenv("DCTOOLS_MAX_OBS_FILES_PER_WORKER", "2")
    monkeypatch.setattr(
        "dctools.metrics.compute_task.create_worker_connect_config",
        lambda config, argo_index=None: (lambda source, alias=None: source),
    )
    monkeypatch.setattr(
        "dctools.metrics.compute_task.filter_by_time",
        lambda dataframe, t0, t1: dataframe,
    )

    result = compute_metric(
        entry=entry,
        pred_source_config=Namespace(protocol="file", keep_variables=["ssh"]),
        ref_source_config=Namespace(protocol="file"),
        model="demo",
        list_metrics=[],
        pred_transform=None,
        ref_transform=None,
    )

    assert result["result"] is None
    assert "silently dropping files" in result["error"]


def test_compute_metric_uses_local_fallback_when_shared_store_unavailable(
    tmp_path, monkeypatch,
):
    """Local observation files should still be processed if shared build fails."""

    class RawObsFrame:
        def get_dataframe(self):
            return pd.DataFrame(
                {
                    "path": ["s3://a.nc", "s3://b.nc", "s3://c.nc"],
                    "date_start": [pd.Timestamp("2024-01-01")] * 3,
                    "date_end": [pd.Timestamp("2024-01-02")] * 3,
                }
            )

    local_map = {}
    for name in ("a.nc", "b.nc", "c.nc"):
        local_path = tmp_path / name
        local_path.write_text("dummy")
        local_map[f"s3://{name}"] = str(local_path)

    pred_data = xr.Dataset(
        {"ssh": (("time", "x"), np.array([[1.0]], dtype=np.float32))},
        coords={
            "time": np.array([np.datetime64("2024-01-01T00:00:00")]),
            "x": np.array([0]),
        },
    )

    entry = {
        "forecast_reference_time": pd.Timestamp("2024-01-01T00:00:00"),
        "lead_time": 0,
        "valid_time": pd.Timestamp("2024-01-01T00:00:00"),
        "pred_coords": None,
        "ref_coords": None,
        "ref_alias": "swot",
        "ref_is_observation": True,
        "pred_data": pred_data,
        "ref_data": {
            "source": RawObsFrame(),
            "keep_vars": ["ssh"],
            "target_dimensions": {},
            "time_bounds": (
                pd.Timestamp("2024-01-01T00:00:00"),
                pd.Timestamp("2024-01-02T00:00:00"),
            ),
            "metadata": {
                "coord_system": SimpleNamespace(coordinates={"time": "time"}),
            },
            "prefetched_local_paths": local_map,
        },
    }

    seen = {}

    class FakeMetric:
        def compute(self, pred, ref, pred_coords, ref_coords):
            return pd.DataFrame({"score": [0.0]})

    def fake_preprocess(self, dataframe, load_to_memory=False):
        seen["paths"] = list(dataframe["path"])
        return xr.Dataset(
            {"ssh": ("n_points", np.array([1.0], dtype=np.float32))},
            coords={
                "n_points": np.array([0]),
                "time": ("n_points", np.array([np.datetime64("2024-01-01T00:00:00")])),
            },
        )

    monkeypatch.setenv("DCTOOLS_MAX_OBS_FILES_PER_WORKER", "2")
    monkeypatch.setattr(
        "dctools.metrics.compute_task.create_worker_connect_config",
        lambda config, argo_index=None: (lambda source, alias=None: source),
    )
    monkeypatch.setattr(
        "dctools.metrics.compute_task.filter_by_time",
        lambda dataframe, t0, t1: dataframe,
    )
    monkeypatch.setattr(
        "dctools.data.datasets.batch_preprocessing.preprocess_batch_obs_files",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "dctools.metrics.compute_task.ObservationDataViewer.preprocess_datasets",
        fake_preprocess,
    )

    result = compute_metric(
        entry=entry,
        pred_source_config=Namespace(protocol="file", keep_variables=["ssh"]),
        ref_source_config=Namespace(protocol="file"),
        model="demo",
        list_metrics=[FakeMetric()],
        pred_transform=None,
        ref_transform=None,
    )

    assert "error" not in result
    assert result["result"] is not None
    assert seen["paths"] == [local_map["s3://a.nc"], local_map["s3://b.nc"], local_map["s3://c.nc"]]