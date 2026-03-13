"""Regression tests for ARGO preprocessing coherence in dataloader."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import xarray as xr

from dctools.data.datasets.dataloader import ObservationDataViewer
from dctools.data.datasets.dataloader import preprocess_argo_profiles


def test_argo_preprocess_skips_first_file_probe(monkeypatch):
    """ARGO path must not open the first path outside preprocess_argo_profiles."""
    called = {"preprocess": False, "open_probe": False}

    def fake_load_fn(*args, **kwargs):
        called["open_probe"] = True
        raise AssertionError("load_fn should not be called before ARGO preprocessing")

    def fake_preprocess_argo_profiles(**kwargs):
        called["preprocess"] = True
        return xr.Dataset(
            {"temperature": ("N_POINTS", np.array([10.0], dtype=np.float32))},
            coords={
                "N_POINTS": np.array([0]),
                "time": np.array([np.datetime64("2024-01-01")]),
            },
        )

    monkeypatch.setattr(
        "dctools.data.datasets.observation_viewer.preprocess_argo_profiles",
        fake_preprocess_argo_profiles,
    )

    metadata = {
        "coord_system": SimpleNamespace(coordinates={"n_points": "N_POINTS"}),
    }

    viewer = ObservationDataViewer(
        source=pd.DataFrame({"path": ["12345:1"]}),
        load_fn=fake_load_fn,
        alias="argo_profiles",
        keep_vars=["temperature"],
        target_dimensions={"depth": np.array([0.0, 10.0], dtype=np.float32)},
        dataset_metadata=metadata,
        time_bounds=(pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")),
        n_points_dim="N_POINTS",
        results_dir="/tmp",
    )
    viewer.save_preprocessed = False

    result = viewer.preprocess_datasets(viewer.meta_df)

    assert result is not None
    assert called["preprocess"] is True
    assert called["open_probe"] is False


def test_monthly_sources_use_kerchunk_window_open(monkeypatch):
    """Monthly-index sources must use Kerchunk window open path."""

    class FakeArgoManager:
        def __init__(self):
            self.window_open_calls = 0

        def open(self, path, *args, **kwargs):
            if isinstance(path, tuple):
                self.window_open_calls += 1
            return xr.Dataset(
                {
                    "depth": ("N_POINTS", np.array([0.0, 10.0], dtype=np.float32)),
                    "TEMP": ("N_POINTS", np.array([10.0, 9.0], dtype=np.float32)),
                },
                coords={
                    "N_POINTS": np.array([0, 1]),
                    "LATITUDE": ("N_POINTS", np.array([45.0, 45.0], dtype=np.float32)),
                    "LONGITUDE": ("N_POINTS", np.array([5.0, 5.0], dtype=np.float32)),
                    "TIME": (
                        "N_POINTS",
                        np.array(
                            [
                                np.datetime64("2024-01-01T00:00:00"),
                                np.datetime64("2024-01-01T01:00:00"),
                            ]
                        ),
                    ),
                },
            )

    fake_manager = FakeArgoManager()

    monkeypatch.setattr(
        "dctools.data.datasets.preprocessing.ArgoManager",
        FakeArgoManager,
    )

    result = preprocess_argo_profiles(
        profile_sources=["2024_01", "2024_02"],
        open_func=fake_manager.open,
        alias="argo_profiles",
        time_bounds=(pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")),
        depth_levels=np.array([0.0, 10.0], dtype=np.float32),
    )

    assert result is not None
    assert fake_manager.window_open_calls == 1
