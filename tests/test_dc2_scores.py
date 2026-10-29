"""Observation scores on synthetic data: what is compared, at which depth, pooled how.

A perfect forecast scores 0, a known offset is all bias, an SSH anomaly is turned into the absolute
height of a model's zos before it is scored, and profile levels are scored separately.
"""

from argparse import Namespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from dctools.metrics.evaluator import _anomaly_to_absolute_height
from dctools.metrics.oceanbench_metrics import (
    _build_class4_bin_specs,
    _model_level,
    _run_class4_with_raw_per_bins,
)

oceanbench_class4 = pytest.importorskip("oceanbench.core.class4_metrics.class4_evaluator")
if not hasattr(oceanbench_class4, "pool_bins"):
    pytest.skip("oceanbench without pool_bins (fork branch dc2-optimisation)", allow_module_level=True)

LAT = np.arange(-60.0, 60.25, 0.25)
LON = np.arange(-180.0, 180.0, 0.25)
DEPTHS = np.array([0.494025, 47.37369, 222.4752, 541.0889, 1245.291])
VALID = np.datetime64("2024-01-10T00:00")


def _model_field(depths=None):
    """A smooth field, so bilinear interpolation of it is nearly exact."""
    la, lo = np.meshgrid(LAT, LON, indexing="ij")
    surf = np.cos(np.deg2rad(la)) + 0.1 * np.sin(np.deg2rad(lo))
    if depths is None:
        return xr.DataArray(surf, dims=("lat", "lon"), coords={"lat": LAT, "lon": LON})
    vals = np.stack([surf * (1 + d / 1000.0) for d in depths])
    return xr.DataArray(vals, dims=("depth", "lat", "lon"),
                        coords={"depth": depths, "lat": LAT, "lon": LON})


def _points(n, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(-50, 50, n), rng.uniform(-170, 170, n)


def _sample(field, lat, lon):
    return field.interp(lat=xr.DataArray(lat, dims="n_points"),
                        lon=xr.DataArray(lon, dims="n_points")).values


def _evaluator():
    return oceanbench_class4.Class4Evaluator(
        metrics=["rmse"], bin_specs=_build_class4_bin_specs(10), apply_qc=False,
    )


def _scores(result):
    df = result["results"] if isinstance(result, dict) else result
    return {(r.Metric, r.Variable): r.Value for r in df.itertuples()}


# ── SSH anomaly -> absolute height ───────────────────────────────────────────

def _alongtrack(names=("ssha", "mdt")):
    n = 50
    data = {names[0]: ("n_points", np.full(n, 0.10))}
    if len(names) > 1:
        data[names[1]] = ("n_points", np.full(n, 0.50))
    return xr.Dataset(data)


def test_anomaly_becomes_absolute_height():
    out = _anomaly_to_absolute_height(_alongtrack(), Namespace(eval_variables=["ssha"]))
    np.testing.assert_allclose(out["ssha"].values, 0.60)
    assert "mdt" not in out
    # again on an already absolute height: unchanged
    again = _anomaly_to_absolute_height(out, Namespace(eval_variables=["ssha"]))
    np.testing.assert_allclose(again["ssha"].values, 0.60)


def test_anomaly_after_renaming_and_group_prefixed_names():
    renamed = _anomaly_to_absolute_height(
        _alongtrack(("ssh", "mean_dynamic_topography")), Namespace(eval_variables=["ssha_filtered"]))
    np.testing.assert_allclose(renamed["ssh"].values, 0.60)
    jason = _anomaly_to_absolute_height(
        _alongtrack(("data_01__ku__ssha", "data_01__mean_dynamic_topography")),
        Namespace(eval_variables=["data_01__ku__ssha"]))
    np.testing.assert_allclose(jason["data_01__ku__ssha"].values, 0.60)


def test_l3_anomaly_names():
    # DUACS L3 along-track: sla_filtered + mdt (sla_filtered was not an "ssh" alias: no conversion)
    out = _anomaly_to_absolute_height(_alongtrack(("sla_filtered", "mdt")),
                                      Namespace(eval_variables=["sla_filtered"]))
    np.testing.assert_allclose(out["sla_filtered"].values, 0.60)
    from dctools.data.coordinates import get_standardized_var_name
    assert get_standardized_var_name("sla_filtered") == "ssh"


def test_a_declared_anomaly_that_is_not_found_is_an_error():
    with pytest.raises(ValueError, match="no such variable"):
        _anomaly_to_absolute_height(_alongtrack(("adt_like", "mdt")),
                                    Namespace(eval_variables=["sla_filtered"]))


def test_unedited_anomalies_are_dropped():
    ds = _alongtrack()
    ds["ssha"][:3] = [25.0, -30.0, 1.9]           # land/ice values of an L2 product, and a real one
    out = _anomaly_to_absolute_height(ds, Namespace(eval_variables=["ssha"]))
    assert np.isnan(out["ssha"].values[:2]).all()
    assert out["ssha"].values[2] == pytest.approx(2.4)
    kept = _anomaly_to_absolute_height(ds, Namespace(eval_variables=["ssha"], max_abs_sla=None))
    assert kept["ssha"].values[0] == pytest.approx(25.5)


def test_anomaly_without_its_mdt_is_an_error():
    with pytest.raises(ValueError, match="mean dynamic topography"):
        _anomaly_to_absolute_height(_alongtrack(("ssha",)), Namespace(eval_variables=["ssha"]))


def test_absolute_references_are_left_alone():
    ds = _alongtrack()
    assert _anomaly_to_absolute_height(ds, Namespace(eval_variables=["zos"])) is ds
    assert _anomaly_to_absolute_height(ds, Namespace(eval_variables=["ssha"], sla_to_adt=False)) is ds


# ── surface (along-track) scores ─────────────────────────────────────────────

def _surface_obs(values, lat, lon):
    return xr.Dataset(
        {"ssh": ("n_points", values)},
        coords={"lat": ("n_points", lat), "lon": ("n_points", lon),
                "time": ("n_points", np.full(lat.size, VALID))},
    )


def test_perfect_surface_forecast_scores_zero_and_an_offset_is_bias():
    model = _model_field()
    lat, lon = _points(2000)
    truth = _sample(model, lat, lon)
    perfect = _scores(_run_class4_with_raw_per_bins(
        _evaluator(), xr.Dataset({"ssh": model}), _surface_obs(truth, lat, lon), ["ssh"]))
    assert perfect[("rmse", "Surface ssh")] < 1e-3
    assert perfect[("count", "Surface ssh")] == 2000

    shifted = _scores(_run_class4_with_raw_per_bins(
        _evaluator(), xr.Dataset({"ssh": model + 0.3}), _surface_obs(truth, lat, lon), ["ssh"]))
    assert shifted[("bias", "Surface ssh")] == pytest.approx(0.3, abs=1e-3)
    assert shifted[("crmse", "Surface ssh")] < 1e-3


def test_noisier_forecast_scores_worse():
    model = _model_field()
    lat, lon = _points(2000)
    obs = _surface_obs(_sample(model, lat, lon), lat, lon)
    rng = np.random.default_rng(1)
    rmse = []
    for sigma in (0.01, 0.1):
        noisy = model + sigma * rng.standard_normal(model.shape)
        rmse.append(_scores(_run_class4_with_raw_per_bins(
            _evaluator(), xr.Dataset({"ssh": noisy}), obs, ["ssh"]))[("rmse", "Surface ssh")])
    assert rmse[0] < rmse[1]


def test_scores_pool_points_not_bins():
    """One far bin with a single bad point must not weigh as much as a full bin."""
    model = _model_field()
    lat, lon = _points(999, seed=2)
    lat = np.append(lat, 55.0)            # alone in its 10-degree bin
    lon = np.append(lon, 175.0)
    truth = _sample(model, lat, lon)
    truth[-1] += 10.0                     # one bad point
    s = _scores(_run_class4_with_raw_per_bins(
        _evaluator(), xr.Dataset({"ssh": model}), _surface_obs(truth, lat, lon), ["ssh"]))
    # cos(lat) weights: the point at 55N weighs cos(55) relative to the others
    w = np.cos(np.deg2rad(lat))
    expected = np.sqrt(w[-1] * 100.0 / w.sum())
    assert s[("rmse", "Surface ssh")] == pytest.approx(expected, rel=0.02)


# ── profile (Argo-like) scores ───────────────────────────────────────────────

def _profiles(model, lat, lon, offsets):
    vals = np.stack([_sample(model.sel(depth=d), lat, lon) + offsets[i]
                     for i, d in enumerate(DEPTHS)], axis=1)
    return xr.Dataset(
        {"temperature": (("n_points", "depth"), vals)},
        coords={"lat": ("n_points", lat), "lon": ("n_points", lon), "depth": DEPTHS,
                "time": ("n_points", np.full(lat.size, VALID))},
    )


def test_profile_levels_are_scored_separately():
    model = _model_field(DEPTHS)
    lat, lon = _points(300, seed=3)
    offsets = np.array([0.0, 0.5, 0.0, 0.0, 9.0])   # error at 50 m, and deep below 550 m
    s = _scores(_run_class4_with_raw_per_bins(
        _evaluator(), xr.Dataset({"temperature": model}), _profiles(model, lat, lon, offsets),
        ["temperature"]))
    assert s[("rmse", "Surface temperature")] < 1e-3
    assert s[("bias", "50m temperature")] == pytest.approx(-0.5, abs=1e-3)   # model - obs
    assert s[("rmse", "200m temperature")] < 1e-3
    assert s[("rmse", "550m temperature")] < 1e-3
    assert s[("count", "Surface temperature")] == 300


def test_profile_levels_match_the_model_by_depth():
    model = _model_field(DEPTHS)
    # the model's levels in another order: matching by index would compare 0.5 m with 1245 m
    reordered = model.isel(depth=slice(None, None, -1))
    assert float(_model_level(reordered, "depth", 0, 0.494025)["depth"]) == pytest.approx(0.494025)
    # no model level within the tolerance: skipped
    assert _model_level(model, "depth", 0, 3000.0) is None
    lat, lon = _points(200, seed=4)
    s = _scores(_run_class4_with_raw_per_bins(
        _evaluator(), xr.Dataset({"temperature": reordered}),
        _profiles(model, lat, lon, np.zeros(DEPTHS.size)), ["temperature"]))
    assert s[("rmse", "Surface temperature")] < 1e-3
    assert s[("rmse", "550m temperature")] < 1e-3


def test_point_observations_are_scored_at_their_own_depth():
    """Argo velocities: surface drift and parking-depth (~1000 m) drift in one table."""
    depths = np.array([0.494025, 47.37369, 541.0889, 902.3393, 1245.291])
    model = _model_field(depths)                    # linear in depth: vertical interpolation is exact
    lat, lon = _points(400, seed=5)
    pt_depth = np.where(np.arange(400) % 2 == 0, 0.0, 1000.0)
    truth_at = {0.0: _sample(model.isel(depth=0), lat, lon),
                1000.0: _sample(model.interp(depth=1000.0), lat, lon)}
    vals = np.where(pt_depth == 0.0, truth_at[0.0], truth_at[1000.0])
    vals = vals + np.where(pt_depth == 1000.0, 0.2, 0.0)       # an offset at depth only
    obs = xr.Dataset(
        {"u_current": ("n_points", vals)},
        coords={"lat": ("n_points", lat), "lon": ("n_points", lon),
                "depth": ("n_points", pt_depth), "time": ("n_points", np.full(400, VALID))},
    )
    s = _scores(_run_class4_with_raw_per_bins(
        _evaluator(), xr.Dataset({"u_current": model}), obs, ["u_current"]))
    assert s[("rmse", "Surface u_current")] < 1e-3
    assert s[("count", "Surface u_current")] == 200
    assert s[("bias", "1000m u_current")] == pytest.approx(-0.2, abs=1e-3)
    assert s[("count", "1000m u_current")] == 200


def test_depth_range_keeps_the_parking_depth_drift():
    from dctools.metrics.evaluator import _has_subsurface_point_depth, _select_point_depths
    ds = xr.Dataset({"U": ("n_points", [0.3, 0.05, 0.04])},
                    coords={"depth": ("n_points", [0.0, 1000.0, 980.0])})
    out = _select_point_depths(ds, Namespace(depth_range=[100, 3000]))
    assert np.isnan(out["U"].values[0]) and np.isfinite(out["U"].values[1:]).all()
    assert _select_point_depths(ds, Namespace()) is ds
    assert _has_subsurface_point_depth(ds)
    assert not _has_subsurface_point_depth(ds.assign_coords(depth=("n_points", [0.0, 0.5, 0.0])))


def test_evaluation_options_reach_the_connection_parameters():
    from types import SimpleNamespace

    from dctools.data.datasets.dataset import _attach_evaluation_options

    params = SimpleNamespace(protocol="s3")

    class FakeDataset:
        def get_connection_config(self):
            return params

    _attach_evaluation_options(FakeDataset(), {"eval_variables": ["ssha_filtered"], "max_abs_sla": 1.5,
                                               "depth_range": [100, 3000], "keep_variables": ["x"]})
    assert params.source_eval_variables == ["ssha_filtered"]
    assert params.max_abs_sla == 1.5 and params.depth_range == [100, 3000]
    assert not hasattr(params, "keep_variables")
    # what compute_metric then does with them: the raw YAML name triggers the conversion
    out = _anomaly_to_absolute_height(_alongtrack(("ssh", "mean_dynamic_topography")), params)
    np.testing.assert_allclose(out["ssh"].values, 0.60)


def test_results_are_flat_rows():
    model = _model_field()
    lat, lon = _points(100)
    res = _run_class4_with_raw_per_bins(
        _evaluator(), xr.Dataset({"ssh": model}), _surface_obs(_sample(model, lat, lon), lat, lon),
        ["ssh"])
    df = res["results"]
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) == {"Metric", "Variable", "Value"}
    assert {"rmse", "mse", "mae", "bias", "crmse", "count"} <= set(df["Metric"])


def test_per_bins_rows_take_numeric_depth_levels():
    from dctools.processing.base import _pb_raw_to_v2_row

    bins = [{"lat_bin": {"left": 0.0, "right": 2.0}, "lon_bin": {"left": 0.0, "right": 2.0},
             "depth_bin": 541.0889, "rmse": 0.1, "count": 3.0, "w_sum": 2.9},
            {"lat_bin": {"left": 0.0, "right": 2.0}, "lon_bin": {"left": 2.0, "right": 4.0},
             "depth_bin": pd.Interval(10.0, 20.0), "rmse": 0.2, "count": 1.0, "w_sum": 1.0}]
    row = _pb_raw_to_v2_row("argo_profiles", "obs", 0, "2024-01-03", {"temperature": bins})
    col = row["pb"]["temperature"]
    assert col["dl"] == [541.09, 10.0] and col["dr"] == [541.09, 20.0]
    assert "w_sum" not in col and col["rmse"] == [0.1, 0.2]


def test_observation_points_are_kept_within_the_task_window():
    from dctools.data.datasets.dataloader import _points_in_time_bounds

    t = pd.to_datetime(["2023-06-01", "2024-01-02T13:00", "2024-01-03", "2024-01-03T11:00", "2024-12-01"], format="ISO8601")
    ds = xr.Dataset({"U": ("n_points", np.arange(5.0))}, coords={"time": ("n_points", t.values)})
    out = _points_in_time_bounds(ds, "n_points", (pd.Timestamp("2024-01-02T12:00"), pd.Timestamp("2024-01-03T12:00")))
    assert out["U"].values.tolist() == [1.0, 2.0, 3.0]
    assert _points_in_time_bounds(ds, "n_points", (pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-02"))) is None
    assert _points_in_time_bounds(ds, "n_points", None) is ds


def test_gridded_and_observation_scores_share_variable_names():
    from dctools.metrics.evaluator import _canonical_variable_label as name
    assert name("Surface height") == "Surface ssh"
    assert name("50m eastward velocity") == "50m u_current"
    assert name("550m northward velocity") == "550m v_current"
    assert name("200m temperature") == "200m temperature"
    assert name("15m u_current") == "15m u_current"          # observation labels unchanged
