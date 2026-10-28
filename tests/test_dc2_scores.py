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
