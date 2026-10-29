"""Ensemble forecasts on synthetic data: fair CRPS, spread, spread-skill ratio.

A perfect ensemble scores 0, a one-member ensemble's CRPS is its MAE, the fair CRPS of a reliable
ensemble does not depend on its size, an under-dispersive ensemble has ssr < 1, and the
deterministic scores of an ensemble are those of its mean.
"""

import numpy as np
import pytest
import xarray as xr

from dctools.metrics.ensemble import (
    fair_crps,
    gridded_ensemble_scores,
    member_dim,
    point_sums,
    scores_from_sums,
)
from dctools.metrics.oceanbench_metrics import (
    _build_class4_bin_specs,
    _run_class4_with_raw_per_bins,
)

oceanbench_class4 = pytest.importorskip("oceanbench.core.class4_metrics.class4_evaluator")
if not hasattr(oceanbench_class4, "pool_bins"):
    pytest.skip("oceanbench without pool_bins (fork branch dc2-optimisation)",
                allow_module_level=True)

LAT = np.arange(-60.0, 60.25, 0.25)
LON = np.arange(-180.0, 180.0, 0.25)
VALID = np.datetime64("2024-01-10T00:00")


def _crps_brute(x, y):
    m = x.shape[0]
    pair = np.abs(x[:, None] - x[None, :]).sum(axis=(0, 1))
    return np.abs(x - y).mean(axis=0) - pair / (2 * m * (m - 1))


# ── the score itself ─────────────────────────────────────────────────────────

def test_fair_crps_is_the_pairwise_formula():
    rng = np.random.default_rng(0)
    x, y = rng.standard_normal((7, 100)), rng.standard_normal(100)
    np.testing.assert_allclose(fair_crps(x, y), _crps_brute(x, y), atol=1e-12)


def test_perfect_ensemble_scores_zero_and_one_member_is_mae():
    y = np.linspace(-1, 1, 50)
    assert np.allclose(fair_crps(np.tile(y, (10, 1)), y), 0.0)
    x = y[None] + 0.3
    np.testing.assert_allclose(fair_crps(x, y), 0.3)


def test_fair_crps_does_not_depend_on_the_ensemble_size():
    # members and truth from the same N(0, 1): expected fair CRPS 1/sqrt(pi) for any M
    rng = np.random.default_rng(1)
    n = 200_000
    y = rng.standard_normal(n)
    for m in (3, 50):
        crps = fair_crps(rng.standard_normal((m, n)), y).mean()
        assert crps == pytest.approx(1 / np.sqrt(np.pi), rel=0.01)


def test_spread_skill_ratio():
    rng = np.random.default_rng(2)
    n, m = 100_000, 20
    centre = rng.standard_normal(n)
    y = centre + rng.standard_normal(n)
    reliable = centre + rng.standard_normal((m, n))
    narrow = centre + 0.3 * rng.standard_normal((m, n))
    s_rel = scores_from_sums(point_sums(reliable, y))
    s_nar = scores_from_sums(point_sums(narrow, y))
    assert s_rel["ssr"] == pytest.approx(1.0, abs=0.02)
    assert s_rel["spread"] == pytest.approx(1.0, abs=0.01)
    assert s_nar["ssr"] < 0.4
    # the wider (reliable) ensemble has the better CRPS, the members being equally centred
    assert s_rel["crps"] < s_nar["crps"]


def test_pooled_bins_are_all_their_points():
    rng = np.random.default_rng(3)
    x, y = rng.standard_normal((5, 1000)), rng.standard_normal(1000)
    bins = []
    for sl in (slice(0, 10), slice(10, 1000)):
        sc = scores_from_sums(point_sums(x[:, sl], y[sl]))
        mse = float(((x[:, sl].mean(0) - y[sl]) ** 2).mean())
        bins.append({"mse": mse, "count": float(y[sl].size), **sc})
    pooled = oceanbench_class4.pool_bins(bins)
    whole = scores_from_sums(point_sums(x, y))
    assert pooled["crps"] == pytest.approx(whole["crps"])
    assert pooled["spread"] == pytest.approx(whole["spread"])
    assert pooled["ssr"] == pytest.approx(whole["ssr"])
    # a deterministic forecast has none of them
    assert "crps" not in oceanbench_class4.pool_bins([{"mse": 1.0, "count": 3.0}])


# ── Class 4: members interpolated to the observations ───────────────────────

def _field():
    la, lo = np.meshgrid(LAT, LON, indexing="ij")
    return xr.DataArray(np.cos(np.deg2rad(la)) + 0.1 * np.sin(np.deg2rad(lo)),
                        dims=("lat", "lon"), coords={"lat": LAT, "lon": LON})


def _obs(field, n=3000, seed=0):
    rng = np.random.default_rng(seed)
    lat, lon = rng.uniform(-50, 50, n), rng.uniform(-170, 170, n)
    vals = field.interp(lat=xr.DataArray(lat, dims="n_points"),
                        lon=xr.DataArray(lon, dims="n_points")).values
    return xr.Dataset({"ssh": ("n_points", vals)},
                      coords={"lat": ("n_points", lat), "lon": ("n_points", lon),
                              "time": ("n_points", np.full(n, VALID))})


def _ensemble(field, offsets):
    return xr.concat([field + o for o in offsets], dim="member").assign_coords(
        member=np.arange(1, len(offsets) + 1))


def _class4(model, obs):
    ev = oceanbench_class4.Class4Evaluator(metrics=["rmse"], bin_specs=_build_class4_bin_specs(10),
                                           apply_qc=False)
    res = _run_class4_with_raw_per_bins(ev, xr.Dataset({"ssh": model}), obs, ["ssh"])
    df = res["results"]
    scores = {r.Metric: r.Value for r in df.itertuples() if r.Variable == "Surface ssh"}
    return scores, res["per_bins"]


def test_class4_perfect_ensemble_and_ensemble_mean():
    field = _field()
    obs = _obs(field)
    perfect, _ = _class4(_ensemble(field, [0.0, 0.0, 0.0]), obs)
    assert perfect["crps"] < 1e-6 and perfect["rmse"] < 1e-3 and perfect["spread"] < 1e-6
    # symmetric members around the truth: the ensemble mean is perfect, the crps is not 0
    s, per_bins = _class4(_ensemble(field, [-0.2, 0.0, 0.2]), obs)
    assert s["rmse"] < 1e-3 and s["bias"] == pytest.approx(0.0, abs=1e-6)
    x = np.array([-0.2, 0.0, 0.2])[:, None]
    assert s["crps"] == pytest.approx(float(_crps_brute(x, np.zeros(1))[0]), rel=1e-6)
    assert s["spread"] == pytest.approx(0.2, rel=1e-6)
    assert s["count"] == 3000
    # the per-bins maps carry the ensemble scores
    assert all("crps" in b and "spread" in b for b in per_bins["ssh"])


def test_class4_offset_ensemble_and_a_single_member():
    field = _field()
    obs = _obs(field)
    s, _ = _class4(_ensemble(field, [0.3, 0.3]), obs)
    assert s["bias"] == pytest.approx(0.3, abs=1e-3) and s["crps"] == pytest.approx(0.3, abs=1e-3)
    one, _ = _class4(_ensemble(field, [0.3]), obs)   # one member: crps = mae
    assert one["crps"] == pytest.approx(one["mae"], rel=1e-6)


def test_class4_wider_reliable_ensemble_beats_a_narrow_one():
    field = _field()
    obs = _obs(field)
    rng = np.random.default_rng(4)
    truth_noise = 0.1 * rng.standard_normal(obs.sizes["n_points"])
    obs = obs.assign(ssh=obs["ssh"] + truth_noise)   # observation error the members should span
    reliable = _ensemble(field, [0.0] * 8) + 0.1 * rng.standard_normal((8,) + field.shape)
    narrow = _ensemble(field, [0.0] * 8) + 0.01 * rng.standard_normal((8,) + field.shape)
    s_rel, _ = _class4(reliable, obs)
    s_nar, _ = _class4(narrow, obs)
    assert s_nar["ssr"] < 0.3 < s_rel["ssr"]


# ── gridded references ───────────────────────────────────────────────────────

def test_gridded_scores_under_the_deterministic_labels():
    from oceanbench.core.rmsd import DEPTH_LABELS, VARIABLE_LABELS, Variable

    depth = np.array([0.494025, 47.37369, 222.4752, 541.0889, 1245.291])
    lat, lon = np.arange(-10.0, 10.25, 0.25), np.arange(0.0, 20.0, 0.25)
    values = np.random.default_rng(5).standard_normal((depth.size, lat.size, lon.size))
    truth = xr.DataArray(values, dims=("depth", "lat", "lon"),
                         coords={"depth": depth, "lat": lat, "lon": lon},
                         attrs={"standard_name": "sea_water_potential_temperature"})
    members = _ensemble(truth, [-0.1, 0.3]).assign_attrs(truth.attrs)   # mean off by 0.1
    pred = xr.Dataset({"temperature": members})
    ref = xr.Dataset({"temperature": truth})
    assert member_dim(pred) == "member"

    def label(v, level, has_depth):
        return f"{DEPTH_LABELS[level]} {VARIABLE_LABELS[v]}".capitalize()

    out = gridded_ensemble_scores(pred, ref, [Variable.SEA_WATER_POTENTIAL_TEMPERATURE],
                                  list(DEPTH_LABELS), labels=label)
    assert set(out["crps"]) == {"Surface temperature", "50m temperature", "200m temperature",
                                "550m temperature"}
    x = np.array([-0.1, 0.3])[:, None]
    assert out["crps"]["50m temperature"] == pytest.approx(float(_crps_brute(x, np.zeros(1))[0]))
    spread = np.sqrt(0.08)                      # variance (ddof 1) of (-0.1, 0.3)
    assert out["spread"]["Surface temperature"] == pytest.approx(spread)
    assert out["ssr"]["Surface temperature"] == pytest.approx(np.sqrt(3 / 2) * spread / 0.1)


# ── centred scores (crps_c, ssr_c): the bias removed, as crmse ───────────────

def test_centred_scores_remove_the_bias_only():
    rng = np.random.default_rng(6)
    n, m = 50_000, 10
    centre = rng.standard_normal(n)
    y = centre + rng.standard_normal(n)
    members = centre + rng.standard_normal((m, n))
    plain = scores_from_sums(point_sums(members, y))
    shifted = scores_from_sums(point_sums(members + 0.7, y))      # a reference-surface offset
    assert shifted["crps"] > plain["crps"] + 0.1
    assert shifted["crps_c"] == pytest.approx(plain["crps_c"], rel=1e-9)
    assert shifted["ssr_c"] == pytest.approx(plain["ssr_c"], rel=1e-9)
    assert shifted["ssr"] < plain["ssr"]
    # a pure offset: crps is the offset, the centred one is 0
    pure = scores_from_sums(point_sums(np.tile(y + 0.3, (4, 1)), y))
    assert pure["crps"] == pytest.approx(0.3) and pure["crps_c"] == pytest.approx(0.0, abs=1e-12)


def test_class4_centred_scores():
    field = _field()
    obs = _obs(field)
    rng = np.random.default_rng(7)
    obs = obs.assign(ssh=obs["ssh"] + 0.1 * rng.standard_normal(obs.sizes["n_points"]))
    members = _ensemble(field, [0.0] * 6) + 0.1 * rng.standard_normal((6,) + field.shape)
    plain, _ = _class4(members, obs)
    shifted, per_bins = _class4(members + 0.5, obs)
    assert shifted["bias"] == pytest.approx(plain["bias"] + 0.5, abs=1e-6)
    assert shifted["crps"] > plain["crps"] + 0.3
    assert shifted["crps_c"] == pytest.approx(plain["crps_c"], rel=1e-6)
    assert shifted["ssr_c"] == pytest.approx(plain["ssr_c"], rel=1e-6)
    assert all("crps_c" in b for b in per_bins["ssh"])


def test_a_nan_score_does_not_drop_its_batch(tmp_path):
    """A NaN in a batch file made the consolidation's orjson reject (and skip) the whole batch."""
    import gzip
    import json

    from dctools.metrics.evaluator import _nan_to_none
    from dctools.processing.base import _iter_batch_items

    rows = [{"ref_alias": "argo_velocities",
             "result": [{"Metric": "ssr_c", "Variable": "550m u_current", "Value": float("nan")},
                        {"Metric": "rmse", "Variable": "550m u_current", "Value": 0.1}]}]
    old = tmp_path / "results_m_batch_0.json.gz"
    with gzip.open(old, "wt") as f:
        json.dump(rows, f)                       # what the writer did: a NaN token
    assert [r["ref_alias"] for r in _iter_batch_items(str(old))] == ["argo_velocities"]
    clean = _nan_to_none(rows)
    assert clean[0]["result"][0]["Value"] is None and clean[0]["result"][1]["Value"] == 0.1
    json.dumps(clean, allow_nan=False)           # strict JSON now
    # undefined ratios are not emitted: a perfect ensemble mean has no ssr
    assert "ssr" not in scores_from_sums(point_sums(np.array([[-1.0, 1.0], [1.0, -1.0]]),
                                                    np.zeros(2)))


def test_class4_bins_carry_the_spread_skill_ratios():
    field = _field()
    obs = _obs(field)
    rng = np.random.default_rng(8)
    obs = obs.assign(ssh=obs["ssh"] + 0.1 * rng.standard_normal(obs.sizes["n_points"]))
    members = _ensemble(field, [0.0] * 6) + 0.1 * rng.standard_normal((6,) + field.shape)
    _, per_bins = _class4(members + 0.5, obs)
    bins = [b for b in per_bins["ssh"] if b["count"] >= 5]
    assert bins and all("ssr" in b and "ssr_c" in b for b in bins)
    # the offset makes ssr small; removing the level's bias brings it back near the reliable 1
    assert np.median([b["ssr"] for b in bins]) < 0.5 < np.median([b["ssr_c"] for b in bins])
