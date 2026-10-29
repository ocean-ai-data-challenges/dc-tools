"""Energy by band of scales: Parseval, the band of a known wave, perfect / smooth / noisy forecasts.

A perfect forecast has energy ratio 1 and error ratio 0 in every band; a smoothed one lacks the
small scales; noisy members have too much there while their mean, smoothed by the averaging,
has less; a deterministic forecast is a one-member ensemble.
"""

import numpy as np
import pytest
import xarray as xr

from dctools.metrics.spectra import (
    BANDS,
    R_EARTH_KM,
    band_scores,
    band_sums,
    segment_power,
    segment_starts,
)

LAT = np.arange(-50.0, 50.25, 0.25)
LON = np.arange(-180.0, 180.0, 0.25)


def test_segment_power_sums_to_the_variance():
    rng = np.random.default_rng(0)
    seg = rng.standard_normal((4000, 64))
    total = segment_power(seg).sum(axis=-1)
    # the trend removed takes ~2/64 of the variance; the window costs a little at the ends
    assert total.mean() == pytest.approx(1.0, abs=0.06)


def test_segments_skip_land():
    ok = np.ones(200, dtype=bool)
    ok[70] = False
    starts = segment_starts(ok, 64)
    assert list(starts) == [0, 71, 135]
    assert all(ok[s:s + 64].all() for s in starts)


def _field(rng, smooth_km=None):
    """A field with energy at all scales (red noise along longitude)."""
    f = rng.standard_normal((LAT.size, LON.size)).cumsum(axis=1)
    f -= f.mean(axis=1, keepdims=True)
    if smooth_km:
        from scipy.ndimage import uniform_filter1d
        f = uniform_filter1d(f, size=max(2, int(smooth_km / 27.8)), axis=1, mode="wrap")
    return f


def test_a_wave_lands_in_its_band():
    # a 300 km wave at the equator: 300 km / 27.8 km per point
    dx = np.deg2rad(0.25) * R_EARTH_KM
    x = np.arange(LON.size) * dx
    ref = np.tile(np.sin(2 * np.pi * x / 300.0), (3, 1))
    sums = band_sums(ref[None], ref, np.zeros(3), LON)
    names = list(BANDS)
    assert np.argmax(sums["ref"]) == names.index("200_500km")
    assert sums["ref"][names.index("200_500km")] > 0.95 * sums["ref"].sum()


def test_perfect_smooth_and_noisy_forecasts():
    rng = np.random.default_rng(1)
    ref = _field(rng)
    perfect = band_scores(band_sums(ref[None], ref, LAT, LON), 1)
    for b in BANDS:
        assert perfect[f"energy_ratio_{b}"] == pytest.approx(1.0)
        assert perfect[f"err_ratio_{b}"] == pytest.approx(0.0, abs=1e-12)
    assert not any(k.startswith("energy_ratio_mean") for k in perfect)   # deterministic

    smooth = band_scores(band_sums(_field(np.random.default_rng(1), 300)[None], ref, LAT, LON), 1)
    assert smooth["energy_ratio_lt100km"] < 0.3 < smooth["energy_ratio_gt500km"]

    noise = 0.5 * np.sqrt(ref.var()) * rng.standard_normal((8,) + ref.shape) / 10
    members = ref[None] + noise
    s = band_scores(band_sums(members, ref, LAT, LON), 8)
    assert s["energy_ratio_lt100km"] > 1.0                    # the members are noisy
    assert s["energy_ratio_mean_lt100km"] < s["energy_ratio_lt100km"]   # their mean smoother
    assert s["err_ratio_lt100km"] > s["err_ratio_gt500km"]


def test_land_is_left_out():
    rng = np.random.default_rng(2)
    ref = _field(rng)
    ref[:, 100:300] = np.nan                                  # a continent
    s = band_scores(band_sums(ref[None], ref, LAT, LON), 1)
    assert s["energy_ratio_100_200km"] == pytest.approx(1.0)


def test_gridded_scores_have_spectra_and_ensemble_maps():
    from dctools.metrics.ensemble import gridded_ensemble_scores
    from dctools.metrics.evaluator import _merge_ensemble_bins

    rng = np.random.default_rng(3)
    ref = xr.DataArray(_field(rng), dims=("lat", "lon"), coords={"lat": LAT, "lon": LON})
    members = xr.concat([ref + 0.1 * k + 0.05 for k in (-1, 0, 1)], dim="member")
    det_scores = gridded_ensemble_scores(xr.Dataset({"ssh": ref}), xr.Dataset({"ssh": ref}),
                                         ["ssh"], None)
    assert det_scores["energy_ratio_100_200km"]["None ssh"] == pytest.approx(1.0)
    assert "crps" not in det_scores                           # a deterministic forecast
    scores, bins = gridded_ensemble_scores(xr.Dataset({"ssh": members}), xr.Dataset({"ssh": ref}),
                                           ["ssh"], None, bin_resolution=10)
    assert scores["crps"]["None ssh"] > 0 and "energy_ratio_mean_lt100km" in scores
    assert {"crps", "crps_c", "spread", "ssr_c"} <= set(bins["ssh"][0])
    # merged into the deterministic bins of the same cells
    det = {"ssh": [{"lat_bin": dict(b["lat_bin"]), "lon_bin": dict(b["lon_bin"]), "rmse": 0.0}
                   for b in bins["ssh"]]}
    _merge_ensemble_bins(det, bins)
    assert all("crps" in b and "rmse" in b for b in det["ssh"])
