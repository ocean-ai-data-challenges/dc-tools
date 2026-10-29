"""Energy by band of scales of a forecast against a gridded reference, for ensembles too.

The point scores cannot tell a smooth forecast from a sharp one slightly off (the double
penalty), nor an ensemble whose members are realistic from one whose members are blurred. As
WeatherBench 2 does with its energy spectra (Rasp et al. 2024), and oceanml3d-eval's
``spectral_score`` / ``alongtrack_psd`` with their bands, the variance of the fields is split by
scale along longitude:

1. every latitude row of a level is cut into segments of ``SEGMENT_POINTS`` consecutive grid
   points where the reference and every member are ocean (land ends a run; a row holds
   several segments or none), the same segments for all the fields;
2. each segment is detrended, Hann-windowed and Fourier-transformed; its power at each
   wavenumber goes to the band (``BANDS``) of the wavelength it stands for there (the segment's
   length in km depends on its latitude: 64 points are 1780 km at the equator, 890 km at 60N);
3. the band variances are summed over segments, for the reference, each member, the ensemble
   mean and the error of the mean.

Scores, per band ``<b>`` (``gt500km``, ``200_500km``, ``100_200km``, ``lt100km``):

* ``energy_ratio_<b>``: the members' variance over the reference's (the mean over the members);
  1 for realistic members, < 1 blurred, > 1 noisy. A deterministic forecast is its one member;
* ``energy_ratio_mean_<b>``: the same for the ensemble mean (ensembles only; < 1 is expected,
  the mean smooths the scales the members disagree on);
* ``err_ratio_<b>``: the variance of the error of the ensemble mean (of the forecast) over the
  reference's; 0 perfect, 1 no skill at those scales.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

R_EARTH_KM = 6371.0
SEGMENT_POINTS = 64
BANDS: Dict[str, tuple] = {
    "gt500km": (500.0, None),
    "200_500km": (200.0, 500.0),
    "100_200km": (100.0, 200.0),
    "lt100km": (None, 100.0),
}


def segment_starts(ok_row: np.ndarray, n: int = SEGMENT_POINTS) -> np.ndarray:
    """Start indices of the non-overlapping n-point runs of ``True`` in one row."""
    starts = []
    i, size = 0, ok_row.size
    while i + n <= size:
        bad = np.flatnonzero(~ok_row[i:i + n])
        if bad.size:
            i += int(bad[-1]) + 1
            continue
        starts.append(i)
        i += n
    return np.asarray(starts, dtype=int)


def band_index(n: int, dx_km: np.ndarray) -> np.ndarray:
    """Band of each positive wavenumber (1..n//2) of segments of ``n`` points spaced ``dx_km``.

    Returns (n_segments, n//2) band indices into ``BANDS`` (-1: outside every band).
    """
    j = np.arange(1, n // 2 + 1, dtype=np.float64)
    wavelength = (n * np.asarray(dx_km, dtype=np.float64))[:, None] / j[None, :]
    out = np.full(wavelength.shape, -1, dtype=int)
    for b, (lo, hi) in enumerate(BANDS.values()):
        sel = np.ones(wavelength.shape, dtype=bool)
        if lo is not None:
            sel &= wavelength >= lo
        if hi is not None:
            sel &= wavelength < hi
        out[sel & (out < 0)] = b
    return out


def segment_power(seg: np.ndarray) -> np.ndarray:
    """Variance per positive wavenumber of segments (..., n): detrended, Hann, Parseval."""
    n = seg.shape[-1]
    x = np.arange(n, dtype=np.float64)
    xc = x - x.mean()
    slope = (seg * xc).sum(-1, keepdims=True) / (xc ** 2).sum()
    seg = seg - seg.mean(-1, keepdims=True) - slope * xc
    w = np.hanning(n)
    # sum over k of |X_k|^2 / (n sum(w^2)) is the variance of the (windowed, renormalised) segment
    spec = np.abs(np.fft.rfft(seg * w, axis=-1)) ** 2 / (n * (w ** 2).sum())
    if n % 2 == 0:
        spec[..., 1:-1] *= 2.0                        # one-sided: all but the mean and Nyquist
    else:
        spec[..., 1:] *= 2.0
    return spec[..., 1:]


def band_sums(members: np.ndarray, ref: np.ndarray, lat: np.ndarray, lon: np.ndarray,
              n: int = SEGMENT_POINTS) -> Optional[Dict[str, np.ndarray]]:
    """Band variances summed over the segments of one level.

    ``members`` (M, lat, lon), ``ref`` (lat, lon), 1-D ``lat``/``lon`` (degrees). Returns
    {"ref", "members" (mean over members), "mean", "err"} -> (n_bands,) arrays, and
    "n_segments"; None without any segment.
    """
    members = np.asarray(members, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    ok = np.isfinite(ref) & np.all(np.isfinite(members), axis=0)
    dlon = float(np.abs(np.diff(lon)).mean())
    rows, starts = [], []
    for j in range(ok.shape[0]):
        s = segment_starts(ok[j], n)
        rows.extend([j] * s.size)
        starts.extend(s.tolist())
    if not rows:
        return None
    rows_a, starts_a = np.asarray(rows), np.asarray(starts)
    cols = starts_a[:, None] + np.arange(n)[None, :]
    ref_seg = ref[rows_a[:, None], cols]                          # (S, n)
    mem_seg = members[:, rows_a[:, None], cols]                   # (M, S, n)
    mean_seg = mem_seg.mean(axis=0)
    dx_km = np.deg2rad(dlon) * R_EARTH_KM * np.cos(np.deg2rad(np.asarray(lat)[rows_a]))
    bands = band_index(n, dx_km)                                  # (S, n//2)
    n_bands = len(BANDS)

    def _per_band(power: np.ndarray) -> np.ndarray:               # power (..., S, n//2)
        out = np.zeros(power.shape[:-2] + (n_bands,))
        for b in range(n_bands):
            out[..., b] = np.where(bands == b, power, 0.0).sum(axis=(-2, -1))
        return out

    return {
        "ref": _per_band(segment_power(ref_seg)),
        "members": _per_band(segment_power(mem_seg)).mean(axis=0),
        "mean": _per_band(segment_power(mean_seg)),
        "err": _per_band(segment_power(mean_seg - ref_seg)),
        "n_segments": np.asarray(float(len(rows))),
    }


def band_scores(sums: Optional[Dict[str, np.ndarray]], n_members: int) -> Dict[str, float]:
    """energy_ratio_<b>, energy_ratio_mean_<b> (ensembles), err_ratio_<b> from ``band_sums``."""
    if sums is None:
        return {}
    out: Dict[str, float] = {}
    for b, name in enumerate(BANDS):
        ref = float(sums["ref"][b])
        if ref <= 0:
            continue
        out[f"energy_ratio_{name}"] = float(sums["members"][b]) / ref
        if n_members > 1:
            out[f"energy_ratio_mean_{name}"] = float(sums["mean"][b]) / ref
        out[f"err_ratio_{name}"] = float(sums["err"][b]) / ref
    return out
