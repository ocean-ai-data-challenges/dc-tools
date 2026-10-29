"""Scores of ensemble (probabilistic) forecasts: CRPS, spread and spread-skill ratio.

A forecast with a member dimension (``member``, ``ens``, ``realization`` or ``number``) is scored
twice: its ensemble mean through the deterministic scores (rmse, bias, crmse...: comparable with
the deterministic models), and its members through

- ``crps``: the fair CRPS (Ferro 2014), mean over the points of
  ``mean_i |x_i - y| - sum_ij |x_i - x_j| / (2 M (M - 1))``, unbiased for the finite ensemble
  (an M-member ensemble is not penalised for having M members). It is the MAE for one member;
- ``spread``: the root of the mean ensemble variance (unbiased, ddof 1);
- ``ssr``: the spread-skill ratio ``sqrt((M + 1) / M) * spread / rmse(ensemble mean)``
  (Fortin et al. 2014), 1 for a reliable ensemble, < 1 under-dispersive;
- ``crps_c`` and ``ssr_c``: the same once the bias is removed, as ``crmse`` is the rmse once the
  bias is removed. The bias is that of the ensemble mean over the points the scores pool (``bias``):
  ``crps_c`` is the CRPS of the members minus it, ``ssr_c`` divides the spread by ``crmse``. They
  are what an SSH score reads, the model's and the observations' reference surfaces differing by
  a constant.

The points are weighted as in the deterministic scores (cos(latitude)). Class 4 bins carry crps,
ens_var and n_members, pooled over bins by oceanbench's ``pool_bins`` with the other scores.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import xarray as xr

MEMBER_DIMS = ("member", "ens", "realization", "number")
ENSEMBLE_SCORES = ("crps", "spread", "ssr", "crps_c", "ssr_c")


def member_dim(ds: Any) -> Optional[str]:
    """The member dimension of a forecast (Dataset or DataArray), None for a deterministic one."""
    dims = getattr(ds, "dims", ())
    return next((d for d in MEMBER_DIMS if d in dims), None)


def ensemble_mean(ds: Any, dim: Optional[str] = None) -> Any:
    """The ensemble mean, attributes kept (they carry the standard names the scores look up)."""
    dim = dim or member_dim(ds)
    if dim is None:
        return ds
    return ds.mean(dim, keep_attrs=True)


def fair_crps(members: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """Fair CRPS per point of ``members`` (M, ...) against ``obs`` (...).

    ``sum_ij |x_i - x_j|`` is ``2 sum_k (2k - M - 1) x_(k)`` over the sorted members (O(M log M)
    instead of O(M^2)).
    """
    x = np.asarray(members, dtype=np.float64)
    y = np.asarray(obs, dtype=np.float64)
    m = x.shape[0]
    skill = np.mean(np.abs(x - y[None]), axis=0)
    if m == 1:
        return skill
    k = np.arange(1, m + 1, dtype=np.float64).reshape((m,) + (1,) * (x.ndim - 1))
    pairs = 2.0 * np.sum((2.0 * k - m - 1.0) * np.sort(x, axis=0), axis=0)
    return skill - pairs / (2.0 * m * (m - 1))


def point_sums(members: np.ndarray, obs: np.ndarray, weights: Optional[np.ndarray] = None,
               ) -> Dict[str, float]:
    """Weighted sums over points (the members finite and the observation too) for pooling.

    ``members`` (M, N), ``obs`` (N). Returns w_sum, w_crps, w_var, w_sq and w_err (squared and
    signed error of the ensemble mean), w_crps_c (crps of the members centred on the points' bias),
    count, n_members.
    """
    x = np.asarray(members, dtype=np.float64)
    y = np.asarray(obs, dtype=np.float64).ravel()
    x = x.reshape(x.shape[0], -1)
    w = np.ones_like(y) if weights is None else np.asarray(weights, dtype=np.float64).ravel()
    ok = np.isfinite(y) & np.all(np.isfinite(x), axis=0) & np.isfinite(w)
    x, y, w = x[:, ok], y[ok], w[ok]
    m = x.shape[0]
    out = {"w_sum": float(w.sum()), "count": float(y.size), "n_members": float(m),
           "w_crps": 0.0, "w_var": 0.0, "w_sq": 0.0, "w_err": 0.0, "w_crps_c": 0.0}
    if y.size == 0:
        return out
    out["w_crps"] = float((w * fair_crps(x, y)).sum())
    if m > 1:
        out["w_var"] = float((w * x.var(axis=0, ddof=1)).sum())
    err = x.mean(axis=0) - y
    out["w_sq"] = float((w * err ** 2).sum())
    out["w_err"] = float((w * err).sum())
    bias = out["w_err"] / out["w_sum"] if out["w_sum"] > 0 else 0.0
    out["w_crps_c"] = float((w * fair_crps(x - bias, y)).sum())
    return out


def add_sums(acc: Dict[str, float], sums: Dict[str, float]) -> Dict[str, float]:
    for key in ("w_sum", "count", "w_crps", "w_var", "w_sq", "w_err", "w_crps_c"):
        acc[key] = acc.get(key, 0.0) + sums.get(key, 0.0)
    acc["n_members"] = sums.get("n_members", acc.get("n_members", 0.0))
    return acc


def scores_from_sums(s: Dict[str, float]) -> Dict[str, float]:
    """Ensemble scores from weighted sums.

    crps, spread, ssr, crps_c, ssr_c, and ens_var (the mean variance, for pooling).
    """
    ws = s.get("w_sum", 0.0)
    if not ws:
        return {}
    m = s.get("n_members", 0.0)
    out = {"crps": s["w_crps"] / ws, "ens_var": s["w_var"] / ws, "n_members": m}
    out["spread"] = float(np.sqrt(out["ens_var"]))
    mse = s["w_sq"] / ws
    bias = s.get("w_err", 0.0) / ws
    rmse, crmse = float(np.sqrt(mse)), float(np.sqrt(max(mse - bias ** 2, 0.0)))
    out["crps_c"] = s.get("w_crps_c", 0.0) / ws
    # the ratios only when defined (one member, or a perfect ensemble mean: none)
    if m > 1:
        factor = float(np.sqrt((m + 1.0) / m))
        if rmse > 0:
            out["ssr"] = factor * out["spread"] / rmse
        if crmse > 0:
            out["ssr_c"] = factor * out["spread"] / crmse
    return out


# --------------------------------------------------------------------------- gridded references

def _find(ds: xr.Dataset, variable: Any, aliases: Dict[str, Any]) -> Optional[str]:
    """The name of ``variable`` (an oceanbench Variable or a name) in ``ds``."""
    if isinstance(variable, str):
        return variable if variable in ds.data_vars else None
    try:
        name = variable.variable_name_from_dataset(ds)
        if name in ds.data_vars:
            return name
    except Exception:
        pass
    return next((raw for raw, v in aliases.items() if v == variable and raw in ds.data_vars), None)


def gridded_levels(pred: xr.Dataset, ref: xr.Dataset, variables: List[Any],
                   depth_levels: Optional[List[Any]], aliases: Optional[Dict[str, Any]] = None):
    """(variable, level, has_depth, name in pred, pred level, ref level) of each scored level.

    The variables, depth levels (those of both datasets within 1e-3 m) and the surface of the
    variables without depth, as the deterministic gridded scores have them.
    """
    aliases = aliases or {}
    for variable in variables:
        p_name, r_name = _find(pred, variable, aliases), _find(ref, variable, aliases)
        if p_name is None or r_name is None:
            continue
        p_da, r_da = pred[p_name], ref[r_name]
        has_depth = "depth" in p_da.dims and "depth" in r_da.dims
        levels = list(depth_levels) if (depth_levels is not None and has_depth) else [None]
        for level in levels:
            if level is not None:
                z = float(getattr(level, "value", level))
                p_lv = p_da.sel(depth=z, method="nearest")
                r_lv = r_da.sel(depth=z, method="nearest")
                if abs(float(p_lv["depth"]) - z) > 1e-3 or abs(float(r_lv["depth"]) - z) > 1e-3:
                    continue
            else:
                p_lv = p_da.isel(depth=0) if "depth" in p_da.dims else p_da
                r_lv = r_da.isel(depth=0) if "depth" in r_da.dims else r_da
            yield variable, level, has_depth, p_name, p_lv, r_lv


def level_arrays(members: xr.DataArray, ref: xr.DataArray, dim: Optional[str]):
    """(members (M, lat, lon), ref (lat, lon), lat, lon) arrays of one level.

    A deterministic forecast (``dim`` None) is a one-member ensemble.
    """
    ref = ref.squeeze(drop=True)
    if dim is None:
        members = members.squeeze(drop=True).expand_dims("__member__")
        dim = "__member__"
    else:
        members = members.squeeze([d for d in members.dims if d != dim and members.sizes[d] == 1],
                                  drop=True)
    others = [d for d in members.dims if d != dim]
    members = members.transpose(dim, *others)
    ref = ref.transpose(*others)
    if ref.shape != members.shape[1:]:
        raise ValueError(
            f"ensemble {dict(members.sizes)} and reference {dict(ref.sizes)} grids differ")
    lat = next(c for c in ("lat", "latitude") if c in ref.coords)
    lon = next(c for c in ("lon", "longitude") if c in ref.coords)
    return (np.asarray(members.values, dtype=np.float64),
            np.asarray(ref.values, dtype=np.float64),
            np.asarray(ref[lat].values, dtype=np.float64),
            np.asarray(ref[lon].values, dtype=np.float64))


def gridded_sums(members: xr.DataArray, ref: xr.DataArray, dim: str) -> Dict[str, float]:
    """Weighted sums of an ensemble field (``dim`` + the reference's dims) against a gridded one."""
    x, y, lat, _ = level_arrays(members, ref, dim)
    w = np.broadcast_to(np.cos(np.deg2rad(lat))[:, None], y.shape) if y.ndim == 2 else None
    return point_sums(x.reshape(x.shape[0], -1), y.ravel(), None if w is None else w.ravel())


def ensemble_bins(x: np.ndarray, y: np.ndarray, lat: np.ndarray, lon: np.ndarray,
                  bin_resolution: float, bias: float) -> List[Dict[str, Any]]:
    """Per-bin crps, crps_c, spread, ssr, ssr_c of one ensemble level.

    ``x`` (M, lat, lon) against ``y`` (lat, lon), in (lat, lon) bins of ``bin_resolution``
    degrees; the centred scores remove the level's ``bias`` (that of the pooled scores), the same
    in every bin.
    """
    m = x.shape[0]
    lat_bins = np.arange(-90, 90 + bin_resolution, bin_resolution)
    lon_bins = np.arange(-180, 180 + bin_resolution, bin_resolution)
    n_lon = len(lon_bins) - 1
    li = np.clip(np.digitize(lat, lat_bins) - 1, 0, len(lat_bins) - 2)
    lj = np.clip(np.digitize(lon, lon_bins) - 1, 0, n_lon - 1)
    idx = (li[:, None] * n_lon + lj[None, :]).ravel()
    xf, yf = x.reshape(m, -1), y.ravel()
    ok = np.isfinite(yf) & np.all(np.isfinite(xf), axis=0)
    xf, yf, idx = xf[:, ok], yf[ok], idx[ok]
    if yf.size == 0:
        return []
    err = xf.mean(axis=0) - yf
    size = len(lat_bins) - 1
    size *= n_lon
    count = np.bincount(idx, minlength=size)

    def _sum(v):
        return np.bincount(idx, weights=v, minlength=size)

    s_crps, s_crps_c = _sum(fair_crps(xf, yf)), _sum(fair_crps(xf - bias, yf))
    s_var = _sum(xf.var(axis=0, ddof=1)) if m > 1 else np.zeros(size)
    s_sq, s_sq_c = _sum(err ** 2), _sum((err - bias) ** 2)
    factor = float(np.sqrt((m + 1.0) / m)) if m > 1 else None
    out = []
    for b in np.flatnonzero(count):
        n = float(count[b])
        spread = float(np.sqrt(s_var[b] / n))
        entry: Dict[str, Any] = {
            "lat_bin": {"left": float(lat_bins[b // n_lon]),
                        "right": float(lat_bins[b // n_lon + 1])},
            "lon_bin": {"left": float(lon_bins[b % n_lon]),
                        "right": float(lon_bins[b % n_lon + 1])},
            "crps": float(s_crps[b] / n), "crps_c": float(s_crps_c[b] / n), "spread": spread,
        }
        rmse, crmse = float(np.sqrt(s_sq[b] / n)), float(np.sqrt(s_sq_c[b] / n))
        if factor is not None and rmse > 0:
            entry["ssr"] = factor * spread / rmse
        if factor is not None and crmse > 0:
            entry["ssr_c"] = factor * spread / crmse
        out.append(entry)
    return out


def gridded_ensemble_scores(
    pred: xr.Dataset,
    ref: xr.Dataset,
    variables: List[Any],
    depth_levels: Optional[List[Any]],
    aliases: Optional[Dict[str, Any]] = None,
    labels: Optional[Any] = None,
    bin_resolution: Optional[float] = None,
    spectra: bool = True,
) -> Any:
    """Scores of a forecast against a gridded reference (e.g. GLORYS), level by level.

    Same variables, depth levels and labels as the deterministic gridded scores (``labels(var,
    level, has_depth)``), one (member, lat, lon) slice in memory at a time:

    * an ensemble: crps, spread, ssr, crps_c, ssr_c (``ENSEMBLE_SCORES``) and, with
      ``bin_resolution``, their per-bin maps;
    * any forecast (a deterministic one as a single member), with ``spectra``: the energy and
      error by band of scales (``dctools.metrics.spectra``).

    Returns ``{score: {label: value}}``, or ``(scores, {variable: [bins]})`` with
    ``bin_resolution`` (each bin carries the level's ``depth`` when the variable has one).
    """
    from dctools.metrics.spectra import band_scores, band_sums

    dim = member_dim(pred)
    out: Dict[str, Dict[str, float]] = {}
    per_bins: Dict[str, List[Dict[str, Any]]] = {}
    for variable, level, has_depth, p_name, p_lv, r_lv in gridded_levels(
            pred, ref, variables, depth_levels, aliases):
        label = labels(variable, level, has_depth) if labels else f"{level} {p_name}"
        x, y, lat, lon = level_arrays(p_lv, r_lv, dim)
        scores: Dict[str, float] = {}
        if dim is not None:
            w = np.broadcast_to(np.cos(np.deg2rad(lat))[:, None], y.shape)
            sums = point_sums(x.reshape(x.shape[0], -1), y.ravel(), w.ravel())
            scores.update({k: v for k, v in scores_from_sums(sums).items() if k in ENSEMBLE_SCORES})
            if bin_resolution is not None:
                bias = sums["w_err"] / sums["w_sum"] if sums["w_sum"] > 0 else 0.0
                bins = ensemble_bins(x, y, lat, lon, bin_resolution, bias)
                if has_depth and level is not None:
                    for b in bins:
                        b["depth"] = float(getattr(level, "value", level))
                per_bins.setdefault(p_name, []).extend(bins)
        if spectra and y.ndim == 2:
            scores.update(band_scores(band_sums(x, y, lat, lon), x.shape[0]))
        for k, v in scores.items():
            out.setdefault(k, {})[label] = v
        del x, y
    if bin_resolution is not None:
        return out, per_bins
    return out
