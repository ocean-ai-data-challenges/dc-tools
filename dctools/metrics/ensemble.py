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

def _lat_weights(da: xr.DataArray) -> Optional[np.ndarray]:
    lat = next((c for c in ("lat", "latitude") if c in da.coords), None)
    if lat is None:
        return None
    w = np.cos(np.deg2rad(np.asarray(da[lat].values, dtype=np.float64)))
    if da[lat].ndim == 1:
        shape = [1] * da.ndim
        shape[da.dims.index(da[lat].dims[0])] = w.size
        return np.broadcast_to(w.reshape(shape), da.shape)
    return np.broadcast_to(w, da.shape)


def gridded_sums(members: xr.DataArray, ref: xr.DataArray, dim: str) -> Dict[str, float]:
    """Weighted sums of an ensemble field (``dim`` + the reference's dims) against a gridded one."""
    ref = ref.squeeze(drop=True)
    members = members.squeeze([d for d in members.dims if d != dim and members.sizes[d] == 1],
                              drop=True)
    others = [d for d in members.dims if d != dim]
    members = members.transpose(dim, *others)
    ref = ref.transpose(*others)
    if ref.shape != members.shape[1:]:
        raise ValueError(
            f"ensemble {dict(members.sizes)} and reference {dict(ref.sizes)} grids differ")
    w = _lat_weights(ref)
    x = np.asarray(members.values, dtype=np.float64).reshape(members.shape[0], -1)
    return point_sums(x, np.asarray(ref.values).ravel(), None if w is None else w.ravel())


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


def gridded_ensemble_scores(
    pred: xr.Dataset,
    ref: xr.Dataset,
    variables: List[Any],
    depth_levels: Optional[List[Any]],
    aliases: Optional[Dict[str, Any]] = None,
    labels: Optional[Any] = None,
) -> Dict[str, Dict[str, float]]:
    """{score: {label: value}} of an ensemble against a gridded reference (e.g. GLORYS).

    Same variables, depth levels and labels as the deterministic gridded scores (``labels(var,
    level, has_depth)``), one (member, lat, lon) slice in memory at a time.
    """
    dim = member_dim(pred)
    out: Dict[str, Dict[str, float]] = {k: {} for k in ENSEMBLE_SCORES}
    if dim is None:
        return {}
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
            scores = scores_from_sums(gridded_sums(p_lv, r_lv, dim))
            label = labels(variable, level, has_depth) if labels else f"{level} {p_name}"
            for k in ENSEMBLE_SCORES:
                if k in scores:
                    out[k][label] = scores[k]
    return {k: v for k, v in out.items() if v}
