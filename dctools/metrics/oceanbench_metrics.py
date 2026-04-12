# -*- coding: UTF-8 -*-

"""Wrapper for functions implemented in Mercator's oceanbench library."""

import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

_OCEANBENCH_IMPORT_ERROR: Exception | None = None
try:
    import oceanbench.metrics as oceanbench_metrics
    from oceanbench.core.class4_metrics import class4_evaluator as oceanbench_class4_module
    from oceanbench.core.class4_metrics.class4_evaluator import Class4Evaluator
    from oceanbench.core.derived_quantities import (
        add_geostrophic_currents,
        add_mixed_layer_depth,
    )
    from oceanbench.core.lagrangian_trajectory import (
        ZoneCoordinates,
        deviation_of_lagrangian_trajectories,
    )
    from oceanbench.core.rmsd import Variable, rmsd

    OCEANBENCH_AVAILABLE = True
except Exception as exc:  # pragma: no cover
    oceanbench_metrics = None
    oceanbench_class4_module = None
    Class4Evaluator = None
    add_geostrophic_currents = None
    add_mixed_layer_depth = None
    ZoneCoordinates = Any
    deviation_of_lagrangian_trajectories = None
    Variable = Any
    rmsd = None
    OCEANBENCH_AVAILABLE = False
    _OCEANBENCH_IMPORT_ERROR = exc
import pandas as pd  # noqa: E402
import numpy as np   # noqa: E402
import xarray as xr  # noqa: E402
from loguru import logger  # noqa: E402

from dctools.data.coordinates import (  # noqa: E402
    COORD_ALIASES,
    EVAL_VARIABLES_GLONET,
    GLOBAL_ZONE_COORDINATES,
    CoordinateSystem,
    get_standardized_var_name,
)

# Dictionary of variables of interest: {generic name -> standard_name(s), common aliases}
if OCEANBENCH_AVAILABLE:
    OCEANBENCH_VARIABLES = {
        "sla": Variable.SEA_SURFACE_HEIGHT_ABOVE_SEA_LEVEL,
        "sst": Variable.SEA_SURFACE_TEMPERATURE,
        "sss": Variable.SEA_WATER_SALINITY,
        "ssh": Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID,
        "temperature": Variable.SEA_WATER_POTENTIAL_TEMPERATURE,
        "salinity": Variable.SEA_WATER_SALINITY,
        "u_current": Variable.NORTHWARD_SEA_WATER_VELOCITY,
        "v_current": Variable.EASTWARD_SEA_WATER_VELOCITY,
        "w_current": Variable.UPWARD_SEA_WATER_VELOCITY,
        "mld": Variable.MIXED_LAYER_THICKNESS,
        "mdt": Variable.MEAN_DYNAMIC_TOPOGRAPHY,
    }
else:  # pragma: no cover
    OCEANBENCH_VARIABLES = {}


def get_variable_alias(variable: str) -> Variable | None:
    """Get the alias for a given variable.

    Args:
        variable (Variable): The variable to get the alias for.

    Returns:
        Optional[str]: The alias of the variable, or None if not found.
    """
    if not OCEANBENCH_VARIABLES:
        return None
    for alias, var in OCEANBENCH_VARIABLES.items():
        if alias == variable or var == variable:
            return var
    return None


# ---------------------------------------------------------------------------
# Per-bins spatial RMSD helper
# ---------------------------------------------------------------------------

def _compute_spatial_per_bins(
    pred_ds: xr.Dataset,
    ref_ds: xr.Dataset,
    eval_variables: List[str],
    has_depth: bool,
    depth_levels: Any,
    bin_resolution: int,
) -> Dict[str, list]:
    """Compute spatial-binned RMSE for grid-to-grid datasets.

    Returns a dict ``{variable_name: [{"lat_bin": ..., "lon_bin": ..., "rmse": ...}, ...]}``.

    Uses fully-vectorised numpy accumulation (np.bincount) instead of nested
    Python loops over lat/lon bins, giving ~30–100× speedup for fine
    bin_resolution values.
    """
    import numpy as np

    per_bins: Dict[str, list] = {}

    # Select first time step if present (per_bins is per-timestep already)
    def _squeeze_time(ds: xr.Dataset) -> xr.Dataset:
        for t_dim in ("time", "lead_time", "forecast_reference_time"):
            if t_dim in ds.dims and ds.sizes[t_dim] > 0:
                ds = ds.isel({t_dim: 0})
        return ds

    pred_ds = _squeeze_time(pred_ds)
    ref_ds = _squeeze_time(ref_ds)

    # Get lat/lon coordinate arrays
    lat_coord = pred_ds["lat"].values if "lat" in pred_ds.coords else None
    lon_coord = pred_ds["lon"].values if "lon" in pred_ds.coords else None
    if lat_coord is None or lon_coord is None:
        return {}

    # Build bin edges and pre-compute per-grid-point 2D flat bin index.
    # combined_idx[i*nlon + j] = which (lat_bin, lon_bin) cell point (i,j) falls into.
    lat_bins = np.arange(-90, 90 + bin_resolution, bin_resolution)
    lon_bins = np.arange(-180, 180 + bin_resolution, bin_resolution)
    n_lat_bins = len(lat_bins) - 1
    n_lon_bins = len(lon_bins) - 1
    n_bins_total = n_lat_bins * n_lon_bins

    lat_idx = np.clip(np.digitize(lat_coord, lat_bins) - 1, 0, n_lat_bins - 1)  # (nlat,)
    lon_idx = np.clip(np.digitize(lon_coord, lon_bins) - 1, 0, n_lon_bins - 1)  # (nlon,)
    # Broadcast to 2D and flatten: shape (nlat * nlon,)
    combined_idx = (lat_idx[:, None] * n_lon_bins + lon_idx[None, :]).ravel()

    # Determine variables to process
    var_names = [v for v in eval_variables if v in pred_ds.data_vars and v in ref_ds.data_vars]
    if not var_names:
        # Fallback: use common data vars
        var_names = [v for v in pred_ds.data_vars if v in ref_ds.data_vars]

    for var in var_names:
        try:
            pred_da = pred_ds[var].squeeze(drop=True)
            ref_da = ref_ds[var].squeeze(drop=True)
        except Exception:
            continue

        if set(pred_da.dims) != set(ref_da.dims):
            continue

        # Determine depth slices to iterate
        has_depth_dim = "depth" in pred_da.dims
        if has_depth_dim:
            depth_values = np.asarray(pred_da["depth"].values, dtype=np.float64)
            n_depths = int(depth_values.size)
            if n_depths == 0:
                continue
            # Emit intervals [d_i, d_{i+1}) for each consecutive depth pair
            if n_depths == 1:
                depth_indices = [0]
                def _depth_bin(idx: int) -> Dict[str, Any]:
                    d = float(depth_values[idx])
                    return {"left": d, "right": d, "closed": "right"}
            else:
                depth_indices = list(range(n_depths - 1))
                def _depth_bin(idx: int) -> Dict[str, Any]:  # type: ignore[misc]
                    return {
                        "left": float(depth_values[idx]),
                        "right": float(depth_values[idx + 1]),
                        "closed": "right",
                    }
        else:
            depth_indices = [None]
            depth_values = None

        bins_list: list = []

        for depth_index in depth_indices:
            if depth_index is not None:
                if ref_da.sizes.get("depth", 0) <= depth_index:
                    continue
                try:
                    pred_slice = pred_da.isel(depth=depth_index).transpose("lat", "lon")
                    ref_slice = ref_da.isel(depth=depth_index).transpose("lat", "lon")
                except Exception:
                    continue
                depth_bin: Optional[Dict[str, Any]] = _depth_bin(depth_index)
            else:
                try:
                    pred_slice = pred_da.transpose("lat", "lon")
                    ref_slice = ref_da.transpose("lat", "lon")
                except Exception:
                    continue
                depth_bin = None

            pred_arr = pred_slice.values.astype(np.float64)
            ref_arr = ref_slice.values.astype(np.float64)
            if pred_arr.shape != ref_arr.shape or pred_arr.ndim != 2:
                continue

            # ── Vectorised accumulation ──────────────────────────────────────
            flat_pred = pred_arr.ravel()
            flat_ref = ref_arr.ravel()
            valid = ~(np.isnan(flat_pred) | np.isnan(flat_ref))
            if not valid.any():
                continue

            diff_sq = (flat_pred[valid] - flat_ref[valid]) ** 2
            valid_idx = combined_idx[valid]

            # np.bincount is O(N) and ~10× faster than np.add.at for dense integer indices
            counts = np.bincount(valid_idx, minlength=n_bins_total)
            sum_sq = np.bincount(valid_idx, weights=diff_sq, minlength=n_bins_total)

            # ── Emit one dict per non-empty bin ──────────────────────────────
            nz_flat = np.flatnonzero(counts)
            if nz_flat.size == 0:
                continue
            nz_lat = nz_flat // n_lon_bins
            nz_lon = nz_flat % n_lon_bins
            nz_rmse = np.sqrt(sum_sq[nz_flat] / counts[nz_flat])
            nz_counts = counts[nz_flat]

            for k in range(nz_flat.size):
                li = int(nz_lat[k])
                lj = int(nz_lon[k])
                entry: Dict[str, Any] = {
                    "lat_bin": {
                        "left": float(lat_bins[li]),
                        "right": float(lat_bins[li + 1]),
                    },
                    "lon_bin": {
                        "left": float(lon_bins[lj]),
                        "right": float(lon_bins[lj + 1]),
                    },
                    "rmse": float(nz_rmse[k]),
                    "count": int(nz_counts[k]),
                }
                if depth_bin is not None:
                    entry["depth_bin"] = depth_bin
                bins_list.append(entry)

        if bins_list:
            per_bins[var] = bins_list

    return per_bins


def _build_class4_bin_specs(bin_resolution: int) -> Dict[str, Any]:
    """Build the spatial binning config expected by Class4Evaluator."""
    import numpy as np

    step = int(bin_resolution)
    if step <= 0:
        raise ValueError(f"bin_resolution must be > 0, got {bin_resolution!r}")

    return {
        "time": "1D",
        "lat": np.arange(-90, 90 + step, step),
        "lon": np.arange(-180, 180 + step, step),
        "depth": None,
    }


def _extract_raw_class4_per_bins(class4_results_df: pd.DataFrame) -> Dict[str, list]:
    """Preserve raw Class4 per-bin payloads in the leaderboard-compatible schema."""
    per_bins_by_var: Dict[str, list] = {}
    if class4_results_df.empty or "per_bins" not in class4_results_df.columns:
        return per_bins_by_var

    for _, row in class4_results_df.iterrows():
        variable = row.get("variable")
        per_bins = row.get("per_bins", [])
        if not variable or not isinstance(per_bins, list) or not per_bins:
            continue
        # Use extend so that multiple rows for the same variable (e.g. when
        # the DataFrame has been built from several per-depth or per-time calls)
        # all accumulate into the same list instead of the last row overwriting.
        existing = per_bins_by_var.setdefault(str(variable), [])
        existing.extend(per_bins)

    return per_bins_by_var


def _class4_compat_helpers_available() -> bool:
    """Return True when the installed oceanbench exposes the helper functions we need."""
    required = (
        "add_model_values",
        "apply_binning",
        "compute_scores_xskillscore",
        "filter_observations_by_qc",
        "format_class4_results",
        "interpolate_model_on_obs",
        "make_superobs",
        "superobs_binning",
        "xr_to_obs_dataframe",
    )
    return oceanbench_class4_module is not None and all(
        hasattr(oceanbench_class4_module, name) for name in required
    )


def _run_class4_with_raw_per_bins(
    evaluator: Any,
    model_ds: xr.Dataset,
    obs_ds: xr.Dataset,
    variables: List[str],
    matching_type: str = "nearest",
) -> Any:
    """Run Class4 evaluation while preserving raw per_bins for leaderboard maps.

    Streaming implementation: observation chunks are processed one at a time
    (via ``xr_to_obs_dataframe(yield_chunks=True)``) to keep peak worker RSS
    low (~32 MB per chunk instead of several GiB for the full DataFrame).
    Partial statistics (count, weighted sum of squared/absolute/signed errors)
    are accumulated per spatial/temporal bin and reconstituted into final
    metrics at the end.
    """
    if not _class4_compat_helpers_available():
        raise RuntimeError("Required oceanbench Class4 helpers are not available")

    import gc
    import ctypes as _ctypes

    try:
        _libc = _ctypes.CDLL("libc.so.6")
        _malloc_trim = _libc.malloc_trim
    except Exception:
        _malloc_trim = None



    all_scores: Dict[str, pd.DataFrame] = {}

    for var in variables:
        obs_da = obs_ds[var]
        model_da = model_ds[var]

        if getattr(evaluator, "apply_qc", False):
            obs_da = oceanbench_class4_module.filter_observations_by_qc(
                ds=obs_da,
                qc_mappings=getattr(evaluator, "qc_mapping", None),
            )

        groupby_cols: List[str] = []

        if matching_type == "nearest":
            # ── Streaming nearest-neighbour path ──────────────────────
            # Accumulate partial statistics per bin across observation
            # chunks so we never hold the full DataFrame in memory.
            #
            # MEMORY OPTIMISATION (critical for SWOT):
            # The previous implementation called interpolate_model_on_obs
            # for each 500 K-point chunk.  That function round-tripped
            # through xarray:
            #   DataFrame → xr.Dataset.from_dataframe → xr.apply_ufunc
            #   → xr.Dataset → .values → obs_df.copy()
            # creating ~80 MB of transient allocations per chunk.  Over
            # 574 chunks (287 M-point SWOT batch), glibc's ptmalloc
            # retained fragmented pages, accumulating 4-5 GiB of
            # "unmanaged" memory per worker.
            #
            # The new implementation pre-builds a single pyinterp.Grid2D
            # from the model and calls grid.bivariate() directly on
            # numpy arrays — zero xarray overhead, zero DataFrame copies.
            # Peak transient allocation per chunk drops from ~80 MB to
            # ~12 MB (3 float64 arrays of 500 K points).
            bin_stats: Dict[Any, Dict[str, float]] = {}
            _groupby_cols_resolved: Optional[List[str]] = None

            _coord_alias_map = {
                "latitude": "lat", "nav_lat": "lat",
                "longitude": "lon", "nav_lon": "lon",
            }

            _chunk_count = 0

            # ── Pre-build pyinterp Grid2D from model (once) ──────────
            # model_da is already an in-memory numpy-backed DataArray
            # (pred_data was .compute()'d earlier in compute_metric).
            _grid2d = None
            try:
                import pyinterp

                # Find lat/lon dimension names in the model
                _lat_dim = next(
                    (d for d in model_da.dims if d in ("latitude", "lat", "nav_lat")),
                    None,
                )
                _lon_dim = next(
                    (d for d in model_da.dims if d in ("longitude", "lon", "nav_lon")),
                    None,
                )
                if _lat_dim and _lon_dim:
                    _src_lat = model_da[_lat_dim].values.astype(np.float64)
                    _src_lon = model_da[_lon_dim].values.astype(np.float64)
                    _model_vals = np.asarray(model_da.values, dtype=np.float64)
                    # Remove leading singleton dims (e.g. time=1, depth=1)
                    _model_vals = _model_vals.squeeze()
                    # pyinterp Grid2D expects (lon, lat) C-order
                    if _model_vals.shape == (len(_src_lat), len(_src_lon)):
                        _model_vals = _model_vals.T  # (lon, lat)
                    _x_axis = pyinterp.Axis(_src_lon)
                    _y_axis = pyinterp.Axis(_src_lat)
                    _grid2d = pyinterp.Grid2D(_x_axis, _y_axis, _model_vals)
                    del _model_vals  # Grid2D holds its own copy
                    logger.debug(
                        f"[perf] Pre-built pyinterp Grid2D for '{var}': "
                        f"lat={len(_src_lat)} lon={len(_src_lon)}"
                    )
            except Exception as _grid_exc:
                logger.debug(
                    f"[perf] Cannot pre-build Grid2D: {_grid_exc!r} "
                    f"— falling back to interpolate_model_on_obs"
                )

            # ── Direct zarr chunk iterator ─────────────────────────
            # We bypass oceanbench's xr_to_obs_dataframe generator to
            # get full control over the memory lifecycle.  The generator
            # holds a reference to each yielded DataFrame until the
            # NEXT iteration, which means our malloc_trim(0) runs while
            # ~48 MB of chunk data is still live.  Over 574 chunks the
            # un-trimmed pages accumulate to 4+ GiB of "unmanaged"
            # memory.
            #
            # By reading zarr chunks directly → numpy → minimal
            # DataFrame (for apply_binning only), we ensure that ALL
            # references are freed before malloc_trim runs.
            import dask as _dask

            # Detect n_points dimension
            _n_pts_dim = None
            for _d in obs_da.dims:
                if "point" in _d.lower() or "obs" in _d.lower() or _d.startswith("n_"):
                    _n_pts_dim = _d
                    break
            if _n_pts_dim is None and obs_da.ndim == 1:
                _n_pts_dim = obs_da.dims[0]
            if _n_pts_dim is None:
                # Fallback: largest dimension
                _n_pts_dim = max(obs_da.dims, key=lambda d: obs_da.sizes[d])

            _n_total = obs_da.sizes[_n_pts_dim]

            # Detect coordinate names in the observation dataset
            _all_coords = set(obs_da.coords) | set(obs_ds.coords)
            _lat_coord = next(
                (c for c in _all_coords
                 if c.lower() in ("lat", "latitude", "nav_lat")),
                None,
            )
            _lon_coord = next(
                (c for c in _all_coords
                 if c.lower() in ("lon", "longitude", "nav_lon")),
                None,
            )
            _time_coord = next(
                (c for c in _all_coords if c.lower() == "time"), None,
            )
            _depth_coord = next(
                (c for c in _all_coords if c.lower() == "depth"), None,
            )

            # Keep a legacy interp_cache only if we need the fallback
            interp_cache: Dict = {} if _grid2d is None else {}

            _CHUNK_SZ = 500_000
            logger.debug(
                f"[perf] Direct zarr streaming: {_n_total} points, "
                f"{(_n_total + _CHUNK_SZ - 1) // _CHUNK_SZ} chunks, "
                f"grid2d={'yes' if _grid2d else 'no'}"
            )

            for _start in range(0, _n_total, _CHUNK_SZ):
                _end = min(_start + _CHUNK_SZ, _n_total)
                _sl = {_n_pts_dim: slice(_start, _end)}

                # Read observation arrays from dask → numpy (one at a time)
                with _dask.config.set(scheduler="synchronous"):
                    _val_arr = obs_da.isel(_sl).values.ravel()
                    _lat_raw = (
                        obs_ds[_lat_coord].isel(_sl).values.ravel()
                        if _lat_coord and _lat_coord in obs_ds
                        else (obs_da.coords[_lat_coord].isel(_sl).values.ravel()
                              if _lat_coord and _lat_coord in obs_da.coords
                              else None)
                    )
                    _lon_raw = (
                        obs_ds[_lon_coord].isel(_sl).values.ravel()
                        if _lon_coord and _lon_coord in obs_ds
                        else (obs_da.coords[_lon_coord].isel(_sl).values.ravel()
                              if _lon_coord and _lon_coord in obs_da.coords
                              else None)
                    )
                    _time_raw = (
                        obs_ds[_time_coord].isel(_sl).values.ravel()
                        if _time_coord and _time_coord in obs_ds
                        else (obs_da.coords[_time_coord].isel(_sl).values.ravel()
                              if _time_coord and _time_coord in obs_da.coords
                              else None)
                    )

                if len(_val_arr) == 0:
                    del _val_arr, _lat_raw, _lon_raw, _time_raw
                    continue

                # Build minimal DataFrame for apply_binning
                _data: Dict[str, Any] = {}
                if _lat_raw is not None:
                    _data["lat"] = _lat_raw
                if _lon_raw is not None:
                    _data["lon"] = _lon_raw
                if _time_raw is not None:
                    _data["time"] = _time_raw
                _data[var] = _val_arr
                chunk_df = pd.DataFrame(_data)
                del _data, _val_arr, _lat_raw, _lon_raw, _time_raw

                # Spatial/temporal binning
                chunk_df, groupby_cols = oceanbench_class4_module.apply_binning(
                    chunk_df,
                    getattr(evaluator, "bin_specs", None),
                )
                if _groupby_cols_resolved is None:
                    _groupby_cols_resolved = list(groupby_cols)

                if chunk_df.empty:
                    continue

                # Resolve variable column name
                if var not in chunk_df.columns:
                    for candidate in ("value", "variable"):
                        if candidate in chunk_df.columns:
                            chunk_df = chunk_df.rename(columns={candidate: var})
                            break
                if var not in chunk_df.columns:
                    continue

                chunk_df = chunk_df.dropna(subset=[var])
                if chunk_df.empty:
                    continue

                obs_col = f"{var}_obs"
                model_col = f"{var}_model"

                # ── Interpolate model → observation locations ─────────
                if _grid2d is not None:
                    # Fast path: direct pyinterp call (no xarray overhead)
                    _obs_lon = chunk_df["lon"].values.astype(np.float64)
                    _obs_lat = chunk_df["lat"].values.astype(np.float64)
                    _interp_vals = pyinterp.bivariate(
                        _grid2d, _obs_lon, _obs_lat,
                        interpolator="bilinear",
                        num_threads=1,
                    )
                    # Build minimal arrays for error computation
                    obs_vals = chunk_df[var].values.astype(np.float64)
                    mod_vals = _interp_vals

                    # NaN mask (from interpolation boundary or missing obs)
                    _valid = np.isfinite(obs_vals) & np.isfinite(mod_vals)
                    if not _valid.all():
                        obs_vals = obs_vals[_valid]
                        mod_vals = mod_vals[_valid]
                        # Need lat for weights and bin cols for grouping
                        _lat_arr = (
                            chunk_df["lat"].values[_valid]
                            if "lat" in chunk_df.columns
                            else None
                        )
                        _bin_df = (
                            chunk_df[_groupby_cols_resolved].iloc[
                                np.where(_valid)[0]
                            ]
                            if _groupby_cols_resolved
                            else None
                        )
                    else:
                        _lat_arr = (
                            chunk_df["lat"].values
                            if "lat" in chunk_df.columns
                            else None
                        )
                        _bin_df = (
                            chunk_df[_groupby_cols_resolved]
                            if _groupby_cols_resolved
                            else None
                        )
                    del _obs_lon, _obs_lat, _interp_vals, _valid
                else:
                    # Fallback: original xarray-based interpolation
                    chunk_df = chunk_df.rename(columns={var: obs_col})
                    chunk_df = oceanbench_class4_module.interpolate_model_on_obs(
                        model_da,
                        chunk_df,
                        var,
                        method=getattr(evaluator, "interp_method", "nearest"),
                        cache=interp_cache,
                    )
                    if len(interp_cache) > 2:
                        interp_cache.clear()
                    chunk_df = chunk_df.dropna(subset=[obs_col, model_col])
                    if chunk_df.empty:
                        del chunk_df
                        continue
                    obs_vals = chunk_df[obs_col].values
                    mod_vals = chunk_df[model_col].values
                    _lat_arr = (
                        chunk_df["lat"].values
                        if "lat" in chunk_df.columns
                        else None
                    )
                    _bin_df = (
                        chunk_df[_groupby_cols_resolved]
                        if _groupby_cols_resolved
                        else None
                    )

                # ── Free the chunk DataFrame ASAP ─────────────────────
                del chunk_df

                if len(obs_vals) == 0:
                    del obs_vals, mod_vals
                    continue

                diff = mod_vals - obs_vals

                if _lat_arr is not None:
                    w = np.cos(np.deg2rad(_lat_arr.astype(np.float64)))
                else:
                    w = np.ones(len(diff), dtype=np.float64)

                # Accumulate weighted partial stats per bin
                agg_cols = list(_groupby_cols_resolved) if _groupby_cols_resolved else []

                if agg_cols and _bin_df is not None:
                    df_agg = _bin_df.copy()
                    df_agg["w_sq_err"] = w * diff ** 2
                    df_agg["w_abs_err"] = w * np.abs(diff)
                    df_agg["w_err"] = w * diff
                    df_agg["w_sum"] = w
                    df_agg["count"] = 1.0

                    grouped = df_agg.groupby(
                        agg_cols, observed=True, dropna=True,
                    ).sum(numeric_only=True)

                    for name, row in grouped.iterrows():
                        if name not in bin_stats:
                            bin_stats[name] = {
                                "count": 0.0, "w_sum": 0.0,
                                "w_sq": 0.0, "w_abs": 0.0, "w_err": 0.0,
                            }
                        s = bin_stats[name]
                        s["count"] += row["count"]
                        s["w_sum"] += row["w_sum"]
                        s["w_sq"] += row["w_sq_err"]
                        s["w_abs"] += row["w_abs_err"]
                        s["w_err"] += row["w_err"]

                    del df_agg, grouped
                else:
                    name = "global"
                    if name not in bin_stats:
                        bin_stats[name] = {
                            "count": 0.0, "w_sum": 0.0,
                            "w_sq": 0.0, "w_abs": 0.0, "w_err": 0.0,
                        }
                    s = bin_stats[name]
                    s["count"] += len(diff)
                    s["w_sum"] += float(w.sum())
                    s["w_sq"] += float((w * diff ** 2).sum())
                    s["w_abs"] += float((w * np.abs(diff)).sum())
                    s["w_err"] += float((w * diff).sum())

                del obs_vals, mod_vals, diff, w, _lat_arr, _bin_df
                _chunk_count += 1
                gc.collect()
                # Release glibc malloc arenas back to the OS after every
                # chunk.  malloc_trim(0) is cheap (~1 ms) and essential
                # to avoid accumulating GiB of "unmanaged memory".
                if _malloc_trim is not None:
                    _malloc_trim(0)

            # Clean up
            del _grid2d
            if interp_cache:
                interp_cache.clear()
            del interp_cache

            # ── Reconstitute metrics from accumulated stats ──────────
            groupby_cols = _groupby_cols_resolved or []
            bin_results = []
            for name, s in bin_stats.items():
                n = s["count"]
                ws = s["w_sum"]
                if n == 0 or ws == 0:
                    continue
                res: Dict[str, Any] = {}
                if groupby_cols:
                    if len(groupby_cols) == 1:
                        res[groupby_cols[0]] = name
                    else:
                        for i, col in enumerate(groupby_cols):
                            res[col] = name[i]

                res["rmse"] = float(np.sqrt(s["w_sq"] / ws))
                res["mse"] = float(s["w_sq"] / ws)
                res["mae"] = float(s["w_abs"] / ws)
                res["bias"] = float(s["w_err"] / ws)
                res["me"] = res["bias"]
                res["count"] = float(n)
                bin_results.append(res)

            # Global aggregation
            global_stats: Dict[str, float] = {}
            if bin_results:
                df_res = pd.DataFrame(bin_results)
                for m in ("rmse", "mse", "mae", "bias", "me"):
                    if m in df_res.columns:
                        global_stats[f"{m}_mean"] = float(df_res[m].mean())
                        global_stats[f"{m}_median"] = float(df_res[m].median())
                        global_stats[f"{m}_std"] = float(df_res[m].std())

            scores_result: Any = {
                "per_bins": bin_results,
                "global": global_stats,
            }

            del bin_stats
            gc.collect()

        elif matching_type == "superobs":
            obs_col = f"{var}_obs"
            model_col = f"{var}_model"
            superobs = oceanbench_class4_module.make_superobs(
                obs_da,
                model_da,
                var,
                reduce="mean",
            )
            obs_binned = oceanbench_class4_module.superobs_binning(
                superobs,
                model_da,
                var=var,
            )
            binned_df = oceanbench_class4_module.xr_to_obs_dataframe(
                obs_binned,
                include_geometry=False,
            )
            if f"{var}_binned" in binned_df.columns:
                binned_df = binned_df.dropna(subset=[f"{var}_binned"]).rename(
                    columns={f"{var}_binned": obs_col}
                )
            final_df = oceanbench_class4_module.add_model_values(
                binned_df,
                model_da,
                var=var,
            )

            _weight_col: Optional[str] = None
            if "lat" in final_df.columns:
                _weight_col = "__cos_lat_weight__"
                final_df[_weight_col] = np.cos(
                    np.deg2rad(final_df["lat"].values.astype(np.float64))
                )

            scores_result = oceanbench_class4_module.compute_scores_xskillscore(
                df=final_df,
                y_obs_col=obs_col,
                y_pred_col=model_col,
                metrics=getattr(evaluator, "metrics", []),
                weights=_weight_col,
                groupby=groupby_cols,
            )
        else:
            raise ValueError(f"Unknown matching_type: {matching_type}")

        if isinstance(scores_result, dict):
            scores_df = pd.DataFrame([scores_result])
            scores_df["variable"] = var
        elif isinstance(scores_result, pd.DataFrame):
            scores_df = scores_result.copy()
            scores_df["variable"] = var
        else:
            scores_df = pd.DataFrame({"variable": [var], "result": [scores_result]})

        all_scores[var] = scores_df

    if not all_scores:
        return pd.DataFrame()

    final_result = pd.concat(list(all_scores.values()), ignore_index=True)
    per_bins_by_var = _extract_raw_class4_per_bins(final_result)
    grid_results = oceanbench_class4_module.format_class4_results(final_result)
    if per_bins_by_var:
        return {"results": grid_results, "per_bins": per_bins_by_var}
    return grid_results


class DCMetric(ABC):
    """Abstract Base Class for Data Challenge Metrics."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the DCMetric.

        Args:
            **kwargs: Configuration parameters for the metric.
                Common arguments include:
                - plot_result (bool): Whether to generate plots.
                - minimum_latitude (float): Min lat bound.
                - maximum_latitude (float): Max lat bound.
                - minimum_longitude (float): Min lon bound.
                - maximum_longitude (float): Max lon bound.
                - spatial_resolution (float): Spatial resolution.
                - small_scale_cutoff_km (float): Cutoff for spectral analysis.
        """
        self.metric_name = None
        no_default_attrs = ["metric_name", "var", "depth"]
        class_default_attrs = ["metric_name"]
        default_attrs: Dict[str, Any] = dict(
            plot_result=False,
            minimum_latitude=None,
            maximum_latitude=None,
            minimum_longitude=None,
            maximum_longitude=None,
            spatial_resolution=None,
            small_scale_cutoff_km=100,
        )
        allowed_attrs = list(default_attrs.keys()) + no_default_attrs
        default_attrs.update(kwargs)
        self.__dict__.update((k, v) for k, v in default_attrs.items() if k in allowed_attrs)

        for attr in class_default_attrs:
            assert hasattr(self, attr)

    def get_metric_name(self) -> Optional[str]:
        """Return the name of the metric.

        Returns:
            str: The name of the metric.
        """
        return self.metric_name

    @abstractmethod
    def compute(
        self, pred_data: xr.Dataset, ref_data: Optional[xr.Dataset] = None, **kwargs: Any
    ) -> Any:
        """Compute the metric wrapper (includes preprocessing).

        Args:
            pred_data (xr.Dataset): Prediction dataset.
            ref_data (xr.Dataset, optional): Reference dataset.
        """
        pass

    @abstractmethod
    def compute_metric(
        self, pred_data: xr.Dataset, ref_data: Optional[xr.Dataset] = None, **kwargs: Any
    ) -> Any:
        """Compute the core metric value.

        Args:
            pred_data (xr.Dataset): Prediction dataset.
            ref_data (xr.Dataset): Reference dataset.
        """
        pass


class OceanbenchMetrics(DCMetric):
    """Central class for calling Oceanbench functions."""

    def __init__(
        self,
        eval_variables: Optional[Optional[List[str]]] = None,
        oceanbench_eval_variables: Optional[Optional[List[str]]] = None,
        is_class4: Optional[Optional[bool]] = None,
        class4_kwargs: Optional[Optional[dict]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize OceanbenchMetrics.

        Args:
            eval_variables (Optional[List[str]]): List of variables to evaluate.
            oceanbench_eval_variables (Optional[List[str]]): OceanBench standard variables.
            is_class4 (Optional[bool]): Enable Class 4 metrics.
            class4_kwargs (Optional[dict]): Arguments for Class4Evaluator.
            **kwargs: Additional arguments.
        """
        if not OCEANBENCH_AVAILABLE:
            msg = (
                "oceanbench is required for OceanbenchMetrics, but it failed to import in this "
                "environment. This is commonly caused by optional dependencies"
                " (e.g. copernicusmarine) "
                "or a broken sqlite3 build in the Python distribution."
            )
            if _OCEANBENCH_IMPORT_ERROR is not None:
                msg += f" Original error: {repr(_OCEANBENCH_IMPORT_ERROR)}"
            raise ImportError(msg)

        super().__init__(**kwargs)
        self.eval_variables = eval_variables
        self.oceanbench_eval_variables = oceanbench_eval_variables
        self.is_class4 = is_class4
        self.class4_kwargs = class4_kwargs or {}
        self.bin_resolution = kwargs.get("bin_resolution", None)
        self.class4_matching_type = self.class4_kwargs.get("matching_type", "nearest")

        self.metrics_set: Dict[str, Optional[Dict[str, Any]]] = {
            "rmsd": {
                "func_with_ref": rmsd,
                "kwargs_with_ref": ["vars"],
                "func_no_ref": oceanbench_metrics.rmsd_of_variables_compared_to_glorys,
            },
            "lagrangian": {
                "func_with_ref": deviation_of_lagrangian_trajectories,
                "kwargs_with_ref": ["zone"],
                "func_no_ref": (
                    oceanbench_metrics.deviation_of_lagrangian_trajectories_compared_to_glorys
                ),
            },
            "rmsd_geostrophic_currents": {
                "func_with_ref": rmsd,
                "kwargs_with_ref": ["vars"],
                "func_no_ref": oceanbench_metrics.rmsd_of_geostrophic_currents_compared_to_glorys,
                "preprocess_ref": add_geostrophic_currents,
            },
            "rmsd_mld": {
                "func_with_ref": rmsd,
                "kwargs_with_ref": ["vars"],
                "func_no_ref": oceanbench_metrics.rmsd_of_mixed_layer_depth_compared_to_glorys,
                "preprocess_ref": add_mixed_layer_depth,
            },
            # --- Addition for class 4 metrics ---
            "class4": None,
        }

        if is_class4:
            class4_args = dict(self.class4_kwargs)
            if self.bin_resolution is not None and "binning" not in class4_args:
                class4_args["binning"] = _build_class4_bin_specs(self.bin_resolution)
            logger.debug(f"Class4Evaluator config: {class4_args}")
            self.class4_evaluator = Class4Evaluator(
                metrics=class4_args["list_scores"],
                interpolation_method=class4_args["interpolation_method"],
                delta_t=class4_args["time_tolerance"],
                bin_specs=class4_args.get("binning", None),
                spatial_mask_fn=class4_args.get("spatial_mask_fn", None),
                cache_dir=class4_args.get("cache_dir", None),
                apply_qc=class4_args.get("apply_qc", False),
                qc_mapping=class4_args.get("qc_mapping", None),
            )

    def compute_metric(
        self,
        pred_data: xr.Dataset,
        ref_data: Optional[xr.Dataset] = None,
        eval_variables: Optional[List[Variable]] = EVAL_VARIABLES_GLONET,
        zone: Optional[ZoneCoordinates] = GLOBAL_ZONE_COORDINATES,
        pred_coords: Optional[CoordinateSystem] = None,
        ref_coords: Optional[CoordinateSystem] = None,
        **extra_kwargs: Any,
    ) -> Optional[Any]:
        """Compute a given metric.

        Args:
            pred_data (xr.Dataset): dataset to evaluate
            ref_data (xr.Dataset): reference dataset

        Returns:
            ndarray, optional: computed metric (if any)
        """
        if self.is_class4 is None:
            self.is_class4 = ref_coords.is_observation_dataset() if ref_coords else False

        if self.is_class4:
            try:
                # ── Promote lat/lon/time to coordinates for obs datasets ──
                # ARGO observation data has lat/lon/time as data_vars, not
                # as coordinates.  Class4Evaluator accesses individual
                # DataArrays via obs_ds[var], and only *coordinates* carry
                # over to the DataArray.  Without this promotion, the
                # resulting DataFrame has no spatial/temporal columns and
                # interpolation cannot match observations to model grid.
                coord_candidates = ["lat", "lon", "time"]
                promote = [
                    c
                    for c in coord_candidates
                    if c in ref_data.data_vars and c not in ref_data.coords  # type: ignore[union-attr]
                ]
                if promote:
                    ref_data = ref_data.set_coords(promote)  # type: ignore[union-attr]

                # ── Harmonize variable names between datasets ──
                # Class4Evaluator.run() uses the same variable name to
                # index into *both* model_ds and obs_ds.  When prediction
                # and observation datasets use different names for the
                # same physical quantity (e.g. "zos" vs "ssh", or "TEMP"
                # vs "thetao"), we must rename one of them so the names
                # match.  Strategy: pick the eval_variable name as the
                # canonical target; rename whichever dataset is missing it.
                variables = list(self.eval_variables) if self.eval_variables else []
                pred_vars = set(pred_data.data_vars)
                ref_vars = set(ref_data.data_vars) if ref_data is not None else set()

                pred_rename: dict[str, str] = {}
                ref_rename: dict[str, str] = {}
                resolved_variables: list[str] = []

                for var in variables:
                    in_pred = var in pred_vars
                    in_ref = var in ref_vars

                    if in_pred and in_ref:
                        resolved_variables.append(var)
                        continue

                    # Find the standardized key for this eval variable
                    std_key = get_standardized_var_name(var)

                    if not in_pred and std_key is not None:
                        # Look for a pred variable that maps to the same
                        # standardized key
                        for dv in pred_vars:
                            if get_standardized_var_name(str(dv)) == std_key:
                                pred_rename[str(dv)] = var
                                in_pred = True
                                break

                    if not in_ref and std_key is not None:
                        for dv in ref_vars:
                            if get_standardized_var_name(str(dv)) == std_key:
                                ref_rename[str(dv)] = var
                                in_ref = True
                                break

                    if in_pred and in_ref:
                        resolved_variables.append(var)
                    else:
                        logger.warning(
                            f"Variable '{var}' (std={std_key}) not found "
                            f"in both model ({sorted(str(v) for v in pred_vars)}) and "
                            f"obs ({sorted(str(v) for v in ref_vars)}) — skipping."
                        )

                if pred_rename:
                    logger.debug(f"Renaming model variables: {pred_rename}")
                    pred_data = pred_data.rename(pred_rename)
                if ref_rename:
                    logger.debug(f"Renaming obs variables: {ref_rename}")
                    ref_data = ref_data.rename(ref_rename)  # type: ignore[union-attr]

                if not resolved_variables:
                    logger.error(
                        f"No common variables between model "
                        f"({sorted(str(v) for v in pred_vars)}) and obs "
                        f"({sorted(str(v) for v in ref_vars)}) for eval_variables="
                        f"{variables}. Cannot compute class4 metrics."
                    )
                    return None

                matching_type = extra_kwargs.get("matching_type", self.class4_matching_type)

                if self.bin_resolution is not None and _class4_compat_helpers_available():
                    res = _run_class4_with_raw_per_bins(
                        evaluator=self.class4_evaluator,
                        model_ds=pred_data,
                        obs_ds=ref_data,
                        variables=resolved_variables,
                        matching_type=matching_type,
                    )
                else:
                    res = self.class4_evaluator.run(
                        model_ds=pred_data,
                        obs_ds=ref_data,
                        variables=resolved_variables,
                        ref_coords=ref_coords,
                        matching_type=matching_type,
                    )

                return res

            except Exception as exc:
                logger.error(f"Failed to compute metric {self.metric_name}: {repr(exc)}")
                raise
        else:
            if eval_variables:
                has_depth = any(
                    depth_alias in list(pred_data.dims) for depth_alias in COORD_ALIASES["depth"]
                )
            if eval_variables and not has_depth:
                if self.metric_name == "lagrangian":
                    logger.warning("Lagrangian metric requires 'depth' variable.")
                    return None
            if self.metric_name is None:
                return None

            metric_name = self.metric_name
            if metric_name not in self.metrics_set:
                logger.warning(f"Metric {metric_name} is not defined in the metrics set.")
                return None
            try:
                metric_info = self.metrics_set[metric_name]
                if metric_info is None:
                    return None

                if ref_data:
                    metric_func = metric_info["func_with_ref"]
                    add_kwargs_list = metric_info.get("kwargs_with_ref", [])
                    if "preprocess_ref" in metric_info:
                        ref_data = metric_info["preprocess_ref"]([ref_data])
                    kwargs = {
                        "challenger_datasets": [pred_data],
                        "reference_datasets": [ref_data],
                    }
                else:
                    metric_func = metric_info["func_no_ref"]
                    add_kwargs_list = None
                    kwargs = {
                        "challenger_datasets": [pred_data],
                    }

                if eval_variables and ref_data:
                    if metric_name != "lagrangian":
                        kwargs["variables"] = self.oceanbench_eval_variables

                # Check for depth as a dimension
                has_depth_dim = "depth" in pred_data.dims
                has_depth_coord = "depth" in pred_data.coords
                if not has_depth_dim and not has_depth_coord:
                    kwargs["depth_levels"] = None
                add_kwargs: Dict[Any, Any] = {}
                if add_kwargs_list:
                    if "vars" in add_kwargs_list:
                        add_kwargs["variables"] = self.oceanbench_eval_variables
                    if "zone" in add_kwargs_list:
                        kwargs["zone"] = zone

                    kwargs.update(add_kwargs)

                # Forward per-bins spatial resolution when set
                if self.bin_resolution is not None:
                    import inspect
                    _sig = inspect.signature(metric_func)
                    if "bin_resolution" in _sig.parameters:
                        kwargs["bin_resolution"] = self.bin_resolution

                result = metric_func(**kwargs)

                # If metric_func (e.g. oceanbench's rmsd()) already returned a
                # {"results": …, "per_bins": …} wrapper, strip the inner per_bins
                # before re-wrapping.  The inner per_bins use the legacy string-
                # label lat_bin format (e.g. "78S-74S") which is incompatible with
                # the dict format expected by _aggregate_per_bins_jsonl.  We
                # always recompute per_bins via _compute_spatial_per_bins below
                # so no scientific data is lost.
                if isinstance(result, dict) and "results" in result and "per_bins" in result:
                    result = result["results"]

                # Compute per-bins spatial RMSD when bin_resolution is set.
                if self.bin_resolution is not None and ref_data is not None:
                    per_bins = _compute_spatial_per_bins(
                        pred_data, ref_data,
                        self.eval_variables or [],
                        has_depth_dim or has_depth_coord,
                        kwargs.get("depth_levels"),
                        self.bin_resolution,
                    )
                    if per_bins:
                        return {"results": result, "per_bins": per_bins}

                return result
            except Exception as exc:
                logger.error(f"Failed to compute metric {self.metric_name}: {repr(exc)}")
                raise
