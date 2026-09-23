"""Temporal extent of a dataset, robust to the stores met on the EDITO S3.

Moved from ``dctools.data.connection.connection_manager`` (which re-exports these names).

``get_time_bound_values`` finds the time variable whatever its role (dimension, coordinate, data
variable, or any datetime64 variable), loads it eagerly (always a small 1-D array; also sidesteps a
dask reduce bug on datetime64), rejects implausible bounds (fill values decoding to 1970, int64
overflow near 1677/2262 from incomplete zarr uploads) and falls back to the ACDD
``time_coverage_start``/``time_coverage_end`` global attributes.
"""
from __future__ import annotations

import logging
import traceback

import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)

# List of possible names for the time dimension
TIME_NAMES = [
    "time",
    "Time",
    "TIME",
    "date",
    "datetime",
    "valid_time",
    "forecast_time",
    "time_counter",
    "profile_date",
]

# Sane bounds for any real observation/model timestamp in this project. Used to
# detect corrupted time bounds (e.g. incomplete/interrupted zarr uploads where
# some declared variables are missing on disk and zarr transparently returns
# fill-values for missing chunks -> min/max reduces to 1970-01-01 epoch and/or
# int64-nanosecond overflow near the pd.Timestamp representable range limits,
# ~1677 / ~2262). A file whose computed date_start/date_end falls outside this
# window is treated as unreliable rather than trusted verbatim.
_TIME_SANITY_MIN_YEAR = 1990
_TIME_SANITY_MAX_YEAR = 2100

# Global attribute names (ACDD/CF convention) sometimes present on satellite
# products (e.g. SWOT) that reliably describe the file's real temporal
# coverage even when the per-point "time" variable's data is missing/corrupted.
_TIME_COVERAGE_ATTRS = ("time_coverage_start", "time_coverage_end")


def _is_sane_timestamp(ts) -> bool:
    try:
        year = pd.Timestamp(ts).year
    except Exception:
        return False
    return _TIME_SANITY_MIN_YEAR <= year <= _TIME_SANITY_MAX_YEAR


def _time_coverage_attrs_fallback(ds: xr.Dataset):
    """Try to recover (date_start, date_end) from ACDD-style global attrs.

    Returns (pd.Timestamp, pd.Timestamp) or None if unavailable/unparseable.
    """
    start_attr, end_attr = _TIME_COVERAGE_ATTRS
    start_raw = ds.attrs.get(start_attr)
    end_raw = ds.attrs.get(end_attr)
    if not start_raw or not end_raw:
        return None
    try:
        # Normalize to timezone-naive, matching the rest of the pipeline
        # (which otherwise always deals in naive datetime64[ns] timestamps),
        # to avoid "can't compare offset-naive and offset-aware" errors
        # downstream in date-range filtering.
        start_ts = pd.Timestamp(start_raw).tz_localize(None)
        end_ts = pd.Timestamp(end_raw).tz_localize(None)
    except Exception:
        return None
    if not (_is_sane_timestamp(start_ts) and _is_sane_timestamp(end_ts)):
        return None
    return (start_ts, end_ts)

def get_time_bound_values(ds: xr.Dataset) -> tuple:
    """
    Returns the time bounds (min, max) of an xarray dataset.

    Regardless of the structure (dimension, coordinate, variable).
    """
    time_vals = None

    try:
        # Search for the time variable in dims, coords, data_vars
        for time_name in TIME_NAMES:
            if time_name in ds.dims:
                time_vals = ds[time_name] if time_name in ds.data_vars else ds.coords.get(time_name)
                if time_vals is not None:
                    break
        if time_vals is None:
            for time_name in TIME_NAMES:
                if time_name in ds.coords:
                    time_vals = ds.coords[time_name]
                    break
        if time_vals is None:
            for time_name in TIME_NAMES:
                if time_name in ds.data_vars:
                    time_vals = ds[time_name]
                    break

        # If nothing found, search for a variable with datetime64 dtype
        if time_vals is None:
            for _, var in ds.data_vars.items():
                if np.issubdtype(var.dtype, np.datetime64):
                    time_vals = var
                    break

        if time_vals is not None:
            # If array is empty
            if time_vals.size == 0:
                return (None, None)
            # If datetime — the "time" coordinate is always a small 1-D array
            # (one value per observation point / profile / time step, never a
            # full N-D field), so loading it eagerly is cheap regardless of
            # dataset size. This also sidesteps a dask/numpy incompatibility
            # where dask's tree-reduce machinery for min()/max() on a lazy
            # (dask-backed) datetime64 array internally attempts a "sum"
            # combine step, which raises
            # `_UFuncBinaryResolutionError: ufunc 'add' cannot use operands
            # with types dtype('<M8[ns]') and dtype('<M8[ns]')` — silently
            # producing (None, None) for every affected file if left uncaught.
            if np.issubdtype(time_vals.dtype, np.datetime64):
                time_vals = time_vals.load()
                dt_min = time_vals.min().values
                dt_max = time_vals.max().values
                ts_min, ts_max = pd.Timestamp(dt_min), pd.Timestamp(dt_max)
                if not (_is_sane_timestamp(ts_min) and _is_sane_timestamp(ts_max)):
                    # Likely an incomplete/corrupted zarr store: some declared
                    # variables have no actual chunk data on disk, so zarr
                    # silently substitutes fill-values (e.g. 0 -> 1970-01-01,
                    # or an int64 overflow near the datetime64[ns] boundaries
                    # ~1677/~2262). Fall back to ACDD global attrs if present,
                    # otherwise treat as having no reliable temporal metadata
                    # so the entry is excluded rather than corrupting date-
                    # range filtering with a bogus (effectively all-matching)
                    # window.
                    fallback = _time_coverage_attrs_fallback(ds)
                    if fallback is not None:
                        logger.warning(
                            "Corrupted time bounds detected "
                            f"(date_start={ts_min}, date_end={ts_max}); "
                            f"using time_coverage_start/end attrs instead: {fallback}"
                        )
                        return fallback
                    logger.warning(
                        "Corrupted time bounds detected "
                        f"(date_start={ts_min}, date_end={ts_max}) and no usable "
                        "time_coverage_start/end attrs fallback; treating as no "
                        "reliable temporal metadata."
                    )
                    return (None, None)
                return (ts_min, ts_max)
            # If numeric
            elif np.issubdtype(time_vals.dtype, np.floating) or np.issubdtype(
                time_vals.dtype, np.integer
            ):
                with np.errstate(all="ignore"):
                    num_min = float(time_vals.min(skipna=True).values)
                    num_max = float(time_vals.max(skipna=True).values)
                if np.isnan(num_min) or np.isnan(num_max):
                    return (None, None)
                return (num_min, num_max)
            elif time_vals.dtype == object:
                # cftime objects (from use_cftime=True) or other object arrays
                try:
                    values = np.asarray(time_vals.values).ravel()
                    if len(values) == 0:
                        return (None, None)
                    t_min = values.min()
                    t_max = values.max()
                    return (pd.Timestamp(t_min.isoformat()), pd.Timestamp(t_max.isoformat()))
                except Exception:
                    try:
                        converted = pd.to_datetime(
                            [str(v) for v in values], errors="coerce"
                        )
                        valid = converted.dropna()
                        if len(valid) > 0:
                            return (valid.min(), valid.max())
                    except Exception:
                        pass
                    return (None, None)
            else:
                logger.warning(f"Unsupported time data type: {time_vals.dtype}")
                return (None, None)
        else:
            logger.debug("No temporal data found in dataset")
            return (None, None)
    except Exception as exc:
        logger.warning(f"Failed to get time bounds for DS: {ds} : {repr(exc)}")
        traceback.print_exc()
        return (None, None)
