#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""NaN-filtering utilities for observation point datasets.

Provides three complementary masking strategies:

- ``_drop_nan_points`` — filter already-computed (numpy) datasets.
- ``_build_nan_mask`` — build a mask lazily via dask (pre-compute).
- ``_nan_mask_numpy`` — build a mask from in-memory numpy arrays (post-compute).
"""

from typing import Any, Optional

import numpy as np
import xarray as xr


def _drop_nan_points(ds: xr.Dataset, n_points_dim: str) -> xr.Dataset:
    """Drop points where ALL data variables are NaN along n_points_dim.

    SWOT swath grids stack every (num_lines, num_pixels) cell, including
    land, ice, and orbital-gap areas that are entirely fill-value.  These
    NaN-only points can represent 60-90 % of each file's size and have
    zero scientific value.  Removing them immediately after per-file
    compute() keeps the accumulator list lean before the final concat.
    """
    if n_points_dim not in ds.dims:
        return ds

    n_pts = ds.sizes[n_points_dim]
    if n_pts == 0:
        return ds

    valid_mask: Optional[np.ndarray] = None
    for vname in ds.data_vars:
        v = ds[vname]
        if n_points_dim not in v.dims:
            continue
        arr = v.values
        if arr.ndim == 1:
            finite = np.isfinite(arr)
        else:
            # Multi-dim variable: a point is valid if any element along
            # non-n_points axes is finite.
            ax0_size = arr.shape[v.dims.index(n_points_dim)]
            try:
                flat = arr.reshape(ax0_size, -1)
                finite = np.any(np.isfinite(flat), axis=1)
            except Exception:
                continue
        valid_mask = finite if valid_mask is None else (valid_mask | finite)

    if valid_mask is None or valid_mask.all():
        return ds

    n_valid = int(valid_mask.sum())
    if n_valid == 0:
        return ds  # keep caller's None-check handling

    return ds.isel({n_points_dim: valid_mask})


def _build_nan_mask(ds: xr.Dataset, n_points_dim: str) -> Optional[np.ndarray]:
    """Return a pre-computed 1-D boolean mask of *valid* (non-all-NaN) points.

    Unlike :func:`_drop_nan_points` (which works on already-computed numpy
    arrays), this function builds the mask from the **lazy** dask graph so
    that xarray never needs to materialise the full dataset in RAM.  Dask
    evaluates the mask chunk-by-chunk (peak ≈ one chunk); the returned numpy
    array is only *n_pts* booleans (≪ full data).

    Returns
    -------
    np.ndarray or None
        1-D boolean mask (True = valid point).  ``None`` when all points are
        valid (caller should skip filtering).
    """
    if n_points_dim not in ds.dims or ds.sizes.get(n_points_dim, 0) == 0:
        return None

    import dask.array as da  # local import to keep module-level deps clean

    combined_mask = None
    for vname in ds.data_vars:
        v = ds[vname]
        if n_points_dim not in v.dims:
            continue
        raw = v.data  # dask array or numpy
        n_pts_axis = v.dims.index(n_points_dim)
        other_axes = tuple(i for i in range(raw.ndim) if i != n_pts_axis)
        if isinstance(raw, da.Array):
            finite: Any = da.isfinite(raw)
            if other_axes:
                finite = da.any(finite, axis=other_axes)
        else:
            finite = np.isfinite(raw)
            if other_axes:
                finite = np.any(finite, axis=other_axes)
        combined_mask = (
            finite if combined_mask is None else (combined_mask | finite)
        )

    if combined_mask is None:
        return None

    # Compute only the small 1-D mask (cheap: n_pts × 1 byte).
    if hasattr(combined_mask, "compute"):
        mask_np: np.ndarray = combined_mask.compute()
    else:
        mask_np = np.asarray(combined_mask)

    if mask_np.all():
        return None  # all valid — caller can skip isel
    return mask_np


def _nan_mask_numpy(ds: xr.Dataset, n_points_dim: str) -> Optional[np.ndarray]:
    """Build NaN mask from an already-computed (in-memory) dataset.

    Unlike :func:`_build_nan_mask` (which creates a dask graph and reads data
    lazily), this operates on numpy arrays directly.  Use it **after**
    ``.compute()`` to avoid the double-read penalty.
    """
    if n_points_dim not in ds.dims or ds.sizes.get(n_points_dim, 0) == 0:
        return None
    combined: Optional[np.ndarray] = None
    for vname in ds.data_vars:
        v = ds[vname]
        if n_points_dim not in v.dims:
            continue
        vals = v.values  # already numpy
        n_pts_axis = v.dims.index(n_points_dim)
        other_axes = tuple(i for i in range(vals.ndim) if i != n_pts_axis)
        finite = np.isfinite(vals)
        if other_axes:
            finite = np.any(finite, axis=other_axes)
        combined = finite if combined is None else (combined | finite)
    if combined is None or combined.all():
        return None
    return combined
