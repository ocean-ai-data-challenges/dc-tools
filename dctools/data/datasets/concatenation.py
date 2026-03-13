#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Dataset concatenation helpers.

Provides eager and delayed concatenation along an arbitrary dimension with
optional sorting.
"""

from typing import Any, List

import dask
import numpy as np
import xarray as xr


def concat_with_dim_delayed(
    datasets: List[xr.Dataset],
    concat_dim: str,
    sort: bool = True,
):
    """Concatenate datasets along a dimension using dask.delayed."""
    datasets_with_dim: List[Any] = []
    for i, ds in enumerate(datasets):
        if concat_dim not in ds.dims:
                ds = ds.expand_dims({concat_dim: [i]})
        datasets_with_dim.append(ds)

    result = dask.delayed(xr.concat)(
        datasets_with_dim,
        dim=concat_dim,
        coords="minimal",
        compat="override",
        join="outer"
    )
    if sort:
        result = dask.delayed(result.sortby)(concat_dim)
    return result


def concat_with_dim(
    datasets: List[xr.Dataset],
    concat_dim: str,
    sort: bool = True,
):
    """Concatenate datasets along a dimension eagerly."""
    datasets_refs: List[Any] = list(datasets)
    datasets_with_dim: List[Any] = []
    for i in range(len(datasets_refs)):
        ds = datasets_refs[i]
        # Release reference in source list immediately (memory optimization)
        datasets_refs[i] = None

        if "time" in ds.coords:
            # Check dtype without loading data
            dtype = ds.coords["time"].dtype
            if np.issubdtype(dtype, np.integer):
                pass
            elif dtype == "O":
                pass
            else:
                 pass

        if concat_dim not in ds.dims:
                ds = ds.expand_dims({concat_dim: [i]})
        datasets_with_dim.append(ds)

    # Optimization: override to avoid comparing all coordinates (slow on S3/Dask)
    # join='override' assumes that non-concatenated coordinates are identical.
    join_mode = "outer"
    compat_mode = "no_conflicts"

    # If we concatenate on time or n_points for massive data, override is much faster
    if len(datasets) > 10:
        join_mode = "override"
        compat_mode = "override"

    result: xr.Dataset = xr.concat(  # type: ignore[call-overload]
        datasets_with_dim, dim=concat_dim,
        coords="minimal",
        compat=compat_mode, join=join_mode,
    )
    if sort:
        result = result.sortby(concat_dim)
    return result
