#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Classes and functions for loading xarray datasets.

The implementation moved to ``dctools_core.loader`` (packages/dctools-core) so that it can be
installed without the evaluator stack; this module re-exports it under its historical name.
"""

from dctools_core.loader import (  # noqa: F401
    FileLoader,
    choose_chunks_automatically,
    list_all_group_paths,
    _fix_nanosecond_time,
    _fix_stringified_scale_attrs,
    _nc_engine_for,
    _zarr_vars_with_stringified_scale_attrs,
)
