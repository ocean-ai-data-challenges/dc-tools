#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""ObservationDataViewer — per-batch observation preprocessing on workers.

This class is used inside evaluator workers to preprocess observation files
(SWOT swaths, nadir tracks, Argo profiles) into a single xarray Dataset
ready for metric computation.
"""

import gc
import os
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import dask
import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger
from oceanbench.core.distributed import DatasetProcessor

from dctools.data.datasets.concatenation import concat_with_dim
from dctools.data.datasets.preprocessing import (
    preprocess_argo_profiles,
    preprocess_one_npoints,
)


class ObservationDataViewer:
    """Class to view and preprocess observation data."""

    def __init__(
        self,
        source: Union[xr.Dataset, List[xr.Dataset], pd.DataFrame, gpd.GeoDataFrame],
        load_fn: Callable[..., xr.Dataset],
        alias: str,
        keep_vars: List[str],
        target_dimensions: Dict[str, Any],
        dataset_metadata: Any,
        time_bounds: Tuple[pd.Timestamp, pd.Timestamp],
        # time_tolerance: pd.Timedelta = pd.Timedelta("12h"),
        n_points_dim: str,
        dataset_processor: Optional[Optional[DatasetProcessor]] = None,
        results_dir: Optional[str] = None,
        include_geometry: bool = False,
        save_preprocessed: bool = False,
    ):
        """
        Initialize the ObservationDataViewer.

        Parameters:
            source: either
                - one or more xarray Datasets (data already loaded)
                - a DataFrame containing metadata, including file links
            load_fn: a callable that loads a dataset given a link
            alias: optional alias to pass to load_fn if needed
            keep_vars: extracted variables to keep
            target_dimensions: target dimensions dict
            dataset_metadata: metadata dict
            time_bounds: time bounds tuple
            n_points_dim: name of points dimension
            dataset_processor: optional processor
            include_geometry: whether to include geometry column
            save_preprocessed: whether to persist preprocessed data to Zarr
        """
        self.is_metadata = isinstance(source, (pd.DataFrame, gpd.GeoDataFrame))
        self.load_fn = load_fn
        # self.time_tolerance = time_tolerance
        self.alias = alias
        self.keep_vars = keep_vars
        self.target_dimensions = target_dimensions
        self.n_points_dim = n_points_dim
        self.dataset_processor = dataset_processor
        self.time_bounds = time_bounds
        self.results_dir = results_dir
        self.save_preprocessed = save_preprocessed

        if self.is_metadata:
            if self.load_fn is None:
                raise ValueError("load_fn must be provided when source is a DataFrame/GeoDataFrame")
            self.meta_df = source
        else:
            self.datasets = source if isinstance(source, list) else [source]
        self.coordinates = dataset_metadata['coord_system'].coordinates
        self.include_geometry = include_geometry

    def save_to_zarr(self, dataset: xr.Dataset, root_path: str):
        """
        Save preprocessed dataset to a Zarr file in the specified root path.

        Parameters:
            dataset: The xarray Dataset to save.
            root_path: The root directory path where the Zarr file will be saved.
        """
        # Save preprocessed dataset in a Zarr file
        time_val = dataset.coords["time"].values

        # If it's an array with a single value
        if isinstance(time_val, (np.ndarray, list)) and len(time_val) == 1:
            time_str = str(pd.to_datetime(time_val[0]))
        else:
            time_str = str(pd.to_datetime(time_val))
        argo_name = f"argo_profiles_{time_str}.zarr"
        path = os.path.join(root_path, argo_name)
        dataset.to_zarr(path, mode="w", consolidated=True)

    def preprocess_datasets(
        self,
        dataframe: pd.DataFrame,
        load_to_memory: bool = False,
    ) -> Optional[xr.Dataset]:
        """
        Preprocess the input DataFrame and single observations files.

        Returns:
            xr.Dataset: The preprocessed dataset.
        """
        # remove "geometry" fields if needed:
        if not self.include_geometry and "geometry" in dataframe.columns:
            dataframe = dataframe.drop(columns=["geometry"])

        # File loading
        dataset_paths = [row["path"] for _, row in dataframe.iterrows()]
        if not dataset_paths:
            logger.warning(f"No dataset paths found for alias '{self.alias}'")
            return None

        # log diagnostic
        total_files = len(dataset_paths)
        if total_files > 100:
            logger.info(
                f"Preprocessing large batch of files for {self.alias}: {total_files} files. "
                "This may generate a large Dask graph."
            )

        # swath_dims = {"num_lines", "num_pixels", "num_nadir"}
        reduced_swath_dims = {"num_lines", "num_pixels"}

        # if argo profiles, special preprocessing:
        # NOTE: This is the FALLBACK path.  The preferred pipeline is the
        # shared-Zarr approach in evaluator._evaluate_batch() which merges
        # all batch time-windows, downloads once, and lets workers filter
        # by searchsorted.  This branch only runs when the evaluator's
        # prefetch did not happen (e.g. standalone ObservationDataViewer
        # usage outside the Evaluator class).

        if self.alias == "argo_profiles":
            try:
                result = preprocess_argo_profiles(
                    profile_sources=dataset_paths,
                    open_func=self.load_fn,
                    alias=self.alias,
                    time_bounds=self.time_bounds,
                    depth_levels=self.target_dimensions.get('depth', np.array([])),
                )
                if result is None:
                    logger.error("No Argo profiles could be processed")
                    return None

                if load_to_memory:
                    result = result.compute()

                if self.save_preprocessed and self.results_dir:
                    save_path = os.path.join(self.results_dir, self.alias + "_preprocessed")
                    os.makedirs(save_path, exist_ok=True)
                    self.save_to_zarr(result, save_path)
                result_ds: xr.Dataset = xr.Dataset(result)
                return result_ds
            except Exception as e:
                logger.error(f"Argo preprocessing failed: {e}")
                traceback.print_exc()
                return None

        first_ds = None
        try:
            if self.alias is not None:
                first_ds = self.load_fn(dataset_paths[0], self.alias)
            else:
                first_ds = self.load_fn(dataset_paths[0])
        except Exception as exc:
            logger.error(
                f"Failed to open first dataset for alias '{self.alias}' "
                f"during preprocessing probe: {exc}"
            )
            traceback.print_exc()
            return None

        if first_ds is None:
            logger.error(
                f"Failed to open first dataset for alias '{self.alias}' "
                "during preprocessing probe (received None)."
            )
            return None

        # --- Capture dimension flags, then immediately free the probe dataset ---
        # first_ds is only needed to decide which preprocessing branch to take.
        # Keeping it alive through the N-file loop wastes RAM equal to one full
        # SWOT file (often 50-300 MB) and can double-open the first file.
        _probe_has_npoints = (
            self.n_points_dim in first_ds.dims
            or ("time" in first_ds.dims and len(first_ds.dims) == 1)
        )
        _probe_is_swath = reduced_swath_dims.issubset(first_ds.dims)
        _probe_dims = dict(first_ds.sizes)  # lightweight copy
        del first_ds
        gc.collect()

        # Data with n_points/N_POINTS dimension only
        # OR special case: unique "time" dimension (saral tracks, etc)
        # OR swath data (num_lines, num_pixels) -- unified path.
        if _probe_has_npoints or _probe_is_swath:
            _is_swath = _probe_is_swath and not _probe_has_npoints
            # When _probe_has_npoints and _probe_is_swath are both True,
            # favour npoints (is_swath=False) to avoid redundant flattening.
            if _probe_has_npoints:
                _is_swath = False
            elif _probe_is_swath:
                _is_swath = True

            try:
                # Clean and process datasets
                if self.dataset_processor is not None and load_to_memory:
                    delayed_tasks: List[Any] = []
                    for idx, dataset_path in enumerate(dataset_paths):
                        delayed_tasks.append(dask.delayed(preprocess_one_npoints)(
                            dataset_path, _is_swath, self.n_points_dim, dataframe, idx,
                            self.alias, self.load_fn,
                            self.keep_vars, self.target_dimensions,
                            self.coordinates,
                            self.time_bounds,
                            load_to_memory,
                        ))
                    batch_results = self.dataset_processor.compute_delayed_tasks(
                        delayed_tasks, sync=False
                    )
                else:
                    # Sequential preprocessing — this viewer runs inside a
                    # Dask worker process which is already a separate OS process.
                    # Using threads for CPU-bound work (numpy, swath_to_points)
                    # gives no benefit under the GIL and oversubscribes the core.
                    from dctools.data.datasets.nan_filtering import (
                        _nan_mask_numpy,
                    )

                    def _process_one_file(idx_path):
                        idx, dataset_path = idx_path
                        result = preprocess_one_npoints(
                            dataset_path, _is_swath, self.n_points_dim, dataframe, idx,
                            self.alias, self.load_fn,
                            self.keep_vars, self.target_dimensions,
                            self.coordinates,
                            self.time_bounds,
                            load_to_memory,
                        )
                        if result is None:
                            return None
                        if hasattr(result, 'chunks') and result.chunks:
                            try:
                                result = result.compute(
                                    scheduler="synchronous"
                                )
                            except Exception:
                                pass
                        _nmask = _nan_mask_numpy(
                            result, self.n_points_dim
                        )
                        if _nmask is not None:
                            if int(_nmask.sum()) == 0:
                                del result
                                return None
                            result = result.isel(
                                {self.n_points_dim: _nmask}
                            )
                        if result.sizes.get(
                            self.n_points_dim, 0
                        ) > 0:
                            return result
                        del result
                        return None

                    # Sequential processing — this code runs INSIDE a Dask
                    # worker process.  Using threads for CPU-bound work
                    # (swath_to_points, NaN filtering, pyinterp) only adds
                    # GIL contention and oversubscribes the core that the
                    # Dask scheduler already assigned to us.  Process files
                    # one by one to keep memory predictable and avoid
                    # thread-level CPU thrashing.
                    batch_results_sync: List[Any] = []

                    for idx, path in enumerate(dataset_paths):
                        try:
                            r = _process_one_file((idx, path))
                            if r is not None:
                                batch_results_sync.append(r)
                        except Exception as exc:
                            logger.warning(
                                f"File {path} failed during sequential "
                                f"preprocessing: {exc}"
                            )

                    batch_results = batch_results_sync

                if not batch_results:
                    return None

                # Combine results — sort=False because n_points is just an
                # integer index; sorting by it would interleave unrelated
                # files and waste CPU + RAM on a full-array reindex copy.
                # Release individual datasets as they are consumed by concat
                # to avoid holding 2× peak memory (list + concatenated result).
                combined = concat_with_dim(batch_results, self.n_points_dim, sort=False)
                del batch_results
                gc.collect()

                return xr.Dataset(combined) if combined is not None else None

            except Exception as e:
                logger.error(f"Preprocessing failed for {self.alias}: {e}")
                traceback.print_exc()
                return None
        else:
            logger.error(
                f"Dataset for {self.alias} has unsupported dimensions: {_probe_dims}"
            )
            return None
