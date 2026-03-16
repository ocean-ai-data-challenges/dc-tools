#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Evaluation dataloader — batch iteration over prediction × reference pairs.

This module contains the main :class:`EvaluationDataloader` class that
drives the evaluation loop by yielding batches of prediction/reference
entries.

Sub-modules (split for readability):
- :mod:`~dctools.data.datasets.preprocessing` — per-file preprocessing helpers
- :mod:`~dctools.data.datasets.nan_filtering` — NaN mask / drop utilities
- :mod:`~dctools.data.datasets.concatenation` — eager and delayed concat
- :mod:`~dctools.data.datasets.batch_preprocessing` — driver-side shared zarr
- :mod:`~dctools.data.datasets.observation_viewer` — ObservationDataViewer class
"""

import traceback
from typing import Any, Dict, Generator, List, Optional, Type

import pandas as pd
import xarray as xr
from loguru import logger

from dctools.data.connection.connection_manager import (
    ArgoManager,
    BaseConnectionManager,
    CMEMSManager,
    FTPManager,
    GlonetManager,
    LocalConnectionManager,
    S3Manager,
    S3WasabiManager,
)

# -- Backward-compatible re-exports --------------------------------------
# These symbols used to live here.  Keep them importable from the old path
# so that external code / tests using
#   ``from dctools.data.datasets.dataloader import <symbol>``
# continue to work without changes.
from dctools.data.datasets.batch_preprocessing import (  # noqa: F401
    preprocess_batch_obs_files,
)
from dctools.data.datasets.concatenation import (  # noqa: F401
    concat_with_dim,
    concat_with_dim_delayed,
)
from dctools.data.datasets.nan_filtering import (  # noqa: F401
    _build_nan_mask,
    _drop_nan_points,
    _nan_mask_numpy,
)
from dctools.data.datasets.observation_viewer import ObservationDataViewer  # noqa: F401
from dctools.data.datasets.preprocessing import (  # noqa: F401
    add_coords_as_dims,
    add_time_dim,
    filter_by_time,
    preprocess_argo_profiles,
    preprocess_one_npoints,
    swath_to_points,
)

# Dictionary mapping names to classes
CLASS_REGISTRY: Dict[str, Type[BaseConnectionManager]] = {
    "S3WasabiManager": S3WasabiManager,
    "FTPManager": FTPManager,
    "GlonetManager": GlonetManager,
    "ArgoManager": ArgoManager,
    "CMEMSManager": CMEMSManager,
    "S3Manager": S3Manager,
    "LocalConnectionManager": LocalConnectionManager,
}


class EvaluationDataloader:
    """Class to manage loading and batching of evaluation data."""

    pred_catalog: Any
    ref_catalogs: Dict[str, Any]
    ref_aliases: List[str]
    forecast_mode: bool
    forecast_index: Optional[pd.DataFrame]
    n_days_forecast: int
    time_tolerance: Any
    keep_variables: Dict[str, List[str]]
    metadata: Dict[str, Any]
    optimize_for_parallel: bool
    min_batch_size_for_parallel: int
    pred_coords: Any
    ref_coords: Any
    pred_manager: Any
    ref_managers: Dict[str, Any]
    target_dimensions: Dict[str, Any]
    lead_time_unit: str
    file_cache: Any
    pred_alias: str
    pred_connection_params: Any
    ref_connection_params: Dict[str, Any]
    pred_transform: Any
    ref_transforms: Optional[Dict[str, Any]]

    def __init__(
        self,
        params: dict,
    ):
        """
        Initializes the dataloader for data collections.

        Args:
            params: parameter dictionary
        """
        for key, value in params.items():
            setattr(self, key, value)
        self.pred_coords = self.pred_catalog.get_global_metadata().get("coord_system", None)
        self.ref_coords = {
            ref_alias: ref_catalog.get_global_metadata().get("coord_system", None)
            for ref_alias, ref_catalog in self.ref_catalogs.items()
        }

        self.optimize_for_parallel = True
        self.min_batch_size_for_parallel = 4

    def __len__(self):
        """Return the number of batches."""
        if self.forecast_mode and self.forecast_index is not None:
            return len(self.forecast_index)
        return len(self.pred_catalog.get_dataframe())

    def __iter__(self):
        """Iterate over batches of data."""
        return self._generate_batches()

    def _find_matching_ref(self, valid_time, ref_alias):
        """Find matching reference file covering valid_time for ref_alias."""
        ref_df = self.ref_catalogs[ref_alias].get_dataframe()
        match = ref_df[(ref_df["date_start"] <= valid_time) & (ref_df["date_end"] >= valid_time)]
        if not match.empty:
            return match.iloc[0]["path"]
        return None

    def _generate_batches(self) -> Generator[List[Dict[str, Any]], None, None]:
        batch: List[Any] = []
        # -- Volume-aware batch splitting for observation datasets ---------
        # Instead of splitting purely by entry count (obs_batch_size), also
        # track the cumulative number of unique observation files across the
        # batch.  When the file count exceeds the threshold the batch is
        # yielded early, preventing a single batch from accumulating far
        # more data than the driver can preprocess in RAM.
        # Configurable via the dataloader param or DCTOOLS_MAX_OBS_FILES_PER_BATCH.
        import os as _os_gen
        _MAX_OBS_FILES: int = int(getattr(
            self, "max_obs_files_per_batch", 0,
        ) or _os_gen.environ.get("DCTOOLS_MAX_OBS_FILES_PER_BATCH", "150"))
        _batch_obs_paths: set = set()
        try:
            # Check maximum available date in reference data
            if self.forecast_index is None:
                logger.error("forecast_index is None, cannot generate batches")
                return

            for ref_alias in self.ref_aliases:
                _batch_obs_paths = set()  # reset per ref_alias
                for _, row in self.forecast_index.iterrows():
                    # Check if enough data for this forecast
                    forecast_reference_time = row["forecast_reference_time"]
                    lead_time = row["lead_time"]
                    valid_time = row["valid_time"]

                    # Calculate full forecast end (last lead time)
                    max_lead_time = self.n_days_forecast - 1  # 0-indexed
                    if (
                        hasattr(self, 'lead_time_unit')
                        and self.lead_time_unit == "hours"
                    ):
                        forecast_end_time = (
                            forecast_reference_time + pd.Timedelta(hours=max_lead_time)
                        )
                    else:
                        forecast_end_time = (
                            forecast_reference_time + pd.Timedelta(days=max_lead_time)
                        )

                    entry = {
                        "forecast_reference_time": forecast_reference_time,
                        "lead_time": lead_time,
                        "valid_time": valid_time,
                        "pred_data": row["file"],
                        "ref_data": None,
                        "ref_alias": ref_alias,
                        "pred_coords": self.pred_coords,
                        "ref_coords": self.ref_coords[ref_alias] if ref_alias else None,
                    }
                    if ref_alias:
                        ref_catalog = self.ref_catalogs[ref_alias]
                        ref_df = ref_catalog.get_dataframe()

                        # Find max available date in reference data
                        max_available_date = ref_df["date_end"].max()

                        # If forecast end > available data, skip this entry
                        if forecast_end_time > max_available_date:
                            logger.debug(
                                f"Skipping forecast starting at {forecast_reference_time}: "
                                f"forecast ends at {forecast_end_time} "
                                f"but data only available until {max_available_date}"
                            )
                            if batch:
                                yield batch
                                batch = []
                                _batch_obs_paths = set()
                            break

                        # ref_catalog = self.ref_catalogs[ref_alias]
                        coord_system = ref_catalog.get_global_metadata().get("coord_system")
                        if coord_system:
                            is_observation = coord_system.is_observation_dataset()
                        else:
                            is_observation = False

                        if is_observation:
                            # Observation logic: filter observation catalog
                            # on forecast_index time interval
                            obs_time_interval = (valid_time, valid_time)
                            keep_vars = self.keep_variables[ref_alias]
                            rename_vars_dict = self.metadata[ref_alias]['variables_dict']
                            keep_vars = [
                                rename_vars_dict[var]
                                for var in keep_vars
                                if var in rename_vars_dict
                            ]

                            t0, t1 = obs_time_interval
                            t0 = t0 - self.time_tolerance
                            t1 = t1 + self.time_tolerance
                            time_bounds = (t0, t1)

                            # -- Volume estimation for this entry ------
                            # Count how many unique observation files this
                            # entry would add to the current batch.  If the
                            # projected total exceeds the threshold, yield
                            # the current batch first to cap driver-side
                            # memory usage during shared zarr construction.
                            _entry_paths = set(
                                filter_by_time(ref_df, t0, t1)["path"].tolist()
                            )
                            _projected = len(_batch_obs_paths | _entry_paths)
                            if batch and _projected > _MAX_OBS_FILES:
                                logger.debug(
                                    f"Volume split ({ref_alias}): {_projected} "
                                    f"unique files would exceed limit "
                                    f"{_MAX_OBS_FILES} — yielding batch "
                                    f"({len(batch)} entries, "
                                    f"{len(_batch_obs_paths)} files)"
                                )
                                yield batch
                                batch = []
                                _batch_obs_paths = set()
                            _batch_obs_paths |= _entry_paths

                            entry["ref_data"] = {
                                "source": ref_catalog,
                                "keep_vars": keep_vars,
                                "target_dimensions": self.target_dimensions,
                                "metadata": self.metadata[ref_alias],
                                "time_bounds": time_bounds,
                            }
                            entry["ref_is_observation"] = True
                            entry["obs_time_interval"] = obs_time_interval
                        else:
                            # Gridded logic: associate reference file covering valid_time
                            ref_path = self._find_matching_ref(valid_time, ref_alias)
                            if ref_path is None:
                                logger.debug(f"No reference data found for valid_time {valid_time}")
                                continue
                            entry["ref_data"] = ref_path
                            entry["ref_is_observation"] = False

                    batch.append(entry)
                    # Adapt batch size according to observation/gridded type:
                    # - observation datasets (SWOT, saral ...): use obs_batch_size
                    #   to limit per-batch S3 download volume.
                    # - gridded datasets (GLORYS ...): use gridded_batch_size
                    #   to limit per-batch 3-D I/O and worker RAM pressure.
                    # - fallback: no limit (single batch).
                    _effective_bs: float = float("inf")
                    if entry.get("ref_is_observation") and getattr(
                        self, "obs_batch_size", None
                    ):
                        _effective_bs = self.obs_batch_size  # type: ignore[attr-defined]
                    elif not entry.get("ref_is_observation") and getattr(
                        self, "gridded_batch_size", None
                    ):
                        _effective_bs = self.gridded_batch_size  # type: ignore[attr-defined]
                    if len(batch) >= _effective_bs:
                        yield batch
                        batch = []
                        _batch_obs_paths = set()
                if batch:  # last batch of ref_alias
                    yield batch
                    batch = []
                    _batch_obs_paths = set()
        except Exception as e:
            logger.error(f"Error generating batches: {e}")
            traceback.print_exc()

    def open_pred(self, pred_entry: str) -> xr.Dataset:
        """Open a prediction dataset."""
        pred_data: xr.Dataset = self.pred_manager.open(pred_entry, self.file_cache)
        return pred_data

    def open_ref(self, ref_entry: str, ref_alias: str) -> xr.Dataset:
        """Open a reference dataset."""
        ref_data: xr.Dataset = self.ref_managers[ref_alias].open(ref_entry, self.file_cache)
        return ref_data
