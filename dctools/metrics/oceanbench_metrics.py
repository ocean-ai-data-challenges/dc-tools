# -*- coding: UTF-8 -*-

"""Wrapper for functions implemented in Mercator's oceanbench library."""

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
    """
    import numpy as np

    per_bins: Dict[str, list] = {}

    def _iter_depth_slices(da: xr.DataArray) -> List[tuple[Optional[int], Optional[Dict[str, float]]]]:
        da = da.squeeze(drop=True)
        if "depth" not in da.dims:
            return [(None, None)]

        depth_values = np.asarray(da["depth"].values, dtype=np.float64)
        if depth_values.ndim != 1 or depth_values.size == 0:
            return []
        if depth_values.size == 1:
            depth = float(depth_values[0])
            return [(0, {"left": depth, "right": depth, "closed": "right"})]

        return [
            (
                depth_index,
                {
                    "left": float(depth_values[depth_index]),
                    "right": float(depth_values[depth_index + 1]),
                    "closed": "right",
                },
            )
            for depth_index in range(depth_values.size - 1)
        ]

    # Determine variables to process
    var_names = [v for v in eval_variables if v in pred_ds.data_vars and v in ref_ds.data_vars]
    if not var_names:
        # Fallback: use common data vars
        var_names = [v for v in pred_ds.data_vars if v in ref_ds.data_vars]

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

    # Build bin edges
    lat_bins = np.arange(-90, 90 + bin_resolution, bin_resolution)
    lon_bins = np.arange(-180, 180 + bin_resolution, bin_resolution)

    for var in var_names:
        try:
            pred_da = pred_ds[var].squeeze(drop=True)
            ref_da = ref_ds[var].squeeze(drop=True)
        except Exception:
            continue

        if set(pred_da.dims) != set(ref_da.dims):
            continue

        depth_slices = _iter_depth_slices(pred_da)
        if not depth_slices:
            continue

        bins_list = []
        for depth_index, depth_bin in depth_slices:
            if depth_index is None:
                pred_slice = pred_da
                ref_slice = ref_da
            else:
                if "depth" not in ref_da.dims or ref_da.sizes.get("depth", 0) <= depth_index:
                    continue

                pred_slice = pred_da.isel(depth=depth_index)
                ref_slice = ref_da.isel(depth=depth_index)

            try:
                pred_slice = pred_slice.transpose("lat", "lon")
                ref_slice = ref_slice.transpose("lat", "lon")
            except Exception:
                continue

            pred_arr = pred_slice.values.astype(np.float64)
            ref_arr = ref_slice.values.astype(np.float64)
            if pred_arr.shape != ref_arr.shape or pred_arr.ndim != 2:
                continue

            for i in range(len(lat_bins) - 1):
                lat_mask = (lat_coord >= lat_bins[i]) & (lat_coord < lat_bins[i + 1])
                if not lat_mask.any():
                    continue
                for j in range(len(lon_bins) - 1):
                    lon_mask = (lon_coord >= lon_bins[j]) & (lon_coord < lon_bins[j + 1])
                    if not lon_mask.any():
                        continue

                    # Extract sub-array for this bin
                    sub_pred = pred_arr[np.ix_(lat_mask, lon_mask)]
                    sub_ref = ref_arr[np.ix_(lat_mask, lon_mask)]

                    # Flatten and mask NaN
                    sp = sub_pred.ravel()
                    sr = sub_ref.ravel()
                    valid = ~(np.isnan(sp) | np.isnan(sr))
                    n_valid = int(valid.sum())
                    if n_valid == 0:
                        continue

                    diff = sp[valid] - sr[valid]
                    rmse_val = float(np.sqrt(np.mean(diff * diff)))

                    bin_entry = {
                        "lat_bin": {
                            "left": float(lat_bins[i]),
                            "right": float(lat_bins[i + 1]),
                        },
                        "lon_bin": {
                            "left": float(lon_bins[j]),
                            "right": float(lon_bins[j + 1]),
                        },
                        "rmse": rmse_val,
                        "count": n_valid,
                    }
                    if depth_bin is not None:
                        bin_entry["depth_bin"] = depth_bin
                    bins_list.append(bin_entry)

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
    """Run Class4 evaluation while preserving raw per_bins for leaderboard maps."""
    if not _class4_compat_helpers_available():
        raise RuntimeError("Required oceanbench Class4 helpers are not available")

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
        obs_col = f"{var}_obs"
        model_col = f"{var}_model"

        if matching_type == "nearest":
            obs_df = oceanbench_class4_module.xr_to_obs_dataframe(
                obs_da,
                include_geometry=False,
            )
            # Standardize coordinate column names so that apply_binning and
            # interpolate_model_on_obs can always find "lat" / "lon".
            # SWOT (and some other observation datasets) expose their position
            # coordinates as "latitude" / "longitude" rather than the short
            # aliases expected by the rest of the pipeline.
            _coord_alias_map = {
                "latitude": "lat", "nav_lat": "lat",
                "longitude": "lon", "nav_lon": "lon",
            }
            _rename = {
                k: v for k, v in _coord_alias_map.items()
                if k in obs_df.columns and v not in obs_df.columns
            }
            if _rename:
                obs_df = obs_df.rename(columns=_rename)

            obs_df, groupby_cols = oceanbench_class4_module.apply_binning(
                obs_df,
                getattr(evaluator, "bin_specs", None),
            )
            if obs_df.empty:
                continue

            if var not in obs_df.columns:
                for candidate in ("value", "variable"):
                    if candidate in obs_df.columns:
                        obs_df = obs_df.rename(columns={candidate: var})
                        break
            if var not in obs_df.columns:
                continue

            obs_df = obs_df.dropna(subset=[var]).copy()
            if obs_df.empty:
                continue

            obs_df = obs_df.rename(columns={var: obs_col})
            final_df = oceanbench_class4_module.interpolate_model_on_obs(
                model_da,
                obs_df,
                var,
                method=getattr(evaluator, "interp_method", "nearest"),
            )
        elif matching_type == "superobs":
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
        else:
            raise ValueError(f"Unknown matching_type: {matching_type}")

        # cos(latitude) area weighting — accounts for the convergence of
        # meridians so that high-latitude points do not receive the same
        # weight as equatorial ones in the global RMSD.
        # Inject as a column so grouped slicing is automatic.
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
