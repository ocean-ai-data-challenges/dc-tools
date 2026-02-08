# -*- coding: UTF-8 -*-

"""Wrapper for functions implemented in Mercator's oceanbench library."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import oceanbench.metrics as oceanbench_metrics
import xarray as xr
from loguru import logger
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

from dctools.data.coordinates import (
    COORD_ALIASES,
    EVAL_VARIABLES_GLONET,
    GLOBAL_ZONE_COORDINATES,
    CoordinateSystem,
)

# Dictionary of variables of interest: {generic name -> standard_name(s), common aliases}
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

def get_variable_alias(variable: str) -> Variable | None:
    """Get the alias for a given variable.

    Args:
        variable (Variable): The variable to get the alias for.

    Returns:
        Optional[str]: The alias of the variable, or None if not found.
    """
    for alias, var in OCEANBENCH_VARIABLES.items():
        if alias == variable or var == variable:
            return var
    return None


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
        no_default_attrs = ['metric_name', 'var', 'depth']
        class_default_attrs = ['metric_name']
        default_attrs: Dict[str, Any] = dict(
            plot_result=False, minimum_latitude=None, maximum_latitude=None,
            minimum_longitude=None, maximum_longitude=None,
            spatial_resolution=None, small_scale_cutoff_km=100,
        )
        allowed_attrs = list(default_attrs.keys()) + no_default_attrs
        default_attrs.update(kwargs)
        self.__dict__.update((k,v) for k,v in default_attrs.items() if k in allowed_attrs)

        for attr in class_default_attrs:
            assert(hasattr(self, attr))

    def get_metric_name(self) -> Optional[str]:
        """Return the name of the metric.

        Returns:
            str: The name of the metric.
        """
        return self.metric_name

    @abstractmethod
    def compute(
        self, pred_data: xr.Dataset,
        ref_data: Optional[xr.Dataset] = None,
        **kwargs: Any
    ) -> Any:
        """Compute the metric wrapper (includes preprocessing).

        Args:
            pred_data (xr.Dataset): Prediction dataset.
            ref_data (xr.Dataset, optional): Reference dataset.
        """
        pass

    @abstractmethod
    def compute_metric(
        self, pred_data: xr.Dataset,
        ref_data: Optional[xr.Dataset] = None,
        **kwargs: Any
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
        super().__init__(**kwargs)
        self.eval_variables = eval_variables
        self.oceanbench_eval_variables = oceanbench_eval_variables
        self.is_class4 = is_class4
        self.class4_kwargs = class4_kwargs or {}

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
            "class4": None
        }


        if is_class4:
            class4_args = dict(self.class4_kwargs)
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
                res = self.class4_evaluator.run(
                    model_ds=pred_data,
                    obs_ds=ref_data,
                    variables=self.eval_variables,
                    ref_coords=ref_coords,
                )

                # logger.info(f"Class4Evaluator.run time: {t_c4_end - t_c4_start:.4f} seconds")

                return res

            except Exception as exc:
                logger.error(f"Failed to compute metric {self.metric_name}: {repr(exc)}")
                raise
        else:
            if eval_variables:
                has_depth = any(
                    depth_alias in list(pred_data.dims)
                    for depth_alias in COORD_ALIASES["depth"]
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
                result = metric_func(**kwargs)

                return result
            except Exception as exc:
                logger.error(f"Failed to compute metric {self.metric_name}: {repr(exc)}")
                raise
