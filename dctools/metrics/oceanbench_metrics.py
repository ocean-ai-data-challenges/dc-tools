#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Wrapper for functions implemented in Mercator's oceanbench library."""

from abc import ABC, abstractmethod
import traceback
from typing import Any, Callable, Dict, List, Optional
from typing_extensions import Unpack

from loguru import logger
#import numpy.typing as npt
from memory_profiler import profile
from numpy import ndarray
import oceanbench.metrics as oceanbench_metrics
from oceanbench.core.rmsd import rmsd, Variable
from oceanbench.core.lagrangian_trajectory import ZoneCoordinates, deviation_of_lagrangian_trajectories
from oceanbench.core.derived_quantities import add_mixed_layer_depth
from oceanbench.core.derived_quantities import add_geostrophic_currents
import xarray as xr

from oceanbench.core.class4_metrics.class4_evaluator import Class4Evaluator
from dctools.data.coordinates import (
    CoordinateSystem,
    EVAL_VARIABLES_GLONET,
    GLOBAL_ZONE_COORDINATES,
    COORD_ALIASES,
)

# Dictionnaire des variables d'intérêt : {nom générique -> standard_name(s), alias courants}
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


class DCMetric(ABC):
    def __init__(self, **kwargs: Dict[str, Any]) -> None:
        """Init func.

        Args:
        """

        self.metric_name = None
        no_default_attrs = ['metric_name', 'var', 'depth']
        class_default_attrs = ['metric_name']
        default_attrs = dict(
            plot_result=False, minimum_latitude=None, maximum_latitude=None,
            minimum_longitude=None, maximum_longitude=None,
            spatial_resolution=None, small_scale_cutoff_km=100,
        )
        allowed_attrs = list(default_attrs.keys()) + no_default_attrs
        default_attrs.update(kwargs)
        self.__dict__.update((k,v) for k,v in default_attrs.items() if k in allowed_attrs)

        for attr in class_default_attrs:
            assert(hasattr(self, attr))

    def get_metric_name(self) -> None:
        return self.metric_name

    @abstractmethod
    def compute(self, pred_data: xr.Dataset, ref_data: xr.Dataset):
        pass

    @abstractmethod
    def compute_metric(self, pred_data: xr.Dataset, ref_data: xr.Dataset):
        pass



class OceanbenchMetrics(DCMetric):
    """Central class for calling Oceanbench functions."""

    def __init__(
        self,is_class4: Optional[bool] = None,
        class4_kwargs: Optional[dict] = None,
        **kwargs: Optional[Dict[str, Any]],
    ) -> None:
        """Init func.

        Args:
        """
        super().__init__(**kwargs)
        self.is_class4 = is_class4
        self.class4_kwargs = class4_kwargs or {}

        if is_class4:
            class4_args = dict(self.class4_kwargs)
            self.class4_evaluator = Class4Evaluator(
                metrics=class4_args["list_scores"],
                interpolation_method=class4_args["interpolation_method"],
                delta_t=class4_args["time_tolerance"],
                bin_specs=class4_args.get("binning", None),
                spatial_mask_fn=class4_args.get("spatial_mask_fn", None),
                cache_dir=class4_args.get("cache_dir", None),
                apply_qc=class4_args.get("apply_qc", False),
                # distributed=class4_args.get("distributed", False),
                qc_mapping=class4_args.get("qc_mapping", None),
            )

        self.metrics_set: dict[str, Callable] = {
            "rmsd": {
                "func_with_ref": rmsd,
                "kwargs_with_ref": ["vars"],
                "func_no_ref": oceanbench_metrics.rmsd_of_variables_compared_to_glorys,
            },

            "lagrangian": {
                "func_with_ref": deviation_of_lagrangian_trajectories,
                "kwargs_with_ref": ["zone"],
                "func_no_ref": oceanbench_metrics.deviation_of_lagrangian_trajectories_compared_to_glorys,
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

            # --- Ajout pour les métriques classe 4 ---
            "class4": None
        }
        logger.debug(f"self.metrics_set: {self.metrics_set}")

    #@profile
    def compute_metric(
        self, 
        eval_dataset: xr.Dataset,
        ref_dataset: Optional[xr.Dataset]=None,
        eval_variables: Optional[List[Variable]] = EVAL_VARIABLES_GLONET,
        zone: Optional[ZoneCoordinates] = GLOBAL_ZONE_COORDINATES,
        pred_coords: Optional[CoordinateSystem] = None,
        ref_coords: Optional[CoordinateSystem] = None,
        **extra_kwargs,
    ) -> Optional[Any]:
        """Compute a given metric.

        Args:
            eval_dataset (xr.Dataset): dataset to evaluate
            ref_dataset (xr.Dataset): reference dataset

        Returns:
            ndarray, optional: computed metric (if any)
        """
        if self.is_class4 is None:
            self.is_class4 = ref_coords.is_observation_dataset() if ref_coords else False

        if self.is_class4:
            try:
                # Préparer les arguments pour compute_class4_metrics
                # Fusionner class4_kwargs, extra_kwargs, et les arguments explicites
                #import pandas as pd
                # Définir la date de début souhaitée
                # start_date = pd.Timestamp("2016-02-12")  # TODO: Remove this

                # Générer la nouvelle série temporelle avec un pas d’un jour
                #new_time = pd.date_range(start=start_date, periods=eval_dataset.sizes["time"], freq="h")

                # emplacer la coordonnée "time"
                #eval_dataset = eval_dataset.assign_coords(time=new_time)

                eval_variables = [var for var in eval_variables if var in ref_dataset.data_vars]
                oceanbench_eval_variables =[
                    self.get_variable_alias(var) for var in eval_variables
                ] if eval_variables else None
                # Ajout des arguments obligatoires
                # Appel
                res = self.class4_evaluator.run(
                    model_ds=eval_dataset,
                    obs_ds=ref_dataset,
                    # pred_coords=pred_coords,
                    variables=eval_variables,
                    ref_coords=ref_coords,
                    # variables=oceanbench_eval_variables,
                )
                return res

            except Exception as exc:
                logger.error(f"Failed to compute metric {self.metric_name}: {traceback.format_exc()}")
                raise
        else:
            if eval_variables:
                has_depth = any(
                    depth_alias in list(eval_dataset.dims) for depth_alias in COORD_ALIASES["depth"])
            if eval_variables and not has_depth:
                if self.metric_name == "lagrangian":
                    logger.warning("Lagrangian metric requires 'depth' variable.")
                    return None
            if self.metric_name not in self.metrics_set:
                logger.warning(f"Metric {self.metric_name} is not defined in the metrics set.")
                return None
            try:
                metric_info = self.metrics_set[self.metric_name]
                if ref_dataset:
                    metric_func = metric_info["func_with_ref"]
                    add_kwargs_list = metric_info.get("kwargs_with_ref", [])
                    if "preprocess_ref" in metric_info:
                        ref_dataset = metric_info["preprocess_ref"]([ref_dataset])
                    kwargs = {
                        "challenger_datasets": [eval_dataset],
                        "reference_datasets": [ref_dataset],
                    }
                else:
                    metric_func = metric_info["func_no_ref"]
                    add_kwargs_list = None
                    kwargs = {
                        "challenger_datasets": [eval_dataset],
                    }
                oceanbench_eval_variables = [
                    self.get_variable_alias(var) for var in eval_variables
                ] if eval_variables else None

                if eval_variables and ref_dataset:
                    if self.metric_name != "lagrangian":
                        kwargs["variables"] = oceanbench_eval_variables

                # Vérifier la présence de depth comme dimension
                has_depth_dim = "depth" in eval_dataset.dims
                has_depth_coord = "depth" in eval_dataset.coords
                if not has_depth_dim and not has_depth_coord:
                    kwargs["depth_levels"] = None
                add_kwargs = {}
                if add_kwargs_list:
                    if "vars" in add_kwargs_list:
                        add_kwargs["variables"] = oceanbench_eval_variables
                    if "zone" in add_kwargs_list:
                        kwargs["zone"] = zone

                    kwargs.update(add_kwargs)
                return metric_func(**kwargs)
            except Exception as exc:
                logger.error(f"Failed to compute metric {self.metric_name}: {traceback.format_exc()}")
                raise

    def get_variable_alias(self, variable: str) -> Variable | None:
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