#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Wrapper for functions implemented in Mercator's oceanbench library."""

from abc import ABC, abstractmethod
import traceback
from typing import Any, Callable, Dict, List, Optional
from typing_extensions import Unpack

from loguru import logger
#import numpy.typing as npt
from numpy import ndarray
import oceanbench.metrics as oceanbench_metrics
from oceanbench.core.rmsd import rmsd, Variable
from oceanbench.core.lagrangian_trajectory import ZoneCoordinates, deviation_of_lagrangian_trajectories
from oceanbench.core.derived_quantities import add_mixed_layer_depth
from oceanbench.core.derived_quantities import add_geostrophic_currents
import xarray as xr

from dctools.data.coordinates import (
    EVAL_VARIABLES_GLONET,
    GLOBAL_ZONE_COORDINATES,
    COORD_ALIASES,
)

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

    #def __init__(self,
    #             metric_name: str, plot_result: bool=False, **kwargs: Dict[str, Any]) -> None:
    def __init__(self, **kwargs: Dict[str, Any]) -> None:
        """Init func.

        Args:
        """
        super().__init__(**kwargs)
        self.metrics_set: dict[str, Callable] = {
            "rmsd": {
                "func_with_ref": rmsd,
                "kwargs_with_ref": ["vars"],
                "func_no_ref": oceanbench_metrics.rmsd_of_variables_compared_to_glorys,
                #"kwargs_no_ref": None,
            },
            "lagrangian": {
                "func_with_ref": deviation_of_lagrangian_trajectories,
                "kwargs_with_ref": ["zone"],
                "func_no_ref": oceanbench_metrics.deviation_of_lagrangian_trajectories_compared_to_glorys,
                #"kwargs_no_ref": None,
            },
            "rmsd_geostrophic_currents": {
                "func_with_ref": rmsd,
                "kwargs_with_ref": ["vars"],
                "func_no_ref": oceanbench_metrics.rmsd_of_geostrophic_currents_compared_to_glorys,
                #"kwargs_no_ref": None,
                "preprocess_ref": add_geostrophic_currents,
            },

            "rmsd_mld": {
                "func_with_ref": rmsd,
                "kwargs_with_ref": ["vars"],
                "func_no_ref": oceanbench_metrics.rmsd_of_mixed_layer_depth_compared_to_glorys,
                #"kwargs_no_ref": None,
                "preprocess_ref": add_mixed_layer_depth,
            }
        }

    def compute_metric(
        self, 
        eval_dataset: xr.Dataset,
        ref_dataset: Optional[xr.Dataset]=None,
        eval_variables: Optional[List[Variable]] = EVAL_VARIABLES_GLONET,
        zone: Optional[ZoneCoordinates] = GLOBAL_ZONE_COORDINATES,
    ) -> Optional[ndarray]:
        """Compute a given metric.

        Args:
            eval_dataset (xr.Dataset): dataset to evaluate
            ref_dataset (xr.Dataset): reference dataset
                Defaults to False.

        Returns:
            ndarray, optional: computed metric (if any)
        """
        logger.debug(f"Computing metric {self.metric_name} with variables: {eval_variables}")
        if eval_variables:
            has_depth = any(depth_alias in eval_variables for depth_alias in COORD_ALIASES)
        if eval_variables and not has_depth:
            if self.metric_name == "lagrangian":
                logger.warning("Lagrangian metric requires 'depth' variable.")
                return None
        if self.metric_name not in self.metrics_set:
            logger.warning(f"Metric {self.metric_name} is not defined in the metrics set.")
            return None
        try:
            if ref_dataset:
                metric_func = self.metrics_set[self.metric_name]["func_with_ref"]
                add_kwargs_list = self.metrics_set[self.metric_name]["kwargs_with_ref"]
                if "preprocess_ref" in self.metrics_set[self.metric_name]:
                    ref_dataset = self.metrics_set[self.metric_name]["preprocess_ref"]([ref_dataset])
                kwargs = {
                    "challenger_datasets": [eval_dataset],
                    "reference_datasets": [ref_dataset],
                    # "variables": eval_variables,
                }
            else:
                logger.debug(f"Computing metric {self.metric_name} without reference dataset.")
                metric_func = self.metrics_set[self.metric_name]["func_no_ref"]
                add_kwargs_list = None
                kwargs = {
                    "challenger_datasets": [eval_dataset],
                    # "variables": eval_variables,
                }
            if eval_variables and ref_dataset:
                kwargs["variables"] = eval_variables
            add_kwargs = {}
            if add_kwargs_list:
                if "vars" in add_kwargs_list:
                    add_kwargs["variables"] = eval_variables
                if "zone" in add_kwargs_list:
                    kwargs["zone"] = zone

                kwargs.update(add_kwargs)
            logger.debug(f"Calling metric function {metric_func.__name__} with kwargs: {kwargs}")
            return metric_func(**kwargs)
        
        except Exception as exc:
            logger.error(f"Failed to compute metric {self.metric_name}: {traceback.format_exc()}")
            raise
