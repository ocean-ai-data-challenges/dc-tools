#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Wrapper for functions implemented in Mercator's oceanbench library."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from typing_extensions import Unpack

from loguru import logger
#import numpy.typing as npt
from numpy import ndarray
import oceanbench.metrics as oceanbench_metrics
#import oceanbench.plot as oceanbench_plot
#import oceanbench.core.evaluate.rmse_core as rmse_metrics
from oceanbench.core.rmsd import rmsd, Variable
import xarray as xr

from dctools.utilities.xarray_utils import (
    get_vars_dims,
    LIST_VARS_GLONET,
    LIST_VARS_GLONET_NO_DIMS,
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

    def compute_metric(
        self, 
        eval_dataset: xr.Dataset,
        ref_dataset: Optional[xr.Dataset]=None,
    ) -> Optional[ndarray]:
        """Compute a given metric.

        Args:
            metric_name (str): name of metric : ['rmse', 'mld', geo', density',
                'euclid_dist', 'energy_cascad', 'kinetic_energy', 'vorticity'
                'mass_conservation']
            eval_dataset (xr.Dataset): dataset to evaluate
            ref_dataset (xr.Dataset): reference dataset
                Defaults to False.

        Returns:
            ndarray, optional: computed metric (if any)
        """
        #vars_2d, vars_3d = get_vars_dims(eval_dataset)
        logger.info(f"Run {self.metric_name} Evaluation.")
        result = None
        match self.metric_name:
            case 'rmse':
                result = self.rmse_evaluation(eval_dataset, ref_dataset) #, vars_2d, vars_3d)
            case _:
                logger.warning("Unknown metric_name.")
        
        '''case 'euclid_dist':
            result = self.euclid_dist_analysis(eval_dataset, ref_dataset)
        case 'energy_cascad':
            result = self.energy_cascad_analysis(eval_dataset)'''
        return result

    def rmse_evaluation(
        self,
        eval_dataset: xr.Dataset,
        ref_dataset: Optional[xr.Dataset],
        #vars_2d: List[str],
        #vars_3d: List[str],
    ) -> ndarray:
        """Compute RMSE metric.

        Args:
            dataset (xr.Dataset): dataset to evaluate
            ref_dataset (xr.Dataset): reference dataset
                Defaults to False.

        Returns:
            Optional[ndarray]: _description_
        """
        logger.info("Compute RMSE metric.")
        try:
            #logger.info(f"Eval dataset: {eval_dataset}")
            #logger.info(f"Ref dataset: {ref_dataset}")
            eval_variables = [
                Variable.HEIGHT,
                Variable.TEMPERATURE,
                Variable.SALINITY,
                Variable.NORTHWARD_VELOCITY,
                Variable.EASTWARD_VELOCITY,
            ]
            if ref_dataset:
                #logger.info("Compute RMSE metric with reference dataset.")
                #logger.info(f"Eval dataset: {eval_dataset}")
                #logger.info(f"Ref dataset: {ref_dataset}")
                nparray = rmsd(
                    challenger_datasets=[eval_dataset],
                    reference_datasets=[ref_dataset],
                    variables=eval_variables,
                    #vars_2d=vars_2d, vars_3d=vars_3d,
                )
                # logger.info(f"RMSE: {nparray}")
            else:
                nparray = oceanbench_metrics.rmsd_of_variables_compared_to_glorys(
                    challenger_datasets=[eval_dataset],
                    #vars_2d=vars_2d, vars_3d=vars_3d,
                )
            """if self.plot_result:
                logger.info("Plot RMSE metric.")
                oceanbench_plot.plot_rmse(
                    rmse_dataarray=nparray, depth=2
                )
                oceanbench_plot.plot_rmse_for_average_depth(
                    rmse_dataarray=nparray
                )
                oceanbench_plot.plot_rmse_depth_for_average_time(
                    rmse_dataarray=nparray, dataset_depth_values=eval_dataset.depth.values
                )"""
        except Exception as exc:
            logger.error(f"Compute RMSE metric error: {repr(exc)}")

        return nparray

    '''def euclid_dist_analysis(
        self,
        eval_dataset: xr.Dataset,
        ref_dataset: Optional[xr.Dataset],
    ):
        """Euclidian distance analysis.

        Args:
            eval_dataset (xr.Dataset): dataset to evaluate
            ref_dataset (xr.Dataset): reference dataset

        Returns:
            Optional[ndarray]: _description_
        """
        logger.info("Run Euclidian distance analysis.")
        """assert(hasattr(self, 'minimum_latitude'))
        assert(hasattr(self, 'maximum_latitude'))
        assert(hasattr(self, 'minimum_longitude'))
        assert(hasattr(self, 'maximum_longitude'))"""

        try:
            if not hasattr(self, 'minimum_latitude') or not hasattr(self, 'maximum_latitude'):
                self.minimum_latitude = min(eval_dataset.lat.values)
                self.maximum_latitude = max(eval_dataset.lat.values)
            if not hasattr(self, 'minimum_longitude') or not hasattr(self, 'maximum_longitude'):
                self.minimum_longitude = min(eval_dataset.lon.values)
                self.maximum_longitude = max(eval_dataset.lon.values)

            if ref_dataset:
                euclid_dist = rmse_metrics._get_euclidean_distance_core(
                    candidate_dataset=eval_dataset,
                    reference_dataset=ref_dataset,
                    minimum_latitude=self.minimum_latitude,
                    maximum_latitude=self.maximum_latitude,
                    minimum_longitude=self.minimum_longitude,
                    maximum_longitude=self.maximum_longitude,
                )
            else:
                euclid_dist = oceanbench_metrics.euclidean_distance_to_glorys(
                    candidate_dataset=eval_dataset,
                    minimum_latitude=self.minimum_latitude,
                    maximum_latitude=self.maximum_latitude,
                    minimum_longitude=self.minimum_longitude,
                    maximum_longitude=self.maximum_longitude,
                )

            """if self.plot_result:
                oceanbench_plot.plot_euclidean_distance(euclid_dist)"""
            return euclid_dist
        except Exception as exc:
            logger.error(f"Euclidian distance analysis error: {repr(exc)}")'''

    '''def energy_cascad_analysis(
        self,
        eval_dataset: xr.Dataset,
    ):
        """Energy cascad analysis.

        Args:
            eval_dataset (xr.Dataset): dataset to evaluate

        Returns:
            Optional[ndarray]: _description_
        """
        logger.info("Run energy cascad Analysis.")
        try:
            assert(hasattr(self, 'var'))
            assert(hasattr(self, 'depth'))
            _, gglonet_sc = oceanbench_metrics.energy_cascade(
                candidate_dataset=eval_dataset,
                var=self.var,
                depth=self.depth,
                spatial_resolution=self.spatial_resolution,
                small_scale_cutoff_km=self.small_scale_cutoff_km,
            )
            """if self.plot_result:
                oceanbench_plot.plot_energy_cascade(gglonet_sc)"""
            return gglonet_sc
        except Exception as exc:
            logger.error(f"Energy cascad analysis error: {repr(exc)}")'''
