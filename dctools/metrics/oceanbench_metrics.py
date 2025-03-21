#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Wrapper for functions implemented in Mercator's oceanbench library."""

import logging
from typing import List, Any, Optional

import numpy.typing as npt
from numpy import ndarray
import oceanbench
import xarray

from dctools.utilities.errors import DCExceptionHandler


class oceanbench_evaluate_funcs(object):
    """Wrapper class.

    Wraps functions in Mercator's oceanbench lib
    inside the oceanbench/evaluate.py file.
    """

    def __init__(self):
        """Init func."""
        pass

    def pointwise_evaluation(
        self,
        glonet_datasets: List[xarray.Dataset], glorys_datasets: List[xarray.Dataset]
    ) -> npt.NDArray[Any]:
        """Compute pointwise evaluation.

        Args:
            glonet_datasets(List[xarray.Dataset]):
            glorys_datasets(List[xarray.Dataset]):
        Returns:
            np.ndarray[Any]
        """
        gnet = oceanbench.evaluate.pointwise_evaluation(
            glonet_datasets, glorys_datasets
        )
        return gnet

    def get_euclidean_distance(
        self,
        first_dataset: xarray.Dataset,
        second_dataset: xarray.Dataset,
        minimum_latitude: float,
        maximum_latitude: float,
        minimum_longitude: float,
        maximum_longitude: float,
    ):
        """Compute Euclidian distance .

        Args:
            first_dataset(xarray.Dataset):
            second_dataset(xarray.Dataset):
            minimum_latitude(float):
            maximum_latitude(float):
            minimum_longitude(float):
            maximum_longitude(float):
        """
        distance = oceanbench.evaluate.get_euclidean_distance(
            first_dataset,
            second_dataset,
            minimum_latitude,
            maximum_latitude,
            minimum_longitude,
            maximum_longitude,
        )
        return distance

    def analyze_energy_cascade(
        self,
        dataset: xarray.Dataset,
        var: str,
        depth: float,
        spatial_resolution: Optional[float] = None,
        small_scale_cutoff_km: Optional[float] = 100,
    ):
        """Compute pointwise evaluation.

        Args:
            dataset(xarray.Dataset):
            var(str):
            depth(float):
            spatial_resolution(Optional[float]):
            small_scale_cutoff_km(Optional[float]):
        Returns:
            time_spectra:
            small_scale_fraction:
        """
        time_spectra, small_scale_fraction = oceanbench.evaluate.analyze_energy_cascade(
            dataset, var, depth, spatial_resolution, small_scale_cutoff_km
        )
        return (time_spectra, small_scale_fraction)


class oceanbench_plotting_funcs(object):
    """Wrapper class.

    Wraps functions in Mercator's oceanbench lib
    inside the oceanbench/plot.py file.
    """

    def __init__(self):
        """Init func."""
        pass

    def plot_density(self, dataset: xarray.Dataset):
        """Plot density.

        Args:
            dataset(xarray.Dataset):
        """
        oceanbench.plot.plot_density(dataset)

    def plot_geo(self, dataset: xarray.Dataset):
        """Plot geo.

        Args:
            dataset(xarray.Dataset):
        """
        oceanbench.plot.plot_geo(dataset)

    def plot_mld(self, dataset: xarray.Dataset):
        """Plot mld.

        Args:
            dataset(xarray.Dataset):
        """
        oceanbench.plot.plot_mld(dataset)

    def plot_pointwise_evaluation(
        self, rmse_dataarray: npt.NDArray[Any], depth: int
    ):
        """Plot pointwise evaluation.

        Args:
            rmse_dataarray(np.ndarray[Any]):
            depth(int):
        """
        oceanbench.plot.plot_pointwise_evaluation(rmse_dataarray, depth)

    def plot_pointwise_evaluation_for_average_depth(
        self, rmse_dataarray: npt.NDArray[Any]
    ):
        """Plot pointwise evaluation for average depth.

        Args:
            rmse_dataarray(np.ndarray[Any]):
        """
        oceanbench.plot.plot_pointwise_evaluation_for_average_depth(rmse_dataarray)

    def plot_pointwise_evaluation_depth_for_average_time(
        self,
        rmse_dataarray: npt.NDArray[Any],
        dataset_depth_values: npt.NDArray,
    ):
        """Plot pointwise evaluation  depth for average time.

        Args:
            rmse_dataarray(np.ndarray[Any]):
            dataset_depth_values(np.ndarray):
        """
        oceanbench.plot.plot_pointwise_evaluation_depth_for_average_time(
            rmse_dataarray, dataset_depth_values
        )

    def plot_euclidean_distance(self, euclidean_distance):
        """Plot euclidean_distance.

        Args:
            euclidean_distance:
        """
        oceanbench.plot.plot_euclidean_distance(euclidean_distance)

    def plot_energy_cascade(self, gglonet_sc):
        """Plot energy cascade.

        Args:
            gglonet_sc:
        """
        oceanbench.plot.plot_energy_cascade(gglonet_sc)

    def plot_kinetic_energy(self, dataset: xarray.Dataset):
        """Plot euclidean_distance.

        Args:
            dataset(xarray.Dataset):
        """
        oceanbench.plot.plot_kinetic_energy(dataset)

    def plot_vorticity(self, dataset: xarray.Dataset):
        """Plot euclidean_distance.

        Args:
            dataset(xarray.Dataset):
        """
        oceanbench.plot.plot_vortocity(dataset)


class oceanbench_processing_funcs(object):
    """Wrapper class.

    Wraps functions in Mercator's oceanbench lib
    inside the oceanbench/process.py file.
    """

    def __init__(self):
        """Init func."""
        pass

    def calc_density(
        self,
        dataset: xarray.Dataset,
        lead: int,
        minimum_latitude: float,
        maximum_latitude: float,
        minimum_longitude: float,
        maximum_longitude: float,
    ) -> xarray.Dataset:
        """Compute density.

        Args:
            dataset(xarray.Dataset):
            lead(int):
            minimum_latitude(float):
            maximum_latitude(float):
            minimum_longitude(float):
            maximum_longitude(float):
        Returns:
            xarray.Dataset
        """
        density = oceanbench.process.calc_density(
            dataset,
            lead,
            minimum_latitude,
            maximum_latitude,
            minimum_longitude,
            maximum_longitude,
        )
        return density

    def calc_geo(self, dataset: xarray.Dataset, lead: int, variable: str) -> xarray.Dataset:
        """Compute geo.

        Args:
            dataset(xarray.Dataset):
            lead(int):
            variable(str):
        Returns:
            xarray.Dataset
        """
        dataset = oceanbench.process.calc_geo(
            dataset,
            lead,
            variable,
        )
        return dataset

    def calc_mld(self, dataset: xarray.Dataset, lead: int) -> xarray.Dataset:
        """Compute geo.

        Args:
            dataset(xarray.Dataset):
            lead(int):
        Returns:
            xarray.Dataset
        """
        dataset = oceanbench.process.calc_mld(
            dataset,
            lead,
        )
        return dataset

    def get_particle_file(
        self,
        dataset: xarray.Dataset,
        minimum_latitude: float,
        maximum_latitude: float,
        minimum_longitude: float,
        maximum_longitude: float,
    ) -> xarray.Dataset:
        """Get particle file.

        Args:
            dataset(xarray.Dataset):
            minimum_latitude(float):
            maximum_latitude(float):
            minimum_longitude(float):
            maximum_longitude(float):
        Returns:
            xarray.Dataset
        """
        dataset = oceanbench.process.get_particle_file(
            dataset,
            minimum_latitude,
            maximum_latitude,
            minimum_longitude,
            maximum_longitude,
        )
        return dataset

    def mass_conservation(
        self,
        dataset: xarray.Dataset, depth: float, deg_resolution: float = 0.25
    ) -> xarray.DataArray:
        """Compute geo.

        Args:
            dataset(xarray.Dataset):
            depth(float):
            deg_resolution(float):
        Returns:
            xarray.DataArray
        """
        dataarray = oceanbench.process.mass_conservation(dataset, depth, deg_resolution)
        return dataarray

class OceanbenchMetrics:
    """Central class for calling Oceanbench functions."""

    def __init__(self, dc_logger: logging.Logger, exc_handler: DCExceptionHandler) -> None:
        """Init func.

        Args:
            dc_logger (logging.logger): _description_
            exc_handler (DCExceptionHandler): _description_
        """
        self.dc_logger = dc_logger
        self.exc_handler = exc_handler
        self.oceanbench_eval = oceanbench_evaluate_funcs()
        self.oceanbench_plot = oceanbench_plotting_funcs()
        self.oceanbench_process = oceanbench_processing_funcs()

    def compute_metric(
        self, metric_name: str,
        eval_dataset: xarray.Dataset,
        ref_dataset: Optional[xarray.Dataset]=None,
        plot_result: bool=False
    ) -> Optional[ndarray]:
        """Compute a given metric.

        Args:
            metric_name (str): name of metric : ['rmse', 'mld', geo', density',
                'euclid_dist', 'energy_cascad', 'kinetic_energy', 'vorticity'
                'mass_conservation']
            eval_dataset (xarray.Dataset): dataset to evaluate
            ref_dataset (xarray.Dataset): reference dataset
            plot_result (bool, optional): whether to display figures or not.
                Defaults to False.

        Returns:
            ndarray, optional: computed metric (if any)
        """
        self.dc_logger.info(f"Run {metric_name} Evaluation.")
        result = None
        match metric_name:
            case 'rmse':
                result = self.rmse_evaluation(eval_dataset, ref_dataset, plot_result)
            case 'mld':
                result = self.mld_analysis(eval_dataset, plot_result)
            case 'geo':
                result = self.geo_analysis(eval_dataset, plot_result)
            case 'density':
                result = self.density_analysis(eval_dataset, plot_result)
            case 'euclid_dist':
                result = self.euclid_dist_analysis(eval_dataset, ref_dataset, plot_result)
            case 'energy_cascad':
                result = self.energy_cascad_analysis(eval_dataset, plot_result)
            case 'kinetic_energy':
                if plot_result:
                    result = self.kinetic_energy_analysis(eval_dataset)
                else:
                    self.dc_logger.info("Kinetic energy: Nothing to do (plotting disabled).")
            case 'vorticity':
                if plot_result:
                    self.vorticity_analysis(eval_dataset)
                else:
                    self.dc_logger.info("Vorticity: Nothing to do (plotting disabled).")
            case 'mass_conservation':
                if plot_result:
                    self.mass_conservation_analysis(eval_dataset)
                else:
                    self.dc_logger.info("Mass_conservation: Nothing to do (plotting disabled).")
            case _:
                self.dc_logger.warning("Unknown metric_name.")
        return result


    def rmse_evaluation(
        self,
        eval_dataset: xarray.Dataset,
        ref_dataset: Optional[xarray.Dataset],
        plot_result: bool
    ):
        """Compute RMSE metric.

        Args:
            dataset (xarray.Dataset): dataset to evaluate
            ref_dataset (xarray.Dataset): reference dataset
            plot_result (bool, optional): display figures ?
                Defaults to False.

        Returns:
            Optional[ndarray]: _description_
        """
        self.dc_logger.info("Compute RMSE metric.")
        try:
            if ref_dataset:
                nparray = self.oceanbench_eval.pointwise_evaluation(
                    glonet_datasets=[eval_dataset],
                    glorys_datasets=[ref_dataset],
                )
                if plot_result:
                    self.oceanbench_plot.plot_pointwise_evaluation(
                        rmse_dataarray=nparray, depth=2
                    )
                    self.oceanbench_plot.plot_pointwise_evaluation_for_average_depth(
                        rmse_dataarray=nparray
                    )
                    self.oceanbench_plot.plot_pointwise_evaluation_depth_for_average_time(
                        rmse_dataarray=nparray, dataset_depth_values=eval_dataset.depth.values
                    )
                return nparray
            else:
                self.dc_logger.warning("Empty reference dataset.")

        except Exception as exc:
            self.exc_handler.handle_exception(exc, "Compute RMSE metric error.")

    def mld_analysis(self,
        eval_dataset: xarray.Dataset,
        plot_result: bool
    ):
        """MLD analysis.

        Args:
            eval_dataset (xarray.Dataset): dataset to evaluate
            plot_result (bool, optional): display figures ?
                Defaults to False.

        Returns:
            Optional[ndarray]: _description_
        """
        self.dc_logger.info("Run MLD Analysis.")
        try:
            result = self.oceanbench_process.calc_mld(
                dataset=eval_dataset.compute(),
                lead=1,
            )
            if plot_result:
                self.oceanbench_plot.plot_mld(dataset=result)
            return result
        except Exception as exc:
            self.exc_handler.handle_exception(exc, "MLD analysis error.")

    def geo_analysis(
        self,
        eval_dataset: xarray.Dataset,
        plot_result: bool
    ):
        """Geo analysis.

        Args:
            dataset (xarray.Dataset): dataset to evaluate
            plot_result (bool, optional): display figures ?
                Defaults to False.

        Returns:
            Optional[ndarray]: _description_
        """
        self.dc_logger.info("Run Geo analysis.")
        try:
            result = self.oceanbench_process.calc_geo(
                dataset=eval_dataset,
                lead=1,
                variable="zos",
            )
            if plot_result:
                self.oceanbench_plot.plot_geo(dataset=result)
            return result
        except Exception as exc:
            self.exc_handler.handle_exception(exc, "Geo analysis error.")

    def density_analysis(
        self,
        eval_dataset: xarray.Dataset,
        plot_result: bool
    ):
        """Density analysis.

        Args:
            eval_dataset (xarray.Dataset): dataset to evaluate
            plot_result (bool, optional): display figures ?
                Defaults to False.

        Returns:
            Optional[ndarray]: _description_
        """
        self.dc_logger.info("Run density Analysis.")
        try:
            dataarray = self.oceanbench_process.calc_density(
                dataset=eval_dataset,
                lead=1,
                minimum_longitude=-100,
                maximum_longitude=-40,
                minimum_latitude=-15,
                maximum_latitude=50,
            )
            if plot_result:
                self.oceanbench_plot.plot_density(dataset=dataarray)
            return dataarray
        except Exception as exc:
            self.exc_handler.handle_exception(exc, "Density analysis error.")

    def euclid_dist_analysis(
        self,
        eval_dataset: xarray.Dataset,
        ref_dataset: Optional[xarray.Dataset],
        plot_result: bool
    ):
        """Euclidian distance analysis.

        Args:
            eval_dataset (xarray.Dataset): dataset to evaluate
            ref_dataset (xarray.Dataset): reference dataset
            plot_result (bool, optional): display figures ?
                Defaults to False.

        Returns:
            Optional[ndarray]: _description_
        """
        self.dc_logger.info("Run Euclidian distance analysis.")
        try:
            if ref_dataset:
                euclidean_distance = self.oceanbench_eval.get_euclidean_distance(
                    first_dataset=eval_dataset,
                    second_dataset=ref_dataset,
                    minimum_latitude=466,
                    maximum_latitude=633,
                    minimum_longitude=400,
                    maximum_longitude=466,
                )
                if plot_result:
                    self.oceanbench_plot.plot_euclidean_distance(euclidean_distance)
                return euclidean_distance
            else:
                self.dc_logger.warning("Empty reference dataset.")
        except Exception as exc:
            self.exc_handler.handle_exception(exc, "Euclidian distance analysis error.")

    def energy_cascad_analysis(
        self,
        eval_dataset: xarray.Dataset,
        plot_result: bool
    ):
        """Energy cascad analysis.

        Args:
            eval_dataset (xarray.Dataset): dataset to evaluate
            plot_result (bool, optional): display figures ?
                Defaults to False.

        Returns:
            Optional[ndarray]: _description_
        """
        self.dc_logger.info("Run energy cascad Analysis.")
        try:
            _, gglonet_sc = self.oceanbench_eval.analyze_energy_cascade(
                eval_dataset, "uo", 0, 1 / 4
            )
            if plot_result:
                self.oceanbench_plot.plot_energy_cascade(gglonet_sc)
            return gglonet_sc
        except Exception as exc:
            self.exc_handler.handle_exception(exc, "Energy cascad analysis error.")

    def kinetic_energy_analysis(
        self,
        eval_dataset: xarray.Dataset,
    ):
        """Kinetic energy analysis.

        Args:
            eval_dataset (xarray.Dataset): dataset to evaluate
            plot_result (bool, optional): display figures ?
                Defaults to False.

        Returns:
            Optional[ndarray]: _description_
        """
        self.dc_logger.info("Plot kinetic energy.")
        try:
            self.oceanbench_plot.plot_kinetic_energy(eval_dataset)
        except Exception as exc:
            self.exc_handler.handle_exception(exc, "Kinetic energy analysis error.")

    def vorticity_analysis(
        self,
        eval_dataset: xarray.Dataset,
    ):
        """Vorticity analysis.

        Args:
            eval_dataset (xarray.Dataset): dataset to evaluate
            plot_result (bool, optional): display figures ?
                Defaults to False.

        Returns:
            Optional[ndarray]: _description_
        """
        self.dc_logger.info("Plot vorticity.")
        try:
            self.oceanbench_plot.plot_vorticity(eval_dataset)
        except Exception as exc:
            self.exc_handler.handle_exception(exc, "Vorticity analysis error.")

    def mass_conservation_analysis(
        self,
        eval_dataset: xarray.Dataset
    ):
        """Mass conservation.

        Args:
            eval_dataset (xarray.Dataset): dataset to evaluate
            plot_result (bool, optional): display figures ?
                Defaults to False.

        Returns:
            Optional[ndarray]: _description_
        """
        self.dc_logger.info("Compute mass conservation.")
        try:
            mean_div_time_series = self.oceanbench_process.mass_conservation(
                eval_dataset, 0, deg_resolution=0.25
            )  # should be close to zero
            return(mean_div_time_series.data)  # time-dependent scores
        except Exception as exc:
            self.exc_handler.handle_exception(exc, "Mass conservation analysis error.")
