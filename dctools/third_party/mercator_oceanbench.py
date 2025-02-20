#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Wrapper for functions implemented in Mercator's oceanbench library."""

from typing import List, Any, Optional

import oceanbench
from oceanbench.core import evaluate as oceanbench_eval
from oceanbench.core import plot as oceanbench_plot
from oceanbench.core import process as oceanbench_process
import numpy as np
import xarray


class oceanbench_core_evaluate(object):
    """Wrapper class.

    Wraps functions in Mercator's oceanbench lib.
    inside the oceanbench/core/evaluate folder.
    """

    '''def __init__(self):
        """Init func."""
        pass'''

    def get_rmse_glonet(self, forecast, ref, var, lead, level):
        """Mercator's plot for temporal RMSE for depth.

        Args:
            forecast:
            ref:
            var:
            lead:
            level:
        """
        rmse = oceanbench_eval.rmse_core.get_rmse_glonet(
            forecast, ref, var, lead, level
        )
        return rmse

    def get_glonet_rmse_for_given_days(
        self,
        depthg,
        var,
        glonet_datasets: List[xarray.Dataset],
        glorys_datasets: List[xarray.Dataset],
    ):
        """RMSE for given days.

        Args:
            depthg:
            var:
            glonet_datasets(List[xarray.Dataset]):
            glorys_datasets(List[xarray.Dataset]):
        """
        rmse = oceanbench_eval.rmse_core.get_glonet_rmse_for_given_days(
            depthg, var, glonet_datasets, glorys_datasets
        )
        return rmse

    def glonet_pointwise_evaluation_core(
        self,
        glonet_datasets: List[xarray.Dataset],
        glorys_datasets: List[xarray.Dataset],
    ):
        """Pointwise evaluation.

        Args:
            glonet_datasets(List[xarray.Dataset])
            glorys_datasets(List[xarray.Dataset])
        """
        gnet = oceanbench_eval.rmse_core.glonet_pointwise_evaluation_core(
            glonet_datasets, glorys_datasets
        )
        return gnet

    def get_euclidean_distance_core(
        self,
        first_dataset: xarray.Dataset,
        second_dataset: xarray.Dataset,
        minimum_latitude: float,
        maximum_latitude: float,
        minimum_longitude: float,
        maximum_longitude: float,
    ):
        """Get Euclidian distance.

        Args:
            first_dataset(xarray.Datase):
            second_dataset(xarray.Datase):
            minimum_latitude(float):
            maximum_latitude(float):
            minimum_longitude(float):
            maximum_longitude(float):
        """
        distance = oceanbench_eval.rmse_core.get_euclidean_distance_core(
            first_dataset,
            second_dataset,
            minimum_latitude,
            maximum_latitude,
            minimum_longitude,
            maximum_longitude,
        )
        return distance

    def analyze_energy_cascade_core(
        self,
        glonet: xarray.Dataset,
        var,
        depth,
        spatial_resolution=None,
        small_scale_cutoff_km: int = 100,
    ):
        """Get Analyze energy cascade.

        Args:
            glonet(xarray.Dataset):
            var:
            depth:
            spatial_resolution:,
            small_scale_cutoff_km(int):
        Return:
            time_spectra:
            small_scale_fraction:
        """
        time_spectra, small_scale_fraction = (
            oceanbench_eval.rmse_core.analyze_energy_cascade_core(
                glonet, var, depth, spatial_resolution, small_scale_cutoff_km
            )
        )
        return (time_spectra, small_scale_fraction)


class oceanbench_core_plot(object):
    """Wrapper class.

    Wraps functions in Mercator's oceanbench lib
    inside the oceanbench/core/plot folder.
    """

    def __init__(self):
        """Init func."""
        pass

    def plot_temporal_rmse_for_depth(rmse_dataarray: np.ndarray[Any], depth: int):
        """Plot temporal RMSE for depth GLONET.

        Args:
            rmse_dataarray(np.ndarray[Any]):
            depth(int):
        """
        rmse = oceanbench_plot.rmse_core.plot_temporal_rmse_for_depth(
            rmse_dataarray, depth
        )

    def plot_temporal_rmse_for_average_depth(rmse_dataarray: np.ndarray[Any]):
        """Plot temporal RMSE for average depth.

        Args:
            rmse_dataarray(np.ndarray[Any]):
        """
        oceanbench_plot.rmse_core.plot_temporal_rmse_for_average_depth(rmse_dataarray)

    def plot_depth_rmse_average_on_time(
        rmse_dataarray: np.ndarray[Any],
        dataset_depth_values: np.ndarray,
    ):
        """Plot depth RMSE average on time.

        Args:
            rmse_dataarray(np.ndarray[Any]):
            dataset_depth_values(np.ndarray):
        """
        oceanbench_plot.rmse_core.plot_depth_rmse_average_on_time(
            rmse_dataarray, dataset_depth_values
        )

    def plot_euclidean_distance_core(e_d):
        """Plot Euclidian distance.

        Args:
            e_d: euclidian distance array
        """
        oceanbench_plot.rmse_core.plot_euclidean_distance_core(e_d)

    def plot_energy_cascade_core(gglonet_sc):
        """Plot Energy cascade.

        Args:
            gglonet_sc:
        """
        oceanbench_plot.rmse_core.plot_energy_cascade_core(gglonet_sc)


class oceanbench_core_process(object):
    """Wrapper class.

    Wraps functions in Mercator's oceanbench lib
    inside the oceanbench/core/process folder.
    """

    def __init__(self):
        """Init func."""
        pass

    def calc_density_core(
        dataset: xarray.Dataset,
        lead: int,
        minimum_latitude: float,
        maximum_latitude: float,
        minimum_longitude: float,
        maximum_longitude: float,
    ) -> xarray.Dataset:
        """Compute density core.

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
        density = oceanbench_process.calc_density_core.calc_density_core(
            dataset,
            lead,
            minimum_latitude,
            maximum_latitude,
            minimum_longitude,
            maximum_longitude,
        )
        return density

    def calc_geo_core(dataset: xarray.Dataset, var: str, lead: int) -> xarray.Dataset:
        """Compute geo.

        Args:
            dataset(xarray.Dataset):
            var(str):
            lead(int):
        Returns:
            xarray.Dataset
        """
        dataset = oceanbench_process.calc_geo_core.calc_geo_core(dataset, var, lead)
        return dataset

    def calc_mld_core(dataset: xarray.Dataset, lead: int) -> xarray.Dataset:
        """Compute MLD.

        Args:
            dataset(xarray.Dataset):
            lead(int):
        Returns:
            xarray.Dataset
        """
        dataset = oceanbench_process.calc_mld_core.calc_mld_core(dataset, lead)
        return dataset

    def get_particle_file_core(
        dataset: xarray.Dataset, latzone, lonzone
    ) -> xarray.Dataset:
        """Get particle file.

        Args:
            dataset(xarray.Dataset):
            latzone:
            lonzone:
        Returns:
            xarray.Dataset
        """
        ds = oceanbench_process.lagrangian_analysis.get_particle_file_core(
            dataset, latzone, lonzone
        )
        return ds


class oceanbench_core_process_utils(object):
    """Wrapper class.

    Wraps functions in Mercator's oceanbench lib
    inside the oceanbench/core/process.utils.py file.
    """

    def __init__(self):
        """Init func."""
        pass

    def compute_kinetic_energy_core(dataset: xarray.Dataset) -> xarray.Dataset:
        """Compute kinetic energy.

        Args:
            dataset(xarray.Dataset):
        Returns:
            xarray.Dataset
        """
        KE = oceanbench_process.utils.compute_kinetic_energy_core(dataset)
        return KE

    def compute_vorticity_core(dataset: xarray.Dataset) -> xarray.Dataset:
        """Compute vorticity.

        Args:
            dataset(xarray.Dataset):
        Returns:
            xarray.Dataset
        """
        vorticity = oceanbench_process.utils.compute_vorticity_core(dataset)
        return vorticity

    def mass_conservation_core(
        dataset: xarray.Dataset, depth, deg_resolution
    ) -> xarray.Dataset:
        """Compute mass conservation.

        Args:
            dataset(xarray.Dataset):
            depth:
            deg_resolution:
        Returns:
            xarray.Dataset
        """
        mean_divergence_time_series = oceanbench_process.utils.mass_conservation_core(
            dataset, depth, deg_resolution
        )
        return mean_divergence_time_series


class oceanbench_bench_evaluate(object):
    """Wrapper class.

    Wraps functions in Mercator's oceanbench lib
    inside the oceanbench/evaluate.py file.
    """

    def __init__(self):
        """Init func."""
        pass

    def pointwise_evaluation(
        glonet_datasets: List[xarray.Dataset], glorys_datasets: List[xarray.Dataset]
    ) -> np.ndarray[Any]:
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


class oceanbench_plotting(object):
    """Wrapper class.

    Wraps functions in Mercator's oceanbench lib
    inside the oceanbench/plot.py file.
    """

    def __init__(self):
        """Init func."""
        pass

    def plot_density(dataset: xarray.Dataset):
        """Plot density.

        Args:
            dataset(xarray.Dataset):
        """
        oceanbench.plot.plot_density(dataset)

    def plot_geo(dataset: xarray.Dataset):
        """Plot geo.

        Args:
            dataset(xarray.Dataset):
        """
        oceanbench.plot.plot_geo(dataset)

    def plot_mld(dataset: xarray.Dataset):
        """Plot mld.

        Args:
            dataset(xarray.Dataset):
        """
        oceanbench.plot.plot_mld(dataset)

    def plot_pointwise_evaluation(rmse_dataarray: np.ndarray[Any], depth: int):
        """Plot pointwise evaluation.

        Args:
            rmse_dataarray(numpy.ndarray[Any]):
            depth(int):
        """
        oceanbench.plot.plot_pointwise_evaluation(rmse_dataarray, depth)

    def plot_pointwise_evaluation_for_average_depth(rmse_dataarray: np.ndarray[Any]):
        """Plot pointwise evaluation for average depth.

        Args:
            rmse_dataarray(np.ndarray[Any]):
        """
        oceanbench.plot.plot_pointwise_evaluation_for_average_depth(rmse_dataarray)

    def plot_pointwise_evaluation_depth_for_average_time(
        rmse_dataarray: np.ndarray[Any],
        dataset_depth_values: np.ndarray,
    ):
        """Plot pointwise evaluation  depth for average time.

        Args:
            rmse_dataarray(np.ndarray[Any]):
            dataset_depth_values(np.ndarray):
        """
        oceanbench.plot.plot_pointwise_evaluation_depth_for_average_time(
            rmse_dataarray, dataset_depth_values
        )

    def plot_euclidean_distance(euclidean_distance):
        """Plot euclidean_distance.

        Args:
            euclidean_distance:
        """
        oceanbench.plot.plot_euclidean_distance(euclidean_distance)

    def plot_energy_cascade(euclidean_distance):
        """Plot energy cascade.

        Args:
            gglonet_sc:
        """
        oceanbench.plot.plot_energy_cascade(gglonet_sc)

    def plot_kinetic_energy(dataset: xarray.Dataset):
        """Plot euclidean_distance.

        Args:
            dataset(xarray.Dataset):
        """
        oceanbench.plot.plot_kinetic_energy(dataset)

    def plot_vorticity(dataset: xarray.Dataset):
        """Plot euclidean_distance.

        Args:
            dataset(xarray.Dataset):
        """
        oceanbench.plot.plot_vortocity(dataset)


class oceanbench_process(object):
    """Wrapper class.

    Wraps functions in Mercator's oceanbench lib
    inside the oceanbench/process.py file.
    """

    def __init__(self):
        """Init func."""
        pass

    def calc_density(
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

    def calc_geo(dataset: xarray.Dataset, lead: int, variable: str) -> xarray.Dataset:
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

    def calc_mld(dataset: xarray.Dataset, lead: int, variable: str) -> xarray.Dataset:
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
