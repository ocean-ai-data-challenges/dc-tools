
from typing import Any, Callable

from memory_profiler import profile
import numpy as np
import xarray as xr

from loguru import logger

from dctools.data.coordinates import CoordinateSystem
from dctools.metrics.oceanbench_metrics import OceanbenchMetrics
from dctools.utilities.misc_utils import add_noise_with_snr  #, to_float32


class MetricComputer(OceanbenchMetrics):
    def __init__(
            self,
            is_class4: bool = False,
            class4_kwargs: dict = None,
            **kwargs,
        ):
        super().__init__(is_class4=is_class4, class4_kwargs=class4_kwargs, **kwargs)
        self.is_class4 = is_class4
        self.class4_kwargs = class4_kwargs or {}
        self.eval_variables = kwargs.get("eval_variables", None)
        self.add_noise = kwargs.get("add_noise", False)
        # self.observations = kwargs.get("observations", None)

    #@profile
    def compute(
        self, pred_data: xr.Dataset, ref_data: xr.Dataset,
        # ref_is_observation: bool,
        pred_coords: CoordinateSystem, ref_coords: CoordinateSystem,
        # list_metrics: list[MetricComputer] = None,
    ):
        try:
            ref_data = ref_data.rename(name_dict={'sla': 'ssh'})   # TODO : compute sla from ssh
        except ValueError as exc:
            logger.warning("Cannot rename SLA")
        #pred_data = to_float32(pred_data)
        #ref_data = to_float32(ref_data)
        try:
            # Restriction des variables à celles présentes dans les deux datasets
            #if self.eval_variables is not None:
            pred_vars = set(pred_data.data_vars)
            ref_vars = set(ref_data.data_vars)
            common_vars = [v for v in self.eval_variables if v in pred_vars and v in ref_vars]
            if not common_vars:
                logger.warning("No common variables found between pred_data and ref_data for evaluation.")
                return {}
            self.eval_variables = common_vars
            if self.is_class4:
                # Appel harmonisé pour la classe 4
                result = self.compute_metric(
                    pred_data,
                    ref_data,
                    self.eval_variables,
                    #list_metrics,
                    # observations=ref_data,
                    # ref_is_observation=ref_is_observation,
                    pred_coords=pred_coords,
                    ref_coords=ref_coords,
                    **self.class4_kwargs,
                )
            else:
                result = self.compute_metric(
                    pred_data,
                    ref_data,
                    self.eval_variables,
                    #list_metrics,
                    # ref_is_observation=ref_is_observation,
                    pred_coords=pred_coords,
                    ref_coords=ref_coords,
                )
            if self.add_noise and result is not None:
                result = add_noise_with_snr(result, snr_db=15)
            if isinstance(result, xr.Dataset) or isinstance(result, xr.DataArray):
                return "plot.jpeg"
            else:
                return result
        except Exception as exc:
            logger.error(
                f"Error while computing metrics: {repr(exc)}"
            )
            return None



'''    def compute(
        self, pred_data: xr.Dataset, ref_data: xr.Dataset = None
    ):
        try:
            # Détection automatique d'une métrique de classe 4 via l'attribut du dataset de référence
            is_class4 = (
                ref_data is not None
                and hasattr(ref_data, "attrs")
                and ref_data.attrs.get("is_observation", False)
            )
            # Pour la classe 4, il faut passer observations et kwargs spécifiques
            if is_class4:
                result = self.compute_metric(
                    pred_data,
                    ref_data,
                    self.eval_variables,
                    observations=ref_data,
                    **self.class4_kwargs
                )
            else:
                result = self.compute_metric(
                    pred_data,
                    ref_data,
                    self.eval_variables,
                )
            if self.add_noise and result is not None:
                result = add_noise_with_snr(result, snr_db=15)
            if isinstance(result, xr.Dataset) or isinstance(result, xr.DataArray):
                return "plot.jpeg"
            else:
                return result
            # formatted_res = self.post_process_result(result)

        except Exception as exc:
            logger.error(
                f"Error while computing metrics: {repr(exc)}"
            )
            return None
'''