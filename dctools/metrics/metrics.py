
import traceback
from typing import List

import xarray as xr

from loguru import logger

from dctools.data.coordinates import CoordinateSystem
from dctools.metrics.oceanbench_metrics import OceanbenchMetrics
from dctools.utilities.misc_utils import add_noise_with_snr  #, to_float32


class MetricComputer(OceanbenchMetrics):
    def __init__(
            self,
            eval_variables: List[str] = None,
            oceanbench_eval_variables: List[str] = None,
            is_class4: bool = False,
            class4_kwargs: dict = None,
            **kwargs,
        ):
        super().__init__(
            is_class4=is_class4, class4_kwargs=class4_kwargs, **kwargs
        )
        self.is_class4 = is_class4
        self.class4_kwargs = class4_kwargs or {}
        self.eval_variables = eval_variables
        self.oceanbench_eval_variables = oceanbench_eval_variables
        self.add_noise = kwargs.get("add_noise", False)

    #@profile
    def compute(
        self, pred_data: xr.Dataset, ref_data: xr.Dataset,
        pred_coords: CoordinateSystem, ref_coords: CoordinateSystem,
    ):

        try:
            if self.is_class4:
                # Appel harmonisé pour la classe 4
                result = self.compute_metric(
                    pred_data,
                    ref_data,
                    self.eval_variables,
                    pred_coords=pred_coords,
                    ref_coords=ref_coords,
                    **self.class4_kwargs,
                )
            else:
                result = self.compute_metric(
                    pred_data,
                    ref_data,
                    self.eval_variables,
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
            traceback.print_exc()
            return None
