
from typing import Any 


import numpy as np
import xarray as xr

from loguru import logger

from dctools.metrics.oceanbench_metrics import OceanbenchMetrics
from dctools.utilities.misc_utils import add_noise_with_snr


class MetricComputer(OceanbenchMetrics):
    def __init__(
            self,
            **kwargs,
        ):
        super().__init__(**kwargs)
        self.add_noise = kwargs.get("add_noise", False)

    def compute(
        self, pred_data: xr.Dataset, ref_data: xr.Dataset
    ):
        try:
            result = self.compute_metric(
                pred_data,
                ref_data,
            )
            if self.add_noise:
                result = add_noise_with_snr(result, snr_db=10)
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
        #return self.metrics_results
