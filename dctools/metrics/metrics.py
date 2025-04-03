
from typing import Any 


import numpy as np
import xarray as xr

from dctools.metrics.oceanbench_metrics import OceanbenchMetrics



class MetricComputer(OceanbenchMetrics):
    def __init__(
            self,
            **kwargs,
        ):
        super().__init__(**kwargs)

    def compute(
        self, ref_data: xr.Dataset, pred_data: xr.Dataset
    ):
        try:
            #metric_name = self.get_metric_name()

            result = self.compute_metric(
                ref_data,
                pred_data,
            )
            if isinstance(result, xr.Dataset) or isinstance(result, xr.DataArray):
                return "plot.jpeg"
            else:
                return result
            # formatted_res = self.post_process_result(result)

        except Exception as e:
            self.exc_handler.handle_exception(
                e, "Error while computing metrics."
            )
            self.logger.info(f"Metric error: {e}")
            return None
        #return self.metrics_results

    def post_process_result(self, result: Any):
        if isinstance(result, xr.Dataset) or isinstance(result, xr.DataArray):
            formatted_res = result.to_dict()
        elif isinstance(result, np.ndarray):
            formatted_res = result.tolist()
        else:
            formatted_res = result
        return formatted_res


