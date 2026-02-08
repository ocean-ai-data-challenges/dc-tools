"""Metrics computation functions for data challenge evaluation."""

import traceback
from typing import Any, Dict, List, Optional

import xarray as xr
from loguru import logger

from dctools.data.coordinates import CoordinateSystem
from dctools.metrics.oceanbench_metrics import OCEANBENCH_VARIABLES, OceanbenchMetrics
from dctools.utilities.misc_utils import add_noise_with_snr

try:
    from oceanbench.core.rmsd import Variable
except ImportError:
    Variable = None


class MetricComputer(OceanbenchMetrics):
    """Computes metrics between prediction and reference datasets.

    Extends OceanbenchMetrics to handle variable mapping and optional noise addition.
    """

    def __init__(
            self,
            eval_variables: Optional[Optional[List[str]]] = None,
            oceanbench_eval_variables: Optional[Optional[List[str]]] = None,
            is_class4: bool = False,
            class4_kwargs: Optional[Optional[Dict[str, Any]]] = None,
            **kwargs,
        ):
        """Initialize the MetricComputer.

        Args:
            eval_variables (Optional[List[str]]): List of variable names to evaluate in the
                prediction/reference datasets.
            oceanbench_eval_variables (Optional[List[str]]): List of variable names corresponding to
                OceanBench standards. If None, tries to map `eval_variables` automatically.
            is_class4 (bool): Whether to use Class 4 metrics (Lagrangian/in-situ comparisons).
                Defaults to False.
            class4_kwargs (Optional[Dict[str, Any]]): Additional arguments for Class 4 evaluation.
            **kwargs: Additional arguments, e.g., `add_noise=True`.
        """
        super().__init__(
            is_class4=is_class4, class4_kwargs=class4_kwargs, **kwargs
        )
        self.is_class4 = is_class4
        self.class4_kwargs = class4_kwargs or {}
        self.eval_variables = eval_variables

        if oceanbench_eval_variables is None:
            if eval_variables and Variable:
                 logger.debug(f"Mapping eval_variables: {eval_variables}")
                 # Try to map
                 mapped_vars: List[Any] = []
                 mapping = {
                    "so": Variable.SEA_WATER_SALINITY,
                    "thetao": Variable.SEA_WATER_POTENTIAL_TEMPERATURE,
                    "uo": Variable.EASTWARD_SEA_WATER_VELOCITY,
                    "vo": Variable.NORTHWARD_SEA_WATER_VELOCITY,
                    "zos": Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID,
                 }
                 mapping.update(OCEANBENCH_VARIABLES)
                 for v in eval_variables:
                     if v in mapping:
                         mapped_vars.append(mapping[v])
                     else:
                         mapped_vars.append(v) # Keep as is if not found
                 self.oceanbench_eval_variables = mapped_vars
                 logger.debug(f"Mapped variables: {self.oceanbench_eval_variables}")
            else:
                 self.oceanbench_eval_variables = eval_variables
        else:
            self.oceanbench_eval_variables = oceanbench_eval_variables

        self.add_noise = kwargs.get("add_noise", False)

    #@profile
    def compute(
        self, pred_data: xr.Dataset, ref_data: Optional[xr.Dataset] = None,
        pred_coords: Optional[CoordinateSystem] = None,
        ref_coords: Optional[CoordinateSystem] = None,
        **kwargs: Any
    ) -> Any:
        """Compute the metrics.

        Args:
            pred_data (xr.Dataset): Prediction dataset.
            ref_data (xr.Dataset, optional): Reference dataset.
            pred_coords (CoordinateSystem, optional): Coordinate system of the prediction data.
            ref_coords (CoordinateSystem, optional): Coordinate system of the reference data.

        Returns:
            Any: The computed metric result, or None if an error occurs.
        """
        try:
            if self.is_class4:
                # Standardized call for class 4
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
