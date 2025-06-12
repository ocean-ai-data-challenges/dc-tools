

from typing import Any, Dict, List, Optional

import ast
# import kornia
from loguru import logger
import numpy as np
from torchvision import transforms
import xarray as xr

from dctools.data.coordinates import (
    CoordinateSystem,
    LIST_VARS_GLONET,
)
from dctools.utilities.xarray_utils import (
    rename_coordinates,
    rename_coords_and_vars,
    subset_variables,
    interpolate_dataset,
    assign_coordinate,
    reset_time_coordinates,
)


class TransformWrapper:
    """ Wraps a transform that operates on only the sample """
    def __init__(self, transf):
        self.transf = transf

    def __call__(self, data):
        """
            data: tuple containing both sample and time_axis
            returns a tuple containing the transformed sample and original time_axis
        """
        sample, time_axis = data
        return self.transf(sample), time_axis


class RenameCoordsVarsTransform:
    """ a custom transform dependent on time axis """
    def __init__(
        self,
        coords_rename_dict: Optional[Dict] = None,
        vars_rename_dict: Optional[Dict] = None
    ):
        self.coords_rename_dict = coords_rename_dict
        self.vars_rename_dict = vars_rename_dict

    def __call__(self, data):
        rename_ds = rename_coords_and_vars(
            data, self.coords_rename_dict, self.vars_rename_dict
        )
        return rename_ds

class SelectVariablesTransform:
    def __init__(self, variables: List[str]):
        self.variables = variables

    def __call__(self, data):
        sub_dataset = subset_variables(data, self.variables)
        return sub_dataset


class InterpolationTransform:
    def __init__(self, ranges: Dict[str, np.arange], weights_filepath: str):
        self.weights_filepath = weights_filepath
        self.ranges = ranges

    def __call__(self, data):
        interp_dataset = interpolate_dataset(data, self.ranges, self.weights_filepath)
        return interp_dataset


class ResetTimeCoordsTransform:
    def __init__(self):
        pass

    def __call__(self, data):
        reset_dataset = reset_time_coordinates(data)
        return reset_dataset


class AssignCoordsTransform:
    """ a custom transform dependent on time axis """
    def __init__(
            self, coord_name: str, coord_vals: List[Any], coord_attrs: Dict[str, str]
        ):
        self.coord_name = coord_name
        self.coord_vals = coord_vals
        self.coord_attrs = coord_attrs

    def __call__(self, data):
        transf_dataset = assign_coordinate(
            data, self.coord_name,
            self.coord_vals, self.coord_attrs)
        return transf_dataset

class SubsetCoordTransform:
    """ a custom transform dependent on time axis """
    def __init__(self, coord_name: str, coord_vals: List[Any]):
        self.coord_name = coord_name
        self.coord_vals = coord_vals

    def approx_inside(self, val1: float, vals: List[float], tolerance: float) -> bool:
        for val2 in vals:
            if abs(val1 - val2) < tolerance:
                return True
        return False

    def __call__(self, data):
        assert(self.coord_name in list(data.dims))
        match self.coord_name:
            case "lat":
                indices = [
                    idx for idx in range(
                        0, data.lat.values.size
                    ) if data.lat.values[idx] in self.coord_vals
                ]
                transf_dataset = data.isel(lat=indices)
            case "lon":
                indices = [
                    idx for idx in range(
                        0, data.lon.values.size
                    ) if data.lon.values[idx] in self.coord_vals
                ]
                transf_dataset = data.isel(lon=indices)
            case "time":
                indices = [
                    idx for idx in range(
                        0, data.time.values.size
                    ) if data.time.values[idx] in self.coord_vals
                ]
                transf_dataset = data.isel(time=indices)
            case "depth":
                indices = [
                    idx for idx in range(
                        0, data.depth.values.size
                    ) if self.approx_inside(data.depth.values[idx], self.coord_vals, 1e-3)  ## TODO : remove this ugly hack: depth values are returned in float64 format which gives wrong approximate values
                ]
                transf_dataset = data.isel(depth=indices)
            case _:
                return data
        return transf_dataset


class CustomTransforms:
    def __init__(self, transform_name: str, **kwargs):
        self.transform_name = transform_name
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __call__(self, dataset):
        match self.transform_name:
            case "rename_subset_vars":
                return self.transform_rename_subset_vars(
                    dataset
                )
            case "interpolate":
                return self.transform_interpolate(
                    dataset
                )
            case "glorys_to_glonet":
                return self.transform_glorys_to_glonet(
                    dataset
                )
            case "subset_dataset":
                return self.transform_subset_dataset(
                    dataset
                )
            case "standardize_dataset":
                return self.transform_standardize_dataset(
                    dataset
                )
            case _:
                return dataset

    def transform_rename_subset_vars(
        self,
        dataset: xr.Dataset,
    ) -> xr.Dataset:
        assert(hasattr(self, "dict_rename") and hasattr(self, "list_vars"))
        if isinstance(self.dict_rename, str):
            dict_rename = ast.literal_eval(self.dict_rename)
        else:
            dict_rename = self.dict_rename
        transform=transforms.Compose([
            RenameCoordsVarsTransform(rename_coords_dict=dict_rename),
            SelectVariablesTransform(self.list_vars),
        ])
        transf_dataset = transform(dataset)
        return transf_dataset

    def transform_standardize_dataset(
        self,
        dataset: xr.Dataset,
    ) -> xr.Dataset:
        assert(hasattr(self, "coords_rename_dict") and hasattr(self, "vars_rename_dict"))
        assert(hasattr(self, "list_vars"))

        # Convert string representations of dictionaries to actual dictionaries
        if isinstance(self.coords_rename_dict, str):
            coords_rename_dict = ast.literal_eval(self.coords_rename_dict)
        else:
            coords_rename_dict = self.coords_rename_dict
        if isinstance(self.vars_rename_dict, str):
            vars_rename_dict = ast.literal_eval(self.vars_rename_dict)
        else:
            vars_rename_dict = self.vars_rename_dict

        transform=transforms.Compose([
            SelectVariablesTransform(self.list_vars),
            RenameCoordsVarsTransform(
                coords_rename_dict=coords_rename_dict,
                vars_rename_dict=vars_rename_dict,
            ),
        ])

        transf_dataset = transform(dataset)
        new_vars = list(transf_dataset.data_vars)

        return transf_dataset

    def transform_interpolate(
        self, dataset: xr.Dataset,
    ) -> xr.Dataset:
        assert(hasattr(self, "interp_ranges"))
        assert(hasattr(self, "weights_path"))

        transform=transforms.Compose([
            InterpolationTransform(self.interp_ranges, self.weights_path)
        ])
        interp_dataset = transform(dataset)
        return interp_dataset

    def transform_glorys_to_glonet(
        self, dataset: xr.Dataset
    ):
        assert(hasattr(self, "depth_coord_vals"))
        assert(hasattr(self, "weights_path"))
        assert(hasattr(self, "interp_ranges"))
        dict_rename = self.dict_rename if hasattr(self, "dict_rename") else None
        depth_coord_name = self.depth_coord_name if hasattr(self, "depth_coord_name") else "depth"
        list_vars = self.list_vars if hasattr(self, "list_vars") else LIST_VARS_GLONET

        transform=transforms.Compose([
            #RenameCoordsVarsTransform(coords_rename_dict=dict_rename),
            #SelectVariablesTransform(list_vars),
            SubsetCoordTransform(depth_coord_name, self.depth_coord_vals),
            #AssignCoordsTransform(time_coord_name, time_coord_vals, time_coord_attrs),
            InterpolationTransform(self.interp_ranges, self.weights_path),
            # ResetTimeCoordsTransform(),
        ])
        transf_dataset = transform(dataset)
        return transf_dataset

    def transform_subset_dataset(
        self, dataset: xr.Dataset,
    ):
        assert(hasattr(self, "list_vars") and hasattr(self, "depth_coord_vals"))
        depth_coord_name = self.depth_coord_name if hasattr(self, "depth_coord_name") else "depth"
        transform=transforms.Compose([
            SelectVariablesTransform(self.list_vars),
            SubsetCoordTransform(depth_coord_name, self.depth_coord_vals),
        ])
        transf_dataset = transform(dataset)
        return transf_dataset
