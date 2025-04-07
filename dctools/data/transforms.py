

from typing import Any, Dict, List, Optional

# import kornia
import numpy as np
from torchvision import transforms
import xarray as xr

from dctools.utilities.xarray_utils import rename_coordinates,\
    subset_variables, interpolate_dataset, create_coords_rename_dict,\
    assign_coordinate


class TransformWrapper:
    """ Wraps a transform that operates on only the sample """
    def __init__(self, transf):
        self.transf = transf

    #@dask.delayed
    def __call__(self, data):
        """
            data: tuple containing both sample and time_axis
            returns a tuple containing the transformed sample and original time_axis
        """
        sample, time_axis = data
        return self.transf(sample), time_axis


class RenameCoordsTransform:
    """ a custom transform dependent on time axis """
    def __init__(self, rename_dict: Optional[Dict] = None):
        self.rename_dict = rename_dict

    #@dask.delayed
    def __call__(self, data):
        if not self.rename_dict:
            self.rename_dict = create_coords_rename_dict(data)

        renamed_dataset = rename_coordinates(data, self.rename_dict)
        return renamed_dataset

class SelectVariablesTransform:
    def __init__(self, variables: List[str]):
        self.variables = variables

    #@dask.delayed
    def __call__(self, data):
        renamed_dataset = subset_variables(data, self.variables)
        return renamed_dataset


class InterpolationTransform:
    def __init__(self, ranges: Dict[str, np.arange], weights_filepath: Optional[str] = None):
        self.weights_filepath = weights_filepath
        self.ranges = ranges

    #@dask.delayed
    def __call__(self, data):
        interp_dataset = interpolate_dataset(data, self.ranges, self.weights_filepath)
        return interp_dataset


class AssignCoordsTransform:
    """ a custom transform dependent on time axis """
    def __init__(
            self, coord_name: str, coord_vals: List[Any], coord_attrs: Dict[str, str]
        ):
        self.coord_name = coord_name
        self.coord_vals = coord_vals
        self.coord_attrs = coord_attrs

    #@dask.delayed
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

    #@dask.delayed
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
        self.weights_path = None
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __call__(self, dataset):
        match self.transform_name:
            case "rename_subset_vars":
                return self.transform_rename_subset_vars(
                    dataset, self.dict_rename, self.list_vars
                )
            case "interpolate":
                return self.transform_interpolate(
                    dataset, self.ranges, weights_filepath=self.weights_path
                )
            case "glorys_to_glonet":
                return self.transform_glorys_to_glonet(
                    dataset, self.list_vars,
                    self.depth_coord_vals, self.interp_ranges,
                    weights_path=self.weights_path
                )
            case "subset_dataset":
                return self.transform_subset_dataset(
                    dataset, self.list_vars,
                    self.depth_coord_vals
                )
            case _:
                return dataset

    def transform_rename_subset_vars(
        self,
        dataset: xr.Dataset,
        dict_rename: Optional[Dict[str, str]] = None,
        list_vars: Optional[List[str]] = None
    ) -> xr.Dataset:
        assert(hasattr(self, "dict_rename") and hasattr(self, "list_vars"))

        transform=transforms.Compose([
            RenameCoordsTransform(dict_rename),
            SelectVariablesTransform(list_vars),
        ])
        transf_dataset = transform(dataset)
        return transf_dataset

    def transform_interpolate(
        self, dataset: xr.Dataset, interp_ranges: Dict[str, np.ndarray],
        weights_path: Optional[str] = None,
    ) -> xr.Dataset:
        assert(hasattr(self, "interp_ranges"))

        transform=transforms.Compose([
            InterpolationTransform(interp_ranges, weights_path)
        ])
        interp_dataset = transform(dataset)
        return interp_dataset

    def transform_glorys_to_glonet(
        self, dataset: xr.Dataset,
        list_vars: List[str],
        depth_coord_vals: List[Any],
        interp_ranges: Dict[str, np.ndarray],
        dict_rename: Optional[Dict[str, str]] = None,
        depth_coord_name: Optional[str] = "depth",
        weights_path: Optional[str] = None,
    ):
        assert(hasattr(self, "list_vars"))
        assert(hasattr(self, "interp_ranges") and hasattr(self, "depth_coord_vals"))
        transform=transforms.Compose([
            RenameCoordsTransform(dict_rename),
            SelectVariablesTransform(list_vars),
            SubsetCoordTransform(depth_coord_name, depth_coord_vals),
            #AssignCoordsTransform(time_coord_name, time_coord_vals, time_coord_attrs),
            InterpolationTransform(interp_ranges, weights_path),
        ])
        transf_dataset = transform(dataset)
        return transf_dataset

    def transform_subset_dataset(
        self, dataset: xr.Dataset,
        list_vars: List[str],
        depth_coord_vals: List[Any],
        depth_coord_name: Optional[str] = "depth",
    ):
        assert(hasattr(self, "list_vars") and hasattr(self, "depth_coord_vals"))
        transform=transforms.Compose([
            SelectVariablesTransform(list_vars),
            SubsetCoordTransform(depth_coord_name, depth_coord_vals),
        ])
        transf_dataset = transform(dataset)
        return transf_dataset
