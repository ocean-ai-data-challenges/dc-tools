
from copy import deepcopy
import profile
import traceback
from typing import Any, Dict, List, Optional

import ast
# import kornia
from loguru import logger
from memory_profiler import profile
import numpy as np
import pandas as pd
from torchvision import transforms
import xarray as xr

from dctools.data.coordinates import (
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
        data = rename_coords_and_vars(
            data, self.coords_rename_dict, self.vars_rename_dict
        )
        return data

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
        data = interpolate_dataset(
            data, self.ranges,
            self.weights_filepath,
            interpolation_lib='pyinterp',
        )
        return data


class ResetTimeCoordsTransform:
    def __init__(self):
        pass

    def __call__(self, data):
        reset_dataset = reset_time_coordinates(data)
        return reset_dataset

class ToTimestampTransform:
    def __init__(self, time_names: List[str]):
        self.time_names = time_names

    def __call__(self, data):
        """
        Convert the time coordinate to a timestamp.
        """
        for time_name in self.time_names:
            time_values = data[time_name].values

            # Vérifier si les valeurs sont déjà des pandas.Timestamp
            are_timestamps = isinstance(time_values[0], pd.Timestamp)

            if not are_timestamps:
                # Convertir toutes les valeurs en pandas.Timestamp
                data[time_name] = pd.to_datetime(data[time_name].values)
        return data


class WrapLongitudeTransform:
    """
    Transforme les longitudes d'un dataset xarray de [0, 360] vers [-180, 180].
    """
    def __init__(self, lon_name: str = "lon"):
        self.lon_name = lon_name

    def __call__(self, ds):

        if self.lon_name not in ds.coords and self.lon_name not in ds.dims:
            # Rien à faire si pas de longitude
            return ds

        # Récupère la DataArray des longitudes
        lon = ds[self.lon_name]
        # Applique la conversion
        lon_wrapped = ((lon + 180) % 360) - 180

        # Trie les longitudes et réindexe le dataset pour garder l'ordre croissant
        order = np.argsort(lon_wrapped)
        lon_wrapped_sorted = lon_wrapped[order]

        # Remplace la coordonnée longitude
        ds = ds.assign_coords({self.lon_name: lon_wrapped_sorted})

        # Réindexe le dataset si la longitude est une dimension
        if self.lon_name in ds.dims:
            ds = ds.sortby(self.lon_name)

        return ds


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
            case "add_spatial_coords":
                return self.transform_add_spatial_coords(
                    dataset
                )
            case "to_timestamp":
                return self.to_timestamp(
                    dataset
                )
            case _:
                return dataset

    #@profile
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

    #@profile
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
        # new_vars = list(transf_dataset.data_vars)

        return transf_dataset

    #@profile
    def transform_interpolate(
        self, dataset: xr.Dataset,
    ) -> xr.Dataset:
        assert(hasattr(self, "interp_ranges"))
        assert(hasattr(self, "weights_path"))

        transform = InterpolationTransform(self.interp_ranges, self.weights_path)
        interp_dataset = transform(dataset)
        return interp_dataset

    #@profile
    def transform_glorys_to_glonet(
        self, dataset: xr.Dataset
    ) -> xr.Dataset:
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

    #@profile
    def transform_subset_dataset(
        self, dataset: xr.Dataset,
    ) -> xr.Dataset:
        assert(hasattr(self, "list_vars") and hasattr(self, "depth_coord_vals"))
        depth_coord_name = self.depth_coord_name if hasattr(self, "depth_coord_name") else "depth"
        transform=transforms.Compose([
            SelectVariablesTransform(self.list_vars),
            SubsetCoordTransform(depth_coord_name, self.depth_coord_vals),
        ])
        transf_dataset = transform(dataset)
        return transf_dataset
    
    #@profile
    def to_timestamp(
        self,
        ds: xr.Dataset,
    ) -> xr.Dataset:
        """
        Convert the time coordinate to a timestamp.
        """
        assert(hasattr(self, "time_names"))
        # Supposons que ds est ton xr.Dataset et "time" la coordonnée temporelle
        transform = ToTimestampTransform(self.time_names)
        return transform(ds)

    '''def transform_add_spatial_coords(self, dataset: xr.Dataset) -> xr.Dataset:
        import ast, traceback
        assert hasattr(self, "coords_rename_dict")
        if isinstance(self.coords_rename_dict, str):
            coords_rename_dict = ast.literal_eval(self.coords_rename_dict)
        else:
            coords_rename_dict = self.coords_rename_dict
        lat_coord_name = coords_rename_dict['lat']
        lon_coord_name = coords_rename_dict['lon']
        depth_coord_name = coords_rename_dict.get('depth', None)
        try:
            coords_to_set = [lat_coord_name, lon_coord_name]
            if depth_coord_name is not None:
                coords_to_set.append(depth_coord_name)

            # S'assurer que les variables sont des coordonnées
            for coord in coords_to_set:
                if coord not in dataset.coords and coord in dataset:
                    dataset = dataset.set_coords(coord)

            # Construire le mapping {ancienne_dim: nouvelle_coordonnée}
            swap_dict = {}
            dims_checked = set()
            for coord in coords_to_set:
                if coord not in dataset.dims and coord in dataset.coords:
                    # Chercher une dimension de même taille à remplacer
                    for dim in dataset.dims:
                        if (
                            dim not in swap_dict  # éviter de swapper deux fois la même dim
                            and dim not in dims_checked
                            and dataset[coord].ndim == 1
                            and dataset[coord].size == dataset.dims[dim]
                            and coord != dim
                        ):
                            swap_dict[dim] = coord
                            dims_checked.add(dim)
                            break
            if swap_dict:
                dataset = dataset.swap_dims(swap_dict)

            return dataset
        except Exception as exc:
            logger.error(f"Erreur lors de l'ajout des coordonnées spatiales : {traceback.format_exc(exc)}")
            raise'''
