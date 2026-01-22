
from typing import Any, Dict, List, Optional

import ast
from loguru import logger
from memory_profiler import profile
import dask
import dask.array as da
import numpy as np
from oceanbench.core.distributed import DatasetProcessor
import pandas as pd
from torchvision import transforms
import xarray as xr
from pyproj import Transformer

from dctools.utilities.xarray_utils import (
    rename_coords_and_vars,
    subset_variables,
    assign_coordinate,
    reset_time_coordinates,
)
from dctools.processing.interpolation import (
    interpolate_dataset,
)

def detect_and_normalize_longitude_system(
    ds: xr.Dataset,
    lon_name: str = "lon"
) -> xr.Dataset:
    """
    Detect the longitude coordinate system and normalize it in the -180° to 180° range.
    """
    
    # Do nothing if ds is gridded in x/y and not lat/lon
    if "x" in ds.dims and "y" in ds.dims:
        return ds

    # Vérifier que la coordonnée longitude existe
    if lon_name not in ds.dims and lon_name not in ds.coords and lon_name not in ds.data_vars:
        logger.warning(f"Longitude coordinate '{lon_name}' not found in dataset")
        from dctools.utilities.xarray_utils import preview_display_dataset
        preview_display_dataset(ds)
        return ds
    
    # Analyser les plages de longitude avec optimisation mémoire et lazy loading
    
    # 1. Tenter d'utiliser les attributs (très rapide)
    attrs = ds[lon_name].attrs
    if "valid_min" in attrs and "valid_max" in attrs:
        lon_min = float(attrs["valid_min"])
        lon_max = float(attrs["valid_max"])
    else:
        # 2. Si dask array, optimiser la réduction
        da_lon = ds[lon_name]
        is_dask = hasattr(da_lon.data, "dask")
        
        if is_dask:
            # Pour détecter le système, un sous-échantillonnage suffit souvent
            # Cela réduit considérablement la charge sur les workers dask
            if da_lon.size > 100_000:
                # Strategy: Sample beginning, middle, and end of the dataset
                # This catches the range in most cases (tracks or grids) without
                # reading all chunks (unlike simple striding).
                
                # Ensure we handle 1D and nD cases reasonably
                main_dim = da_lon.dims[0]
                dim_size = da_lon.sizes[main_dim]
                
                # Define block size
                block = min(dim_size, 500)
                
                parts = []
                # Start
                parts.append(da_lon.isel({main_dim: slice(0, block)}).data.flatten())
                
                # Middle (if distinct)
                mid = dim_size // 2
                if mid > block:
                    parts.append(da_lon.isel({main_dim: slice(mid, mid + block)}).data.flatten())
                
                # End (if distinct)
                if dim_size > 2 * block:
                     parts.append(da_lon.isel({main_dim: slice(dim_size - block, dim_size)}).data.flatten())
                
                # Concatenate samples to form the subset
                da_lon_subset = da.concatenate(parts)
            else:
                da_lon_subset = da_lon.data
                
            # Calculer min et max en une seule passe graphe dask
            # Use dask.nanmin/nanmax which are lazy and handle NaNs correctly for dask arrays
            min_lazy = da.nanmin(da_lon_subset)
            max_lazy = da.nanmax(da_lon_subset)
            
            # Un seul compute pour les deux
            lon_min, lon_max = dask.compute(min_lazy, max_lazy)
            
            lon_min = float(lon_min.item())
            lon_max = float(lon_max.item())
        else:
            # NumPy array standard - calcul immédiat
            lon_min = float(da_lon.min(skipna=True).values)
            lon_max = float(da_lon.max(skipna=True).values)

    # Détecter le système de coordonnées
    lon_system = _detect_longitude_system(lon_min, lon_max)
    
    # Si déjà dans le bon système, retourner le dataset original
    if lon_system == "[-180, 180]":
        return ds
    
    # Normaliser si nécessaire
    if lon_system != "[-180, 180]":
        ds_normalized = _convert_longitude_to_180(ds, lon_name)
        return ds_normalized
    else:
        logger.warning(f"Unknown longitude system: {lon_system}, returning original dataset")
        return ds


def _detect_longitude_system(lon_min: float, lon_max: float) -> str:
    """Détecte le système de coordonnées longitude basé sur les valeurs min/max."""
    
    # Système [0, 360]
    if lon_min >= -5 and lon_max >= 355:  # Tolérance pour les arrondis
        return "[0, 360]"
    
    # Système [-180, 180]
    elif lon_min >= -185 and lon_max <= 185:  # Tolérance pour les arrondis
        return "[-180, 180]"
    
    # Système mixte ou autre
    elif lon_min < -5 and lon_max > 185:
        return "mixed"
    
    else:
        return "unknown"


def _convert_longitude_to_180(ds: xr.Dataset, lon_name: str) -> xr.Dataset:
    """
    Convertit les longitudes de [0, 360] vers [-180, 180].
    Retourne un dataset complet avec toutes les coordonnées, variables et attributs préservés.
    
    Args:
        ds: Dataset à convertir
        lon_name: Nom de la coordonnée longitude
        
    Returns:
        xr.Dataset: Dataset avec longitudes normalisées
    """
    ds_work = ds.copy(deep=True)
    lon_data = ds_work[lon_name]
    
    # Conversion : lon > 180 devient lon - 360
    lon_converted = xr.where(lon_data > 180, lon_data - 360, lon_data)
    
    # Préserver les attributs de la coordonnée longitude
    lon_attrs = lon_data.attrs.copy()
    
    # Remplacer la coordonnée longitude dans le dataset
    if lon_name in ds_work.coords:
        # Si c'est une coordonnée
        ds_work = ds_work.assign_coords({lon_name: lon_converted})
        # Réassigner les attributs
        ds_work[lon_name].attrs.update(lon_attrs)
        
    elif lon_name in ds_work.data_vars:
        # Si c'est une variable de données
        ds_work[lon_name] = lon_converted
        # Réassigner les attributs
        ds_work[lon_name].attrs.update(lon_attrs)
    
    # Trier par longitude si c'est une dimension pour maintenir l'ordre croissant
    if lon_name in ds_work.dims:
        try:
            ds_work = ds_work.sortby(lon_name)
            logger.debug("Dataset sorted by longitude")
        except Exception as e:
            logger.warning(f"Could not sort by longitude: {e}")
    
    # Préserver les attributs globaux du dataset
    ds_work.attrs.update(ds.attrs)
    
    return ds_work


class TransformWrapper:
    """Wraps a transform that operates on only the sample."""
    def __init__(self, transf):
        self.transf = transf

    def __call__(self, data):
        """
            data: tuple containing both sample and time_axis
            returns a tuple containing the transformed sample and original time_axis
        """
        sample, time_axis = data
        return self.transf(sample), time_axis


class StdLongitudeTransform:
    """A custom transform dependent on time axis."""
    def __init__(
        self
    ):
        pass

    def __call__(self, data):
        data = detect_and_normalize_longitude_system(
            data
        )
        return data


class RenameCoordsVarsTransform:
    """A custom transform dependent on time axis."""
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
    def __init__(
        self,
        dataset_processor: DatasetProcessor,
        ranges: Dict[str, np.arange], weights_filepath: str
    ):
        self.weights_filepath = weights_filepath
        self.ranges = ranges
        self.dataset_processor = dataset_processor

    def __call__(self, data):
        data = interpolate_dataset(
            data,
            self.ranges,
            self.dataset_processor,
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
            if time_name not in data:
                continue
                
            # Quick check on dtype to avoid loading data
            if np.issubdtype(data[time_name].dtype, np.datetime64):
                continue

            # If not visibly datetime, check first element lazily
            # This avoids loading the whole array just to check type
            try:
                if hasattr(data[time_name].data, "map_blocks"): # Is it a Dask array?
                    # Compute just the first value
                    first_val = data[time_name].isel({d:0 for d in data[time_name].dims}).compute().item()
                else:
                    first_val = data[time_name].values.flat[0]
                
                if isinstance(first_val, (pd.Timestamp, np.datetime64)):
                    continue
            except Exception:
                pass


            try:
                # If we really must convert, explicit load (unavoidable for pd.to_datetime usually)
                # But we can check if we can defer it or map_blocks it
                 data[time_name] = pd.to_datetime(data[time_name].values)
            except Exception as e:
                logger.warning(f"Could not convert {time_name} to datetime: {e}")

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
        # Lazy operation compatible with Dask
        lon_wrapped = ((lon + 180) % 360) - 180

        # On remplace la coordonnée longitude
        # Note: on ne trie QUE si la longitude est une dimension d'indexation (grille)
        # Si c'est une coordonnée 2D (swath) ou une coordonnée de points, trier casserait la correspondance !
        
        if self.lon_name in ds.dims:
            # C'est une dimension : on assigne et on trie pour garder une grille valide
            ds = ds.assign_coords({self.lon_name: lon_wrapped})
            ds = ds.sortby(self.lon_name)
        else:
             # Ce n'est pas une dimension (ex: swath along 'n_points')
             # On assigne juste les valeurs wrappées, SANS trier
             ds = ds.assign_coords({self.lon_name: lon_wrapped})

        return ds


class AssignCoordsTransform:
    """A custom transform dependent on time axis."""
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
    """A custom transform dependent on time axis. """
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

class ToSurfaceTransform:
    """
    Réduit la dimension 'depth' à sa première valeur (la plus proche de la surface).
    """
    def __init__(self, depth_coord_name: str = "depth"):
        self.depth_coord_name = depth_coord_name

    def __call__(self, ds: xr.Dataset):
        if self.depth_coord_name in ds.dims:
            # Sélectionne la première valeur de la dimension depth en gardant la dimension
            surface_ds = ds.isel({self.depth_coord_name: slice(0, 1)})
            return surface_ds
        return ds

class StdPercentageTransform:
    """
    Transform percentage variables in the [0, 100] range to [0,1].
    """
    def __init__(self, var_names: str | List[str]):
        """Init function for StdPercentageTransform.

        Parameters
        ----------
        var_names : str | List[str]
            Name(s) of the variable(s) to convert to the [0, 1] range.
        """
        
        self.var_names = [var_names] if var_names is str else var_names

    def __call__(self, ds: xr.Dataset) -> xr.Dataset:
        """Transform the variables in `ds` specified in `self.var_names`.

        Parameters
        ----------
        ds : xr.Dataset
            Dataset containing variables whose names match `self.var_names`.

        Returns
        -------
        xr.Dataset
            The dataset with the transformed variables.
        """
        for var_name in self.var_names:

            try:
                var_da = ds[var_name]
            except KeyError:
                logger.error(
                    f"Percentage variable '{var_name}' not found in dataset. " \
                        "Skipping variable."
                    )
                continue
            
            # Check that values are between 0 and 100
            if (var_da < 0).any().item() or (var_da > 100).any().item():
                logger.error(
                    f"Variable '{var_name}' does not represent a percentage. " \
                       "Skipping variable."
                    )
                continue
            
            # Check if variable is between 0 and 1
            if (var_da > 1).any().item():
                # Convert to 0 to 1 range
                new_var_da = var_da / 100.

                # Set units
                new_var_da = new_var_da.assign_attrs(units="1")
                ds[var_name] = new_var_da
            else:
                # NOTE: What happens if the variable is represented as a 0-100
                # percentage but the data just so happens to be between 0 and 1%?
                logger.warning(
                    f"Percentage variable '{var_name}' is already in the 0 " \
                        "to 1 range. Skipping variable."
                    )

        return ds


class CustomTransforms:
    def __init__(
        self, 
        transform_name: str,
        dataset_processor: DatasetProcessor,
        **kwargs,
    ):
        self.transform_name = transform_name
        self.dataset_processor = dataset_processor
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
            case "to_epsg3413":
                return self.transform_to_epsg3413(
                    dataset
                )
            case "standardize_to_surface":
                return self.transform_standardize_to_surface(
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
            StdLongitudeTransform(),
        ])

        transf_dataset = transform(dataset)

        return transf_dataset

    def transform_interpolate(
        self, dataset: xr.Dataset,
    ) -> xr.Dataset:
        assert(hasattr(self, "interp_ranges"))
        assert(hasattr(self, "weights_path"))

        transform = InterpolationTransform(
            self.dataset_processor,
            self.interp_ranges, self.weights_path
        )
        interp_dataset = transform(dataset)
        return interp_dataset

    def transform_glorys_to_glonet(
        self, dataset: xr.Dataset
    ) -> xr.Dataset:
        assert(hasattr(self, "depth_coord_vals"))
        assert(hasattr(self, "weights_path"))
        assert(hasattr(self, "interp_ranges"))
        depth_coord_name = self.depth_coord_name if hasattr(self, "depth_coord_name") else "depth"

        transform=transforms.Compose([
            SubsetCoordTransform(depth_coord_name, self.depth_coord_vals),
            InterpolationTransform(self.dataset_processor,
                                   self.interp_ranges, self.weights_path,
            ),
        ])
        transf_dataset = transform(dataset)
        return transf_dataset

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
    
    def to_timestamp(
        self,
        ds: xr.Dataset,
    ) -> xr.Dataset:
        """
        Convert the time coordinate to a timestamp.
        """
        assert(hasattr(self, "time_names"))
        transform = ToTimestampTransform(self.time_names)
        return transform(ds)

    def transform_to_epsg3413(
      self,
      dataset: xr.Dataset,      
    ) -> xr.Dataset:
        """
        Converts a dataset with lat/lon coordinates into the EPSG 3413 CRS.

        Parameters
        ----------
        dataset : xr.Dataset
            The dataset to transform

        Returns
        -------
        xr.Dataset
            A copy of the dataset with added `x` and `y` coordinates
        """

        # NOTE: Maybe this should be put into a class like the other transforms but
        #       I don't really get what would be the point of that

        # Create transformer from WGS84 to EPSG:3413
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:3413", always_xy=True)

        # Extract lon and lat arrays
        lons = dataset['lon'].values
        lats = dataset['lat'].values

        # Transform to EPSG:3413
        x, y = transformer.transform(lons, lats)

        # Add x and y as coordinates to the dataset
        transf_dataset = dataset.assign_coords(x=("n_points", x), y=("n_points", y))
        return transf_dataset


    def transform_standardize_to_surface(
        self,
        dataset: xr.Dataset,
    ) -> xr.Dataset:
        """
        Applique la standardisation puis réduit à la surface (première valeur de depth).
        """
        assert(hasattr(self, "coords_rename_dict") and hasattr(self, "vars_rename_dict"))
        assert(hasattr(self, "list_vars"))
        depth_coord_name = self.depth_coord_name if hasattr(self, "depth_coord_name") else "depth"

        # Convert string representations of dictionaries to actual dictionaries
        if isinstance(self.coords_rename_dict, str):
            coords_rename_dict = ast.literal_eval(self.coords_rename_dict)
        else:
            coords_rename_dict = self.coords_rename_dict
        if isinstance(self.vars_rename_dict, str):
            vars_rename_dict = ast.literal_eval(self.vars_rename_dict)
        else:
            vars_rename_dict = self.vars_rename_dict

        transform = transforms.Compose([
            SelectVariablesTransform(self.list_vars),
            RenameCoordsVarsTransform(
                coords_rename_dict=coords_rename_dict,
                vars_rename_dict=vars_rename_dict,
            ),
            StdLongitudeTransform(),
            ToSurfaceTransform(depth_coord_name=depth_coord_name),
        ])

        transf_dataset = transform(dataset)
        return transf_dataset
