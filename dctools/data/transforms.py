"""Data transformation registry and pipeline utilities."""

from typing import Any, Dict, List, Optional, cast

import ast
from loguru import logger
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

from dctools.data.coordinates import (
    TARGET_DIM_RANGES,
    TARGET_DEPTH_VALS,
)

def detect_and_normalize_longitude_system(
    ds: xr.Dataset,
    lon_name: str = "lon"
) -> xr.Dataset:
    """Detect the longitude coordinate system and normalize it in the -180° to 180° range."""
    # Do nothing if ds is gridded in x/y and not lat/lon
    if "x" in ds.dims and "y" in ds.dims:
        return ds

    # Check that the longitude coordinate exists
    if lon_name not in ds.dims and lon_name not in ds.coords and lon_name not in ds.data_vars:
        logger.warning(f"Longitude coordinate '{lon_name}' not found in dataset")
        from dctools.utilities.xarray_utils import preview_display_dataset
        preview_display_dataset(ds)
        return ds

    # Analyze longitude ranges with memory optimization and lazy loading

    # 1. Try to use attributes (very fast)
    attrs = ds[lon_name].attrs
    if "valid_min" in attrs and "valid_max" in attrs:
        lon_min = float(attrs["valid_min"])
        lon_max = float(attrs["valid_max"])
    else:
        # 2. If dask array, optimize reduction
        da_lon = ds[lon_name]
        is_dask = hasattr(da_lon.data, "dask")

        if is_dask:
            # For system detection, subsampling is often sufficient
            # This significantly reduces load on dask workers
            if da_lon.size > 100_000:
                # Strategy: Sample beginning, middle, and end of the dataset
                # This catches the range in most cases (tracks or grids) without
                # reading all chunks (unlike simple striding).

                # Ensure we handle 1D and nD cases reasonably
                main_dim = da_lon.dims[0]
                dim_size = da_lon.sizes[main_dim]

                # Define block size
                block = min(dim_size, 500)

                parts: List[Any] = []
                # Start
                parts.append(da_lon.isel({main_dim: slice(0, block)}).data.flatten())

                # Middle (if distinct)
                mid = dim_size // 2
                if mid > block:
                    parts.append(da_lon.isel({main_dim: slice(mid, mid + block)}).data.flatten())

                # End (if distinct)
                if dim_size > 2 * block:
                    parts.append(
                        da_lon.isel({main_dim: slice(dim_size - block, dim_size)}).data.flatten()
                    )

                # Concatenate samples to form the subset
                da_lon_subset = da.concatenate(parts)
            else:
                da_lon_subset = da_lon.data

            # Calculate min and max in a single dask graph pass
            # Use dask.nanmin/nanmax which are lazy and handle NaNs correctly for dask arrays
            min_lazy = da.nanmin(da_lon_subset)
            max_lazy = da.nanmax(da_lon_subset)

            # One compute for both
            lon_min, lon_max = dask.compute(min_lazy, max_lazy)

            lon_min = float(lon_min)
            lon_max = float(lon_max)
        else:
            # Standard NumPy array - immediate calculation
            lon_min = float(da_lon.min(skipna=True).values)
            lon_max = float(da_lon.max(skipna=True).values)

    # Detect coordinate system
    lon_system = _detect_longitude_system(lon_min, lon_max)

    # If already in correct system, return original dataset
    if lon_system == "[-180, 180]":
        return ds

    # Normalize if necessary
    if lon_system != "[-180, 180]":
        ds_normalized = _convert_longitude_to_180(ds, lon_name)
        return ds_normalized
    else:
        logger.warning(f"Unknown longitude system: {lon_system}, returning original dataset")
        return ds


def _detect_longitude_system(lon_min: float, lon_max: float) -> str:
    """Detects the longitude coordinate system based on min/max values."""
    # System [0, 360]
    if lon_min >= -5 and lon_max >= 355:  # Tolerance for rounding
        return "[0, 360]"

    # System [-180, 180]
    elif lon_min >= -185 and lon_max <= 185:  # Tolerance for rounding
        return "[-180, 180]"

    # Mixed system or other
    elif lon_min < -5 and lon_max > 185:
        return "mixed"

    else:
        return "unknown"


def _convert_longitude_to_180(ds: xr.Dataset, lon_name: str) -> xr.Dataset:
    """
    Converts longitudes from [0, 360] to [-180, 180].

    Returns a complete dataset with all coordinates, variables and attributes preserved.

    Args:
        ds: Dataset to convert
        lon_name: Name of longitude coordinate

    Returns:
        xr.Dataset: Dataset with normalized longitudes
    """
    ds_work = ds.copy(deep=True)
    lon_data = ds_work[lon_name]

    # Conversion: lon > 180 becomes lon - 360
    lon_converted = xr.where(lon_data > 180, lon_data - 360, lon_data)

    # Preserve longitude coordinate attributes
    lon_attrs = lon_data.attrs.copy()

    # Replace longitude coordinate in the dataset
    if lon_name in ds_work.coords:
        # If it's a coordinate
        ds_work = ds_work.assign_coords({lon_name: lon_converted})
        # Reassign attributes
        ds_work[lon_name].attrs.update(lon_attrs)

    elif lon_name in ds_work.data_vars:
        # If it's a data variable
        ds_work[lon_name] = lon_converted
        # Reassign attributes
        ds_work[lon_name].attrs.update(lon_attrs)

    # Sort by longitude if it is a dimension to maintain increasing order
    if lon_name in ds_work.dims:
        try:
            ds_work = ds_work.sortby(lon_name)
            logger.debug("Dataset sorted by longitude")
        except Exception as e:
            logger.warning(f"Could not sort by longitude: {e}")

    # Preserve global dataset attributes
    ds_work.attrs.update(ds.attrs)

    return ds_work


class TransformWrapper:
    """Wraps a transform that operates on only the sample."""

    def __init__(self, transf):
        self.transf = transf

    def __call__(self, data):
        """
        Convert data.

        data: tuple containing both sample and time_axis
        returns a tuple containing the transformed sample and original time_axis
        """
        sample, time_axis = data
        return self.transf(sample), time_axis

# Registry for transforms
TRANSFORM_REGISTRY = {}

def register_transform(name: str):
    """Decorator to register a transform class."""
    def decorator(cls):
        TRANSFORM_REGISTRY[name] = cls
        return cls
    return decorator

def build_transform_pipeline(
    config_list: Optional[List[Dict[str, Any]]],
    dataset_processor: Optional[DatasetProcessor] = None
):
    """Factory to build a composition of transforms from a config list."""
    if config_list is None:
        return None  # Or empty pipeline?

    pipeline: List[Any] = []
    for step in config_list:
        name = step["name"]
        kwargs = step.get("kwargs", {}).copy()

        # Inject dataset_processor if expected by the transform (naive injection)
        # InterpolationTransform needs it.
        # We can check signature or just know which ones need it.
        # Ideally, the kwargs in config should not contain runtime objects like dataset_processor.

        if name == "interpolate_dataset" and dataset_processor:
            kwargs["dataset_processor"] = dataset_processor

        if name not in TRANSFORM_REGISTRY:
            raise ValueError(
                f"Transform '{name}' not found in registry. "
                f"Available: {list(TRANSFORM_REGISTRY.keys())}"
            )

        cls = TRANSFORM_REGISTRY[name]
        pipeline.append(cls(**kwargs))

    return transforms.Compose(pipeline)

@register_transform("detect_normalize_longitude")
class StdLongitudeTransform:
    """A custom transform dependent on time axis."""

    def __init__(self):
        """Initialize."""
        pass

    def __call__(self, data):
        """Apply transform."""
        data = detect_and_normalize_longitude_system(data)
        return data


@register_transform("rename_coords_vars")
class RenameCoordsVarsTransform:
    """A custom transform dependent on time axis."""

    def __init__(
        self,
        coords_rename_dict: Optional[Optional[Dict]] = None,
        vars_rename_dict: Optional[Optional[Dict]] = None
    ):
        """Initialize."""
        self.coords_rename_dict = coords_rename_dict
        self.vars_rename_dict = vars_rename_dict

    def __call__(self, data):
        """Apply transform."""
        data = rename_coords_and_vars(
            data, self.coords_rename_dict, self.vars_rename_dict
        )
        return data

@register_transform("select_variables")
class SelectVariablesTransform:
    """Transform to select specific variables from the dataset."""

    def __init__(self, variables: List[str]):
        """Initialize."""
        self.variables = variables

    def __call__(self, data):
        """Apply transform."""
        sub_dataset = subset_variables(data, self.variables)
        return sub_dataset


@register_transform("interpolate_dataset")
class InterpolationTransform:
    """Transform to interpolate the dataset to target ranges."""

    def __init__(
        self,
        dataset_processor: DatasetProcessor,
        ranges: Dict[str, Any], weights_filepath: str,
        reduce_precision: bool = False
    ):
        """Initialize."""
        self.weights_filepath = weights_filepath
        self.ranges = ranges
        self.dataset_processor = dataset_processor
        self.reduce_precision = reduce_precision

    def __call__(self, data):
        """Apply transform."""
        data = interpolate_dataset(
            data,
            self.ranges,
            self.dataset_processor,
            self.weights_filepath,
            interpolation_lib='pyinterp',
            reduce_precision=self.reduce_precision,
        )
        return data


@register_transform("reset_time_coords")
class ResetTimeCoordsTransform:
    """Transform to reset time coordinates."""

    def __init__(self):
        """Initialize."""
        pass

    def __call__(self, data):
        """Apply transform."""
        reset_dataset = reset_time_coordinates(data)
        return reset_dataset

@register_transform("to_timestamp")
class ToTimestampTransform:
    """Transform to convert time coordinates to timestamps."""

    def __init__(self, time_names: List[str]):
        """Initialize."""
        self.time_names = time_names

    def __call__(self, data):
        """Convert the time coordinate to a timestamp."""
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
                    first_val = data[time_name].isel(
                        {d:0 for d in data[time_name].dims}
                    ).compute().item()
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


@register_transform("wrap_longitude")
class WrapLongitudeTransform:
    """Transforms xarray dataset longitudes from [0, 360] to [-180, 180]."""

    def __init__(self, lon_name: str = "lon"):
        """Initialize."""
        self.lon_name = lon_name

    def __call__(self, ds):
        """Apply transform."""
        if self.lon_name not in ds.coords and self.lon_name not in ds.dims:
            # Nothing to do if no longitude
            return ds

        # Retrieve longitudes DataArray
        lon = ds[self.lon_name]
        # Apply conversion
        # Lazy operation compatible with Dask
        lon_wrapped = ((lon + 180) % 360) - 180

        # Replace longitude coordinate
        # Note: sort only if longitude is an indexing dimension (grid)
        # If it's a 2D coordinate (swath) or point coordinate,
        # sorting would break correspondence!

        if self.lon_name in ds.dims:
            # It is a dimension: assign and sort to keep a valid grid
            ds = ds.assign_coords({self.lon_name: lon_wrapped})
            ds = ds.sortby(self.lon_name)
        else:
             # It is not a dimension (e.g. swath along 'n_points')
             # Just assign wrapped values, without sorting
             ds = ds.assign_coords({self.lon_name: lon_wrapped})

        return ds


@register_transform("assign_coords")
class AssignCoordsTransform:
    """A custom transform dependent on time axis."""

    def __init__(
            self, coord_name: str, coord_vals: List[Any], coord_attrs: Dict[str, str]
        ):
        """Initialize."""
        self.coord_name = coord_name
        self.coord_vals = coord_vals
        self.coord_attrs = coord_attrs

    def __call__(self, data):
        """Apply transform."""
        transf_dataset = assign_coordinate(
            data, self.coord_name,
            self.coord_vals, self.coord_attrs)
        return transf_dataset

@register_transform("subset_coord")
class SubsetCoordTransform:
    """A custom transform dependent on time axis."""

    def __init__(self, coord_name: str, coord_vals: List[Any]):
        """Initialize."""
        self.coord_name = coord_name
        self.coord_vals = coord_vals

    def approx_inside(self, val1: float, vals: List[float], tolerance: float) -> bool:
        """Check if value is approximately inside list."""
        for val2 in vals:
            if abs(val1 - val2) < tolerance:
                return True
        return False

    def __call__(self, data):
        """Apply transform."""
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
                    # TODO : remove this ugly hack: depth values are returned in float64 format
                    ) if self.approx_inside(data.depth.values[idx], self.coord_vals, 1e-3)
                ]
                transf_dataset = data.isel(depth=indices)
            case _:
                return data
        return transf_dataset

@register_transform("to_surface")
class ToSurfaceTransform:
    """Reduces 'depth' dimension to its first value (closest to surface)."""

    def __init__(self, depth_coord_name: str = "depth"):
        """Initialize."""
        self.depth_coord_name = depth_coord_name

    def __call__(self, ds: xr.Dataset):
        """Apply transform."""
        if self.depth_coord_name in ds.dims:
            # Selects first value of depth dimension keeping the dimension
            surface_ds = ds.isel({self.depth_coord_name: slice(0, 1)})
            return surface_ds
        return ds

@register_transform("std_percentage")
class StdPercentageTransform:
    """Transform percentage variables in the [0, 100] range to [0,1]."""

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


@register_transform("to_epsg3413")
class ToEpsg3413Transform:
    """Converts a dataset with lat/lon coordinates into the EPSG 3413 CRS."""

    def __init__(self):
        """Initialize."""
        pass

    def __call__(self, dataset: xr.Dataset) -> xr.Dataset:
        """Apply transform."""
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


def get_dataset_transform(
        alias: str,
        metadata: Dict[str, Any],
        dataset_processor: DatasetProcessor,
        transform_name: Optional[Optional[str]] = None,
        config: Optional[Optional[Dict[str, Any]]] = None,
    ) -> "CustomTransforms":
    """
    Retrieves the appropriate transformation pipeline for a dataset based on its alias and name.

    Args:
        alias (str): The alias of the dataset.
        metadata (Dict[str, Any]): Dictionary containing dataset metadata
            (renaming dicts, keep vars).
        dataset_processor (DatasetProcessor): The processor for distributed/lazy operations.
        transform_name (str, optional): The name of the transformation pipeline to build.
            If None, infers the default pipeline for the alias.
        config (Optional[Dict[str, Any]]): Additional configuration (paths, precision, ranges).

    Returns:
        CustomTransforms: The configured transformation pipeline.
    """
    if config is None:
        config = {}

    keep_vars = metadata.get("keep_vars", [])
    coords_rename_dict = metadata.get("coords_rename_dict", {})
    vars_rename_dict = metadata.get("vars_rename_dict", {})

    # Base pipeline common to nearly all datasets: selection, renaming, lon normalization
    # Used by "standardize" based transforms
    std_dataset_params = [
            {"name": "select_variables", "kwargs": {"variables": keep_vars}},
            {
                "name": "rename_coords_vars",
                "kwargs": {
                    "coords_rename_dict": coords_rename_dict,
                    "vars_rename_dict": vars_rename_dict
                }
            },
            {"name": "detect_normalize_longitude", "kwargs": {}}
    ]

    # Infer default transform name if not provided
    if transform_name is None:
        if alias == "glorys_cmems":
            transform_name = "standardize_glorys"
        else:
            transform_name = "standardize"

    pipeline: List[Any] = []

    if transform_name == "standardize_glorys":
        # Specific pipeline for GLORYS interpolation
        pipeline.extend(std_dataset_params)

        weights_filepath = config.get("regridder_weights")
        reduce_precision = config.get("reduce_precision", False)
        interp_ranges = config.get("interp_ranges", TARGET_DIM_RANGES)
        depth_coord_vals = config.get("depth_coord_vals", TARGET_DEPTH_VALS)

        pipeline.append({
            "name": "subset_coord",
            "kwargs": {
                "coord_name": "depth",
                "coord_vals": depth_coord_vals
            }
        })
        pipeline.append({
            "name": "interpolate_dataset",
            "kwargs": {
                "ranges": interp_ranges,
                "weights_filepath": weights_filepath,
                "reduce_precision": reduce_precision
            }
        })

    elif transform_name == "standardize":
        pipeline.extend(std_dataset_params)

    elif transform_name == "standardize_add_coords":
        # Assuming this adds EPSG coords or similar
        pipeline.append({"name": "to_epsg3413", "kwargs": {}})
        pipeline.extend(std_dataset_params)

    elif transform_name == "standardize_to_surface":
        pipeline.extend(std_dataset_params)
        pipeline.append({
            "name": "to_surface",
            "kwargs": {"depth_coord_name": "depth"}
        })
    else:
        # Fallback or error
        # Check if it matches a specific dataset alias special case,
        # usually handled via transform_name
        logger.warning(
            f"Transform name '{transform_name}' is not standard. " \
            "Returning empty pipeline or minimal pipeline."
        )
        # Assuming minimal standardization or empty
        pass

    return CustomTransforms(pipeline, dataset_processor=dataset_processor)


class CustomTransforms:
    """Wrapper for dataset transformations."""

    def __init__(
        self,
        transform_name: str | List[Dict[str, Any]],
        dataset_processor: DatasetProcessor,
        **kwargs,
    ):
        """Initialize."""
        self.dataset_processor = dataset_processor

        if isinstance(transform_name, list):
            self.pipeline = build_transform_pipeline(transform_name, dataset_processor)
            self.is_pipeline = True
        else:
            self.transform_name = transform_name
            self.is_pipeline = False
            for key, value in kwargs.items():
                setattr(self, key, value)

    def __call__(self, dataset):
        """Apply transform."""
        if self.is_pipeline:
            return self.pipeline(dataset)

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
        """Rename and subset variables."""
        assert(hasattr(self, "dict_rename") and hasattr(self, "list_vars"))
        if isinstance(self.dict_rename, str):
            dict_rename = ast.literal_eval(self.dict_rename)
        else:
            dict_rename = self.dict_rename
        transform=transforms.Compose([
            RenameCoordsVarsTransform(coords_rename_dict=dict_rename),
            SelectVariablesTransform(self.list_vars),
        ])
        return cast(xr.Dataset, transform(dataset))

    def transform_standardize_dataset(
        self,
        dataset: xr.Dataset,
    ) -> xr.Dataset:
        """Standardize dataset (rename, subset, longitude)."""
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

        return cast(xr.Dataset, transform(dataset))

    def transform_interpolate(
        self, dataset: xr.Dataset,
    ) -> xr.Dataset:
        """Interpolate dataset."""
        assert(hasattr(self, "interp_ranges"))
        assert(hasattr(self, "weights_path"))

        reduce_precision = getattr(self, "reduce_precision", False)

        transform = InterpolationTransform(
            self.dataset_processor,
            self.interp_ranges, self.weights_path,
            reduce_precision=reduce_precision
        )
        return cast(xr.Dataset, transform(dataset))

    def transform_glorys_to_glonet(
        self, dataset: xr.Dataset
    ) -> xr.Dataset:
        """Transform GLORYS to GloNet format."""
        assert(hasattr(self, "depth_coord_vals"))
        assert(hasattr(self, "weights_path"))
        assert(hasattr(self, "interp_ranges"))
        depth_coord_name = self.depth_coord_name if hasattr(self, "depth_coord_name") else "depth"

        reduce_precision = getattr(self, "reduce_precision", False)

        transform=transforms.Compose([
            SubsetCoordTransform(depth_coord_name, self.depth_coord_vals),
            InterpolationTransform(self.dataset_processor,
                                   self.interp_ranges, self.weights_path,
                                   reduce_precision=reduce_precision
            ),
        ])
        return cast(xr.Dataset, transform(dataset))

    def transform_subset_dataset(
        self, dataset: xr.Dataset,
    ) -> xr.Dataset:
        """Subset dataset."""
        assert(hasattr(self, "list_vars") and hasattr(self, "depth_coord_vals"))
        depth_coord_name = self.depth_coord_name if hasattr(self, "depth_coord_name") else "depth"
        transform=transforms.Compose([
            SelectVariablesTransform(self.list_vars),
            SubsetCoordTransform(depth_coord_name, self.depth_coord_vals),
        ])
        return cast(xr.Dataset, transform(dataset))

    def to_timestamp(
        self,
        ds: xr.Dataset,
    ) -> xr.Dataset:
        """Convert the time coordinate to a timestamp."""
        assert(hasattr(self, "time_names"))
        transform = ToTimestampTransform(self.time_names)
        return cast(xr.Dataset, transform(ds))

    def transform_add_spatial_coords(
        self,
        dataset: xr.Dataset,
    ) -> xr.Dataset:
        """Add spatial coordinates to the dataset."""
        # This method was missing but referenced in __call__.
        # Implementation is unclear, returning dataset as is for now.
        logger.warning("transform_add_spatial_coords called but not implemented.")
        return dataset

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
        """Applies standardization then reduces to surface (first value of depth)."""
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

        return cast(xr.Dataset, transform(dataset))
