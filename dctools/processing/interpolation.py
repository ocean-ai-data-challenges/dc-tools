"""Interpolation functions."""

from pathlib import Path
import traceback
from typing import Any, Dict, List, Optional, Sequence

import dask
import numpy as np
import xarray as xr

try:
    import xesmf as xe
except Exception:
    xe = None

try:
    import pyinterp # noqa: F401
    # import pyinterp.backends.xarray
except ImportError:
    pass

from loguru import logger
from oceanbench.core.distributed import DatasetProcessor
from scipy.interpolate import RegularGridInterpolator

from dctools.data.coordinates import (
    GEO_STD_COORDS
)
from dctools.utilities.xarray_utils import rename_coords_and_vars, create_empty_dataset


def interpolate_scipy(
    ds: xr.Dataset,
    target_grid: Dict[str, Sequence],
    var_names: Optional[Optional[Sequence[str]]] = None,
    depth_name: str = "depth",
    lat_name: str = "latitude",
    lon_name: str = "longitude",
    tol_depth: float = 1e-3,
    output_mode: str = "lazy",  # 'zarr'|'lazy'|'inmemory'
    output_path: Optional[str] = None,
    zarr_target_chunks: Optional[dict] = None,
    pairwise: bool = False,
) -> xr.Dataset:
    """
    Interpolate an xarray Dataset onto a target grid using SciPy interpolation.

    Parameters
    ----------
    ds : xr.Dataset
        Input xarray Dataset (must contain latitude and longitude dimensions).
    target_grid : dict of {str: Sequence}
        Dictionary defining the target grid coordinates.
        Must include keys for latitude and longitude.
    var_names : sequence of str, optional
        List of variable names to interpolate.
        If None, all variables with latitude and longitude dimensions are used.
    depth_name : str, default="depth"
        Name of the depth dimension in the Dataset.
    lat_name : str, default="latitude"
        Name of the latitude dimension in the Dataset.
    lon_name : str, default="longitude"
        Name of the longitude dimension in the Dataset.
    tol_depth : float, default=1e-3
        Tolerance for depth selection during interpolation.
    output_mode : {'zarr', 'lazy', 'inmemory'}, default='zarr'
        Output mode:
        - 'zarr': write to a Zarr file,
        - 'lazy': return a lazily evaluated Dataset,
        - 'inmemory': return an in-memory Dataset.
    output_path : str, optional
        Path to the output file if `output_mode` is 'zarr'. If None, a temporary file is created.
    zarr_target_chunks : dict, optional
        Chunking configuration for writing to Zarr.
    pairwise : bool, default=False
        If True, performs pairwise interpolation (Grid-to-Track) instead of
        Cartesian (Grid-to-Grid).
        Requires target_grid['lat'] and target_grid['lon'] to have the same length.

    Returns
    -------
    xr.Dataset
        Interpolated xarray Dataset on the target grid.
    """
    # target grid
    tgt_lat = np.asarray(target_grid["lat"])
    tgt_lon = np.asarray(target_grid["lon"])

    if pairwise and len(tgt_lat) != len(tgt_lon):
        raise ValueError(
            f"Pairwise interpolation requires lat/lon of same length. "
            f"Got {len(tgt_lat)} and {len(tgt_lon)}."
        )

    # var selection
    if var_names is None:
        var_names = [
            str(v) for v, da in ds.data_vars.items() if lat_name in da.dims and lon_name in da.dims
        ]

    ds_subset = ds[var_names]

    # Use xarray lazy interpolation or optimized manual kernel
    try:
        # Critical optimization for Dask:
        # If assume_sorted=True, xarray.interp avoids costly sorts on dask coords
        # We verify if source coords are monotonic (fast since 1D)

        # FORCED OPTIMIZATION for Pairwise: Use our specialized apply_ufunc implementation
        # standard .interp() can be slow/memory-intensive for Grid-to-Track on Dask
        if pairwise:
             raise NotImplementedError(
                 "Force fallback to optimized apply_over_time_depth"
             )

        # Grid-to-Grid: Cartesian product (default behavior)
        interp_coords = {lat_name: tgt_lat, lon_name: tgt_lon}

        # Constructing lazy call
        ds_out: xr.Dataset = ds_subset.interp(  # type: ignore[assignment]
            interp_coords,
            method="linear",
            kwargs={"fill_value": np.nan},
            assume_sorted=True  # Important for RAM/CPU with Dask
        )

    except Exception as e:
        if not pairwise and not isinstance(e, NotImplementedError):
             logger.warning(f"Lazy interpolation failed: {e}. Falling back to eager.")

        # Fallback to optimized manual kernel implementation
        # (This uses apply_ufunc now, so it is actually the PREFERRED path for pairwise)
        lat_src = np.asarray(ds[lat_name])
        lon_src = np.asarray(ds[lon_name])

        out_vars = apply_over_time_depth(
            ds, var_names, depth_name, lat_name, lon_name,
            lat_src, lon_src, tgt_lat, tgt_lon,
            tol_depth=tol_depth,
            pairwise=pairwise,
        )
        if len(out_vars) == 0:
            logger.warning("No variables interpolated. Returning empty dataset.")
            return create_empty_dataset({}) # Fallback empty

        ds_out = xr.Dataset(out_vars)


    # Writing / returning according to mode
    if output_mode == "zarr":
        # Use temporary file system
        if output_path is None:
            import tempfile
            temp_dir = tempfile.mkdtemp()
            output_path = str(Path(temp_dir) / "pairwise_interp.zarr")
            logger.info(f"Using temporary file: {output_path}")

        if zarr_target_chunks is None:
            zarr_target_chunks_dict: Dict[Any, Any] = {}
            for d in ("time", depth_name):
                if d in ds_out.dims:
                    zarr_target_chunks_dict[d] = 1
            ds_out = ds_out.chunk(zarr_target_chunks_dict)
        else:
             ds_out = ds_out.chunk(zarr_target_chunks)

        outp = Path(output_path)
        if outp.exists():
            import shutil
            shutil.rmtree(outp)
        ds_out.to_zarr(output_path, mode="w", consolidated=True)
        ds_out.close()
        ds_out = xr.open_zarr(output_path, chunks=zarr_target_chunks)

    elif output_mode == "lazy":
        pass # return ds_out

    elif output_mode == "inmemory":
        ds_out = ds_out.compute()

    else:
        raise ValueError(f"Unknown mode {output_mode}")

    return ds_out


def apply_over_time_depth(
    ds: xr.Dataset,
    var_names: Sequence[str],
    depth_name: str,
    lat_name: str,
    lon_name: str,
    lat_src: np.ndarray,
    lon_src: np.ndarray,
    tgt_lat: np.ndarray,
    tgt_lon: np.ndarray,
    tol_depth: float = 1e-3,
    pairwise: bool = False,
) -> Dict[str, xr.DataArray]:
    """
    Apply an operation over time and depth dimensions for selected variables in an xarray Dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Input xarray Dataset (must contain latitude and longitude dimensions).
    var_names : sequence of str
        List of variable names to process.
    depth_name : str
        Name of the depth dimension in the Dataset.
    lat_name : str
        Name of the latitude dimension in the Dataset.
    lon_name : str
        Name of the longitude dimension in the Dataset.
    lat_src : np.ndarray
        Source latitude coordinates (1D array).
    lon_src : np.ndarray
        Source longitude coordinates (1D array).
    tgt_lat : np.ndarray
        Target latitude coordinates (1D array).
    tgt_lon : np.ndarray
        Target longitude coordinates (1D array).
    tol_depth : float, default=1e-3
        Tolerance for depth selection during processing.
    pairwise : bool, default=False
        If True, use pairwise interpolation (Grid-to-Track).

    Returns
    -------
    dict of {str: xr.DataArray}
        Dictionary containing the resulting variables after applying the operation.
    """
    out_vars: Dict[Any, Any] = {}

    # Define target coords / dims based on mode
    if pairwise:
        out_dims = ["points"]
        out_coords = {"points": np.arange(len(tgt_lat))}
        # Optionally keep track of lat/lon as non-dim coords
        out_coords[lat_name] = ("points", tgt_lat) # type: ignore[assignment]
        out_coords[lon_name] = ("points", tgt_lon) # type: ignore[assignment]
    else:
        out_dims = [lat_name, lon_name]
        out_coords = {lat_name: tgt_lat, lon_name: tgt_lon}

    for var in var_names:
        if var not in ds:
            continue

        da = ds[var]
        if not {lat_name, lon_name}.issubset(da.dims):
            continue

        # Use apply_ufunc for efficient lazy execution/parallelism
        # This replaces the explicit Python loops over time/depth

        # Definition of wrapper to handle argument mapping
        def _wrapper_scipy_bilinear(data, l_src=lat_src, l_tgt=tgt_lat,
                                  ln_src=lon_src, ln_tgt=tgt_lon,
                                  pw=pairwise):
            return scipy_bilinear(
                data, l_src, ln_src, l_tgt, ln_tgt,
                pairwise=pw
            )

        output_sizes: Dict[Any, Any] = {}
        if pairwise:
            output_sizes["points"] = len(tgt_lat)
        else:
            output_sizes[lat_name] = len(tgt_lat)
            output_sizes[lon_name] = len(tgt_lon)

        # Determine output core dims
        output_core_dims = [["points"]] if pairwise else [[lat_name, lon_name]]

        try:
            da_out = xr.apply_ufunc(
                _wrapper_scipy_bilinear,
                da,
                input_core_dims=[[lat_name, lon_name]],
                output_core_dims=output_core_dims,
                vectorize=True,  # Loop over non-core dims (time, depth) in numpy/dask
                dask="parallelized",
                output_dtypes=[da.dtype],
                output_sizes=output_sizes,
            )

            # Restore coords
            if pairwise:
                da_out = da_out.assign_coords(points=out_coords["points"])
                # Note: other coords (time, depth) are preserved by apply_ufunc
            else:
                da_out = da_out.assign_coords({lat_name: tgt_lat, lon_name: tgt_lon})

            # Copy attrs
            da_out.attrs = da.attrs
            da_out.name = da.name
            out_vars[var] = da_out

        except Exception as e:
            logger.warning(f"Vectorized interpolation failed for {var}, falling back to loop: {e}")
            # Fallback legacy implementation (slow loop)
            time_slices_out: List[Any] = []
            has_time = "time" in da.dims
            has_depth = depth_name in da.dims

            for t in (ds.time.values if has_time else [None]):
                depth_slices_out: List[Any] = []
                # ... existing loop logic ...
                if depth_name in ds:
                    subset_vals: Any = ds[depth_name].values
                else:
                    subset_vals = [None]

                for z in subset_vals:
                    # Selection
                    sel_kwargs: Dict[Any, Any] = {}
                    if t is not None:
                        sel_kwargs["time"] = t
                    if z is not None:
                        sel_kwargs[depth_name] = z

                    da_sel = da.sel(**sel_kwargs, method="nearest")

                    if z is not None:
                        try:
                            actual_depth = float(da_sel[depth_name].values)
                        except Exception:
                            actual_depth = float(z)

                        if abs(actual_depth - float(z)) > tol_depth:
                            continue

                    result_array = scipy_bilinear(
                        da_sel.values,
                        lat_src, lon_src,
                        tgt_lat, tgt_lon,
                        pairwise=pairwise
                    )

                    # Convert numpy.ndarray to DataArray
                    da_out_slice = xr.DataArray(
                        result_array,
                        dims=out_dims,  # Output coordinate dimensions
                        coords=out_coords,
                        attrs=da_sel.attrs.copy(),
                        name=da_sel.name
                    )

                    # add time/depth dims if needed
                    if z is not None:
                        da_out_slice = da_out_slice.expand_dims({depth_name: [actual_depth]})
                    if t is not None:
                        da_out_slice = da_out_slice.expand_dims({"time": [t]})

                    depth_slices_out.append(da_out_slice)

                if len(depth_slices_out) > 0:
                    depth_concat = (
                        xr.concat(depth_slices_out, dim=depth_name)
                        if has_depth else depth_slices_out[0]
                    )
                    time_slices_out.append(depth_concat)

            if len(time_slices_out) > 0:
                var_out = xr.concat(time_slices_out, dim="time") if has_time else time_slices_out[0]
                out_vars[var] = var_out

    return out_vars


def interpolate_pyinterp(
    ds: xr.Dataset,
    target_grid: Dict[str, Sequence],
    reduce_precision: bool = False,
    pairwise: bool = False,
) -> xr.Dataset:
    """
    Interpolate Dataset using PyInterp (High Performance).

    Uses Grid2D/3D logic.
    Supports both Cartesian Grid (default) and Pairwise (Grid-to-Track) interpolation.
    """
    try:
        import pyinterp
        import pyinterp.backends.xarray
    except ImportError:
        logger.error("PyInterp not found. Please install it for high performance.")
        raise

    # Prepare Coordinates
    tgt_lat = np.asarray(target_grid["lat"])
    tgt_lon = np.asarray(target_grid["lon"])

    if reduce_precision:
        tgt_lat = tgt_lat.astype(np.float32)
        tgt_lon = tgt_lon.astype(np.float32)
        ds = ds.astype(np.float32)

    out_vars: Dict[Any, Any] = {}

    # Prepare Target Coordinates for Interpolation
    if pairwise:
        if len(tgt_lat) != len(tgt_lon):
            raise ValueError(
                "Pairwise interpolation requires lat/lon of same length. "
                f"Got {len(tgt_lat)} vs {len(tgt_lon)}"
            )
        x_target = tgt_lon
        y_target = tgt_lat
        output_dim_size = len(tgt_lat)
        out_coords = {"points": np.arange(output_dim_size)}
    else:
        # Cartesian Meshgrid
        mesh_lon, mesh_lat = np.meshgrid(tgt_lon, tgt_lat)
        x_target = mesh_lon.flatten()
        y_target = mesh_lat.flatten()
        out_coords = {"lat": tgt_lat, "lon": tgt_lon}

    for var_name in ds.data_vars:
        da = ds[var_name]

        # We need to recognize Lat/Lon dims
        lat_dim = 'latitude' if 'latitude' in da.dims else 'lat'
        lon_dim = 'longitude' if 'longitude' in da.dims else 'lon'

        # Check if variable has spatial dims
        if lat_dim not in da.dims or lon_dim not in da.dims:
            continue

        # Define wrapper for ufunc
        def _pyinterp_wrapper(
            data,
            lat_src,
            lon_src,
            xt=x_target,
            yt=y_target,
            pw=pairwise,
            t_lat=tgt_lat,
            t_lon=tgt_lon
        ):
            # Use PyInterp Core directly
            try:
                # Transpose to (x, y) if needed by PyInterp, assuming (lat, lon) input which
                # is (y, x)
                # PyInterp Grid2D expects (x_axis, y_axis, values) where values shape is
                # (len(x), len(y)).
                # Our data is usually (lat, lon) -> (y, x). So we transpose to (lon, lat) -> (x, y).

                x_axis = pyinterp.Axis(lon_src)
                y_axis = pyinterp.Axis(lat_src)

                # Check shapes
                if data.shape == (len(lat_src), len(lon_src)):
                     grid_data = data.T # (lon, lat)
                elif data.shape == (len(lon_src), len(lat_src)):
                     grid_data = data
                else:
                     # Attempt reshape if flattened
                     grid_data = data.reshape(len(lon_src), len(lat_src))

                grid = pyinterp.Grid2D(x_axis, y_axis, grid_data)

                res = grid.bivariate(
                    x=xt,
                    y=yt,
                    interpolator="bilinear"
                )

                if pw:
                    return res # 1D array
                else:
                    return res.reshape(len(t_lat), len(t_lon))
            except Exception as e:
                # Fallback or logging could go here
                raise e

        # Check existing coords
        src_lat = da[lat_dim].values
        src_lon = da[lon_dim].values

        # Determine output setup
        if pairwise:
             output_core_dims = [['points']]
             output_sizes = {'points': output_dim_size}
        else:
             output_core_dims = [['lat', 'lon']]
             output_sizes = {'lat': len(tgt_lat), 'lon': len(tgt_lon)}

        # apply_ufunc
        res = xr.apply_ufunc(
            _pyinterp_wrapper,
            da,
            src_lat,
            src_lon,
            input_core_dims=[[lat_dim, lon_dim], [], []],
            output_core_dims=output_core_dims,
            vectorize=True, # Loop over Time/Depth
            dask='parallelized', # Enable Dask
            output_dtypes=[np.float32 if reduce_precision else np.float64],
            dask_gufunc_kwargs={'allow_rechunk': True},
            output_sizes=output_sizes
        )

        # Assign coords
        if pairwise:
             res = res.assign_coords(points=out_coords["points"])
        else:
             res = res.assign_coords(lat=tgt_lat, lon=tgt_lon)

        out_vars[var_name] = res

    return xr.Dataset(out_vars)


def interpolate_dataset(
    ds: xr.Dataset,
    target_grid: dict,
    dataset_processor: Optional[DatasetProcessor] = None,
    weights_filepath: Optional[str] = None,
    interpolation_lib: str = 'pyinterp',
    reduce_precision: bool = False,
    pairwise: bool = False,
) -> xr.Dataset:
    """Unified interface which uses only scatter logic with Dask."""
    # Add missing standard_names
    for variable_name in ds.variables:
        var_std_name = str(ds[variable_name].attrs.get("standard_name", '')).lower()
        if not var_std_name:
            ds[variable_name].attrs["standard_name"] = str(variable_name).lower()

    # Save coordinate attributes before interpolation
    coords_attrs: Dict[Any, Any] = {}
    for coord in ds.coords:
        coords_attrs[coord] = ds.coords[coord].attrs.copy()

    # Save variable attributes as well
    vars_attrs: Dict[Any, Any] = {}
    for var in ds.data_vars:
        vars_attrs[var] = ds[var].attrs.copy()

    # Build output dictionary
    out_dict: Dict[Any, Any] = {}
    for key in target_grid.keys():
        out_dict[key] = target_grid[key]

    for dim in GEO_STD_COORDS.keys():
        if dim not in out_dict.keys():
            out_dict[dim] = ds.coords.get(dim, ds.sizes.get(dim))

    # Filter only dimensions that exist in the dataset
    # ranges = {k: v for k, v in target_grid.items() if k in ds.sizes}

    # Choose interpolation method
    # Standardize coordinate names first for all methods relying on standard names
    ds_renamed, coord_mapping = rename_to_standard_pyinterp(ds, 'lat', 'lon')

    # OPTIMIZATION: Subset dataset around target track to minimize I/O and memory
    # This prevents loading the full global grid when only a sparse track is needed.
    track_chunking_strategy = False

    if pairwise:
        try:
            tgt_lats: Any = target_grid.get('lat')
            tgt_lons: Any = target_grid.get('lon')

            # Identify actual coordinate names in ds_renamed
            d_lat = next(
                (d for d in ds_renamed.coords if d in ['latitude', 'lat', 'nav_lat']), None
            )
            d_lon = next(
                (d for d in ds_renamed.coords if d in ['longitude', 'lon', 'nav_lon']), None
            )

            if tgt_lats is not None and tgt_lons is not None and d_lat and d_lon:
                # buffer in degrees
                buffer = 2.0

                lat_min, lat_max = np.min(tgt_lats), np.max(tgt_lats)
                lon_min, lon_max = np.min(tgt_lons), np.max(tgt_lons)

                # Check lat increasing
                lat_vals = ds_renamed[d_lat]
                lat_increasing = True
                if lat_vals.size > 1:
                    lat_increasing = bool(lat_vals[0] < lat_vals[-1])

                lat_slice = (
                    slice(lat_min - buffer, lat_max + buffer) if lat_increasing
                    else slice(lat_max + buffer, lat_min - buffer)
                )
                ds_renamed = ds_renamed.sel({d_lat: lat_slice})

                # Check lon increasing / wrap
                lon_vals = ds_renamed[d_lon]
                if (lon_max - lon_min) < 350:
                        lon_increasing = True
                        if lon_vals.size > 1:
                            lon_increasing = bool(lon_vals[0] < lon_vals[-1])

                        lon_slice = (
                            slice(lon_min - buffer, lon_max + buffer) if lon_increasing
                            else slice(lon_max + buffer, lon_min - buffer)
                        )
                        ds_renamed = ds_renamed.sel({d_lon: lon_slice})

                # Log volume
                vol_points = ds_renamed[d_lat].size * ds_renamed[d_lon].size

                if vol_points > 1_000_000:
                    logger.warning(
                        f"Large subset detected for track! Grid Size: {vol_points} points. "
                        "Switching to Iterative Track Tiling strategy."
                    )
                    track_chunking_strategy = True

                logger.info(
                    f"Subsetted dataset for pairwise interp: "
                    f"Lat [{lat_min:.2f}, {lat_max:.2f}] Lon [{lon_min:.2f}, {lon_max:.2f}] | "
                    f"Grid Size: {ds_renamed[d_lat].size}x{ds_renamed[d_lon].size} = "
                    f"{vol_points} pts"
                )

        except Exception as e:
            logger.warning(f"Could not subset dataset (using full grid): {e}")


    if interpolation_lib == 'pyinterp':
        # Apply Track Chunking if needed, otherwise run directly
        if pairwise and track_chunking_strategy:
             # Iterate just like Scipy but calling interpolate_pyinterp
             track_chunk_size = 10000
             n_points = len(tgt_lats)
             datasets_list: List[Any] = []

             logger.info(
                 f"Starting Iterative Track Tiling (PyInterp): {n_points} points."
             )

             for i in range(0, n_points, track_chunk_size):
                sub_lats = tgt_lats[i:i+track_chunk_size]
                sub_lons = tgt_lons[i:i+track_chunk_size]

                buffer = 2.0
                loc_lat_min, loc_lat_max = np.min(sub_lats), np.max(sub_lats)
                loc_lon_min, loc_lon_max = np.min(sub_lons), np.max(sub_lons)

                ds_chunk = ds_renamed

                # ... Lat Slice ...
                lat_vals = ds_chunk[d_lat]
                lat_inc = True
                if lat_vals.size > 1:
                    lat_inc = bool(lat_vals[0] < lat_vals[-1])
                l_slice = (
                    slice(loc_lat_min - buffer, loc_lat_max + buffer) if lat_inc
                    else slice(loc_lat_max + buffer, loc_lat_min - buffer)
                )
                ds_chunk = ds_chunk.sel({d_lat: l_slice})

                # ... Lon Slice ...
                lon_vals = ds_chunk[d_lon]
                if (loc_lon_max - loc_lon_min) < 350:
                    lon_inc = True
                    if lon_vals.size > 1:
                        lon_inc = bool(lon_vals[0] < lon_vals[-1])
                    ln_slice = (
                        slice(loc_lon_min - buffer, loc_lon_max + buffer) if lon_inc
                        else slice(loc_lon_max + buffer, loc_lon_min - buffer)
                    )
                    ds_chunk = ds_chunk.sel({d_lon: ln_slice})

                ds_chunk_res = interpolate_pyinterp(
                    ds_chunk,
                    target_grid={"lat": sub_lats, "lon": sub_lons},
                    reduce_precision=reduce_precision,
                    pairwise=True,
                )
                datasets_list.append(ds_chunk_res)

             # Concatenate all chunks after the loop
             ds_interp_internal = xr.concat(datasets_list, dim="points")
        else:
            ds_interp_internal = interpolate_pyinterp(
                ds_renamed, target_grid, reduce_precision, pairwise=pairwise
            )

        ds_interp = rename_back(ds, ds_interp_internal, coord_mapping)

    elif interpolation_lib == 'scipy':

        if pairwise and track_chunking_strategy:
            # OPTIMIZATION: Iterative Track Tiling
            track_chunk_size = 10000  # Process 10000 points at a time
            n_points = len(tgt_lats)
            datasets_list_scipy: List[Any] = []

            logger.info(
                f"Starting Iterative Track Tiling (Scipy): {n_points} points."
            )

            for i in range(0, n_points, track_chunk_size):
                # 1. Extract track segment
                sub_lats = tgt_lats[i:i+track_chunk_size]
                sub_lons = tgt_lons[i:i+track_chunk_size]

                # 2. Identify local BBox + Buffer
                buffer = 2.0
                loc_lat_min, loc_lat_max = np.min(sub_lats), np.max(sub_lats)
                loc_lon_min, loc_lon_max = np.min(sub_lons), np.max(sub_lons)

                # 3. Create lazy subset of source grid
                ds_chunk = ds_renamed

                # ... Lat Slice ...
                lat_vals = ds_chunk[d_lat]
                lat_inc = True
                if lat_vals.size > 1:
                    lat_inc = bool(lat_vals[0] < lat_vals[-1])
                l_slice = (
                    slice(loc_lat_min - buffer, loc_lat_max + buffer) if lat_inc
                    else slice(loc_lat_max + buffer, loc_lat_min - buffer)
                )
                ds_chunk = ds_chunk.sel({d_lat: l_slice})

                # ... Lon Slice ...
                lon_vals = ds_chunk[d_lon]
                if (loc_lon_max - loc_lon_min) < 350:
                    lon_inc = True
                    if lon_vals.size > 1:
                        lon_inc = bool(lon_vals[0] < lon_vals[-1])
                    ln_slice = (
                        slice(loc_lon_min - buffer, loc_lon_max + buffer) if lon_inc
                        else slice(loc_lon_max + buffer, loc_lon_min - buffer)
                    )
                    ds_chunk = ds_chunk.sel({d_lon: ln_slice})

                # 4. Interpolate this small chunk
                ds_chunk_res = interpolate_scipy(
                    ds_chunk,
                    target_grid={"lat": sub_lats, "lon": sub_lons},
                    output_mode="lazy",
                    pairwise=True,
                )
                datasets_list_scipy.append(ds_chunk_res)

            # 5. Concatenate all segments along 'points' dimension
            ds_renamed_res = xr.concat(datasets_list_scipy, dim="points")

        elif dataset_processor is not None:
            # We assume apply_single distributes the call to interpolate_scipy
            ds_renamed_res = dataset_processor.apply_single(
                ds_renamed, interpolate_scipy,
                target_grid={"lat": target_grid['lat'], "lon": target_grid['lon']},
                output_mode="lazy",
                pairwise=pairwise,
            )
        else:
            ds_renamed_res = interpolate_scipy(
                ds_renamed,
                target_grid={"lat": target_grid['lat'], "lon": target_grid['lon']},
                output_mode="lazy",
                pairwise=pairwise,
            )
        ds_interp = rename_back(ds, ds_renamed_res, coord_mapping)



    elif interpolation_lib == "xesmf":
        ds_interp = interpolate_xesmf(
            ds,
            target_grid=out_dict,
            weights_filepath=weights_filepath,
            method="bilinear",
        )
    else:
        raise ValueError(f"Unknown interpolation library: {interpolation_lib}")

    # Reassign variable attributes
    for var in ds_interp.data_vars:
        if var in vars_attrs:
            ds_interp[var].attrs.update(vars_attrs[var])

    # Reassign coordinate attributes
    for coord in ds_interp.coords:
        if coord in coords_attrs:
            ds_interp[coord].attrs.update(coords_attrs[coord])

    # Add missing standard_names to result
    for variable_name in ds_interp.variables:
        var_std_name = str(ds_interp[variable_name].attrs.get("standard_name", '')).lower()
        if not var_std_name:
            ds_interp[variable_name].attrs["standard_name"] = str(variable_name).lower()

    return xr.Dataset(ds_interp)

def scipy_bilinear(
    data2d: np.ndarray,
    lat_src,
    lon_src,
    target_lat,
    target_lon,
    batch_size=100,
    pairwise=False
):
    """
    Memory-efficient bilinear interpolation using scipy.

    Parameters
    ----------
    pairwise : bool, default=False
        If True, interpolate at (target_lat[i], target_lon[i]) pairs (Grid-to-Track).
        If False, interpolate on the grid defined by target_lat x target_lon (Grid-to-Grid).
    """
    # Ensure numpy arrays
    try:
        data2d = np.asarray(data2d, dtype=np.float64)
        lat_src = np.asarray(lat_src, dtype=np.float64)
        lon_src = np.asarray(lon_src, dtype=np.float64)
        target_lat = np.asarray(target_lat, dtype=np.float64)
        target_lon = np.asarray(target_lon, dtype=np.float64)
    except Exception:
        traceback.print_exc()
    # Fix shape if needed
    if data2d.shape == (len(lon_src), len(lat_src)):
        data2d = data2d.T
    elif data2d.shape != (len(lat_src), len(lon_src)):
        if data2d.ndim == 1 and data2d.size == len(lat_src) * len(lon_src):
            data2d = data2d.reshape(len(lat_src), len(lon_src))
        else:
            raise ValueError(f"Cannot match data shape {data2d.shape}")

    # Interpolator
    interpolator = RegularGridInterpolator(
        (lat_src, lon_src), data2d,
        method="linear", bounds_error=False, fill_value=np.nan
    )

    if pairwise:
        # Output is 1D array of length N (N points along track)
        out = np.empty((len(target_lat),), dtype=np.float64)

        # Process in batches
        for i in range(0, len(target_lat), batch_size):
            lat_batch = target_lat[i:i+batch_size]
            lon_batch = target_lon[i:i+batch_size]
            points = np.column_stack([lat_batch, lon_batch])
            out[i:i+batch_size] = interpolator(points)
    else:
        # Output is 2D array (M x N)
        out = np.empty((len(target_lat), len(target_lon)), dtype=np.float64)

        # Process in batches of rows
        for i in range(0, len(target_lat), batch_size):
            lat_batch = target_lat[i:i+batch_size]
            # Build points for this batch
            lat_grid, lon_grid = np.meshgrid(lat_batch, target_lon, indexing="ij")
            points = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])
            out[i:i+batch_size, :] = interpolator(points).reshape(len(lat_batch), len(target_lon))

    return out

def rename_to_standard_pyinterp(ds, lat_name, lon_name):
    """Rename coordinates to 'lat' and 'lon' for pyinterp compatibility."""
    # save original names

    coord_mapping: Dict[Any, Any] = {}
    if lat_name != 'latitude':
        coord_mapping[lat_name] = 'latitude'
    if lon_name != 'longitude':
        coord_mapping[lon_name] = 'longitude'

    # Apply renaming
    if len(coord_mapping) > 0:
        logger.debug(f"Renaming coordinates for pyinterp compatibility: {coord_mapping}")
        ds_renamed = rename_coords_and_vars(
            ds, coord_mapping, coord_mapping
        )
    else:
        ds_renamed = ds
    return ds_renamed, coord_mapping


def rename_back(orig_ds, renamed_ds, coord_mapping):
    """Rename coordinates back to their original names."""
    # rennomage inverse
    reverse_mapping: Dict[Any, Any] = {}
    if len(coord_mapping) > 0:
        for orig_name, std_name in coord_mapping.items():
            reverse_mapping[std_name] = orig_name

    if reverse_mapping:
        logger.debug(f"Restoring original coordinate names: {reverse_mapping}")
        if isinstance(renamed_ds, xr.DataArray):
            ds_final = renamed_ds.rename(reverse_mapping)
        if isinstance(renamed_ds, xr.Dataset):
            ds_final = rename_coords_and_vars(
                renamed_ds, reverse_mapping, reverse_mapping
            )
    else:
        ds_final = renamed_ds

    # copy original attributes
    ds_final.attrs.update(orig_ds.attrs)

    # Copy variable attributes
    if isinstance(ds_final, xr.Dataset):
        for var_name in ds_final.data_vars:
            if var_name in orig_ds.data_vars:
                ds_final[var_name].attrs.update(orig_ds[var_name].attrs)
    if isinstance(ds_final, xr.DataArray):
        ds_final.attrs.update(orig_ds.attrs)
    return ds_final


def interpolate_xesmf(  # TODO : check and uncomment
        ds: xr.Dataset, target_grid: Dict[str, np.ndarray],
        weights_filepath: Optional[Optional[str]] = None,
        method: str = "bilinear",
    ) -> xr.Dataset:
    """
    Interpolate Dataset using xESMF.

    Args:
        ds (xr.Dataset): Input dataset.
        target_grid (Dict): Target grid coordinates.
        weights_filepath (str, optional): Path to weights file.
        method (str): Interpolation method (default: bilinear).

    Returns:
        xr.Dataset: Interpolated dataset.
    """
    # Save coordinate and variable attributes
    coords_attrs = {coord: ds.coords[coord].attrs.copy() for coord in ds.coords}
    vars_attrs = {var: ds[var].attrs.copy() for var in ds.data_vars}

    # Prepare target lat/lon grid
    lat_out = target_grid["lat"]
    lon_out = target_grid["lon"]
    target_grid_ds = xr.Dataset({
        "lat": ("lat", lat_out),
        "lon": ("lon", lon_out)
    })

    # Regridder cache mechanism
    regridder_cache: Dict[Any, Any] = {}

    def get_regridder(src, tgt, method, cache_key=None):
        if cache_key and cache_key in regridder_cache:
            return regridder_cache[cache_key]
        with dask.config.set(scheduler="synchronous"):
            regridder = xe.Regridder(src, tgt, method)
        if cache_key:
            regridder_cache[cache_key] = regridder
        return regridder

    out_vars: Dict[Any, Any] = {}
    for var in ds.data_vars:
        da = ds[var]
        # Loop over each depth
        if "depth" in da.dims:
            depth_vals = target_grid['depth']
            slices: List[Any] = []
            for d in depth_vals:
                da_sel = da.sel(depth=d)
                # Use grid size as cache key
                src_shape = tuple(da_sel["lat"].shape) + tuple(da_sel["lon"].shape)
                tgt_shape = tuple(lat_out.shape) + tuple(lon_out.shape)
                cache_key = (src_shape, tgt_shape, method)
                regridder = get_regridder(da_sel, target_grid_ds, method, cache_key)
                da_interp = regridder(da_sel)
                # Add depth dimension
                da_interp = da_interp.expand_dims({"depth": [d]})
                slices.append(da_interp)
            # Concatenate along depth dimension
            out_vars[var] = xr.concat(slices, dim="depth")
        else:
            src_shape = tuple(da["lat"].shape) + tuple(da["lon"].shape)
            tgt_shape = tuple(lat_out.shape) + tuple(lon_out.shape)
            cache_key = (src_shape, tgt_shape, method)
            regridder = get_regridder(da, target_grid_ds, method, cache_key)
            out_vars[var] = regridder(da)

    # Explicitly clear regridder cache to release ESMF resources
    regridder_cache.clear()

    # Build final dataset
    ds_out = xr.Dataset(out_vars)

    # Reassign variable attributes
    for var in ds_out.data_vars:
        if var in vars_attrs:
            ds_out[var].attrs = vars_attrs[var].copy()

    # Reassign coordinate attributes
    for coord in ds_out.coords:
        if coord in coords_attrs:
            new_coord = xr.DataArray(
                ds_out.coords[coord].values,
                dims=ds_out.coords[coord].dims,
                attrs=coords_attrs[coord].copy()
            )
            ds_out = ds_out.assign_coords({coord: new_coord})

    return ds_out
