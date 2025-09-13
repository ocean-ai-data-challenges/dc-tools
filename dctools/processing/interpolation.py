
import numpy as np
import xarray as xr

import pyinterp

import pyinterp.backends.xarray
from loguru import logger
from oceanbench.core.distributed import DatasetProcessor

from dctools.data.coordinates import (
    GEO_STD_COORDS
)
# from dctools.processing.distributed import ParallelExecutor
from dctools.utilities.xarray_utils import rename_coords_and_vars



def interpolate_dataset(
    ds: xr.Dataset, 
    target_grid: dict,
    dataset_processor: DatasetProcessor,
    weights_filepath: str = None,
    interpolation_lib: str = 'pyinterp', 
    max_workers: int = None,
    chunk_strategy: str = 'auto',
    **kwargs
) -> xr.Dataset:
    """
    Interface unifiée qui utilise uniquement la logique scatter avec Dask.
    """
    
    # Ajouter les standard_names manquants
    for variable_name in ds.variables:
        var_std_name = ds[variable_name].attrs.get("standard_name", '').lower()
        if not var_std_name:
            ds[variable_name].attrs["standard_name"] = variable_name.lower()

    # Sauvegarder les attributs des coordonnées avant interpolation
    coords_attrs = {}
    for coord in ds.coords:
        coords_attrs[coord] = ds.coords[coord].attrs.copy()

    # Sauvegarder aussi les attrs des variables
    vars_attrs = {}
    for var in ds.data_vars:
        vars_attrs[var] = ds[var].attrs.copy()

    # Construire le dictionnaire de sortie
    out_dict = {}
    for key in target_grid.keys():
        out_dict[key] = target_grid[key]
    
    for dim in GEO_STD_COORDS.keys():
        if dim not in out_dict.keys():
            out_dict[dim] = ds.coords.get(dim, ds.sizes.get(dim))
    
    # Filtrer seulement les dimensions qui existent dans le dataset
    # ranges = {k: v for k, v in target_grid.items() if k in ds.sizes}

    # Choisir la méthode d'interpolation
    '''if interpolation_lib == 'scipy':
        method = kwargs.get('method', 'linear')
        ds_out = interpolate_scipy_optimized(ds, target_grid, method)'''
    
    if interpolation_lib == 'pyinterp':
        # ds_out = interpolate_dataset_time_depth_vars(ds, target_grid)

        ds_renamed, coord_mapping = rename_to_standard_pyinterp(ds, 'lat', 'lon')
        ds_renamed = dataset_processor.apply_single(
            ds_renamed, scipy_bilinear,
            target_grid={"lat": target_grid['lat'], "lon": target_grid['lon']},
            output_mode="zarr",
        )
        ds_interp = rename_back(ds, ds_renamed, coord_mapping)
    #elif interpolation_lib == "xesmf":
    #    if weights_filepath and Path(weights_filepath).is_file():
    #        # Use precomputed weights
    #        logger.debug(f"Using interpolation precomputed weights from {weights_filepath}")
    #        #regridder = xe.Regridder(
    #        #    ds, ds_out, "bilinear", reuse_weights=True, filename=weights_filepath
    #        #)
    #        ds_interp = interpolate_xesmf(
    #            ds,
    #            target_grid=out_dict,
    #            reuse_weights=True,
    #            weights_file=weights_filepath,
    #            method="bilinear",
    #        )
    #    else:
    #        ds_interp = interpolate_xesmf(
    #            ds,
    #            target_grid=out_dict,
    #            reuse_weights=False,
    #            weights_file=weights_filepath,
    #            method="bilinear",
    #        )
    else:
        raise ValueError(f"Unknown interpolation library: {interpolation_lib}")
        
    # Réaffecter les attributs des variables
    for var in ds_interp.data_vars:
        if var in vars_attrs:
            ds_interp[var].attrs.update(vars_attrs[var])

    # Réaffecter les attributs des coordonnées
    for coord in ds_interp.coords:
        if coord in coords_attrs:
            ds_interp[coord].attrs.update(coords_attrs[coord])

    # Ajouter les standard_names manquants au résultat
    for variable_name in ds_interp.variables:
        var_std_name = ds_interp[variable_name].attrs.get("standard_name", '').lower()
        if not var_std_name:
            ds_interp[variable_name].attrs["standard_name"] = variable_name.lower()

    return ds_interp

def scipy_bilinear(data2d: np.ndarray, lat_src, lon_src, target_lat, target_lon, batch_size=100):
    """Memory-efficient bilinear interpolation using scipy, processed by rows."""
    from scipy.interpolate import RegularGridInterpolator
    import numpy as np

    # Ensure numpy arrays
    data_np = np.asarray(data2d, dtype=np.float64)
    lat_src = np.asarray(lat_src, dtype=np.float64)
    lon_src = np.asarray(lon_src, dtype=np.float64)
    target_lat = np.asarray(target_lat, dtype=np.float64)
    target_lon = np.asarray(target_lon, dtype=np.float64)

    # Fix shape if needed
    if data_np.shape == (len(lon_src), len(lat_src)):
        data_np = data_np.T
    elif data_np.shape != (len(lat_src), len(lon_src)):
        if data_np.ndim == 1 and data_np.size == len(lat_src) * len(lon_src):
            data_np = data_np.reshape(len(lat_src), len(lon_src))
        else:
            raise ValueError(f"Cannot match data shape {data_np.shape}")

    # Interpolator
    interpolator = RegularGridInterpolator(
        (lat_src, lon_src), data_np,
        method="linear", bounds_error=False, fill_value=np.nan
    )

    # Output array
    out = np.empty((len(target_lat), len(target_lon)), dtype=np.float64)

    # Process in batches of rows
    for i in range(0, len(target_lat), batch_size):
        lat_batch = target_lat[i:i+batch_size]
        # Build points for this batch
        lat_grid, lon_grid = np.meshgrid(lat_batch, target_lon, indexing="ij")
        points = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])
        out[i:i+batch_size, :] = interpolator(points).reshape(len(lat_batch), len(target_lon))

    return out


def _pyinterp_2d_block(arr2d, lat_src, lon_src, lat_tgt, lon_tgt):
    """
    Interpole bilinéairement une carte 2D (lat x lon) sur (lat_tgt, lon_tgt) avec PyInterp.
    - arr2d: np.ndarray 2D [lat, lon]
    - lat_src, lon_src: 1D numpy arrays (coords sources)
    - lat_tgt, lon_tgt: 1D numpy arrays (coords cibles)
    Retourne un np.ndarray 2D [lat_tgt, lon_tgt]
    """
    # Construire la grille PyInterp à partir d'un DataArray temporaire
    da_src = xr.DataArray(
        arr2d,
        coords={"lat": np.asarray(lat_src), "lon": np.asarray(lon_src)},
        dims=("lat", "lon")
    )
    grid = pyinterp.backends.xarray.Grid2D(da_src)

    xx, yy = np.meshgrid(np.asarray(lon_tgt), np.asarray(lat_tgt))
    out = grid.bilinear({"lon": xx.ravel(), "lat": yy.ravel()}).reshape(len(lat_tgt), len(lon_tgt))
    return out

def rename_to_standard_pyinterp(ds, lat_name, lon_name):
    # sauvegarde des noms originaux

    coord_mapping = {}
    if lat_name != 'latitude':
        coord_mapping[lat_name] = 'latitude'
    if lon_name != 'longitude':
        coord_mapping[lon_name] = 'longitude'

    # Appliquer le renommage si nécessaire
    if len(coord_mapping) > 0:
        logger.debug(f"Renaming coordinates for pyinterp compatibility: {coord_mapping}")
        ds_renamed = rename_coords_and_vars(
            ds, coord_mapping, coord_mapping
        )
    else:
        ds_renamed = ds
    return ds_renamed, coord_mapping


def rename_back(orig_ds, renamed_ds, coord_mapping):

    # rennomage inverse
    reverse_mapping = {}
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
    
    # copier les attributs d'origine
    ds_final.attrs.update(orig_ds.attrs)
    
    # Copier les attributs des variables
    if isinstance(ds_final, xr.Dataset):
        for var_name in ds_final.data_vars:
            if var_name in orig_ds.data_vars:
                ds_final[var_name].attrs.update(orig_ds[var_name].attrs)
    if isinstance(ds_final, xr.DataArray):
        ds_final.attrs.update(orig_ds.attrs)
    return ds_final


'''def interpolate_xesmf(  # TODO : check and uncomment
        ds: xr.Dataset, ranges: Dict[str, np.ndarray],
        weights_filepath: Optional[str] = None,
    ) -> xr.Dataset:

    for variable_name in ds.variables:
        var_std_name = ds[variable_name].attrs.get("standard_name",'').lower()
        if not var_std_name:
            var_std_name = ds[variable_name].attrs.get("std_name", '').lower()

    # 1. Sauvegarder les attributs des coordonnées AVANT interpolation
    coords_attrs = {}
    for coord in ds.coords:
        coords_attrs[coord] = ds.coords[coord].attrs.copy()

    # (optionnel) Sauvegarder aussi les attrs des variables si besoin
    vars_attrs = {}
    for var in ds.data_vars:
        vars_attrs[var] = ds[var].attrs.copy()

    for key in ranges.keys():
        assert(key in list(ds.dims))

    out_dict = {}
    for key in ranges.keys():
        out_dict[key] = ranges[key]
    for dim in GEO_STD_COORDS.keys():
        if dim not in out_dict.keys():
            out_dict[dim] = ds.coords[dim].values
    ds_out = create_empty_dataset(out_dict)

    # TODO : adapt chunking depending on the dataset type
    ds_out = ds_out.chunk(chunks={"lat": -1, "lon": -1, "time": 1})

    if weights_filepath and Path(weights_filepath).is_file():
        # Use precomputed weights
        logger.debug(f"Using interpolation precomputed weights from {weights_filepath}")
        regridder = xe.Regridder(
            ds, ds_out, "bilinear", reuse_weights=True, filename=weights_filepath
        )
    else:
        # Compute weights
        regridder = xe.Regridder(
            ds, ds_out, "bilinear"
        )
        # Save the weights to a file
        regridder.to_netcdf(weights_filepath)
    # Regrid the dataset
    ds_out = regridder(ds)

    # 2. Réaffecter les attributs des variables (déjà fait dans ton code)
    for var in ds_out.data_vars:
        if var in vars_attrs:
            ds_out[var].attrs = vars_attrs[var].copy()

    # 3. Réaffecter les attributs des coordonnées
    for coord in ds_out.coords:
        if coord in coords_attrs:
            # Crée un nouveau DataArray avec les attrs sauvegardés
            new_coord = xr.DataArray(
                ds_out.coords[coord].values,
                dims=ds_out.coords[coord].dims,
                attrs=coords_attrs[coord].copy()
            )
            ds_out = ds_out.assign_coords({coord: new_coord})

    for variable_name in ds.variables:
        var_std_name = ds[variable_name].attrs.get("standard_name",'').lower()
        if not var_std_name:
            var_std_name = ds[variable_name].attrs.get("std_name", '').lower()

    return ds_out'''