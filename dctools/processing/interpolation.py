
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence

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




def interpolate_scipy(
    ds: xr.Dataset,
    target_grid: Dict[str, Sequence],
    var_names: Optional[Sequence[str]] = None,
    depth_name: str = "depth",
    lat_name: str = "latitude",
    lon_name: str = "longitude",
    tol_depth: float = 1e-3,
    output_mode: str = "zarr",  # 'zarr'|'lazy'|'inmemory'
    output_path: str = None,
    zarr_target_chunks: dict = None,
) -> xr.Dataset:
    """
    Applique la fonction callable_fct sur un Dataset en utilisant un maillage cible.
    Parameters
    ----------
    ds: xr.Dataset
        Dataset d'entrée (doit contenir lat/lon)
    callable_fct: Callable
        Fonction à appliquer. Doit avoir la signature:
            fct(data2d, lat_src, lon_src, target_lat, target_lon, **kwargs)
        où data2d est une tranche 2D (lat, lon) de la variable.
    target_grid: Dict[str, Sequence]
        Dictionnaire avec 'lat' et 'lon' comme clés et les valeurs cibles.
    var_names: Optional[Sequence[str]]
        Liste des noms de variables à traiter. Si None, toutes les variables avec lat/lon sont utilisées.
    depth_name, lat_name, lon_name: str
        Noms des dimensions dans le Dataset.
    callable_kwargs: Optional[Dict[str, Any]]
        Arguments supplémentaires à passer à callable_fct.
    dask_gufunc_kwargs: Optional[Dict[str, Any]]
        Arguments supplémentaires pour dask.apply_ufunc.
    tol_depth: float
        Tolérance pour la sélection de profondeur.
    output_mode: str
        Mode de sortie: 'zarr' pour écrire dans un fichier Zarr, 'lazy' pour un Dataset paresseux, 'inmemory' pour un Dataset en mémoire.
    output_path: str
        Chemin du fichier de sortie si output_mode est 'zarr'. Si None, un fichier temporaire est créé.
    zarr_target_chunks: dict
        Spécification des chunks pour l'écriture Zarr.
    Returns
    -------
    xr.Dataset
        Dataset résultant après application de la fonction.
    """
    # target grid
    tgt_lat = np.asarray(target_grid["lat"])
    tgt_lon = np.asarray(target_grid["lon"])

    # var selection
    if var_names is None:
        var_names = [v for v, da in ds.data_vars.items() if lat_name in da.dims and lon_name in da.dims]

    lat_src = np.asarray(ds[lat_name])
    lon_src = np.asarray(ds[lon_name])

    out_vars = apply_over_time_depth(
        ds, var_names, depth_name, lat_name, lon_name,
        lat_src, lon_src, tgt_lat, tgt_lon,
        #safe_kwargs, dask_gufunc_kwargs,
        tol_depth=tol_depth,
    )

    if len(out_vars) == 0:
        raise RuntimeError("No metrics computed.")

    ds_out = xr.Dataset(out_vars)

    # writing / returning according to mode
    if output_mode == "zarr":
        # Utiliser le système de fichiers temporaires
        if output_path is None:
            output_path = self.create_temp_file(suffix=".zarr", prefix="pairwise_")
            logger.info(f"Using temporary file: {output_path}")
        
        if zarr_target_chunks is None:
            zarr_target_chunks = {}
            for d in ("time", depth_name):
                if d in ds_out.dims:
                    zarr_target_chunks[d] = 1
        ds_out = ds_out.chunk(zarr_target_chunks)
        outp = Path(output_path)
        if outp.exists():
            import shutil; shutil.rmtree(outp)
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
    #callable_kwargs: Dict[str, Any],
    #dask_gufunc_kwargs: Dict[str, Any],
    tol_depth: float = 1e-3,
) -> Dict[str, xr.DataArray]:
    """
    Generic loop over time/depth that applies a callable either on one or two datasets.
    Parameters
    ----------
    ds: xr.Dataset
        Dataset d'entrée (doit contenir lat/lon)
    var_names: Sequence[str]
        Liste des noms de variables à traiter.
    depth_name, lat_name, lon_name: str
        Noms des dimensions dans le Dataset.
    lat_src, lon_src: np.ndarray
        Coordonnées source (1D arrays).
    tgt_lat, tgt_lon: np.ndarray
        Coordonnées cibles (1D arrays).
    callable_fct: Callable
        Fonction à appliquer. Doit avoir la signature:
            fct(data2d, lat_src, lon_src, target_lat, target_lon, **kwargs)
        où data2d est une tranche 2D (lat, lon) de la variable.
    callable_kwargs: Dict[str, Any]
        Arguments supplémentaires à passer à callable_fct.
    dask_gufunc_kwargs: Dict[str, Any]
        Arguments supplémentaires pour dask.apply_ufunc.
    tol_depth: float
        Tolérance pour la sélection de profondeur.
    mode: str
        "single" pour une seule dataset, "double" pour deux datasets (ds2 requis).
    ds2: Optional[xr.Dataset]
        Second dataset si mode="double".
    Returns
    -------
    Dict[str, xr.DataArray]
        Dictionnaire des variables résultantes après application de la fonction.
    """
    out_vars = {}

    for var in var_names:
        if var not in ds:
            continue

        da = ds[var]
        if not {lat_name, lon_name}.issubset(da.dims):
            continue

        has_time = "time" in da.dims
        has_depth = depth_name in da.dims

        time_slices_out = []
        for t in (ds.time.values if has_time else [None]):
            depth_slices_out = []
            for z in (ds[depth_name].values if has_depth else [None]):
                # Sélection
                sel_kwargs = {}
                if t is not None:
                    sel_kwargs["time"] = t
                if z is not None:
                    sel_kwargs[depth_name] = z

                da_sel = da.sel(**sel_kwargs, method="nearest")

                if z is not None:
                    actual_depth = float(da_sel[depth_name].values)
                    if abs(actual_depth - float(z)) > tol_depth:
                        continue

                result_array = scipy_bilinear(
                    da_sel,
                    lat_src, lon_src,
                    tgt_lat, tgt_lon,
                )

                # Convert numpy.ndarray to DataArray
                da_out = xr.DataArray(
                    result_array,
                    dims=[lat_name, lon_name],  # Dimensions des coordonnées de sortie
                    coords={
                        lat_name: tgt_lat,
                        lon_name: tgt_lon
                    },
                    attrs=da_sel.attrs.copy(),  # Conserver les attributs de la variable originale
                    name=da_sel.name
                )
                # add time/depth dims if needed
                if z is not None:
                    da_out = da_out.expand_dims({depth_name: [actual_depth]})
                if t is not None:
                    da_out = da_out.expand_dims({"time": [t]})

                depth_slices_out.append(da_out)

            if len(depth_slices_out) > 0:
                depth_concat = xr.concat(depth_slices_out, dim=depth_name) if has_depth else depth_slices_out[0]
                time_slices_out.append(depth_concat)

        if len(time_slices_out) > 0:
            var_out = xr.concat(time_slices_out, dim="time") if has_time else time_slices_out[0]
            out_vars[var] = var_out

    return out_vars


def interpolate_dataset(
    ds: xr.Dataset, 
    target_grid: dict,
    dataset_processor: DatasetProcessor = None,
    weights_filepath: str = None,
    interpolation_lib: str = 'pyinterp', 
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
        if dataset_processor is not None:
            ds_renamed = dataset_processor.apply_single(
                ds_renamed, scipy_bilinear,
                target_grid={"lat": target_grid['lat'], "lon": target_grid['lon']},
                output_mode="zarr",
            )
        else:
            ds_renamed = interpolate_scipy(
                ds_renamed,
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