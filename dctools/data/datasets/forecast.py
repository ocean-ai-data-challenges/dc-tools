from collections import OrderedDict
from datetime import timedelta
import os
from typing import Callable

import pandas as pd
import xarray as xr

'''def evaluate_forecast_leadtimes(
    forecast_index: pd.DataFrame,
    open_pred: Callable[[str], xr.Dataset],
    open_ref: Callable[[str], xr.Dataset],
    eval_variables: list,
    compute_metrics: Callable,
    binning: dict,
    metrics: list,
    max_open_files: int = 20,  # Limite le nombre de fichiers ouverts/téléchargés
    remove_file: Callable[[str], None] = None,  # Fonction pour supprimer un fichier du disque
):
    results = {}
    pred_cache = OrderedDict()
    ref_cache = OrderedDict()

    for ref_time, group in forecast_index.groupby("forecast_reference_time"):
        results[ref_time] = {}
        for _, row in group.iterrows():
            lead_time = row["lead_time"]
            valid_time = row["valid_time"]
            pred_path = row["pred_data"]
            ref_path = row["ref_data"]

            # --- Gestion du cache et suppression des anciens fichiers ---
            if pred_path not in pred_cache:
                pred_ds = open_pred(pred_path)
                pred_cache[pred_path] = pred_ds
                # Si le fichier a été téléchargé, on peut le supprimer plus tard
                if len(pred_cache) > max_open_files:
                    old_path, _ = pred_cache.popitem(last=False)
                    if remove_file:
                        remove_file(old_path)
            else:
                pred_ds = pred_cache[pred_path]

            if ref_path not in ref_cache:
                ref_ds = open_ref(ref_path)
                ref_cache[ref_path] = ref_ds
                if len(ref_cache) > max_open_files:
                    old_path, _ = ref_cache.popitem(last=False)
                    if remove_file:
                        remove_file(old_path)
            else:
                ref_ds = ref_cache[ref_path]

            # (Optionnel) Sélectionne la tranche temporelle si besoin
            # pred_ds_sel = pred_ds.sel(time=valid_time, method="nearest")
            # ref_ds_sel = ref_ds.sel(time=valid_time, method="nearest")
            
            # Calcul des scores
            score_dict = compute_metrics(
                pred_ds, ref_ds,
                eval_variables=eval_variables,
                binning=binning,
                metrics=metrics,
            )
            results[ref_time][lead_time] = {
                "valid_time": valid_time,
                "scores": score_dict,
            }
    return results'''
    

def build_forecast_index_from_catalog(
    catalog: pd.DataFrame,
    init_date: str,
    end_date: str,
    forecast_time_col="date_start",
    valid_time_col="date_end",
    file_col="path",
    n_days_forecast: int = 10,
    n_days_interval: int = 7,
    lead_time_unit="days",
) -> pd.DataFrame:
    """
    Generate a forecast index mapping (init_time, lead_time) pairs to files 
    covering the corresponding valid_time (= init_time + lead_time).
    
    Parameters
    ----------
    catalog : pd.DataFrame
        Must contain at least the columns:
        - 'path' : path to the file
        - 'date_start' : earliest datetime in the file (forecast_reference_time)
        - 'date_end'   : latest datetime in the file
        
    init_date : str
        Start of the forecast initialization window
        
    end_date : str
        End of the forecast initialization window
        
    n_days_forecast : int
        Number of forecast days
        
    n_days_interval : int
        Interval between successive forecast initializations
        
    lead_time_unit : str
        Unit of lead time ("days" or "hours")
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - 'forecast_reference_time'
        - 'lead_time'
        - 'valid_time' (= forecast_reference_time + lead_time)
        - 'file' (path to file covering valid_time)
    """
    
    assert 'date_start' in catalog.columns and 'date_end' in catalog.columns and 'path' in catalog.columns, \
        "Catalog must contain 'date_start', 'date_end', and 'path' columns"

    init_step = pd.Timedelta(days=n_days_interval)
    max_lead_time = pd.Timedelta(days=n_days_forecast)
    init_start = pd.Timestamp(init_date)
    init_end = pd.Timestamp(end_date)
    lead_times = [i for i in range(n_days_forecast)]
    
    # Generate init times
    init_times = pd.date_range(start=init_start, end=init_end, freq=init_step)

    records = []

    for init_time in init_times:
        for lead_time in lead_times:
            if lead_time_unit == "days":
                valid_time = init_time + pd.Timedelta(days=lead_time)
                lead_time_value = lead_time 
            elif lead_time_unit == "hours":
                valid_time = init_time + pd.Timedelta(hours=lead_time)
                lead_time_value = lead_time 
            else:
                raise ValueError("lead_time_unit must be 'days' or 'hours'")

            # Select all files containing valid_time
            matching_files = catalog[
                (catalog["date_start"] <= valid_time) &
                (catalog["date_end"] >= valid_time)
            ]
            
            # Si plusieurs fichiers correspondent, choisir celui avec le même forecast_reference_time
            if len(matching_files) > 1:
                # Priorité 1: fichier dont date_start == init_time (même forecast_reference_time)
                exact_forecast_match = matching_files[matching_files["date_start"] == init_time]
                if not exact_forecast_match.empty:
                    selected_file = exact_forecast_match.iloc[0]
                else:
                    # Priorité 2: fichier avec date_start la plus proche (et <= init_time)
                    # pour éviter d'utiliser un forecast "du futur"
                    past_forecasts = matching_files[matching_files["date_start"] <= init_time]
                    if not past_forecasts.empty:
                        # Prendre le forecast le plus récent (date_start la plus proche de init_time)
                        selected_file = past_forecasts.loc[past_forecasts["date_start"].idxmax()]
                    else:
                        # Fallback: prendre le premier fichier disponible
                        selected_file = matching_files.iloc[0]
            elif len(matching_files) == 1:
                selected_file = matching_files.iloc[0]
            else:
                # Aucun fichier ne contient ce valid_time, passer au suivant
                continue
            
            records.append({
                "forecast_reference_time": init_time,
                "lead_time": lead_time_value,
                "valid_time": valid_time,
                "file": selected_file["path"]
            })
    return pd.DataFrame.from_records(records)


