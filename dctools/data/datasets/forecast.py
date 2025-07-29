from collections import OrderedDict
from datetime import timedelta
import os
from typing import Callable

import pandas as pd
import xarray as xr

def evaluate_forecast_leadtimes(
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
    return results
    

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



'''def infer_init_times(date_start, date_end, forecast_horizon, delta_t):
    """
    Parameters
    ----------
    date_start : pd.Timestamp
    date_end : pd.Timestamp
    forecast_horizon : int  # nb de lead days (N)
    delta_t : pd.Timedelta  # ex: Timedelta(days=1)

    Returns
    -------
    list of init_times such that for each init_time,
    the full forecast (init_time + [0...N] * delta_t) is within [date_start, date_end]
    """
    times = []
    current = date_start
    while current + forecast_horizon * delta_t <= date_end:
        times.append(current)
        current += delta_t
    return times'''


'''def build_forecast_index_from_catalog(
    catalog: pd.DataFrame,
    init_date: str,
    end_date: str,
    forecast_time_col="date_start",
    valid_time_col="date_end",
    file_col="path",
    n_days_forecast: int = 10,
    n_days_interval: int = 7,
    lead_time_unit="days",
    #time_delta = timedelta(days=1),
) -> pd.DataFrame:
    """
    Generate a forecast index mapping (init_time, lead_time) pairs to files 
    covering the corresponding valid_time (= init_time + lead_time).
    
    Parameters
    ----------
    catalog : pd.DataFrame
        Must contain at least the columns:
        - 'path' : path to the file
        - 'date_start' : earliest datetime in the file
        - 'date_end'   : latest datetime in the file
        
    init_start : pd.Timestamp
        Start of the forecast initialization window (included)
        
    init_end : pd.Timestamp
        End of the forecast initialization window (included)
        
    init_step : timedelta
        Interval between successive forecast initializations
        
    lead_times : list of timedelta, optional
        List of lead times to evaluate for each init_time.
        If not provided, will be inferred from max_lead_time.
        
    max_lead_time : timedelta, optional
        Maximum lead time to consider. Used only if lead_times is not provided.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - 'init_time'
        - 'lead_time'
        - 'valid_time' (= init_time + lead_time)
        - 'path' (to file covering valid_time)
    """
    
    assert 'date_start' in catalog.columns and 'date_end' in catalog.columns and 'path' in catalog.columns, \
        "Catalog must contain 'date_start', 'date_end', and 'path' columns"

    init_step=pd.Timedelta(days=n_days_interval)
    max_lead_time=pd.Timedelta(days=n_days_forecast)
    init_start=pd.Timestamp(init_date)
    init_end=pd.Timestamp(end_date)
    # lead_times = [timedelta(days=i) for i in range(n_days_forecast)]
    lead_times = [i for i in range(n_days_forecast)]
    # Generate init times
    init_times = pd.date_range(start=init_start, end=init_end, freq=init_step)

    # Generate lead_times if not provided
    records = []

    for init_time in init_times:
        for lead_time in lead_times:
            # valid_time = init_time + lead_time

            if lead_time_unit == "days":
                #valid_time = init_time + pd.Timedelta(days=lead_time)
                #lead_time = pd.Timedelta(days=lead_time)

                valid_time = init_time + pd.Timedelta(days=lead_time)
                lead_time_value = lead_time 

            elif lead_time_unit == "hours":
                #valid_time = init_time + pd.Timedelta(hours=lead_time)
                #lead_time = pd.Timedelta(days=lead_time)
                valid_time = init_time + pd.Timedelta(hours=lead_time)
                lead_time_value = lead_time 
            else:
                raise ValueError("lead_time_unit must be 'days' or 'hours'")

            # Select all files containing valid_time
            matching_files = catalog[
                (catalog["date_start"] <= valid_time) &
                (catalog["date_end"] >= valid_time)
            ]
            for _, row in matching_files.iterrows():
                records.append({
                    "forecast_reference_time": init_time,
                    "lead_time": lead_time_value,
                    "valid_time": valid_time,
                    "file": row["path"]
                })
    return pd.DataFrame.from_records(records)'''


'''def build_forecast_index_from_catalog(
    catalog_df: pd.DataFrame,
    init_date: str,
    end_date: str,
    forecast_time_col: str = "date_start",
    valid_time_col: str = "date_end",
    file_col: str = "path",
    n_days_forecast: int = 10,
    n_days_interval: int = 7,
    lead_time_unit: str = "days",  # ou "hours"
) -> pd.DataFrame:

def build_forecast_index(
        catalog_df,
        n_lead_days,
        delta_t
    ):
    """
    Parameters
    ----------
    catalog_df : pd.DataFrame with columns ['path', 'date_start', 'date_end']
    n_lead_days : int
    delta_t : pd.Timedelta

    Returns
    -------
    pd.DataFrame : forecast index with one row per (init_time, lead_day)
    """
    rows = []

    for _, row in catalog_df.iterrows():
        path = row["path"]
        date_start = pd.Timestamp(row["date_start"])
        date_end = pd.Timestamp(row["date_end"])

        init_times = infer_init_times(date_start, date_end, n_lead_days, delta_t)

        for init_time in init_times:
            for lead_day in range(n_lead_days):
                target_time = init_time + lead_day * delta_t
                rows.append({
                    "init_time": init_time,
                    "lead_day": lead_day,
                    "target_time": target_time,
                    "path": path
                })

    forecast_index = pd.DataFrame(rows)
    return forecast_index.sort_values(by=["init_time", "lead_day"]).reset_index(drop=True)'''



'''def build_forecast_index_from_catalog(
    catalog_df: pd.DataFrame,
    init_date: str,
    end_date: str,
    forecast_time_col: str = "date_start",
    valid_time_col: str = "date_end",
    file_col: str = "path",
    n_days_forecast: int = 10,
    n_days_interval: int = 7,
    lead_time_unit: str = "days",  # ou "hours"
) -> pd.DataFrame:
    """
    Construit un index forecast à partir d'un DataFrame issu du DCCatalog,
    en générant les n premiers lead times à partir de date_start et date_end.
    Les forecast_reference_time sont espacées de n_days_interval et comprises entre init_date et end_date.
    """
    rows = []
    
    # Convertir les dates limites en Timestamp
    init_date_ts = pd.to_datetime(init_date)
    end_date_ts = pd.to_datetime(end_date)
    
    # Obtenir toutes les dates de début uniques et les trier
    unique_dates = sorted(catalog_df[forecast_time_col].dropna().unique())
    
    # Filtrer les dates pour avoir un intervalle de n_days_interval
    filtered_dates = []
    if unique_dates:
        # Commencer par la première date dans la plage
        for date in unique_dates:
            current_date = pd.to_datetime(date)
            # Vérifier que la date est dans la plage [init_date, end_date]
            if init_date_ts <= current_date <= end_date_ts:
                if not filtered_dates:
                    # Première date dans la plage
                    filtered_dates.append(current_date)
                    last_selected = current_date
                else:
                    # Vérifier l'intervalle avec la dernière date sélectionnée
                    if (current_date - last_selected).days >= n_days_interval:
                        filtered_dates.append(current_date)
                        last_selected = current_date
    
    # Pour chaque date de référence forecast sélectionnée
    for init_time in filtered_dates:
        # Trouver la ligne correspondante dans le catalogue
        matching_rows = catalog_df[catalog_df[forecast_time_col] == init_time]
        if matching_rows.empty:
            continue
            
        row = matching_rows.iloc[0]
        # end_time = pd.to_datetime(row[valid_time_col])
        end_time = init_time + pd.Timedelta(days=n_days_forecast)
        
        # Générer les lead times pour cette date de référence
        for i in range(n_days_forecast):
            if lead_time_unit == "days":
                valid_time = init_time + pd.Timedelta(days=i)
                lead_time = i
            elif lead_time_unit == "hours":
                valid_time = init_time + pd.Timedelta(hours=i)
                lead_time = i
            else:
                raise ValueError("lead_time_unit must be 'days' or 'hours'")
            
            if valid_time > end_time:
                break
                
            rows.append({
                "forecast_reference_time": init_time,
                "lead_time": lead_time,
                "valid_time": valid_time,
                "file": row[file_col]
            })
    
    return pd.DataFrame(rows)'''
  

'''def build_forecast_index_from_catalog(
    catalog_df: pd.DataFrame,
    forecast_time_col: str = "date_start",
    valid_time_col: str = "date_end",
    file_col: str = "path",
    n_days_forecast: int = 10,
    n_days_interval: int = 7,
    lead_time_unit: str = "days",  # ou "hours"
) -> pd.DataFrame:
    """
    Construit un index forecast à partir d'un DataFrame issu du DCCatalog,
    en générant les n premiers lead times à partir de date_start et date_end.
    """
    rows = []
    for _, row in catalog_df.iterrows():
        init_time = pd.to_datetime(row[forecast_time_col])
        end_time = pd.to_datetime(row[valid_time_col])
        for i in range(n_days_forecast):
            if lead_time_unit == "days":
                valid_time = init_time + pd.Timedelta(days=i)
                lead_time = i
            elif lead_time_unit == "hours":
                valid_time = init_time + pd.Timedelta(hours=i)
                lead_time = i
            else:
                raise ValueError("lead_time_unit must be 'days' or 'hours'")
            if valid_time > end_time:
                break
            rows.append({
                "forecast_reference_time": init_time,
                "lead_time": lead_time,
                "valid_time": valid_time,
                "file": row[file_col]
            })
    return pd.DataFrame(rows)'''


'''
def build_forecast_index_from_catalog(
    catalog_df: pd.DataFrame,
    forecast_time_col: str = "date_start",
    valid_time_col: str = "date_end",
    file_col: str = "path",
    n_days_forecast: int = 10,
    lead_time_unit: str = "days",  # ou "hours"
    time_dim: str = "time",        # nom de la dimension temps dans les fichiers
    open_dataset: callable = None, # fonction pour ouvrir un fichier et lire les dates si besoin
) -> pd.DataFrame:
    """
    Construit un index forecast robuste à partir d'un DataFrame issu du DCCatalog.
    Gère les fichiers multi-leadtime (avec une dimension 'time' ou 'leadtime').
    """
    rows = []
    for _, row in catalog_df.iterrows():
        init_time = pd.to_datetime(row[forecast_time_col])
        # Cas 1 : le fichier contient déjà la date de validité (cas simple)
        if valid_time_col in row and pd.notnull(row[valid_time_col]):
            valid_time = pd.to_datetime(row[valid_time_col])
            if lead_time_unit == "days":
                lead_time = (valid_time - init_time).days
            elif lead_time_unit == "hours":
                lead_time = int((valid_time - init_time).total_seconds() // 3600)
            else:
                raise ValueError("lead_time_unit must be 'days' or 'hours'")
            if 0 <= lead_time < n_days_forecast:
                rows.append({
                    "forecast_reference_time": init_time,
                    "lead_time": int(lead_time),
                    "valid_time": valid_time,
                    "file": row[file_col]
                })
        # Cas 2 : le fichier contient plusieurs lead times (dimension 'time' ou 'leadtime')
        elif open_dataset is not None:
            ds = open_dataset(row[file_col])
            if time_dim in ds:
                for t in ds[time_dim].values:
                    valid_time = pd.to_datetime(t)
                    if lead_time_unit == "days":
                        lead_time = (valid_time - init_time).days
                    elif lead_time_unit == "hours":
                        lead_time = int((valid_time - init_time).total_seconds() // 3600)
                    else:
                        raise ValueError("lead_time_unit must be 'days' or 'hours'")
                    if 0 <= lead_time < n_days_forecast:
                        rows.append({
                            "forecast_reference_time": init_time,
                            "lead_time": int(lead_time),
                            "valid_time": valid_time,
                            "file": row[file_col]
                        })
    return pd.DataFrame(rows)
'''
