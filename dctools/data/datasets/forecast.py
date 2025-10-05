from collections import OrderedDict
from datetime import timedelta
import os
from typing import Callable

from loguru import logger
import pandas as pd



def build_forecast_index_from_catalog(
    catalog: pd.DataFrame,
    init_date: str,
    end_date: str,
    start_time_col="date_start",
    end_time_col="date_end",
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
        Interval (in days) between successive forecast initializations
        
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

    # Vérification des colonnes
    assert start_time_col in catalog.columns and \
           end_time_col in catalog.columns and \
           file_col in catalog.columns, \
           f"Catalog must contain '{start_time_col}', '{end_time_col}', and '{file_col}' columns"

    init_start = pd.Timestamp(init_date)
    init_end = pd.Timestamp(end_date)

    # Dernière date de début possible pour qu'un forecast soit complet
    last_possible_init = init_end - pd.Timedelta(days=n_days_forecast - 1)

    # Déterminer le premier init_time valide (date disponible dans le modèle à évaluer)

    available_inits = catalog[catalog[start_time_col] >= init_start][start_time_col]
    if available_inits.empty:
        raise ValueError("No available initialization times in the catalog after init_date")
    init_start = available_inits.min()

    # Générer les dates de début des forecasts (chevauchantes si n_days_interval < n_days_forecast)
    init_times = pd.date_range(start=init_start, end=last_possible_init, freq=pd.Timedelta(days=n_days_interval))

    records = []

    for init_time in init_times:
        if init_time not in catalog[start_time_col].values:
            continue
        complete_sequence = True
        sequence_records = []

        for lead_time in range(n_days_forecast):
            if lead_time_unit == "days":
                valid_time = init_time + pd.Timedelta(days=lead_time)
                valid_time_plus1 = valid_time + pd.Timedelta(days=1)
                lead_time_value = lead_time
            elif lead_time_unit == "hours":
                valid_time = init_time + pd.Timedelta(hours=lead_time)
                valid_time_plus1 = valid_time + pd.Timedelta(hours=1)
                lead_time_value = lead_time
            else:
                raise ValueError("lead_time_unit must be 'days' or 'hours'")

            # Sélection des fichiers contenant valid_time
            matching_files = catalog[
                (catalog[start_time_col] <= valid_time) &
                (catalog[end_time_col] >= valid_time_plus1)
            ]

            if matching_files.empty:
                complete_sequence = False
                break

            # Priorité : fichier dont date_start == init_time, sinon le plus récent <= init_time
            exact_match = matching_files[matching_files[start_time_col] == init_time]
            if not exact_match.empty:
                selected_file = exact_match.iloc[0]
            else:
                past_forecasts = matching_files[matching_files[start_time_col] <= init_time]
                if not past_forecasts.empty:
                    selected_file = past_forecasts.loc[past_forecasts[start_time_col].idxmax()]
                else:
                    selected_file = matching_files.iloc[0]

            sequence_records.append({
                "forecast_reference_time": init_time,
                "lead_time": lead_time_value,
                "valid_time": valid_time,
                "file": selected_file[file_col]
            })

        # Ajouter la séquence seulement si complète
        if complete_sequence and len(sequence_records) == n_days_forecast:
            records.extend(sequence_records)

    df_result = pd.DataFrame.from_records(records)

    logger.info(f"Built forecast index with {len(df_result)} entries "
                f"({len(df_result)//n_days_forecast if not df_result.empty else 0} complete forecast sequences)")

    return df_result







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
        # check if a complete sequence can be built for this init_time
        complete_sequence = True
        sequence_records = []
        
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
                # Aucun fichier ne contient ce valid_time, séquence incomplète
                complete_sequence = False
                break
            
            sequence_records.append({
                "forecast_reference_time": init_time,
                "lead_time": lead_time_value,
                "valid_time": valid_time,
                "file": selected_file["path"]
            })
        
        # Ajouter la séquence seulement si elle est complète (tous les lead_times de 0 à n_days_forecast-1)
        if complete_sequence and len(sequence_records) == n_days_forecast:
            # Vérifier que la séquence commence bien par lead_time=0 et finit par lead_time=n_days_forecast-1
            lead_times_in_sequence = [r["lead_time"] for r in sequence_records]
            expected_lead_times = list(range(n_days_forecast))
            
            if lead_times_in_sequence == expected_lead_times:
                records.extend(sequence_records)
            else:
                logger.warning(f"Incomplete lead time sequence for init_time {init_time}: "
                             f"got {lead_times_in_sequence}, expected {expected_lead_times}")
    
    # Créer le DataFrame final
    df_result = pd.DataFrame.from_records(records)
    
    # Vérification finale : s'assurer qu'on n'a que des séquences complètes
    if not df_result.empty:
        # Grouper par forecast_reference_time et vérifier que chaque groupe a exactement n_days_forecast entrées
        grouped = df_result.groupby('forecast_reference_time')
        valid_groups = []
        
        for init_time, group in grouped:
            if len(group) == n_days_forecast:
                # Vérifier que les lead_times vont de 0 à n_days_forecast-1
                lead_times_sorted = sorted(group['lead_time'].tolist())
                expected_lead_times = list(range(n_days_forecast))
                
                if lead_times_sorted == expected_lead_times:
                    valid_groups.append(group)
                else:
                    logger.warning(f"Removing incomplete sequence for {init_time}: "
                                 f"lead_times {lead_times_sorted} != expected {expected_lead_times}")
            else:
                logger.warning(f"Removing incomplete sequence for {init_time}: "
                             f"got {len(group)} entries, expected {n_days_forecast}")
        
        if valid_groups:
            df_result = pd.concat(valid_groups, ignore_index=True)
        else:
            df_result = pd.DataFrame(columns=['forecast_reference_time', 'lead_time', 'valid_time', 'file'])
            logger.warning("No complete forecast sequences found after filtering")
    
    logger.info(f"Built forecast index with {len(df_result)} entries "
               f"({len(df_result)//n_days_forecast if not df_result.empty else 0} complete forecast sequences)")
    
    return df_result'''
