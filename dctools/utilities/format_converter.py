#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Format conversion utilities for metrics results."""

from typing import Any, Dict, List, Union
from loguru import logger


def convert_format1_to_format2(
    format1_results: Union[Dict[str, List[float]], Dict[str, Dict[str, float]]], 
    metric_name: str = None
) -> List[Dict[str, Union[str, float]]]:
    """
    Convertit les résultats du Format1 au Format2.
    
    Args:
        format1_results: Résultats au Format1
            Format simple: {'Variable name': [value], ...}
            Format imbriqué: {'metric_name': {'Variable name': value, ...}, ...}
        metric_name (str, optional): Nom de la métrique (ex: 'rmse', 'mae', etc.)
            Requis pour le format simple, optionnel pour le format imbriqué
        
    Returns:
        List[Dict[str, Union[str, float]]]: Résultats au Format2
            Format: [{'Metric': 'metric_name', 'Variable': 'variable_name', 'Value': value}, ...]
    
    Examples:
        >>> # Format simple
        >>> format1 = {
        ...     'Surface salinity': [0.7800133290485501], 
        ...     '50m salinity': [0.36502441182776185]
        ... }
        >>> convert_format1_to_format2(format1, 'rmse')
        [
            {'Metric': 'rmse', 'Variable': 'Surface salinity', 'Value': 0.7800133290485501},
            {'Metric': 'rmse', 'Variable': '50m salinity', 'Value': 0.36502441182776185}
        ]
        
        >>> # Format imbriqué
        >>> format1_nested = {
        ...     'rmsd': {
        ...         'Surface salinity': 0.7957517181075711,
        ...         '50m salinity': 0.35141580091326013
        ...     }
        ... }
        >>> convert_format1_to_format2(format1_nested)
        [
            {'Metric': 'rmsd', 'Variable': 'Surface salinity', 'Value': 0.7957517181075711},
            {'Metric': 'rmsd', 'Variable': '50m salinity', 'Value': 0.35141580091326013}
        ]
    """
    if not isinstance(format1_results, dict):
        logger.error(f"format1_results must be a dictionary, got {type(format1_results)}")
        return []
    
    format2_results = []
    
    # Déterminer le format (simple ou imbriqué)
    if format1_results:
        first_value = next(iter(format1_results.values()))
        is_nested_format = isinstance(first_value, dict)
        
        if is_nested_format:
            # Format imbriqué: {'metric_name': {'Variable name': value, ...}, ...}
            for metric, variables_dict in format1_results.items():
                if not isinstance(variables_dict, dict):
                    logger.warning(f"Expected dict for metric '{metric}', got {type(variables_dict)}")
                    continue
                    
                for variable_name, value in variables_dict.items():
                    if value is not None:
                        format2_results.append({
                            'Metric': metric,
                            'Variable': variable_name,
                            'Value': value
                        })
        else:
            # Format simple: {'Variable name': [value], ...} ou {'Variable name': value, ...}
            if metric_name is None:
                logger.error("metric_name is required for simple format")
                return []
                
            if not isinstance(metric_name, str):
                logger.error(f"metric_name must be a string, got {type(metric_name)}")
                return []
            
            for variable_name, values in format1_results.items():
                # Gérer les listes ou les valeurs directes
                if isinstance(values, list):
                    if len(values) == 0:
                        logger.warning(f"Empty values list for variable '{variable_name}'")
                        continue
                    value = values[0]
                else:
                    value = values
                
                if value is not None:
                    format2_results.append({
                        'Metric': metric_name,
                        'Variable': variable_name,
                        'Value': value
                    })
    
    return format2_results


def convert_format2_to_format1(
    format2_results: List[Dict[str, Union[str, float]]]
) -> Dict[str, List[float]]:
    """
    Convertit les résultats du Format2 au Format1.
    
    Args:
        format2_results (List[Dict[str, Union[str, float]]]): Résultats au Format2
            Format: [{'Metric': 'metric_name', 'Variable': 'variable_name', 'Value': value}, ...]
        
    Returns:
        Dict[str, List[float]]: Résultats au Format1
            Format: {'Variable name': [value], ...}
    
    Example:
        >>> format2 = [
        ...     {'Metric': 'rmse', 'Variable': 'Surface salinity', 'Value': 0.78},
        ...     {'Metric': 'rmse', 'Variable': '50m salinity', 'Value': 0.365}
        ... ]
        >>> convert_format2_to_format1(format2)
        {'Surface salinity': [0.78], '50m salinity': [0.365]}
    """
    if not isinstance(format2_results, list):
        logger.error(f"format2_results must be a list, got {type(format2_results)}")
        return {}
    
    format1_results = {}
    
    for result_dict in format2_results:
        if not isinstance(result_dict, dict):
            logger.warning(f"Each result should be a dictionary, got {type(result_dict)}")
            continue
            
        required_keys = {'Metric', 'Variable', 'Value'}
        if not required_keys.issubset(result_dict.keys()):
            missing_keys = required_keys - set(result_dict.keys())
            logger.warning(f"Missing required keys {missing_keys} in result: {result_dict}")
            continue
        
        variable_name = result_dict['Variable']
        value = result_dict['Value']
        
        if variable_name in format1_results:
            format1_results[variable_name].append(value)
        else:
            format1_results[variable_name] = [value]
    
    return format1_results


def group_format2_by_metric(
    format2_results: List[Dict[str, Union[str, float]]]
) -> Dict[str, List[Dict[str, Union[str, float]]]]:
    """
    Groupe les résultats Format2 par métrique.
    
    Args:
        format2_results (List[Dict[str, Union[str, float]]]): Résultats au Format2
        
    Returns:
        Dict[str, List[Dict[str, Union[str, float]]]]: Résultats groupés par métrique
        
    Example:
        >>> format2 = [
        ...     {'Metric': 'rmse', 'Variable': 'Surface salinity', 'Value': 0.78},
        ...     {'Metric': 'mae', 'Variable': 'Surface salinity', 'Value': 0.65},
        ...     {'Metric': 'rmse', 'Variable': '50m salinity', 'Value': 0.365}
        ... ]
        >>> group_format2_by_metric(format2)
        {
            'rmse': [
                {'Metric': 'rmse', 'Variable': 'Surface salinity', 'Value': 0.78},
                {'Metric': 'rmse', 'Variable': '50m salinity', 'Value': 0.365}
            ],
            'mae': [
                {'Metric': 'mae', 'Variable': 'Surface salinity', 'Value': 0.65}
            ]
        }
    """
    grouped_results = {}
    
    for result in format2_results:
        if not isinstance(result, dict) or 'Metric' not in result:
            logger.warning(f"Invalid result format: {result}")
            continue
            
        metric = result['Metric']
        if metric not in grouped_results:
            grouped_results[metric] = []
        grouped_results[metric].append(result)
    
    return grouped_results


def filter_format2_by_variables(
    format2_results: List[Dict[str, Union[str, float]]],
    variables: List[str]
) -> List[Dict[str, Union[str, float]]]:
    """
    Filtre les résultats Format2 pour ne garder que certaines variables.
    
    Args:
        format2_results (List[Dict[str, Union[str, float]]]): Résultats au Format2
        variables (List[str]): Liste des noms de variables à conserver
        
    Returns:
        List[Dict[str, Union[str, float]]]: Résultats filtrés
    """
    filtered_results = []
    
    for result in format2_results:
        if not isinstance(result, dict) or 'Variable' not in result:
            logger.warning(f"Invalid result format: {result}")
            continue
            
        if result['Variable'] in variables:
            filtered_results.append(result)
    
    return filtered_results
