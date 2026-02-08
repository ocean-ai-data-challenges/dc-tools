#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Format conversion utilities for metrics results."""

from typing import Dict, List, Optional, Union
from loguru import logger


def convert_format1_to_format2(
    format1_results: Union[Dict[str, List[float]], Dict[str, Dict[str, float]]],
    metric_name: Optional[Optional[str]] = None
) -> List[Dict[str, Union[str, float]]]:
    """
    Convert Format1 results to Format2.

    Args:
        format1_results: Format1 Results
            Simple format: {'Variable name': [value], ...}
            Nested format: {'metric_name': {'Variable name': value, ...}, ...}
        metric_name (str, optional): Metric name (e.g., 'rmse', 'mae')
            Required for simple format, optional for nested format

    Returns:
        List[Dict[str, Union[str, float]]]: Format2 Results
            Format: [{'Metric': 'metric_name', 'Variable': 'variable_name', 'Value': value}, ...]

    Examples:
        >>> # Simple format
        >>> format1 = {
        ...     'Surface salinity': [0.7800133290485501],
        ...     '50m salinity': [0.36502441182776185]
        ... }
        >>> convert_format1_to_format2(format1, 'rmse')
        [
            {'Metric': 'rmse', 'Variable': 'Surface salinity', 'Value': 0.7800133290485501},
            {'Metric': 'rmse', 'Variable': '50m salinity', 'Value': 0.36502441182776185}
        ]

        >>> # Nested format
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

    format2_results: List[Dict[str, Union[str, float]]] = []

    # Determine the format (simple or nested)
    if format1_results:
        first_value = next(iter(format1_results.values()))
        is_nested_format = isinstance(first_value, dict)

        if is_nested_format:
            # Nested format: {'metric_name': {'Variable name': value, ...}, ...}
            for metric, variables_dict in format1_results.items():
                if not isinstance(variables_dict, dict):
                    logger.warning(
                        f"Expected dict for metric '{metric}', got {type(variables_dict)}"
                    )
                    continue

                for variable_name, var_value in variables_dict.items():
                    if var_value is not None:
                        format2_results.append({
                            'Metric': str(metric),
                            'Variable': str(variable_name),
                            'Value': var_value
                        })
        else:
            # Simple format: {'Variable name': [value], ...} or {'Variable name': value, ...}
            if metric_name is None:
                logger.error("metric_name is required for simple format")
                return []

            if not isinstance(metric_name, str):
                logger.error(f"metric_name must be a string, got {type(metric_name)}")
                return []

            for variable_name, values in format1_results.items():
                # Handle lists or direct values
                result_value: Union[int, float]
                if isinstance(values, list):
                    if len(values) == 0:
                        logger.warning(f"Empty values list for variable '{variable_name}'")
                        continue
                    result_value = values[0] if isinstance(values[0], (int, float)) else 0.0
                else:
                    result_value = values if isinstance(values, (int, float)) else 0.0

                format2_results.append({
                    'Metric': str(metric_name),
                    'Variable': str(variable_name),
                    'Value': result_value
                })

    return format2_results


def convert_format2_to_format1(
    format2_results: List[Dict[str, Union[str, float]]]
) -> Dict[str, List[float]]:
    """
    Convert Format2 results to Format1.

    Args:
        format2_results (List[Dict[str, Union[str, float]]]): Format2 Results
            Format: [{'Metric': 'metric_name', 'Variable': 'variable_name', 'Value': value}, ...]

    Returns:
        Dict[str, List[float]]: Format1 Results
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

    format1_results: Dict[str, List[float]] = {}

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

        # Convert to float if possible
        try:
            float_value = float(value) if not isinstance(value, str) else float(value)
        except (ValueError, TypeError):
            logger.warning(f"Cannot convert value {value} to float for variable {variable_name}")
            continue

        if variable_name in format1_results:
            format1_results[str(variable_name)].append(float_value)
        else:
            format1_results[str(variable_name)] = [float_value]

    return format1_results


def group_format2_by_metric(
    format2_results: List[Dict[str, Union[str, float]]]
) -> Dict[str, List[Dict[str, Union[str, float]]]]:
    """
    Group Format2 results by metric.

    Args:
        format2_results (List[Dict[str, Union[str, float]]]): Format2 Results

    Returns:
        Dict[str, List[Dict[str, Union[str, float]]]]: Results grouped by metric

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
    grouped_results: Dict[str, List[Dict[str, Union[str, float]]]] = {}

    for result in format2_results:
        if not isinstance(result, dict) or 'Metric' not in result:
            logger.warning(f"Invalid result format: {result}")
            continue

        metric = result['Metric']
        if not isinstance(metric, str):
            logger.warning(f"Metric must be a string, got {type(metric)}")
            continue

        if metric not in grouped_results:
            grouped_results[metric] = []
        grouped_results[metric].append(result)

    return grouped_results


def filter_format2_by_variables(
    format2_results: List[Dict[str, Union[str, float]]],
    variables: List[str]
) -> List[Dict[str, Union[str, float]]]:
    """
    Filter Format2 results to keep only specific variables.

    Args:
        format2_results (List[Dict[str, Union[str, float]]]): Format2 Results
        variables (List[str]): List of variable names to keep

    Returns:
        List[Dict[str, Union[str, float]]]: Filtered results
    """
    filtered_results: List[Dict[str, Union[str, float]]] = []

    for result in format2_results:
        if not isinstance(result, dict) or 'Variable' not in result:
            logger.warning(f"Invalid result format: {result}")
            continue

        if result['Variable'] in variables:
            filtered_results.append(result)

    return filtered_results
