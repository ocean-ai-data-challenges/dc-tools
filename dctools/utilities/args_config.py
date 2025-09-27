#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Read config files and parse command-line arguments."""

from argparse import ArgumentParser, Namespace
from typing import Dict, List, Optional, Tuple

from loguru import logger
import torch
import yaml


TIME_VARIABLES = [
    "start_times",
    "end_times",
    "lead_time_start",
    "lead_time_stop",
    "lead_time_frequency",
]

LOGGER_CONFIG = {
    'log_level': "DEBUG",
    'log_format': "<green>{time:YYYY-MM-DD HH:mm:ss.SSS zz}</green> | <level>{level: <8}</level> | <yellow>Line {line: >4} ({file}):</yellow> <b>{message}</b>"
}

def parse_arguments(cli_args: Optional[List[str]] = None) -> Namespace:
    """Command-line argument parser.

    Args:
        cli_args (List[str], optional): List of arguments.

    Returns:
        Namespace: Namespace with parsed args.
    """
    parser = ArgumentParser(description='DC Tools Argument Parser')
    parser.add_argument(
        '-d', '--data_directory', type=str,
        help="Folder where to store downloaded data",
        required=True,
    )
    parser.add_argument(
        '-c', '--config_name', type=str,
        default=None,
        help="Folder where to store downloaded data",
    )
    parser.add_argument(
        '-l', '--logfile', type=str,
        help="File where to store log info.",
    ),
    parser.add_argument(
        '-j', '--jsonfile', type=str,
        help="File where to store results.",
    ),
    parser.add_argument(
        '-m', '--metric', type=str,
        help="Type of metric to compute."
        "Choose from list: [rmse, mld, geo, density, euclid_dist,"
        "energy_cascad, kinetic_energy, vorticity, mass_conservation]",
        default="rmse",
    )
    return parser.parse_args(args=cli_args)  # None defaults to sys.argv[1:]


def load_configs(
    config_filepath: str
) -> Dict:
    """Load configuration from yaml file.

    Args:
        args (Namespace): parsed arguments Namespace.

    Raises:
        err: error

    Returns:
        Dict: Dict of cofig elements.
    """
    try:
        with open(config_filepath, 'r') as fp:
            config = yaml.safe_load(fp)
            # for time_var in TIME_VARIABLES:
            #    if time_var in config.keys():
            #        config[time_var] = config[time_var]
    except Exception as err:
        logger.error(
            f"Error while loading config from {config_filepath}: {repr(err)}"
        )
        raise
    return config


def load_args_and_config(
    config_filepath: str, args: Namespace = parse_arguments()
) -> Optional[Namespace]:
    """Load config file and parsing comman-line arguments.

    Args:
        args (Namespace, optional): Namespace of parsed arguments.

    Returns:
        args(Namespace): a Namespace with variables from config file and command-line 
    """
    try:
        # init logger and exception handler
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # initialize exception handler
        vars(args)['device'] = device

        if config_filepath:
            config = load_configs(config_filepath)
            for key, value in config.items():
                vars(args)[key] = value
        return args
    except Exception as err:
        print(f"App configuration has failed with error: {err}.")
        return None
