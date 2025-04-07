#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Read config files and parse command-line arguments."""

from argparse import ArgumentParser, Namespace
import os
from typing import Dict, List, Optional, Tuple

import torch
import yaml

from dctools.dcio.dclogger import DCLogger
from dctools.utilities.errors import DCExceptionHandler


def parse_arguments(cli_args: Optional[List[str]] = None) -> Namespace:
    """Command-line argument parser.

    Args:
        cli_args (List[str], optional): List of arguments.

    Returns:
        Namespace: Namespace with parsed args.
    """
    folder_base = os.path.join("/home", "k24aitmo", "IMT", "software", "dc-tools")
    parser = ArgumentParser()
    parser = ArgumentParser(description='Run DC1 Evaluation on Glorys data')
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
        default=os.path.join(folder_base, "tests", "logs", "result_logs.json")
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
    config_filepath: str, exception_handler: DCExceptionHandler
) -> Dict:
    """Load configuration from yaml file.

    Args:
        args (Namespace): parsed arguments Namespace.
        exception_handler (DCExceptionHandler):

    Raises:
        err: error

    Returns:
        Dict: Dict of cofig elements.
    """
    try:
        with open(config_filepath, 'r') as fp:
            #print(f"OPEN FILE: {config_filepath}")
            config = yaml.safe_load(fp)
            if('list_glonet_start_dates' in config.keys()):
                list_dates = config['list_glonet_start_dates'].split(',')
                config['list_glonet_start_dates'] = list_dates
            #print(f"CONFIG: {config}")
    except Exception as err:
        exception_handler.handle_exception(
            err,
            f"Error while loading config from: {config_filepath}."
        )
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
        #print("ENTER CONF")
        # init logger and exception handler
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger_instance = DCLogger(
            name="DCLogger", logfile=args.logfile, jsonfile=args.jsonfile
        )
        dclogger = logger_instance.get_logger()
        json_logger = logger_instance.get_json_logger()
        # initialize exception handler
        exception_handler = DCExceptionHandler(dclogger)
        vars(args)['device'] = device
        vars(args)['dclogger'] = dclogger
        vars(args)['json_logger'] = json_logger
        vars(args)['exception_handler'] = exception_handler

        # TODO : Put these in config files:
        patch_size: Optional[Tuple[float, float]] = None
        stride_size: Optional[Tuple[float, float]] = None
        vars(args)['patch_size'] = patch_size
        vars(args)['stride_size'] = stride_size

        if config_filepath:
            config = load_configs(config_filepath, exception_handler)
            for key, value in config.items():
                vars(args)[key] = value
        #print('LOAD CONF OK')
        return args
    except Exception as err:
        print(f"App configuration has failed with error: {err}.")
        return None
