#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Read config files and parse command-line arguments."""

from argparse import ArgumentParser, Namespace
import os
import sys
from typing import Dict, List, Optional, Tuple

import boto3
from botocore import UNSIGNED
from botocore.config import Config
import dask
import torch
import yaml

from dctools.dcio.dclogger import DCLogger
from dctools.metrics.oceanbench_metrics import OceanbenchMetrics
from dctools.utilities.errors import DCExceptionHandler
from dctools.utilities.net_utils import CMEMSManager


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
        '-l', '--jsonfile', type=str,
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


def load_configs(args: Namespace, exception_handler: DCExceptionHandler) -> Dict:
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
        config_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'config',
            f"{args.config_name}.yaml",
        )
        with open(config_path, 'r') as file_pointer:
            config = yaml.safe_load(file_pointer)
    except Exception as err:
        exception_handler.handle_exception(
            err, f"Error while loading config from: {config_path}."
        )
    return config


def load_args_and_config(args: Namespace = parse_arguments()) -> Optional[Namespace]:
    """Main function.

    Args:
        args (Namespace, optional): Namespace of parsed arguments.

    Returns:
        args(Namespace): a Namespace with variables from config file and command-line 
    """
    try:
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

        if args.config_name:
            config = load_configs(args, exception_handler)
            for key, value in config.items():
                vars(args)[key] = value
        return args
    except Exception as err:
        exception_handler.handle_exception(err, "App configuration has failed.")
        return None
