# -*- coding: UTF-8 -*-

"""Read config files and parse command-line arguments."""

from argparse import ArgumentParser, Namespace
from typing import Any, Dict, List, Optional

import sys

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
    "log_level": "DEBUG",
    # Format inspired by cargo / uv: dim timestamp, level badge, clean message.
    # File/line info is omitted from the default user-facing format – it's
    # available via --log-level DEBUG and the optional log-file sink.
    "log_format": (
        "<dim>{time:HH:mm:ss}</dim>"
        "  <level>{level: <8}</level>"
        "  {message}"
    ),
}


def _normalize_loguru_level(level: Any) -> Any:
    if level is None:
        return None
    if isinstance(level, int):
        return level
    if isinstance(level, str):
        lvl = level.strip().upper()
        return lvl
    return level


def _normalize_python_logging_level(level: Any) -> int | None:
    if level is None:
        return None
    if isinstance(level, int):
        return level
    if isinstance(level, str):
        import logging as _py_logging

        name = level.strip().upper()
        return int(getattr(_py_logging, name, _py_logging.WARNING))
    return None


def configure_logging_from_args(args: Namespace) -> None:
    """Configure Loguru handlers from args/config.

    Supported config styles:
    - Top-level `log_level: INFO`
    - Nested:
        logging:
          level: INFO
          format: "..."
          console: true
          colorize: true
          backtrace: false
          diagnose: false

    CLI `--logfile` adds a file sink at the same level.
    """
    cfg: Dict[str, Any] = {}
    if hasattr(args, "logging") and isinstance(args.logging, dict):
        cfg = args.logging

    level = cfg.get("level") if isinstance(cfg, dict) else None
    if level is None and hasattr(args, "log_level"):
        level = args.log_level
    if level is None:
        level = LOGGER_CONFIG.get("log_level", "INFO")

    fmt = cfg.get("format") if isinstance(cfg, dict) else None
    if not fmt:
        fmt = LOGGER_CONFIG.get("log_format")

    console_enabled = bool(cfg.get("console", True)) if isinstance(cfg, dict) else True
    colorize = bool(cfg.get("colorize", True)) if isinstance(cfg, dict) else True
    backtrace = bool(cfg.get("backtrace", False)) if isinstance(cfg, dict) else False
    diagnose = bool(cfg.get("diagnose", False)) if isinstance(cfg, dict) else False
    enqueue = bool(cfg.get("enqueue", True)) if isinstance(cfg, dict) else True

    level = _normalize_loguru_level(level) or "INFO"

    try:
        logger.remove()
    except Exception:
        pass

    if console_enabled:
        logger.add(
            sys.stderr,
            level=level,
            format=fmt,  # type: ignore[arg-type]
            colorize=colorize,
            backtrace=backtrace,
            diagnose=diagnose,
            enqueue=enqueue,
        )
    logfile = getattr(args, "logfile", None)
    if logfile:
        try:
            import datetime as _dt
            from pathlib import Path as _LogPath
            _lp = _LogPath(logfile)
            _ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            # Insert timestamp before the extension: dc2.log --> dc2_20260301_032630.log
            _timed_logfile = str(_lp.parent / f"{_lp.stem}_{_ts}{_lp.suffix}")
            logger.add(
                _timed_logfile,
                level=level,
                format=fmt,  # type: ignore[arg-type]
                colorize=False,
                backtrace=backtrace,
                diagnose=diagnose,
                enqueue=enqueue,
                mode="w",
            )
            logger.info(f"Log file: {_timed_logfile}")
        except Exception as exc:
            # Avoid crashing evaluation because of a bad path.
            logger.warning(f"Cannot add logfile sink {logfile!r}: {exc!r}")

    # Optional: also filter standard library logging (e.g. dask.distributed).
    # This controls messages that do NOT go through Loguru.
    py_level = None
    if isinstance(cfg, dict):
        py_level = cfg.get("python_level")
    py_level_n = _normalize_python_logging_level(py_level)
    if py_level_n is not None:
        import logging as _py_logging

        try:
            _py_logging.getLogger().setLevel(py_level_n)
            _py_logging.getLogger("distributed").setLevel(py_level_n)
            _py_logging.getLogger("dask").setLevel(py_level_n)
        except Exception:
            pass

    # Install a permanent filter on distributed.* loggers so benign INFO
    # connection-close chatter is suppressed even when distributed resets
    # its own log levels during cluster creation/teardown.
    try:
        from dctools.utilities.init_dask import _install_distributed_noise_filter
        _install_distributed_noise_filter()
    except Exception:
        pass


def parse_arguments(cli_args: Optional[List[str]] = None) -> Namespace:
    """Command-line argument parser.

    Args:
        cli_args (List[str], optional): List of arguments.

    Returns:
        Namespace: Namespace with parsed args.
    """
    parser = ArgumentParser(description="DC Tools Argument Parser")
    parser.add_argument(
        "-d",
        "--data_directory",
        type=str,
        help="Folder where to store downloaded data",
        required=True,
    )
    parser.add_argument(
        "-c",
        "--config_name",
        type=str,
        default=None,
        help="Folder where to store downloaded data",
    )
    (
        parser.add_argument(
            "-l",
            "--logfile",
            type=str,
            help="File where to store log info.",
        ),
    )
    (
        parser.add_argument(
            "-j",
            "--jsonfile",
            type=str,
            help="File where to store results.",
        ),
    )
    parser.add_argument(
        "-m",
        "--metric",
        type=str,
        help="Type of metric to compute."
        "Choose from list: [rmse, mld, geo, density, euclid_dist,"
        "energy_cascad, kinetic_energy, vorticity, mass_conservation]",
        default="rmse",
    )
    return parser.parse_args(args=cli_args)  # None defaults to sys.argv[1:]


def load_configs(config_filepath: str) -> Dict[str, Any]:
    """Load configuration from yaml file.

    Args:
        args (Namespace): parsed arguments Namespace.

    Raises:
        err: error

    Returns:
        Dict: Dict of cofig elements.
    """
    try:
        with open(config_filepath, "r") as fp:
            config: Dict[str, Any] = yaml.safe_load(fp)
    except Exception as err:
        logger.error(f"Error while loading config from {config_filepath}: {repr(err)}")
        raise
    return config


def load_args_and_config(
    config_filepath: str, args: Namespace | None = None
) -> Optional[Namespace]:
    """Load config file and parse command-line arguments.

    Args:
        args (Namespace, optional): Namespace of parsed arguments.

    Returns:
        args(Namespace): a Namespace with variables from config file and command-line.
    """
    if args is None:
        args = parse_arguments()

    try:
        # init logger and exception handler
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # initialize exception handler
        vars(args)["device"] = device

        if config_filepath:
            config = load_configs(config_filepath)
            # Auto-tune parallelism parameters based on machine capabilities.
            # Params already set to explicit numbers are never overridden.
            from dctools.utilities.machine_profile import auto_tune_config
            config = auto_tune_config(
                config,
                data_directory=getattr(args, "data_directory", None),
            )
            for key, value in config.items():
                vars(args)[key] = value

        # Configure Loguru once args+config have been merged.
        configure_logging_from_args(args)
        return args
    except Exception as err:
        print(f"App configuration has failed with error: {err}.")
        return None
