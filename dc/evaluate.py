#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Evaluation of a model against a given reference."""

import sys
from pathlib import Path

# Ensure the repository root is importable when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dctools.processing.runner import run_from_config, resolve_config_path  # noqa: E402
from dctools.utilities.args_config import _has_arg, parse_arguments  # noqa: E402

# Absolute path to the leaderboard display config shipped with this repo.
_LEADERBOARD_CONFIG_YAML = PROJECT_ROOT / "dc" / "config" / "leaderboard_config.yaml"

# Default config name (without .yaml) – overridable via --config_name.
DEFAULT_CONFIG_NAME = "dc2"


def _inject_default_paths(argv: list) -> None:
    """Inject ``--data_directory`` and ``--logfile`` defaults when absent."""
    default_output_dir = PROJECT_ROOT / "output"
    default_log_dir = default_output_dir / "logs"
    default_logfile = default_log_dir / "dc2.log"

    if not _has_arg(argv, "-d", "--data_directory"):
        argv.extend(["--data_directory", str(default_output_dir)])

    if not _has_arg(argv, "-l", "--logfile"):
        default_log_dir.mkdir(parents=True, exist_ok=True)
        argv.extend(["--logfile", str(default_logfile)])


if __name__ == "__main__":
    _inject_default_paths(sys.argv)
    cli_args = parse_arguments()
    # Inject the leaderboard config path so DC2Evaluation can find it without
    # relying on a relative path baked into dc2.py.
    if not getattr(cli_args, "leaderboard_config", None):
        vars(cli_args)["leaderboard_config"] = str(_LEADERBOARD_CONFIG_YAML)
    config_path = resolve_config_path(DEFAULT_CONFIG_NAME, cli_args=cli_args)
    sys.exit(run_from_config(config_path, cli_args=cli_args))
