#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Evaluation of a model against a given reference."""

import sys
from pathlib import Path

# Ensure the repository root is importable when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dctools.processing.runner import run_from_cli  # noqa: E402


if __name__ == "__main__":
    sys.exit(run_from_cli(default_config_name="dc2"))
