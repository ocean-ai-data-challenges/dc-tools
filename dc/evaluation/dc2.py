#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""DC2 evaluation class – DC2-specific wiring only.

All generic evaluation logic lives in :class:`.base.BaseDCEvaluation`.
"""

from argparse import Namespace
from pathlib import Path

import yaml

from dctools.processing.base import BaseDCEvaluation


class DC2Evaluation(BaseDCEvaluation):
    """Class that manages evaluation of Data Challenge 2."""

    CHALLENGE_NAME = "DC2"

    def __init__(self, arguments: Namespace) -> None:
        """Init class.

        Args:
            arguments (Namespace): Namespace with config.
        """
        super().__init__(arguments)

        # Load leaderboard display config (path injected by evaluate.py into args).
        _lb_config_path = getattr(arguments, "leaderboard_config", None)
        if _lb_config_path:
            _lb_yaml = Path(_lb_config_path)
            if not _lb_yaml.is_absolute():
                _lb_yaml = Path.cwd() / _lb_yaml
            if _lb_yaml.is_file():
                try:
                    self.leaderboard_custom_config = (
                        yaml.safe_load(_lb_yaml.read_text(encoding="utf-8")) or {}
                    )
                except Exception:  # noqa: BLE001
                    pass  # leave self.leaderboard_custom_config as None (base default)

        # Use dataset_references from YAML config if provided (e.g. submission
        # mode), otherwise fall back to the hardcoded DC2 default.
        config_refs = getattr(self.args, "dataset_references", None)
        if config_refs and isinstance(config_refs, dict):
            self.dataset_references = config_refs
        else:
            self.dataset_references = {
                "glonet": [
                    "saral"  # "argo_profiles", "glorys", "jason3", "saral", "swot" # "argo_velocities",
                    # "SSS_fields", "SST_fields",
                ],
            }
        self._build_all_datasets()
        self._init_cluster()
