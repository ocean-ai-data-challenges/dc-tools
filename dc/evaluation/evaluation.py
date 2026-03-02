#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""DC2 evaluation class – DC2-specific wiring only.

All generic evaluation logic lives in :class:`.base.BaseDCEvaluation`.
"""

from argparse import Namespace
from pathlib import Path

import yaml

from dctools.processing.base import BaseDCEvaluation

# Path to the leaderboard text/name customisation shipped with this package.
_LEADERBOARD_TEXTS_YAML = Path(__file__).parent.parent / "config" / "leaderboard_texts.yaml"


class DC2Evaluation(BaseDCEvaluation):
    """Class that manages evaluation of Data Challenge 2."""

    def __init__(self, arguments: Namespace) -> None:
        """Init class.

        Args:
            arguments (Namespace): Namespace with config.
        """
        super().__init__(arguments)

        # Load leaderboard display config from the YAML shipped in dc/config/.
        if _LEADERBOARD_TEXTS_YAML.is_file():
            try:
                self.leaderboard_custom_config = (
                    yaml.safe_load(_LEADERBOARD_TEXTS_YAML.read_text(encoding="utf-8")) or {}
                )
            except Exception:  # noqa: BLE001
                pass  # leave self.leaderboard_custom_config as None (base default)

        self.dataset_references = {
            "glonet": [
                "argo_profiles", "glorys", "jason3", "saral", "swot", # "argo_velocities",
                # "SSS_fields", "SST_fields",
            ],
        }
        self.all_datasets = list(
            set(
                list(self.dataset_references.keys())
                + [item for sublist in self.dataset_references.values() for item in sublist]
            )
        )
        self._init_cluster()
