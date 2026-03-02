#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""DC2 evaluation class – DC2-specific wiring only.

All generic evaluation logic lives in :class:`.base.BaseDCEvaluation`.
"""

from argparse import Namespace

from dctools.processing.base import BaseDCEvaluation


class DC2Evaluation(BaseDCEvaluation):
    """Class that manages evaluation of Data Challenge 2."""

    def __init__(self, arguments: Namespace) -> None:
        """Init class.

        Args:
            arguments (Namespace): Namespace with config.
        """
        super().__init__(arguments)

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
