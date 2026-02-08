#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Evaluation of a model against a given reference."""

import os
import sys

from dask.distributed import performance_report
from dctools.utilities.args_config import load_args_and_config

from dc.evaluation.evaluation import DC2Evaluation


def main() -> int:
    """Main function.

    Args:
        args (Namespace, optional): Namespace of parsed arguments.

    Returns:
        int: return code.
    """
    try:
        config_name = "dc2"
        config_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'config',
            f"{config_name}.yaml",
        )
        args = load_args_and_config(config_path)
        if args is None:
            print("Config loading failed.")
            return 1

        vars(args)['regridder_weights'] = os.path.join(args.data_directory, 'weights')
        vars(args)['catalog_dir'] = os.path.join(args.data_directory, "catalogs")
        vars(args)['result_dir'] = os.path.join(args.data_directory, "results")

        if os.path.exists(args.regridder_weights):
            os.remove(args.regridder_weights)

        os.makedirs(args.catalog_dir, exist_ok=True)
        os.makedirs(args.result_dir, exist_ok=True)

        evaluator_instance = DC2Evaluation(args)

        report_path = os.path.join(args.result_dir, "dask-report.html")
        logger_msg = f"Generating Dask performance report at: {report_path}"
        print(logger_msg)

        with performance_report(filename=report_path):
            evaluator_instance.run_eval()

        evaluator_instance.close()

        print("Evaluation has finished successfully.")
        return 0

    except KeyboardInterrupt:
        # raise Exception("Manual abort.")
        print("Manual abort.")
        return 1
    except SystemExit:
        # SystemExit is raised when the user calls sys.exit()
        # or when an error occurs in the argument parsing
        print("SystemExit.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
