"""CLI runner for DC evaluations.

This module centralizes the boilerplate needed to run an evaluation from a YAML
config file shipped by the challenge package (dc2/config/*.yaml):
- ensure the project root is on sys.path
- set HDF5/NetCDF/argopy environment variables early
- load CLI args + YAML config into a Namespace
- create derived directories (catalogs/results) and paths
- run the evaluation with a Dask performance report

Keeping this logic here makes dc2/evaluate.py a thin wrapper.
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path
from typing import Optional, Type, Any

# Remove Loguru's default DEBUG handler immediately.
# Loguru installs a stderr/DEBUG sink the moment `loguru` is first imported.
# Without this, all DEBUG messages emitted during the import phase (dask,
# oceanbench, etc.) are visible even when the config sets level=INFO.
# configure_logging_from_args() will add the properly-configured sinks later.
try:
    from loguru import logger as _early_logger
    _early_logger.remove()
except Exception:
    pass


def _project_root() -> Path:
    # runner.py -> dctools/processing/runner.py
    # parents[0]=processing, parents[1]=dctools, parents[2]=repo root
    return Path(__file__).resolve().parents[2]


def _challenge_config_dir() -> Path:
    # runner.py -> dctools/processing/runner.py
    # config lives at <repo root>/dc2/config
    return _project_root() / "dc2" / "config"


def _ensure_project_on_syspath() -> None:
    root = str(_project_root())
    if root not in sys.path:
        sys.path.insert(0, root)


def _configure_hdf5_netcdf_env() -> None:
    """Disable locking/version checks before importing xarray/netcdf libs."""
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
    os.environ.setdefault("NETCDF4_DEACTIVATE_MPI", "1")
    os.environ.setdefault("NETCDF4_USE_FILE_LOCKING", "FALSE")
    os.environ.setdefault("HDF5_DISABLE_VERSION_CHECK", "1")
    os.environ.setdefault("ARGOPY_NETCDF_LOCKING", "FALSE")


def resolve_config_path(default_config_name: str, cli_args: Any) -> Path:
    """Resolve the YAML config path from CLI args (if any) or default."""
    config_name = getattr(cli_args, "config_name", None) or default_config_name
    return _challenge_config_dir() / f"{config_name}.yaml"


_BANNER_STYLES = {
    "success": {"icon": "\u2714", "border": "\u2550", "corner": ("\u2554", "\u2557", "\u255a", "\u255d"), "side": "\u2551", "color": "\033[1;32m"},
    "warning": {"icon": "\u26a0", "border": "\u2550", "corner": ("\u2554", "\u2557", "\u255a", "\u255d"), "side": "\u2551", "color": "\033[1;33m"},
    "error":   {"icon": "\u2718", "border": "\u2550", "corner": ("\u2554", "\u2557", "\u255a", "\u255d"), "side": "\u2551", "color": "\033[1;31m"},
}
_RESET = "\033[0m"


def _print_banner(
    message: str,
    *,
    details: Optional[list[str]] = None,
    style: str = "success",
    width: int = 60,
) -> None:
    """Print a prominent box-styled banner to the terminal."""
    s = _BANNER_STYLES.get(style, _BANNER_STYLES["success"])
    c, r = s["color"], _RESET
    tl, tr, bl, br = s["corner"]
    side, border, icon = s["side"], s["border"], s["icon"]

    inner = width - 2  # space inside the side borders
    top = f"{tl}{border * inner}{tr}"
    bot = f"{bl}{border * inner}{br}"
    empty = f"{side}{' ' * inner}{side}"

    title = f" {icon}  {message}  {icon} "
    pad = max(inner - len(title), 0)
    left_pad = pad // 2
    right_pad = pad - left_pad
    title_line = f"{side}{' ' * left_pad}{title}{' ' * right_pad}{side}"

    print(f"\n{c}{top}")
    print(empty)
    print(title_line)
    if details:
        print(empty)
        for d in details:
            d_pad = max(inner - len(d) - 4, 0)
            print(f"{side}  {d}{' ' * (d_pad + 2)}{side}")
    print(empty)
    print(f"{bot}{r}\n")


def run_from_config(
    config_path: Path,
    evaluation_cls: Optional[Type[object]] = None,
    cli_args: Any = None,
) -> int:
    """Run an evaluation from a resolved config path.

    Parameters
    ----------
    config_path:
        Path to a YAML file similar to dc2.yaml.
    evaluation_cls:
        Evaluation class to instantiate. Defaults to DC2Evaluation.
    """
    evaluator_instance = None

    _ensure_project_on_syspath()
    _configure_hdf5_netcdf_env()

    try:
        from dask.distributed import performance_report
        from dctools.utilities.args_config import load_args_and_config

        if evaluation_cls is None:
            from dc2.evaluation.evaluation import DC2Evaluation as _DefaultEval  # noqa: E402

            evaluation_cls = _DefaultEval

        args = load_args_and_config(str(config_path), args=cli_args)
        if args is None:
            print("Config loading failed.")
            return 1

        vars(args)["regridder_weights"] = os.path.join(args.data_directory, "weights")
        vars(args)["catalog_dir"] = os.path.join(args.data_directory, "catalogs")
        vars(args)["result_dir"] = os.path.join(args.data_directory, "results")
        vars(args)["logs_dir"] = os.path.join(args.data_directory, "logs")

        # If no explicit --logfile was given, default to <data_directory>/logs/dc2.log
        if not getattr(args, "logfile", None):
            vars(args)["logfile"] = os.path.join(args.logs_dir, "dc2.log")
            # Re-apply logging config so the file sink is registered.
            from dctools.utilities.args_config import configure_logging_from_args
            configure_logging_from_args(args)

        # Backward-compatible: previously this tried to delete the weights path.
        # Only remove it when it is a file.
        try:
            if os.path.isfile(args.regridder_weights):
                os.remove(args.regridder_weights)
        except Exception:
            pass

        os.makedirs(args.catalog_dir, exist_ok=True)
        os.makedirs(args.result_dir, exist_ok=True)
        os.makedirs(args.logs_dir, exist_ok=True)

        evaluator_instance = evaluation_cls(args)  # type: ignore[call-arg]

        report_path = os.path.join(args.result_dir, "dask-report.html")
        print(f"Generating Dask performance report at: {report_path}")

        try:
            with performance_report(filename=report_path):
                evaluator_instance.run_eval()  # type: ignore[attr-defined]
        except ValueError as _pr_exc:
            # The Dask cluster is shut down inside run_eval() before
            # post-processing to free worker RAM.  performance_report's
            # __exit__ calls get_client() which raises ValueError when no
            # client exists.  The evaluation itself succeeded — just skip
            # the performance report.
            if "No global client found" in str(_pr_exc):
                pass  # cluster already closed during post-processing; report skipped
            else:
                raise

        lb_warnings = getattr(evaluator_instance, "_leaderboard_warnings", [])
        if lb_warnings:
            _print_banner(
                "Evaluation finished — leaderboard INCOMPLETE",
                details=[f"[!] {w}" for w in lb_warnings],
                style="warning",
            )
            return 0
        _print_banner("Evaluation has finished successfully", style="success")
        return 0

    except KeyboardInterrupt:
        _print_banner("Evaluation aborted (manual interrupt)", style="error")
        return 1
    except SystemExit:
        _print_banner("Evaluation aborted (SystemExit)", style="error")
        return 1
    except Exception as exc:
        traceback.print_exc()
        _print_banner(f"Evaluation failed: {exc}", style="error")
        return 1
    finally:
        if evaluator_instance is not None:
            if hasattr(evaluator_instance, "close") and callable(evaluator_instance.close):
                evaluator_instance.close()
            elif hasattr(evaluator_instance, "dataset_processor"):
                try:
                    evaluator_instance.dataset_processor.close()
                except Exception:
                    pass


def run_from_cli(default_config_name: str = "dc2") -> int:
    """Entry point used by dc2/evaluate.py."""
    _ensure_project_on_syspath()
    _configure_hdf5_netcdf_env()

    from dctools.utilities.args_config import parse_arguments

    cli_args = parse_arguments()
    config_path = resolve_config_path(default_config_name, cli_args=cli_args)
    return run_from_config(config_path, cli_args=cli_args)
