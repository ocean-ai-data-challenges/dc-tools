#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""CLI entrypoint for submitting a model to the Data Challenge benchmark.

Usage
-----
::

    # Validate only (quick pre-check):
    python -m dc.submit validate /path/to/my_model.zarr --model-name MyModel

    # Full submission (validate --> evaluate --> leaderboard):
    python -m dc.submit run /path/to/my_model.zarr \\
        --model-name MyModel \\
        --data-directory ./output \\
        --team "Ocean AI Lab" \\
        --description "1/4° global 10-day forecast"

    # Submit with forced evaluation (skip validation failures):
    python -m dc.submit run /path/to/my_model.zarr --model-name MyModel --force
"""

import argparse
import sys
from pathlib import Path

# Ensure the repository root is importable when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dc-submit",
        description=(
            "Submit a model's prediction dataset to the Data Challenge benchmark.\n\n"
            "Performs data conformance checks and (optionally) launches the full\n"
            "evaluation pipeline with leaderboard generation."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Quick validation:\n"
            "  python -m dc.submit validate /data/my_model.zarr --model-name MyModel\n\n"
            "  # Full submission:\n"
            "  python -m dc.submit run /data/my_model.zarr \\\n"
            "      --model-name MyModel --data-directory ./output\n"
        ),
    )

    subparsers = parser.add_subparsers(dest="command", help="Action to perform")

    # -- validate ---------------------------------------------------
    val_parser = subparsers.add_parser(
        "validate",
        help="Validate a dataset against the DC specification (no evaluation).",
    )
    _add_common_args(val_parser)
    val_parser.add_argument(
        "--quick",
        action="store_true",
        help="Skip expensive NaN full-scan for a faster check.",
    )
    val_parser.add_argument(
        "--save-report",
        type=str,
        default=None,
        metavar="PATH",
        help="Save the validation report as JSON to this path.",
    )
    val_parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        metavar="PATH",
        help="Save the validation report as JSON (alias for --save-report).",
    )

    # -- run (full submission) --------------------------------------
    run_parser = subparsers.add_parser(
        "run",
        help="Full pipeline: validate --> evaluate --> leaderboard.",
    )
    _add_common_args(run_parser)
    run_parser.add_argument(
        "-d", "--data-directory",
        type=str,
        default=None,
        help="Root directory for evaluation outputs (default: ./output).",
    )
    run_parser.add_argument(
        "--force",
        action="store_true",
        help="Proceed with evaluation even if validation fails.",
    )
    run_parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip the validation step entirely.",
    )
    run_parser.add_argument(
        "--quick-validation",
        action="store_true",
        help="Run a quick validation (skip NaN full scan).",
    )

    # -- info -------------------------------------------------------
    info_parser = subparsers.add_parser(
        "info",
        help="Print the expected dataset specification for the DC config.",
    )
    info_parser.add_argument(
        "--config",
        type=str,
        default="dc2",
        help="Data Challenge config name (default: dc2).",
    )

    return parser


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments shared by validate and run commands."""
    parser.add_argument(
        "data_path",
        type=str,
        help=(
            "Path to prediction data.  Accepted layouts:\n"
            "  - Single .zarr store  (one directory covering the full period)\n"
            "  - Single .nc file\n"
            "  - Directory of forecast files  (e.g. one 20240103.zarr per init date)\n"
            "  - Glob pattern  (e.g. '/data/model/*.nc')"
        ),
    )
    parser.add_argument(
        "--variables",
        type=str,
        nargs="+",
        default=None,
        metavar="VAR",
        help=(
            "Subset of variables to validate/evaluate (partial submission). "
            "Example: --variables zos ssh. If omitted, all DC-required "
            "variables must be present."
        ),
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="unnamed_model",
        help="Short model identifier (used in filenames and leaderboard).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="dc2",
        help="Data Challenge config name (default: dc2).",
    )
    parser.add_argument(
        "--description",
        type=str,
        default="",
        help="Free-text model description.",
    )
    parser.add_argument(
        "--team",
        type=str,
        default="",
        help="Submitting team name.",
    )
    parser.add_argument(
        "--email",
        type=str,
        default="",
        help="Contact email.",
    )
    parser.add_argument(
        "--url",
        type=str,
        default="",
        help="Model paper/repository URL.",
    )
    parser.add_argument(
        "--max-nan-fraction",
        type=float,
        default=0.10,
        help="Maximum NaN fraction per variable (0-1, default 0.10).",
    )


def _cmd_validate(args: argparse.Namespace) -> int:
    """Handle the 'validate' command."""
    from dctools.submission import ModelSubmission

    sub = ModelSubmission(
        model_name=args.model_name,
        data_path=args.data_path,
        dc_config=args.config,
        model_description=args.description,
        team_name=args.team,
        contact_email=args.email,
        model_url=args.url,
        max_nan_fraction=args.max_nan_fraction,
        variables=args.variables,
    )

    report = sub.validate(quick=args.quick)
    print(report.pretty())

    # --output is a shorter alias for --save-report
    save_path = args.output or args.save_report
    if save_path:
        report.save_json(save_path)
        print(f"Report saved to {save_path}")

    return 0 if report.overall_pass else 1


def _cmd_run(args: argparse.Namespace) -> int:
    """Handle the 'run' command."""
    from dctools.submission import ModelSubmission

    sub = ModelSubmission(
        model_name=args.model_name,
        data_path=args.data_path,
        dc_config=args.config,
        model_description=args.description,
        team_name=args.team,
        contact_email=args.email,
        model_url=args.url,
        max_nan_fraction=args.max_nan_fraction,
        variables=args.variables,
    )

    return sub.submit(
        data_directory=args.data_directory,
        skip_validation=args.skip_validation,
        quick_validation=args.quick_validation,
        force=args.force,
    )


def _cmd_info(args: argparse.Namespace) -> int:
    """Handle the 'info' command — print expected specification."""
    from dctools.submission.validator import SubmissionValidator

    try:
        v = SubmissionValidator.from_dc_config(args.config)
    except Exception as exc:
        print(f"Error loading config '{args.config}': {exc}")
        return 1

    sep = "-" * 72
    print(f"\n┌{sep}┐")
    print(f"│{'  DATA CHALLENGE SPECIFICATION':^72}│")
    print(f"├{sep}┤")
    print(f"│  Config     : {args.config:<56}│")
    print(f"│  Time range : {v.start_time} --> {v.end_time:<38}│")
    print(f"│  Forecast   : {v.n_days_forecast} days{'':<53}│")
    print(f"├{sep}┤")

    if v.target_lat is not None:
        lat = v.target_lat
        step_lat = float(lat[1] - lat[0]) if len(lat) > 1 else 0
        print(f"│  Latitude   : [{lat[0]:.2f}, {lat[-1]:.2f}]  step={step_lat:.4f}°"
              f"  ({len(lat)} pts){'':<10}│")
    if v.target_lon is not None:
        lon = v.target_lon
        step_lon = float(lon[1] - lon[0]) if len(lon) > 1 else 0
        print(f"│  Longitude  : [{lon[0]:.2f}, {lon[-1]:.2f}]  step={step_lon:.4f}°"
              f"  ({len(lon)} pts){'':<10}│")
    if v.target_depth is not None:
        d = v.target_depth
        print(f"│  Depth      : {len(d)} levels  [{d[0]:.1f}m .. {d[-1]:.1f}m]{'':<20}│")
    if v.target_time_values is not None:
        print(f"│  Lead times : {v.target_time_values}{'':<40}│")

    print(f"├{sep}┤")
    if v.required_variables:
        print(f"│  Required variables:{'':<51}│")
        for var in v.required_variables:
            print(f"│    • {var:<65}│")
    else:
        print(f"│  Required variables: (auto-detected from first pred source){'':<10}│")

    print(f"├{sep}┤")
    print(f"│  Accepted input layouts:{'':<47}│")
    print(f"│    • Single .zarr directory (one store, full period){'':<19}│")
    print(f"│    • Single .nc / .nc4 file{'':<43}│")
    print(f"│    • Directory of per-date files (.zarr or .nc){'':<24}│")
    print(f"│      e.g. model_output/20240103.zarr, 20240110.zarr, …{'':<16}│")
    print(f"│    • Glob pattern (e.g. '/data/model/*.nc'){'':<29}│")
    print(f"├{sep}┤")
    print(f"│  Coordinate naming conventions:{'':<40}│")
    print(f"│    lat/latitude  lon/longitude  depth/lev  time{'':<23}│")
    print(f"│  Variable naming examples:{'':<45}│")
    print(f"│    SSH: zos, ssh, ssha   |  Temp: thetao, temperature{'':<17}│")
    print(f"│    Sal: so, salinity     |  Vel:  uo/vo, u/v{'':<24}│")
    print(f"└{sep}┘\n")

    return 0


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "validate":
        return _cmd_validate(args)
    elif args.command == "run":
        return _cmd_run(args)
    elif args.command == "info":
        return _cmd_info(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
