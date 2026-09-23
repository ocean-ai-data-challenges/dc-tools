"""Base tools common to all Data Challenges."""

import os

# Configuration for xarray/netcdf/dask compatibility
# Must be set before importing netCDF4 or running dask tasks
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
os.environ["NETCDF4_DEACTIVATE_MPI"] = "1"


def _route_core_logs_to_loguru() -> None:
    """dctools-core logs through stdlib logging; forward it to loguru for the DC2 logs."""
    import logging

    from loguru import logger as _loguru

    class _Intercept(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            _loguru.opt(depth=6, exception=record.exc_info).log(
                record.levelname, record.getMessage()
            )

    core = logging.getLogger("dctools_core")
    if not any(isinstance(h, _Intercept) for h in core.handlers):
        core.addHandler(_Intercept())
        core.setLevel(logging.DEBUG)
        core.propagate = False


_route_core_logs_to_loguru()
