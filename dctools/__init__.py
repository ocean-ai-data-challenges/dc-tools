"""Base tools common to all Data Challenges."""

import os

# Configuration for xarray/netcdf/dask compatibility
# Must be set before importing netCDF4 or running dask tasks
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
os.environ["NETCDF4_DEACTIVATE_MPI"] = "1"

# Lazy access to the submission module
__all__ = ["submission"]
