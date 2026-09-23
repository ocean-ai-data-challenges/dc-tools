#!/usr/bin/env bash

set -euo pipefail

# End-to-end Datarmor helper:
# 1) stage DC2 source data locally
# 2) recursively convert all staged NetCDF files to Zarr
#
# Glonet is intentionally excluded.

STAGE_ROOT="${STAGE_ROOT:-$HOME/dc2_staging_netcdf}"
ZARR_ROOT="${ZARR_ROOT:-$HOME/dc2_staging_zarr}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE_SCRIPT="$SCRIPT_DIR/stage_dc2_datarmor_2023_2024.sh"
CONVERT_SCRIPT="$SCRIPT_DIR/convert_recursive_netcdf_to_zarr.py"

mkdir -p "$STAGE_ROOT" "$ZARR_ROOT"

echo "STAGE_ROOT=$STAGE_ROOT"
echo "ZARR_ROOT=$ZARR_ROOT"

echo "[1/2] Staging source data"
STAGE_ROOT="$STAGE_ROOT" bash "$STAGE_SCRIPT"

echo "[2/2] Converting staged NetCDF trees to Zarr"

run_convert() {
  local src="$1"
  local dst="$2"
  if [[ -d "$src" ]]; then
    echo "--- $src -> $dst"
    python "$CONVERT_SCRIPT" "$src" "$dst" --overwrite
  else
    echo "--- skip missing source: $src"
  fi
}

run_convert "$STAGE_ROOT/saral" "$ZARR_ROOT/saral"
run_convert "$STAGE_ROOT/swot" "$ZARR_ROOT/swot"
run_convert "$STAGE_ROOT/jason3" "$ZARR_ROOT/jason3"
run_convert "$STAGE_ROOT/SSS_fields" "$ZARR_ROOT/SSS_fields"
run_convert "$STAGE_ROOT/SST_fields" "$ZARR_ROOT/SST_fields"
run_convert "$STAGE_ROOT/argo_velocities" "$ZARR_ROOT/argo_velocities"
run_convert "$STAGE_ROOT/argo_profiles" "$ZARR_ROOT/argo_profiles"

echo
echo "Conversion complete. Zarr outputs are under: $ZARR_ROOT"
