#!/usr/bin/env bash

set -euo pipefail

# Datarmor staging for DC2 source data.
# This script copies the full 2023/2024 coverage for each dataset into a local
# staging tree on the Datarmor filesystem, without trying to patch holes.

STAGE_ROOT="${STAGE_ROOT:-$HOME/dc2_staging_netcdf}"
mkdir -p "$STAGE_ROOT"

echo "STAGE_ROOT=$STAGE_ROOT"

copy_selected_files() {
  local src_root="$1"
  local dst="$2"
  local pattern="$3"
  local list_file
  list_file="$(mktemp)"
  find "$src_root" -type f | grep -E "$pattern" > "$list_file" || true
  mkdir -p "$dst"
  if [[ -s "$list_file" ]]; then
    rsync -a --files-from="$list_file" / "$dst/"
  else
    echo "No files matched in $src_root for pattern: $pattern"
  fi
  rm -f "$list_file"
}

echo "[1/7] saral"
copy_selected_files \
  "/home/datawork-cersat-public/provider/aviso/satellite/l2/" \
  "$STAGE_ROOT/saral" \
  '(^|/)(SARAL|Saral|sral|srl|SRL).*2023.*\.nc$|(^|/)(SARAL|Saral|sral|srl|SRL).*2024.*\.nc$'

echo "[2/7] swot"
copy_selected_files \
  "/home/datawork-cersat-public/provider/aviso/satellite/l3/swot/karin/l3_lr_ssh_expert/" \
  "$STAGE_ROOT/swot/expert" \
  '2023.*\.nc$|2024.*\.nc$'
copy_selected_files \
  "/home/datawork-cersat-public/provider/aviso/satellite/l3/swot/karin/l3_lr_ssh_unsmoothed/v2.0.1/" \
  "$STAGE_ROOT/swot/unsmoothed" \
  '2023.*\.nc$|2024.*\.nc$'

echo "[3/7] jason3"
copy_selected_files \
  "/home/datawork-cersat-public/provider/aviso/satellite/l2/" \
  "$STAGE_ROOT/jason3" \
  '(^|/)(JA3|Jason-3|Jason3|jason3).*2023.*\.nc$|(^|/)(JA3|Jason-3|Jason3|jason3).*2024.*\.nc$'

echo "[4/7] SSS_fields"
copy_selected_files \
  "/home/datawork-cersat-public/project/pimep/data/smos/l3/catds_cpdc/RE07/MIR_CS3G09/2023/" \
  "$STAGE_ROOT/SSS_fields/2023" \
  '\.nc$'
copy_selected_files \
  "/home/datawork-cersat-public/project/pimep/data/smos/l3/catds_cpdc/RE07/MIR_CS3G09/2024/" \
  "$STAGE_ROOT/SSS_fields/2024" \
  '\.nc$'

echo "[5/7] SST_fields"
copy_selected_files \
  "/home/ref-cersat-public/sea-surface-temperature/odyssea/l3s/data/glob/nrt/data/v3.0/2023/" \
  "$STAGE_ROOT/SST_fields/2023" \
  '\.nc$'
copy_selected_files \
  "/home/ref-cersat-public/sea-surface-temperature/odyssea/l3s/data/glob/nrt/data/v3.0/2024/" \
  "$STAGE_ROOT/SST_fields/2024" \
  '\.nc$'

echo "[6/7] argo_velocities"
copy_selected_files \
  "/home/ref-copernicus-insitu/INSITU_GLO_PHY_UV_DISCRETE_MY_013_044/" \
  "$STAGE_ROOT/argo_velocities/MY" \
  '(2023|2024).*\.(csv|nc)$'
copy_selected_files \
  "/home/ref-copernicus-insitu/INSITU_GLO_PHY_UV_DISCRETE_NRT_013_048/" \
  "$STAGE_ROOT/argo_velocities/NRT" \
  '(2023|2024).*\.(csv|nc)$'

echo "[7/7] argo_profiles"
copy_selected_files \
  "/dataref/coriolis/public/argo/gdac/" \
  "$STAGE_ROOT/argo_profiles" \
  '/(2023|2024)/.*\.(nc|json|zst)$'

echo "[8/8] glorys"
copy_selected_files \
  "/home/ref-ocean-reanalysis/" \
  "$STAGE_ROOT/glorys" \
  '(2023|2024).*\.nc$'

echo
echo "Staging complete. Summary:"
find "$STAGE_ROOT" -type f | sed 's#^#  #'
