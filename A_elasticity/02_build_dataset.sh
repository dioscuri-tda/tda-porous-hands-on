#!/usr/bin/env bash
# Step 2 - join the descriptors with the labels (the elastic stiffness tensor).
#
#   bash 02_build_dataset.sh [--no-julia] [--reference]
#
# data/stiffness.csv + descriptors/{ph,ecp}.csv -> dataset.csv (one row per structure)
# --reference takes the precomputed tables of reference/descriptors/ instead, so
# the rest of the tutorial can be followed without running step 1 (or without Julia).
source "$(dirname "$0")/../common/cli.sh"

DESC=descriptors; [ "$REFERENCE" = 1 ] && DESC=reference/descriptors
TABLES=("$DESC/ph.csv")
# ECP is required unless --no-julia; the reference tables always contain it
if [ "$USE_JULIA" = 1 ] || { [ "$REFERENCE" = 1 ] && [ -f "$DESC/ecp.csv" ]; }; then
  TABLES+=("$DESC/ecp.csv")
fi

"$PYTHON" "$COMMON/build_dataset.py" --labels data/stiffness.csv \
    --descriptors "${TABLES[@]}" --output dataset.csv
