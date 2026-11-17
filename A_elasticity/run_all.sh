#!/usr/bin/env bash
# Run the whole of task A: data -> descriptors -> dataset -> models.
#
#   bash run_all.sh [--no-julia] [--axes-only] [--workers=N]
set -euo pipefail
cd "$(dirname "$0")"
bash 00_prepare_data.sh "$@"
bash 01_compute_descriptors.sh "$@"
bash 02_build_dataset.sh "$@"
bash 03_train.sh "$@"
