#!/usr/bin/env bash
# Step 1 - direction-aware topological descriptors of every structure.
#
#   bash 01_compute_descriptors.sh [--no-julia] [--axes-only] [--workers=N]
#
# For each loading direction d:
#   1. cone + PCA filtration along d            (Python + numba)  -> work/filt/<d>/
#   2. persistent homology -> persistence images (Python + gudhi) -> work/desc/ph/<d>/
#   3. Euler characteristic profile              (Julia, Ecp)     -> work/desc/ecp/<d>/
#      skipped with --no-julia
# and finally all directions are collected into descriptors/ph.csv and
# descriptors/ecp.csv (one row per structure). The filtrations are large
# (6 MB per structure and direction) and are deleted as soon as they are used.
# The script is resumable: finished directions are skipped.
source "$(dirname "$0")/../common/cli.sh"

# ---- settings (the values of the paper) -------------------------------------
DIRECTIONS=("1 0 0" "0 1 0" "0 0 1" "1 1 0" "1 0 1" "0 1 1" "1 1 1")
[ "$AXES_ONLY" = 1 ] && DIRECTIONS=("1 0 0" "0 1 0" "0 0 1")
RADIUS=3          # radius of the PCA sphere [voxels]
CONE_RADIUS=3     # base radius of the cone [voxels]
CONE_HEIGHT=6     # half-height of the cone [voxels]
PH_RESOLUTION="12 12"            # persistence image: 12x12 pixels per homology dimension
PH_RANGE="0 1.25 0 1.25"         # birth and persistence range of the image
ECP_GRID_RES=8                   # ECP sampled on a (8+1)^3 grid
# -----------------------------------------------------------------------------

[ -d data/structures ] || bash 00_prepare_data.sh
N=$(count_npy data/structures)
require_julia
echo "structures: $N   directions: ${#DIRECTIONS[@]}   workers: $WORKERS   Julia/ECP: $USE_JULIA"

for D in "${DIRECTIONS[@]}"; do
  TAG=${D// /_}
  FILT=work/filt/$TAG; PH=work/desc/ph/$TAG; ECP=work/desc/ecp/$TAG
  NEED=0
  [ "$(count_npy "$PH")" = "$N" ] || NEED=1
  [ "$USE_JULIA" = 0 ] || [ "$(count_npy "$ECP")" = "$N" ] || NEED=1
  if [ "$NEED" = 0 ]; then echo "direction [$D]: done, skipping"; continue; fi

  step "direction [$D]: filtration"
  "$PYTHON" scripts/cone_filtration.py --structures data/structures --outputdir "$FILT" \
      --direction $D --radius $RADIUS --cone-radius $CONE_RADIUS --cone-height $CONE_HEIGHT \
      --workers "$WORKERS"

  step "direction [$D]: persistent homology"
  # the re-oriented grid is periodic along the loading axis (array axis 2) only
  "$PYTHON" "$COMMON/persistence_images.py" --inputdir "$FILT" --outputdir "$PH" \
      --weight constant --bandwidth 1.0 --resolution $PH_RESOLUTION --im_range $PH_RANGE \
      --periodic 0 0 1 --workers "$WORKERS"

  if [ "$USE_JULIA" = 1 ]; then
    step "direction [$D]: Euler characteristic profile (Julia)"
    "$JULIA" --project="$JULIA_ENV" --threads "$WORKERS" "$COMMON/ecp.jl" "$FILT" "$ECP" \
        --grid-res $ECP_GRID_RES
  fi
  rm -rf "$FILT"
done

step "collect"
KINDS="ph"; [ "$USE_JULIA" = 1 ] && KINDS="ph ecp"
"$PYTHON" "$COMMON/collect_descriptors.py" --descroot work/desc --outdir descriptors --kinds $KINDS
echo "DESCRIPTORS DONE -> $TASK_DIR/descriptors/"
