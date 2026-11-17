#!/usr/bin/env bash
# Step 1 - direction-aware topological descriptors of every structure.
#
#   bash 01_compute_descriptors.sh [--no-julia] [--axes-only] [--workers=N]
#
# Two descriptor families, each computed for every probing direction d:
#   TDA  cone filtration of the solid phase along d            -> work/filt/cone_<d>/
#          persistent homology -> persistence images  (gudhi)  -> work/desc/ph/<d>/
#          Euler characteristic profile        (Julia, Ecp)    -> work/desc/ecp/<d>/
#          (the ECP is skipped with --no-julia)
#   THR  throat-scale filtration of the pore space along d     -> work/filt/throat_<d>/
#          persistent homology -> persistence images  (gudhi)  -> work/desc/thr/<d>/
# Finally all directions are collected into descriptors/{ph,ecp,thr}.csv
# (one row per structure). Filtrations are deleted as soon as they are used.
# The script is resumable: finished directions are skipped.
source "$(dirname "$0")/../common/cli.sh"

# ---- settings (the values of the paper) -------------------------------------
# three axes and six signed face diagonals; the sign matters for the off-diagonal K_ij
DIRECTIONS=("1 0 0" "0 1 0" "0 0 1" "1 1 0" "1 -1 0" "1 0 1" "1 0 -1" "0 1 1" "0 1 -1")
[ "$AXES_ONLY" = 1 ] && DIRECTIONS=("1 0 0" "0 1 0" "0 0 1")
# cone filtration (TDA)
RADIUS=4; CONE_RADIUS=3; CONE_HEIGHT=6     # PCA sphere, cone base radius, cone half-height [voxels]
ECP_GRID_RES=6                             # ECP sampled on a (6+1)^3 grid
# throat filtration (THR)
RCAP=10; HORIZON=6                         # cap of the distance transform, walk length [voxels]
# persistence images: 10x10 pixels, bars weighted by persistence; per homology
# dimension: "birth range, persistence range" and Gaussian bandwidth
PH_RES="10 10"
C_RANGE_H0="0 1.25 0 1.25";  C_BW_H0=0.125
C_RANGE_H1="0 1.25 0 0.80";  C_BW_H1=0.100
C_RANGE_H2="0.2 1.0 0 0.06"; C_BW_H2=0.019
T_RANGE_H0="0 0.9 0 0.40";   T_BW_H0=0.060
T_RANGE_H1="0 0.9 0 0.75";   T_BW_H1=0.082
T_RANGE_H2="0 0.9 0 0.50";   T_BW_H2=0.067
# -----------------------------------------------------------------------------

ph () {   # ph <filtration dir> <output dir> <range H0> <bw H0> <range H1> <bw H1> <range H2> <bw H2>
  "$PYTHON" "$COMMON/persistence_images.py" --inputdir "$1" --outputdir "$2" \
      --weight linear --resolution $PH_RES --periodic 1 1 1 \
      --im_range_h0 $3 --bandwidth_h0 "$4" --im_range_h1 $5 --bandwidth_h1 "$6" \
      --im_range_h2 $7 --bandwidth_h2 "$8" --workers "$WORKERS"
}

[ -d data/structures ] || bash 00_prepare_data.sh
N=$(find data/structures -name '*.raw' | wc -l | tr -d ' ')
require_julia
echo "structures: $N   directions: ${#DIRECTIONS[@]}   workers: $WORKERS   Julia/ECP: $USE_JULIA"

for D in "${DIRECTIONS[@]}"; do
  TAG=${D// /_}

  # ---- TDA: cone filtration -> PH (+ ECP) ----
  FILT=work/filt/cone_$TAG; PH=work/desc/ph/$TAG; ECP=work/desc/ecp/$TAG
  NEED=0
  [ "$(count_npy "$PH")" = "$N" ] || NEED=1
  [ "$USE_JULIA" = 0 ] || [ "$(count_npy "$ECP")" = "$N" ] || NEED=1
  if [ "$NEED" = 1 ]; then
    step "direction [$D]: cone filtration"
    "$PYTHON" scripts/filtration3d.py --structures data/structures --outputdir "$FILT" \
        --direction $D --radius $RADIUS --cone-radius $CONE_RADIUS --cone-height $CONE_HEIGHT \
        --workers "$WORKERS"
    step "direction [$D]: persistent homology of the cone filtration"
    ph "$FILT" "$PH" "$C_RANGE_H0" $C_BW_H0 "$C_RANGE_H1" $C_BW_H1 "$C_RANGE_H2" $C_BW_H2
    if [ "$USE_JULIA" = 1 ]; then
      step "direction [$D]: Euler characteristic profile (Julia)"
      "$JULIA" --project="$JULIA_ENV" --threads "$WORKERS" "$COMMON/ecp.jl" "$FILT" "$ECP" \
          --grid-res $ECP_GRID_RES
    fi
    rm -rf "$FILT"
  else
    echo "direction [$D]: cone descriptors done, skipping"
  fi

  # ---- THR: throat filtration -> PH ----
  FILT=work/filt/throat_$TAG; THR=work/desc/thr/$TAG
  if [ "$(count_npy "$THR")" != "$N" ]; then
    step "direction [$D]: throat filtration"
    "$PYTHON" scripts/filtration_throat3d.py --structures data/structures --outputdir "$FILT" \
        --mode dir --direction $D --rcap $RCAP --horizon $HORIZON --phase void --workers "$WORKERS"
    step "direction [$D]: persistent homology of the throat filtration"
    ph "$FILT" "$THR" "$T_RANGE_H0" $T_BW_H0 "$T_RANGE_H1" $T_BW_H1 "$T_RANGE_H2" $T_BW_H2
    rm -rf "$FILT"
  else
    echo "direction [$D]: throat descriptors done, skipping"
  fi
done

step "collect"
KINDS="ph thr"; [ "$USE_JULIA" = 1 ] && KINDS="ph thr ecp"
"$PYTHON" "$COMMON/collect_descriptors.py" --descroot work/desc --outdir descriptors --kinds $KINDS
echo "DESCRIPTORS DONE -> $TASK_DIR/descriptors/"
