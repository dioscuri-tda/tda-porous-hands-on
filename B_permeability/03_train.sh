#!/usr/bin/env bash
# Step 3 - train CatBoost, evaluate on the test set, plot.
#
#   bash 03_train.sh [--no-julia]
#
# One model per permeability component and per feature set, all on the same
# train/validation/test split (60/20/20 structures). Feature sets:
#   porosity   porosity alone (the baseline to beat)
#   ph         persistence images of the cone filtration
#   thr        persistence images of the throat filtration ("THR" in the paper)
#   ph+thr
#   tda        ecp + ph, the cone descriptor "TDA" of the paper   (needs the Julia step)
#   tda+thr                                                        (needs the Julia step)
# Results: results/<feature set>/{metrics.csv, predictions.csv, parity.png,
# importance.png} and the comparison results/summary.{md,csv,png}.
source "$(dirname "$0")/../common/cli.sh"

# ---- settings ---------------------------------------------------------------
TARGETS="K_xx K_yy K_zz k_xy k_xz k_yz"     # diagonal and (symmetrised) off-diagonal components
SEED=0
# -----------------------------------------------------------------------------

[ -f dataset.csv ] || { echo "dataset.csv not found - run 02_build_dataset.sh first"; exit 1; }
HAS_ECP=0; head -1 dataset.csv | grep -q "ecp_" && HAS_ECP=1
if [ "$USE_JULIA" = 1 ] && [ "$HAS_ECP" = 0 ]; then
  echo "ERROR: dataset.csv has no ECP columns. Either compute them (Julia), rebuild the"
  echo "       dataset with 'bash 02_build_dataset.sh --reference', or pass --no-julia."
  exit 1
fi

train () {   # train <label> <feature groups...>
  local label=$1; shift
  "$PYTHON" "$COMMON/train_catboost.py" --dataset dataset.csv --targets $TARGETS \
      --features "$@" --label "$label" --outdir results --seed $SEED --threads "$WORKERS"
}

train porosity porosity
train ph       ph
train thr      thr
train ph+thr   ph thr
if [ "$HAS_ECP" = 1 ]; then
  train tda      ecp ph
  train tda+thr  ecp ph thr
fi

step "summary"
"$PYTHON" "$COMMON/summarize_results.py" --results results \
    --order porosity ph thr ph+thr tda tda+thr
