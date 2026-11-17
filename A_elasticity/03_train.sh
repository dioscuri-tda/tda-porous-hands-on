#!/usr/bin/env bash
# Step 3 - train CatBoost, evaluate on the test set, plot.
#
#   bash 03_train.sh [--no-julia]
#
# One model per stiffness component and per feature set, all on the same
# train/validation/test split (60/20/20 structures). Feature sets:
#   porosity          solid volume fraction alone (the baseline to beat)
#   ph                persistence images of the cone filtration, all directions
#   ph+porosity
#   ecp               Euler characteristic profiles           (needs the Julia step)
#   ecp+ph+porosity   the full descriptor of the paper        (needs the Julia step)
# Results: results/<feature set>/{metrics.csv, predictions.csv, parity.png,
# importance.png} and the comparison results/summary.{md,csv,png}.
source "$(dirname "$0")/../common/cli.sh"

# ---- settings ---------------------------------------------------------------
UNIAXIAL="C_xx_xx C_yy_yy C_zz_zz"
POISSON="C_xx_yy C_xx_zz C_yy_zz"
SHEAR="C_yz_yz C_xz_xz C_xy_xy"
OFFDIAG="C_xx_xy C_yy_xy C_xx_xz C_zz_xz C_yy_yz C_zz_yz"   # 6 of the 12 normal-shear couplings
TARGETS="$UNIAXIAL $POISSON $SHEAR $OFFDIAG"
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

train porosity        solid_fraction
train ph              ph
train ph+porosity     ph solid_fraction
if [ "$HAS_ECP" = 1 ]; then
  train ecp             ecp
  train ecp+ph+porosity ecp ph solid_fraction
fi

step "summary"
"$PYTHON" "$COMMON/summarize_results.py" --results results \
    --order porosity ph ph+porosity ecp ecp+ph+porosity
