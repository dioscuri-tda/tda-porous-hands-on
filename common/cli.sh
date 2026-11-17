# Shared settings and command-line flags of the task scripts. Sourced, not run.
#
# Flags (every NN_*.sh script accepts the same ones and ignores those it does not need):
#   --no-julia     do not require Julia: skip the Euler characteristic profiles (ECP)
#                  and everything that depends on them. Same as USE_JULIA=0.
#   --reference    use the precomputed descriptor tables in reference/descriptors
#                  instead of the ones in descriptors/ (02_build_dataset.sh).
#   --axes-only    compute descriptors along the three coordinate axes only
#                  (about 2-3x faster; 01_compute_descriptors.sh).
#   --workers=N    number of CPU cores to use (default: all). Same as WORKERS=N.
set -euo pipefail

TASK_DIR=$(cd "$(dirname "${BASH_SOURCE[1]}")" && pwd)
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
COMMON=$ROOT/common

USE_JULIA=${USE_JULIA:-1}
REFERENCE=${REFERENCE:-0}
AXES_ONLY=${AXES_ONLY:-0}
WORKERS=${WORKERS:-$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN)}
JULIA=${JULIA:-julia}
JULIA_ENV=${JULIA_ENV:-$ROOT/julia}

for arg in "$@"; do
  case $arg in
    --no-julia)  USE_JULIA=0 ;;
    --reference) REFERENCE=1 ;;
    --axes-only) AXES_ONLY=1 ;;
    --workers=*) WORKERS=${arg#*=} ;;
    -h|--help)   awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "${BASH_SOURCE[1]}"
                 echo; sed -n '3,10p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $arg (try --help)"; exit 1 ;;
  esac
done

# Python: the repository's virtual environment if it exists, else whatever is active.
if [ -z "${PYTHON:-}" ]; then
  if [ -x "$ROOT/.venv/bin/python" ]; then PYTHON=$ROOT/.venv/bin/python; else PYTHON=python; fi
fi

require_julia () {   # stop early, with advice, if the ECP step cannot run
  [ "$USE_JULIA" = 1 ] || return 0
  if ! command -v "$JULIA" >/dev/null 2>&1; then
    echo "ERROR: Julia not found. Install it (https://julialang.org/downloads/) and run"
    echo "         julia --project=julia julia/setup.jl"
    echo "       or rerun this script with --no-julia to skip the ECP descriptors."
    exit 1
  fi
  if ! "$JULIA" --project="$JULIA_ENV" -e 'import Ecp, NPZ, Glob' >/dev/null 2>&1; then
    echo "ERROR: the Julia package Ecp is not installed. From the repository root run"
    echo "         julia --project=julia julia/setup.jl"
    echo "       or rerun this script with --no-julia to skip the ECP descriptors."
    exit 1
  fi
}

count_npy () { find "$1" -maxdepth 1 -name '*.npy' 2>/dev/null | wc -l | tr -d ' '; }
step () { echo; echo "=== $* ==="; }

cd "$TASK_DIR"
