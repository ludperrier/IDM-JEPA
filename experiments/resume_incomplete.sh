#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR="$HOME/CODE/latent-jepa"
PYTHON="$PROJECT_DIR/.venv/bin/python"
SWEEP_TAG=${SWEEP_TAG:-v3}
TARGET_EPOCH=${TARGET_EPOCH:-10}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-$SCRATCH/.stable-wm/checkpoints}
JOB_NAME=${JOB_NAME:-la_jepa_train}

mapfile -t RUNS < <(cd "$PROJECT_DIR" && SWEEP_TAG="$SWEEP_TAG" "$PYTHON" -m sweep --names)
mapfile -t ACTIVE_INDICES < <(squeue -h -r --me --name="$JOB_NAME" -o '%K' | sort -u)

is_active() {
 local candidate=$1
 local active
 for active in "${ACTIVE_INDICES[@]}"; do
  [[ "$active" == "$candidate" ]] && return 0
 done
 return 1
}

indices=()
for index in "${!RUNS[@]}"; do
 run=${RUNS[$index]}
 run_dir="$CHECKPOINT_ROOT/$run"

 if [[ -f "$run_dir/weights_epoch_${TARGET_EPOCH}.pt" ]]; then
  continue
 fi
 if is_active "$index"; then
  echo "skip active:     index=$index run=$run"
  continue
 fi
 if [[ ! -f "$run_dir/last.ckpt" ]]; then
  echo "skip no resume:  index=$index run=$run" >&2
  continue
 fi

 latest_epoch=$(find "$run_dir" -maxdepth 1 -type f -name 'weights_epoch_*.pt' -printf '%f\n' \
  | sed -n 's/^weights_epoch_\([0-9]\+\)\.pt$/\1/p' | sort -n | tail -n 1)
 echo "resume:          index=$index run=$run epoch=${latest_epoch:-unknown}"
 indices+=("$index")
done

if [[ ${#indices[@]} -eq 0 ]]; then
 echo "No inactive, checkpointed $SWEEP_TAG runs need resuming."
 exit 0
fi

array=$(IFS=,; echo "${indices[*]}")
command=(sbatch --array="$array" --export="ALL,SWEEP_TAG=$SWEEP_TAG,RESUME_POLICY=always" "$SCRIPT_DIR/train.sh")

if [[ ${1:-} == "--dry-run" ]]; then
 printf 'Would submit:'
 printf ' %q' "${command[@]}"
 printf '\n'
else
 "${command[@]}"
fi