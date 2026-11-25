#!/bin/bash
#SBATCH --job-name=la_jepa_v13_plan
#SBATCH --partition=long
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --array=0-29
#SBATCH --output=analysis_out/slurm_logs/%x_%A_%a.out
#SBATCH --error=analysis_out/slurm_logs/%x_%A_%a.err

set -euo pipefail

module load python/3.10 cuda/12.2.2

REPO_DIR="$HOME/CODE/latent-jepa"
EXPERIMENTS_DIR="$HOME/CODE/latent-jepa-experiments"
cd "$REPO_DIR"
source .venv/bin/activate

export STABLEWM_HOME="$SCRATCH/.stable-wm"
export HYDRA_FULL_ERROR=1

RUN_INDEX=$((SLURM_ARRAY_TASK_ID / 5))
SEED_INDEX=$((SLURM_ARRAY_TASK_ID % 5))
# CHECKPOINT_RUN is the actual on-disk checkpoint dir; it can differ from the
# canonical RUN_NAME (used for output paths, and matched by the plotting/
# manifest regexes) when a run was resubmitted under a suffixed name to avoid
# colliding with an earlier failed attempt's checkpoint/W&B history.
case "$RUN_INDEX" in
    0) RUN_NAME=v13_idm_frac_0_s0; CHECKPOINT_RUN=v13_idm_frac_0_s0_retry3; LABEL_FRAC=0.00 ;;
    1) RUN_NAME=v13_idm_frac_5_s0; CHECKPOINT_RUN=v13_idm_frac_5_s0_retry3; LABEL_FRAC=0.05 ;;
    2) RUN_NAME=v13_idm_frac_25_s0; CHECKPOINT_RUN=v13_idm_frac_25_s0_retry3; LABEL_FRAC=0.25 ;;
    3) RUN_NAME=v13_idm_frac_50_s0; CHECKPOINT_RUN=v13_idm_frac_50_s0_retry3; LABEL_FRAC=0.50 ;;
    4) RUN_NAME=v13_idm_frac_75_s0; CHECKPOINT_RUN=v13_idm_frac_75_s0_retry3; LABEL_FRAC=0.75 ;;
    5) RUN_NAME=v13_idm_frac_100_s0; CHECKPOINT_RUN=v13_idm_frac_100_s0_retry3; LABEL_FRAC=1.00 ;;
    *) echo "Unexpected run index: $RUN_INDEX" >&2; exit 2 ;;
esac
PLANNING_EVAL_SEED=$((41 + SEED_INDEX))
EVAL_EPOCH=8
SOLVER=cem
OUTPUT_DIR="$EXPERIMENTS_DIR/analysis_out/planning_eval/$RUN_NAME/epoch_${EVAL_EPOCH}/solver_${SOLVER}/eval_seed_${PLANNING_EVAL_SEED}"
RESULT_FILE="$OUTPUT_DIR/pusht_results.txt"
CHECKPOINT="$STABLEWM_HOME/checkpoints/$CHECKPOINT_RUN/weights_epoch_${EVAL_EPOCH}.pt"

if [[ ! -f "$CHECKPOINT" ]]; then
    echo "Missing checkpoint: $CHECKPOINT" >&2
    exit 1
fi
if [[ -s "$RESULT_FILE" && "${EVAL_OVERWRITE:-0}" != "1" ]]; then
    echo "[skip] $RUN_NAME: existing planning results at $RESULT_FILE"
    exit 0
fi

mkdir -p "$OUTPUT_DIR"
cp "$SCRATCH/datasets/pusht_expert_train.h5" "$SLURM_TMPDIR/"

srun python eval.py \
    "solver=$SOLVER" \
    "policy=$CHECKPOINT_RUN/weights_epoch_${EVAL_EPOCH}.pt" \
    "label_frac=$LABEL_FRAC" \
    "label_seed=0" \
    "seed=$PLANNING_EVAL_SEED" \
    "eval.dataset_name=$SLURM_TMPDIR/pusht_expert_train" \
    "output.filename=$RESULT_FILE" \
    2>&1 | tee "$OUTPUT_DIR/run.log"
