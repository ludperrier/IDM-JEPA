#!/bin/bash
#SBATCH --job-name=la_jepa_plan_eval
#SBATCH --partition=long
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --array=0-44
#SBATCH --output=analysis_out/slurm_logs/%x_%A_%a.out
#SBATCH --error=analysis_out/slurm_logs/%x_%A_%a.err


set -euo pipefail


module load python/3.10 cuda/12.2.2


REPO_DIR="$HOME/CODE/latent-jepa"
EXPERIMENTS_DIR="$HOME/CODE/latent-jepa-experiments"
cd "$REPO_DIR"
source .venv/bin/activate


export STABLEWM_HOME="$SCRATCH/.stable-wm"
: "${SWEEP_TAG:?Set SWEEP_TAG to the tag used for training, for example seed0_retry}"
export SWEEP_TAG
export HYDRA_FULL_ERROR=1


cp "$SCRATCH/datasets/pusht_expert_train.h5" "$SLURM_TMPDIR/"


eval "$(python -m sweep "$SLURM_ARRAY_TASK_ID")"   # sets ARM RUN SEED ...
RUN_NAME="$RUN"
EVAL_EPOCH=${EVAL_EPOCH:-8}
SOLVER=${SOLVER:-cem}
PLANNING_EVAL_SEED=${PLANNING_EVAL_SEED:-42}
OUTPUT_DIR="$EXPERIMENTS_DIR/analysis_out/planning_eval/$RUN_NAME/epoch_${EVAL_EPOCH}/solver_${SOLVER}/eval_seed_${PLANNING_EVAL_SEED}"
mkdir -p "$OUTPUT_DIR"
RESULT_FILE="$OUTPUT_DIR/pusht_results.txt"
if [[ -s "$RESULT_FILE" && "${EVAL_OVERWRITE:-0}" != "1" ]]; then
 echo "[skip] $RUN_NAME: existing planning results at $RESULT_FILE"
 exit 0
fi

CHECKPOINT="$STABLEWM_HOME/checkpoints/$RUN_NAME/weights_epoch_${EVAL_EPOCH}.pt"
if [[ ! -f "$CHECKPOINT" ]]; then
 echo "[skip] $RUN_NAME: checkpoint not available at epoch $EVAL_EPOCH ($CHECKPOINT)"
 exit 0
fi
RUN_CONFIG="$STABLEWM_HOME/checkpoints/$RUN_NAME/config.yaml"
if [[ ! -f "$RUN_CONFIG" ]]; then
 echo "[skip] $RUN_NAME: saved training config not available ($RUN_CONFIG)"
 exit 0
fi


# dataset_name is passed WITHOUT the .h5 suffix: eval.py builds the path via
# HDF5Dataset(name=...), which appends ".h5" itself. The other eval scripts go
# through swm.data.load_dataset(), which takes the full filename -- hence the
# apparent inconsistency between the two.
srun python eval.py \
 "solver=$SOLVER" \
 "policy=$RUN_NAME/weights_epoch_${EVAL_EPOCH}.pt" \
 "label_frac=$FRAC" \
 "label_seed=$SEED" \
 "seed=$PLANNING_EVAL_SEED" \
 "eval.dataset_name=$SLURM_TMPDIR/pusht_expert_train" \
 "output.filename=$RESULT_FILE" \
 2>&1 | tee "$OUTPUT_DIR/run.log"






