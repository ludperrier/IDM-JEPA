#!/bin/bash
#SBATCH --job-name=la_jepa_eval
#SBATCH --partition=long
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

module load python/3.10 cuda/12.2.2
cd ~/CODE/latent-jepa
source .venv/bin/activate

export STABLEWM_HOME="$SCRATCH/.stable-wm"
 : "${SWEEP_TAG:?Set SWEEP_TAG to the tag used for training, for example seed0_retry}"
export SWEEP_TAG
: "${EVAL_SEEDS:?Set EVAL_SEEDS to the trained seeds, for example 0 or 0,1,2}"
export EVAL_SEEDS
export EVAL_OVERWRITE=${EVAL_OVERWRITE:-0}
export HYDRA_FULL_ERROR=1

cp "$SCRATCH/datasets/pusht_expert_train.h5" "$SLURM_TMPDIR/"

EVAL_EPOCH=${EVAL_EPOCH:-8}
OUTPUT_DIR="$HOME/CODE/latent-jepa-experiments/analysis_out/next_state_eval/${SWEEP_TAG}/epoch_${EVAL_EPOCH}"
mkdir -p "$OUTPUT_DIR"
RUN_SUFFIX=${EVAL_RUNS:-all}
RUN_SUFFIX=${RUN_SUFFIX//,/_}

srun python evaluate_next_state.py \
  data.dataset.name="$SLURM_TMPDIR/pusht_expert_train.h5" \
  eval_epoch="$EVAL_EPOCH" \
  eval_output_dir="$OUTPUT_DIR" \
  2>&1 | tee "$OUTPUT_DIR/run_${RUN_SUFFIX}.log"