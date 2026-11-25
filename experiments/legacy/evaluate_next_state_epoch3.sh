#!/bin/bash
#SBATCH --job-name=la_jepa_eval_e3
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
export HYDRA_FULL_ERROR=1

cp "$SCRATCH/datasets/pusht_expert_train.h5" "$SLURM_TMPDIR/"

OUTPUT_DIR="$HOME/CODE/latent-jepa-experiments/analysis_out/next_state_eval/epoch_3"
mkdir -p "$OUTPUT_DIR"

srun python evaluate_next_state.py \
  data.dataset.name="$SLURM_TMPDIR/pusht_expert_train.h5" \
  eval_epoch=3 \
  eval_output_dir="$OUTPUT_DIR" \
  2>&1 | tee "$OUTPUT_DIR/run.log"