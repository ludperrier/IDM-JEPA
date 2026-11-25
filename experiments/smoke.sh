#!/bin/bash
#SBATCH --job-name=la_jepa_smoke
#SBATCH --partition=long
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

module load python/3.10 cuda/12.2.2
cd ~/CODE/latent-jepa
source .venv/bin/activate

export STABLEWM_HOME="$SCRATCH/.stable-wm"
export SWEEP_TAG="${SWEEP_TAG:-v1}"
export HYDRA_FULL_ERROR=1
export WANDB_MODE=online
export WANDB_INIT_TIMEOUT=600

cp "$SCRATCH/datasets/pusht_expert_train.h5" "$SLURM_TMPDIR/"
EPISODE_KEY=${EPISODE_KEY:-episode_idx}

srun python smoke.py \
 --dataset "$SLURM_TMPDIR/pusht_expert_train.h5" \
 --wandb \
 --episode-key "$EPISODE_KEY"
