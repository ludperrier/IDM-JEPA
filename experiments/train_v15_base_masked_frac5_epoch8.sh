#!/bin/bash
#SBATCH --job-name=la_jepa_v15_train
#SBATCH --partition=long
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --signal=TERM@300
#SBATCH --requeue

# v15: action-conditioned base model (no IDM) trained on 100% of clips, with
# only 5% of transitions carrying their real action label (base_action_masking
# path, action_label_mode=grouped) -- the other 95% get their normalized
# action replaced by a fixed constant (masked_action_value=0.0, i.e. the
# zero vector) instead of being dropped from training like episode_frac does.
# Distinct from v10_base_frac_5_s0, which instead subsampled to 5% of
# episodes (episode_frac=0.05) and kept real actions on every kept clip.

set -euo pipefail

module load python/3.10 cuda/12.2.2
PROJECT_DIR=~/CODE/latent-jepa
cd "$PROJECT_DIR"
source .venv/bin/activate

export STABLEWM_HOME=$SCRATCH/.stable-wm
export WANDB_MODE=online
export WANDB_INIT_TIMEOUT=600
export HYDRA_FULL_ERROR=1

RUN=v15_base_masked_frac5_s0
RESUME_POLICY=${RESUME_POLICY:-auto}
MAX_EPOCHS=8

CHECKPOINT_ROOT="$STABLEWM_HOME/checkpoints"
RUN_DIR="$CHECKPOINT_ROOT/$RUN"
if [[ -f "$RUN_DIR/weights_epoch_${MAX_EPOCHS}.pt" ]]; then
    echo "$RUN already has weights_epoch_${MAX_EPOCHS}.pt; skipping"
    exit 0
fi

cp "$SCRATCH/datasets/pusht_expert_train.h5" "$SLURM_TMPDIR/"

WORK_DIR="$SLURM_TMPDIR/$RUN"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

set -f
srun python "$PROJECT_DIR/train.py" \
    data.dataset.name=$SLURM_TMPDIR/pusht_expert_train.h5 \
    seed=0 \
    conditioning=action \
    model=lewm \
    output_model_name=$RUN \
    run_group=v15_base_masked_frac5 \
    label_frac=0.05 \
    action_label_mode=grouped \
    base_action_masking=true \
    masked_action_value=0.0 \
    episode_frac=1.0 \
    code_dim=10 \
    subdir=$RUN \
    resume_policy=$RESUME_POLICY \
    trainer.max_epochs=$MAX_EPOCHS \
    hydra.run.dir=$SCRATCH/\${now:%Y%m%d_%H%M%S}_${RUN}
