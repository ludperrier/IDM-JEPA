#!/bin/bash
#SBATCH --job-name=la_jepa_train
#SBATCH --partition=long
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=36:00:00
#SBATCH --array=0-44
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --signal=TERM@300  # Send SIGTERM 300 seconds (5 mins) before the end
#SBATCH --requeue

set -euo pipefail

module load python/3.10 cuda/12.2.2
PROJECT_DIR=~/CODE/latent-jepa
cd "$PROJECT_DIR"
source .venv/bin/activate


export STABLEWM_HOME=$SCRATCH/.stable-wm
export SWEEP_TAG="${SWEEP_TAG:-v1}"
export WANDB_MODE=online
export WANDB_INIT_TIMEOUT=600
export HYDRA_FULL_ERROR=1


cp $SCRATCH/datasets/pusht_expert_train.h5 $SLURM_TMPDIR/

EXPECTED_JOBS=$(python -m sweep --count)
# defaults to the v1 45-job sweep; other tags must pass --export=EXPECTED_ARRAY_SIZE=<count>
# alongside a matching --array to submit a differently sized sweep through this script.
EXPECTED_ARRAY_SIZE="${EXPECTED_ARRAY_SIZE:-45}"
if [[ "$EXPECTED_JOBS" -ne "$EXPECTED_ARRAY_SIZE" ]]; then
 echo "sweep.py declares $EXPECTED_JOBS jobs for SWEEP_TAG=$SWEEP_TAG but EXPECTED_ARRAY_SIZE=$EXPECTED_ARRAY_SIZE; update EXPECTED_ARRAY_SIZE and --array to match" >&2
 exit 1
fi

i=$SLURM_ARRAY_TASK_ID
eval "$(python -m sweep "$i")"   # sets ARM RUN SEED COND MODEL FRAC EXTRA
# append to get a fresh checkpoint dir + W&B run id when resubmitting after a
# crash, instead of colliding with an earlier attempt's last.ckpt/W&B history
RUN="${RUN}${RUN_SUFFIX:-}"
echo "[array task $i] ARM=$ARM RUN=$RUN SEED=$SEED COND=$COND MODEL=$MODEL LABEL_FRAC=$FRAC EXTRA='$EXTRA'"

# epochs of plain IDM training before the cycle term switches on. shared by
# every cyc_* arm; ignored by the others.
CYC_WARMUP=${CYC_WARMUP:-3}
RESUME_POLICY=${RESUME_POLICY:-auto}
MAX_EPOCHS=${MAX_EPOCHS:-10}
if [[ -n "$EXTRA" ]]; then
 EXTRA="$EXTRA loss.cycle.warmup_epochs=$CYC_WARMUP"
fi
# free-form extra hydra overrides (e.g. multi-GPU: trainer.devices=2 optimizer.lr=1e-4),
# for one-off submissions that shouldn't change sweep.py's per-arm defaults.
MANUAL_EXTRA=${MANUAL_EXTRA:-}
# -------------------------------------------------------------------------------

# stable-pretraining records a W&B resume ID in the current directory. Array
# tasks must not share it, or one arm can resume another arm's W&B run.
WORK_DIR="$SLURM_TMPDIR/$RUN"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

python -c "import stable_worldmodel.data as d; assert 'hdf5' in d.list_formats(), d.list_formats(); print('env OK')"

set -f  # disable globbing so bracketed overrides (e.g. fsq_levels=[6,5,4]) pass literally
srun python "$PROJECT_DIR/train.py" \
data.dataset.name=$SLURM_TMPDIR/pusht_expert_train.h5 \
seed=$SEED \
conditioning=$COND \
model=$MODEL \
output_model_name=$RUN \
run_group=$ARM \
label_frac=$FRAC \
subdir=$RUN \
resume_policy=$RESUME_POLICY \
trainer.max_epochs=$MAX_EPOCHS \
$EXTRA \
$MANUAL_EXTRA \
hydra.run.dir=$SCRATCH/\${now:%Y%m%d_%H%M%S}_${RUN}

