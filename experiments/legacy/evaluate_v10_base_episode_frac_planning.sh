#!/bin/bash
#SBATCH --job-name=la_jepa_v10_base_plan
#SBATCH --partition=long
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --array=0-8
#SBATCH --output=analysis_out/%x_%A_%a.out
#SBATCH --error=analysis_out/%x_%A_%a.err

set -euo pipefail

module load python/3.10 cuda/12.2.2

REPO_DIR="$HOME/CODE/latent-jepa"
EXPERIMENTS_DIR="$HOME/CODE/latent-jepa-experiments"
cd "$REPO_DIR"
source .venv/bin/activate

export STABLEWM_HOME="$SCRATCH/.stable-wm"
export HYDRA_FULL_ERROR=1

case "$SLURM_ARRAY_TASK_ID" in
    0) RUN_NAME=v11_cyc_ramp_frac_5_s0 ;;
    1) RUN_NAME=v11_cyc_ramp_frac_25_s0 ;;
    2) RUN_NAME=v11_cyc_ramp_frac_50_s0 ;;
    3) RUN_NAME=v11_cyc_ramp_frac_75_s0 ;;
    4) RUN_NAME=v11_cyc_ramp_frac_100_s0 ;;
    5) RUN_NAME=v11_idm_frac_5_s0 ;;
    6) RUN_NAME=v11_idm_frac_25_s0 ;;
    7) RUN_NAME=v11_idm_frac_50_s0 ;;
    8) RUN_NAME=v11_idm_frac_75_s0 ;;
    *) echo "Unexpected array task: $SLURM_ARRAY_TASK_ID" >&2; exit 2 ;;
esac

EVAL_EPOCH=5
SOLVER=cem
OUTPUT_DIR="$EXPERIMENTS_DIR/analysis_out/planning_eval/$RUN_NAME/epoch_${EVAL_EPOCH}/solver_${SOLVER}"
CHECKPOINT="$STABLEWM_HOME/checkpoints/$RUN_NAME/weights_epoch_${EVAL_EPOCH}.pt"

if [[ ! -f "$CHECKPOINT" ]]; then
    echo "Missing checkpoint: $CHECKPOINT" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
cp "$SCRATCH/datasets/pusht_expert_train.h5" "$SLURM_TMPDIR/"

srun python eval.py \
    "solver=$SOLVER" \
    "policy=$RUN_NAME/weights_epoch_${EVAL_EPOCH}.pt" \
    "label_frac=$LABEL_FRAC" \
    "label_seed=0" \
    "eval.dataset_name=$SLURM_TMPDIR/pusht_expert_train" \
    "output.filename=$OUTPUT_DIR/pusht_results.txt" \
    2>&1 | tee "$OUTPUT_DIR/run.log"