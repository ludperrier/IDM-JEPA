#!/bin/bash
# Submit through submit_paper_epoch8_evaluations.sh; it sizes the array from
# the available epoch-8 checkpoints rather than relying on a stale fixed list.
#SBATCH --job-name=la_jepa_paper_plan
#SBATCH --partition=long
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
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

manifest_args=()
if [[ -n "${PAPER_MANIFEST_CSV:-}" ]]; then
    manifest_args+=(--manifest-csv "$PAPER_MANIFEST_CSV")
fi
if [[ -n "${PAPER_PLANNING_TASK_CSV:-}" ]]; then
    manifest_args+=(--planning-task-csv "$PAPER_PLANNING_TASK_CSV")
fi
if [[ -n "${PAPER_EXCLUDE_TAGS:-}" ]]; then
    IFS=',' read -r -a excluded_tags <<< "$PAPER_EXCLUDE_TAGS"
    for tag in "${excluded_tags[@]}"; do
        manifest_args+=(--exclude-tag "$tag")
    done
fi
if [[ "${PAPER_MISSING_PLANNING:-0}" == "1" ]]; then
    manifest_args+=(--missing-planning)
fi
if [[ -n "${PAPER_PLANNING_DIR:-}" ]]; then
    manifest_args+=(--planning-dir "$PAPER_PLANNING_DIR")
fi
eval "$(python "$EXPERIMENTS_DIR/paper_epoch8_manifest.py" --epoch 8 --format shell --index "$SLURM_ARRAY_TASK_ID" "${manifest_args[@]}")"

: "${EVAL_EPOCH:=8}"
: "${CHECKPOINT_RUN:=$RUN_NAME}"
SOLVER=cem
OUTPUT_DIR="$EXPERIMENTS_DIR/analysis_out/planning_eval/$RUN_NAME/epoch_${EVAL_EPOCH}/solver_${SOLVER}/eval_seed_${PLANNING_EVAL_SEED}"
RESULT_FILE="$OUTPUT_DIR/pusht_results.txt"
CHECKPOINT="$STABLEWM_HOME/checkpoints/$CHECKPOINT_RUN/weights_epoch_${EVAL_EPOCH}.pt"

if [[ ! -f "$CHECKPOINT" ]]; then
    echo "Missing checkpoint: $CHECKPOINT" >&2
    exit 1
fi
if [[ -s "$RESULT_FILE" && "${EVAL_OVERWRITE:-0}" != "1" ]]; then
    echo "[skip] $RUN_NAME eval seed $PLANNING_EVAL_SEED: existing planning results"
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