#!/bin/bash
# Submit the prescribed paper model matrix and regenerate its figures when done.
set -euo pipefail

EXPERIMENTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$HOME/CODE/latent-jepa}"
CHECKPOINT_ROOT="${STABLEWM_HOME:-${SCRATCH:?Set SCRATCH}/.stable-wm}/checkpoints"
PYTHON_BIN="${PYTHON_BIN:-$REPO_DIR/.venv/bin/python}"
cd "$EXPERIMENTS_DIR"

planning_manifest="$EXPERIMENTS_DIR/analysis_out/paper_export/epoch_8/csv/planning_submission_manifest.csv"
"$PYTHON_BIN" "$EXPERIMENTS_DIR/paper_epoch8_manifest.py" \
    --epoch 8 --checkpoint-root "$CHECKPOINT_ROOT" --output "$planning_manifest"
planning_count=$("$PYTHON_BIN" "$EXPERIMENTS_DIR/paper_epoch8_manifest.py" \
    --manifest-csv "$planning_manifest" --format count)
planning_job=${PAPER_PLANNING_JOB_ID:-}
if [[ -z "$planning_job" ]]; then
    array_spec="0-$((planning_count - 1))%16"
    planning_job=$(PAPER_MANIFEST_CSV="$planning_manifest" sbatch --parsable --array="$array_spec" --export=ALL \
        "$EXPERIMENTS_DIR/evaluate_paper_epoch8_planning.sh")
fi
job_ids=("$planning_job")

for tag in v3 v10 v11 v12 v13; do
    for eval_epoch in 7 8; do
        runs=$("$PYTHON_BIN" "$EXPERIMENTS_DIR/paper_epoch8_manifest.py" \
            --epoch 8 --checkpoint-root "$CHECKPOINT_ROOT" --tag "$tag" \
            --checkpoint-epoch "$eval_epoch" --format runs)
        [[ -n "$runs" ]] || continue
        export SWEEP_TAG="$tag"
        export EVAL_SEEDS=0
        export EVAL_EPOCH="$eval_epoch"
        export EVAL_RUNS="$runs"
        export EVAL_RUN_ALIASES="v11_idm_frac_75_s0=v11_idm_frac_75_retry_s0_retry_20260817_151047"
        export EVAL_OVERWRITE=0
        job_ids+=("$(sbatch --parsable --export=ALL "$EXPERIMENTS_DIR/evaluate_latent_geometry.sh")")
        job_ids+=("$(sbatch --parsable --export=ALL "$EXPERIMENTS_DIR/evaluate_next_state.sh")")
        job_ids+=("$(sbatch --parsable --export=ALL "$EXPERIMENTS_DIR/evaluate_probing.sh")")
    done
done
unset SWEEP_TAG EVAL_SEEDS EVAL_EPOCH EVAL_RUNS EVAL_RUN_ALIASES EVAL_OVERWRITE
dependency=$(IFS=:; echo "${job_ids[*]}")
export PAPER_EPOCH=8
export PAPER_CHECKPOINT_ROOT="$CHECKPOINT_ROOT"
export_job=$(sbatch --parsable --dependency="afterany:$dependency" --export=ALL \
    "$EXPERIMENTS_DIR/refresh_paper_epoch8_export.sh")
unset PAPER_EPOCH PAPER_CHECKPOINT_ROOT

printf 'Planning: %s\nExport (after evaluations): %s\n' "$planning_job" "$export_job"
printf 'All submitted job IDs: %s\n' "$dependency"