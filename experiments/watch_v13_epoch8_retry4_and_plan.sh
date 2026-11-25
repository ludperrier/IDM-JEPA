#!/bin/bash
#SBATCH --job-name=la_jepa_v13_retry4_watch
#SBATCH --partition=long
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=3:00:00
#SBATCH --output=analysis_out/slurm_logs/%x_%j.out
#SBATCH --error=analysis_out/slurm_logs/%x_%j.err

# Same polling scheme as watch_v13_epoch8_and_plan.sh, but tracks the _retry4
# checkpoints (submitted on a100 instead of the rtx8000 nodes retry3 landed
# on). Runs alongside the retry3 watcher without touching it.

set -euo pipefail

EXPERIMENTS_DIR="$HOME/CODE/latent-jepa-experiments"
CHECKPOINT_ROOT="${STABLEWM_HOME:-${SCRATCH:?Set SCRATCH}/.stable-wm}/checkpoints"
EVAL_EPOCH=8
POLL_SECONDS="${POLL_SECONDS:-300}"
TIME_BUDGET_SECONDS="${TIME_BUDGET_SECONDS:-9900}"  # stay under the 3h --time limit
SENTINEL="$EXPERIMENTS_DIR/analysis_out/.v13_epoch8_retry4_planning_submitted"

cd "$EXPERIMENTS_DIR"
mkdir -p analysis_out

# frac_0 and frac_100 were cancelled for this retry; only wait on the rest.
RUNS=(v13_idm_frac_5_s0_retry4 v13_idm_frac_25_s0_retry4 v13_idm_frac_50_s0_retry4 v13_idm_frac_75_s0_retry4)

all_ready() {
    local run
    for run in "${RUNS[@]}"; do
        [[ -f "$CHECKPOINT_ROOT/$run/weights_epoch_${EVAL_EPOCH}.pt" ]] || return 1
    done
    return 0
}

submit_planning() {
    echo "All v13 retry4 epoch-${EVAL_EPOCH} checkpoints found; submitting planning eval."
    local planning_job
    planning_job=$(sbatch --parsable "$EXPERIMENTS_DIR/evaluate_v13_epoch8_retry4_planning.sh")
    sbatch --parsable --dependency="afterany:$planning_job" "$EXPERIMENTS_DIR/refresh_v13_planning_benchmark.sh"
    touch "$SENTINEL"
    echo "Submitted planning job $planning_job and its dependent plot refresh."
}

if [[ -f "$SENTINEL" ]]; then
    echo "Planning already submitted per $SENTINEL; exiting."
    exit 0
fi

while ! all_ready; do
    for run in "${RUNS[@]}"; do
        [[ -f "$CHECKPOINT_ROOT/$run/weights_epoch_${EVAL_EPOCH}.pt" ]] \
            || echo "waiting on: $run/weights_epoch_${EVAL_EPOCH}.pt"
    done
    if (( SECONDS + POLL_SECONDS >= TIME_BUDGET_SECONDS )); then
        echo "Time budget reached without all checkpoints ready; resubmitting watcher."
        sbatch "$EXPERIMENTS_DIR/watch_v13_epoch8_retry4_and_plan.sh"
        exit 0
    fi
    sleep "$POLL_SECONDS"
done

submit_planning
