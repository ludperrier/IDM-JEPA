#!/bin/bash
#SBATCH --job-name=la_jepa_paper_export
#SBATCH --partition=long
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:20:00
#SBATCH --output=analysis_out/slurm_logs/%x_%j.out
#SBATCH --error=analysis_out/slurm_logs/%x_%j.err

set -euo pipefail

REPO_DIR="$HOME/CODE/latent-jepa"
EXPERIMENTS_DIR="$HOME/CODE/latent-jepa-experiments"
source "$REPO_DIR/.venv/bin/activate"
cd "$EXPERIMENTS_DIR"

python plot_planning_benchmark.py \
    --sweep-tag v3 --sweep-tag v10 --sweep-tag v11 --sweep-tag v12 --sweep-tag v13 \
    --epoch "${PAPER_EPOCH:-8}" --solver cem --benchmark both

python build_paper_epoch8_export.py \
    --epoch "${PAPER_EPOCH:-8}" \
    --checkpoint-root "${PAPER_CHECKPOINT_ROOT:-${SCRATCH}/.stable-wm/checkpoints}"