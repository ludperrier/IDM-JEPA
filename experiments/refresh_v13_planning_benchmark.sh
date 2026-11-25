#!/bin/bash
#SBATCH --job-name=la_jepa_v13_plot
#SBATCH --partition=long
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=analysis_out/slurm_logs/%x_%j.out
#SBATCH --error=analysis_out/slurm_logs/%x_%j.err

set -euo pipefail

module load python/3.10

REPO_DIR="$HOME/CODE/latent-jepa"
EXPERIMENTS_DIR="$HOME/CODE/latent-jepa-experiments"
cd "$REPO_DIR"
source .venv/bin/activate
cd "$EXPERIMENTS_DIR"

python plot_planning_benchmark.py \
    --sweep-tag v3 \
    --sweep-tag v10 \
    --sweep-tag v11 \
    --sweep-tag v12 \
    --sweep-tag v13 \
    --epoch 8 \
    --solver cem \
    --benchmark both
