#!/bin/bash
#SBATCH --job-name=la_jepa_pca_plot
#SBATCH --partition=long
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=0:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

module load python/3.10
cd "$HOME/CODE/latent-jepa"
source .venv/bin/activate

GEOMETRY_DIR="$HOME/CODE/latent-jepa-experiments/analysis_out/latent_geometry"
OUTPUT="$GEOMETRY_DIR/v10_v3/epoch_4/pca_spectra_comparison.png"

python "$HOME/CODE/latent-jepa-experiments/plot_combined_pca_spectra.py" \
  --geometry-dir "$GEOMETRY_DIR" \
  --epoch 4 \
  --output "$OUTPUT"

python "$HOME/CODE/latent-jepa-experiments/build_analysis_figure_book.py"