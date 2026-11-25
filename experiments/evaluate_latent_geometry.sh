#!/bin/bash
#SBATCH --job-name=la_jepa_geometry
#SBATCH --partition=long
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=3:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

module load python/3.10 cuda/12.2.2
cd ~/CODE/latent-jepa
source .venv/bin/activate

export STABLEWM_HOME="$SCRATCH/.stable-wm"
: "${SWEEP_TAG:?Set SWEEP_TAG to the tag used for training, for example v7}"
export SWEEP_TAG
: "${EVAL_SEEDS:?Set EVAL_SEEDS to the trained seeds, for example 0 or 0,1,2}"
export EVAL_SEEDS
export HYDRA_FULL_ERROR=1

cp "$SCRATCH/datasets/pusht_expert_train.h5" "$SLURM_TMPDIR/"

EVAL_EPOCH=${EVAL_EPOCH:-8}
GEOMETRY_QUANTITY=${GEOMETRY_QUANTITY:-block_orientation}
GEOMETRY_NUM_SAMPLES=${GEOMETRY_NUM_SAMPLES:-50000}
GEOMETRY_TSNE_NUM_SAMPLES=${GEOMETRY_TSNE_NUM_SAMPLES:-5000}
GEOMETRY_TSNE_SEED=${GEOMETRY_TSNE_SEED:-3072}
OUTPUT_DIR=${GEOMETRY_OUTPUT_DIR:-"$HOME/CODE/latent-jepa-experiments/analysis_out/latent_geometry/${SWEEP_TAG}/epoch_${EVAL_EPOCH}"}
mkdir -p "$OUTPUT_DIR"
RUN_SUFFIX=${EVAL_RUNS:-all}
RUN_SUFFIX=${RUN_SUFFIX//,/_}

srun python evaluate_latent_geometry.py \
  data.dataset.name="$SLURM_TMPDIR/pusht_expert_train.h5" \
  eval_epoch="$EVAL_EPOCH" \
  eval_output_dir="$OUTPUT_DIR" \
  geometry_quantity="$GEOMETRY_QUANTITY" \
  geometry_num_samples="$GEOMETRY_NUM_SAMPLES" \
  geometry_tsne_num_samples="$GEOMETRY_TSNE_NUM_SAMPLES" \
  geometry_tsne_seed="$GEOMETRY_TSNE_SEED" \
  2>&1 | tee "$OUTPUT_DIR/run_${RUN_SUFFIX}.log"