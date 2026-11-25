#!/bin/bash
#SBATCH --job-name=laj_train
#SBATCH --partition=long
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --array=4
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --signal=TERM@300  # Send SIGTERM 300 seconds (5 mins) before the end
#SBATCH --requeue     

set -euo pipefail


module load python/3.10 cuda/12.2.2
cd ~/CODE/latent-jepa
source .venv/bin/activate

export STABLEWM_HOME=$SCRATCH/.stable-wm
export WANDB_MODE=online
export WANDB_INIT_TIMEOUT=600
export HYDRA_FULL_ERROR=1

cp $SCRATCH/datasets/pusht_expert_train.h5 $SLURM_TMPDIR/

SEED=${SEED:-0}

# --- map array index -> one run ------------------------------------------------
# 0: baseline (no IDM, full action labels)
# 1-5: IDM-conditioned model, swept over decreasing fractions of true action labels
RUNS=(       baseline_100_test   idm_frac_100_test idm_frac_75_test  idm_frac_50_test  idm_frac_25_test  idm_frac_0_test  )
CONDS=(      action     code        code        code        code        code       )
MODELS=(     lewm       lewm_idm    lewm_idm    lewm_idm    lewm_idm    lewm_idm   )
LABEL_FRAC=( 1.0        1.0         0.75        0.5         0.25        0.0        )
EXTRAS=(     ""         ""          ""          ""          ""          ""         )

i=$SLURM_ARRAY_TASK_ID
RUN=${RUNS[$i]}
COND=${CONDS[$i]}
MODEL=${MODELS[$i]}
FRAC=${LABEL_FRAC[$i]}
EXTRA=${EXTRAS[$i]}
echo "[array task $i] RUN=$RUN COND=$COND MODEL=$MODEL LABEL_FRAC=$FRAC EXTRA='$EXTRA'"
# -------------------------------------------------------------------------------

python -c "import stable_worldmodel.data as d; assert 'hdf5' in d.list_formats(), d.list_formats(); print('env OK')"


set -f  # disable globbing so bracketed overrides (e.g. fsq_levels=[6,5,4]) pass literally
srun python train.py \
 data.dataset.name=$SLURM_TMPDIR/pusht_expert_train.h5 \
 seed=$SEED \
 conditioning=$COND \
 model=$MODEL \
 output_model_name=$RUN \
 label_frac=$FRAC \
 subdir=$RUN \
 $EXTRA \
 hydra.run.dir=$SCRATCH/\${now:%Y%m%d_%H%M%S}_${RUN}