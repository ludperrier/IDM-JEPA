# IDM-JEPA

**Semi-Supervised Action Inference via Inverse Dynamics in Joint-Embedding Predictive Architectures**
— *World Models in Physical AI* workshop, NeurIPS 2026.

JEPA world models usually need an action label for every transition. IDM-JEPA adds an
Inverse Dynamics Model (IDM) to the LeWorldModel training setup. The IDM infers latent
action codes from consecutive frame embeddings, and those codes go through the same
action encoder as the real actions. That lets the model train end-to-end on partially
labeled data while the MPC planner still searches over recorded actions.

On continuous PushT, IDM-JEPA reaches **48.4%** planning success with **5%** action
supervision. An action-conditioned baseline trained only on the labeled 5% reaches
**24.4%** with the same number of gradient steps. The gain only shows up when labels are
very scarce: at 25% supervision the baseline already reaches 72.4%. See the paper for the
full results.

## Repository layout

```
latent-action-jepa/   Training and evaluation code (model, IDM, Hydra configs)
  train.py            Training entry point (Hydra, config/train/)
  eval.py             Planning evaluation with MPC (CEM / Adam solvers, config/eval/)
  jepa.py, module.py  JEPA world model + inverse dynamics model
  probing.py, evaluate_next_state.py, evaluate_latent_geometry.py
experiments/          SLURM launchers, analysis/plotting scripts, paper tables
  paper/              LaTeX source of the paper (main.tex, refs.bib, figs/)
  results/            Selected evaluation outputs (CSV summaries, plots, paper export)
  legacy/             Superseded one-off scripts, kept for reference
workshop/             Paper PDFs and posters for the NeurIPS workshop
```

The README in each subfolder has more detail:
- [`latent-action-jepa/README.md`](latent-action-jepa/README.md): benchmarks (PushT,
  DMC Reacher), latent geometry and downstream evaluation.
- [`experiments/README.md`](experiments/README.md): sweep tags (v10 to v13 ablations),
  launching sweeps and multi-GPU notes.

## Quick start

The code builds on [`stable_worldmodel`](https://github.com/galilai-group/stable-worldmodel)
and `stable_pretraining`, plus PyTorch Lightning, Hydra, einops, scikit-learn,
pandas and matplotlib.

```bash
cd latent-action-jepa
# train on PushT (default) with the IDM model
python train.py model=lewm_idm
# planning evaluation of a checkpoint
python eval.py --config-name pusht policy=<run>/weights_epoch_<epoch>.pt
```

## Notes

- The experiment scripts were written for the Mila SLURM cluster. They expect the
  training code at `~/CODE/latent-jepa` and this folder at
  `~/CODE/latent-jepa-experiments`, so adjust those paths for your own setup.
- `sweep.py` (the sweep definition used by `experiments/train.sh`, `probing.py` and
  `evaluate_next_state.py`) is not yet in this
  repository.
- Model weights and raw per-checkpoint artifacts (latent embeddings `.npz`, SLURM logs,
  t-SNE/PCA dumps) are not included because of their size.
