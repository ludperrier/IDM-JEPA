README FILE

 Problem statement : Latent world models, particularly ones based on the joint embedding predictive architecture (JEPA) paradigm, have shown promise in their ability to achieve considerable performance on robotics benchmarks through a SSL end-to-end training approach. LeWorldModel (Maes et al. 2026), incorporates the LeJEPA (Balestriero et al. 2025) model architecture along with Model Predictive Control (MPC) to achieve efficient planning in various environments. However, such models require an action signal to be provided along each frame observation to recover a subsequent latent state, which might not always be present in a given dataset. This work aims to incorporate an inverse dynamics model (IDM) to the LeWorldModel training architecture, recovering a latent action signal from consecutive frame embeddings, and evaluating model planning performance in downstream tasks across varying levels of action-label sparsity. (draft)

## Benchmarks

PushT remains the default benchmark. The original leWM Reacher benchmark is available through the `dmc` training-data preset and the `reacher` evaluation config.

Train on an HDF5 Reacher dataset (for example, a local `reacher.h5`):

```bash
python train.py data=dmc data.dataset.name=/path/to/reacher.h5
```

Run planning evaluation against the canonical DM Control qpos-matching task. `eval.py` expects the dataset name without its `.h5` suffix, matching `stable_worldmodel.data.HDF5Dataset` behavior:

```bash
python eval.py --config-name reacher \
	policy=<run>/weights_epoch_<epoch>.pt \
	eval.dataset_name=/path/to/reacher_random
```

## Latent geometry

Generate held-out PCA visualizations for every available checkpoint in a sweep:

```bash
cd ../latent-jepa-experiments
SWEEP_TAG=v7 EVAL_SEEDS=0 EVAL_EPOCH=8 \
  sbatch --export=ALL evaluate_latent_geometry.sh
```

Set `EVAL_RUNS` to a comma-separated subset of run names. Set
`GEOMETRY_QUANTITY` to `agent_x`, `agent_y`, `block_x`, `block_y`, or
`block_orientation`; orientation is the default. `GEOMETRY_NUM_SAMPLES`
controls the maximum number of held-out embeddings per checkpoint. t-SNE uses
a reproducible random subset controlled by `GEOMETRY_TSNE_NUM_SAMPLES`
(default `5000`) and `GEOMETRY_TSNE_SEED`.

Results are written under
`analysis_out/latent_geometry/<tag>/epoch_<epoch>/`. Each run receives a
three-panel PCA figure, a t-SNE embedding figure, a full PCA spectrum CSV,
and compressed PCA and t-SNE coordinates. The root directory also contains a
cross-run spectrum figure and summary CSV.

## Downstream evaluation

Next-state and probe evaluations write one metrics CSV per completed model and
skip those files on later invocations. This makes scheduler retries safe: submit
the same launcher again to evaluate only newly available checkpoints. Set
`EVAL_OVERWRITE=1` to recompute existing metrics, or set `EVAL_RUNS` to a
comma-separated subset.

Use the experiment-side report to see which available models still need each
evaluation artifact:

```bash
cd ../latent-jepa-experiments
../latent-jepa/.venv/bin/python evaluation_status.py \
	--sweep-tag v10 --epoch 8 --checkpoint-root "$SCRATCH/.stable-wm/checkpoints"
```

It writes `analysis_out/evaluation_status/<tag>/epoch_<epoch>/evaluation_status.csv`
and reports checkpoint, next-state, probe, planning, PCA, and t-SNE coverage.

Planning repeats use independent evaluation seeds and are saved separately. For
example, run three CEM evaluations of the IDM 50% model with array task `3`:

```bash
cd ../latent-jepa-experiments
for planning_seed in 41 42 43; do
	SWEEP_TAG=v10 EVAL_EPOCH=8 SOLVER=cem PLANNING_EVAL_SEED="$planning_seed" \
		sbatch --export=ALL --array=3 evaluate_planning.sh
done
```

The outputs live in `planning_eval/<run>/epoch_<epoch>/solver_<solver>/eval_seed_<seed>/`.
`plot_planning_benchmark.py` writes per-repeat results plus
`planning_repeat_summary.csv`; plotted error bars are one standard deviation
across evaluation seeds.
