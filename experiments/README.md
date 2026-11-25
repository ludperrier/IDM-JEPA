# latent-jepa-experiments

SLURM orchestration, evaluation, and plotting for the `latent-jepa` project
(training code lives in the sibling `~/CODE/latent-jepa` repo; this repo only
holds sbatch scripts, analysis code, and evaluation outputs).

## Layout

```
sweep.py, train.py, eval.py, module.py, ...   -> live in ~/CODE/latent-jepa (not here)
train.sh                                      -> generic training array launcher (reads sweep.py)
evaluate_*.sh, refresh_*.sh, watch_*.sh        -> evaluation / planning / plotting pipeline
paper_epoch8_manifest.py                      -> discovers which epoch-8 checkpoints exist and
                                                  classifies them by (tag, family, supervision)
plot_planning_benchmark.py                    -> planning-accuracy plots, keyed by --sweep-tag
build_paper_epoch8_export.py                  -> assembles the full paper export bundle
analysis_out/                                 -> all evaluation outputs (git-ignored except a
                                                  few curated top-level CSV/PNG summaries)
analysis_out/slurm_logs/                      -> stdout/stderr from evaluation/plotting sbatch jobs
logs/                                         -> stdout/stderr from training sbatch jobs (git-ignored)
```

## Sweep-tag conventions

Training is organized into `SWEEP_TAG`s (defined in `latent-jepa/sweep.py`).
The default (`SWEEP_TAG` unset / `v1`) is the original 3-seed, 45-job sweep.
Later tags are single-seed ablations layered on top, each isolating one axis:

| tag | IDM input     | action-label masking | seeds | notes |
|-----|---------------|-----------------------|-------|-------|
| v10 | difference    | grouped (per-transition) | 0 | baseline ablation grid |
| v11 | difference    | episode (per-episode)    | 0 | episode-masking ablation |
| v12 | concatenation | grouped (per-transition) | 0 | concat-input ablation |
| v13 | concatenation | episode (per-episode)    | 0 | concat + episode-masking |

All ablation tags use label fractions `0, 5, 25, 50, 75, 100` (%) and
`code_dim=10` (overriding the checked-in default of 3, for comparability
across tags). These per-arm knobs are Hydra CLI overrides emitted by
`sweep.py`'s `_emit()`:

- `model.idm.input_mode=difference|concatenation`
- `action_label_mode=grouped|episode`
- `code_dim=10`

## Launching a new tagged sweep

`train.sh`'s `#SBATCH --array=0-44` / job-count assertion is only the default
for the untagged v1 sweep. For any other tag, override at submit time rather
than editing the file:

```bash
sbatch --array=0-5 --export=ALL,SWEEP_TAG=v13,EXPECTED_ARRAY_SIZE=6 train.sh
```

`EXPECTED_ARRAY_SIZE` must match `python -m sweep --count` for that tag (the
script asserts this and refuses to run otherwise, so a mismatched `--array`
fails fast instead of silently only training part of the sweep).

For one-off overrides that shouldn't become permanent `sweep.py` defaults
(e.g. multi-GPU DDP for a single submission), use `MANUAL_EXTRA`:

```bash
sbatch --array=0-5 --ntasks-per-node=2 --gres=gpu:rtx8000:2 \
  --export=ALL,SWEEP_TAG=v13,EXPECTED_ARRAY_SIZE=6,MANUAL_EXTRA="trainer.devices=2 optimizer.lr=1e-4" \
  train.sh
```

Notes on multi-GPU: `pl.Trainer(**cfg.trainer)` and the checkpoint callback
(gated on `trainer.is_global_zero`) already support DDP. `trainer.num_nodes`
is *not* a key in `config/train/lewm.yaml`'s `trainer:` block, so don't pass
it as a plain override (Hydra struct mode rejects unknown keys) — it defaults
to 1 anyway, which is what you want for single-node DDP. Scale
`optimizer.lr` with GPU count if asked to; per-GPU `loader.batch_size` stays
fixed, so the effective batch size multiplies by the GPU count.

Cluster note: `a100` supply on `long` is tiny (a couple of 8-GPU nodes);
`rtx8000` is much more plentiful. Prefer `rtx8000` if you need a shorter
queue wait. Requesting *more* GPUs per job increases wait time, it doesn't
reduce it.

## Planning-on-ready automation

Each concat/episode-masking ablation (v12, v13, ...) has a three-script chain:

1. `watch_v<N>_epoch8_and_plan.sh` — cheap CPU job, polls for all epoch-8
   checkpoints, self-resubmits via `sbatch` if its time budget runs out
   before they're ready, and once ready submits (2) then (3).
2. `evaluate_v<N>_epoch8_planning.sh` — GPU array job, one task per
   (label fraction x planning eval seed).
3. `refresh_v<N>_planning_benchmark.sh` — regenerates the planning-accuracy
   plots via `plot_planning_benchmark.py --sweep-tag ... --benchmark both`.

To add this for a new tag, copy the three v13 scripts and rename the tag /
run list. Copy this pattern instead of manually polling checkpoints.

The "official" paper-wide pipeline (`submit_paper_epoch8_evaluations.sh` +
`refresh_paper_epoch8_export.sh` + `paper_epoch8_manifest.py`) is separate:
it evaluates every tag together and builds the full export bundle under
`analysis_out/paper_export/`. When adding a tag there, update:

- `paper_epoch8_manifest.py`: `TAGS`, `RUN_PATTERN`, `PAPER_MODELS`, and the
  `parse_model()` branch (family / supervision / category / subset strings).
- `build_paper_epoch8_export.py`: `paper_groups()` — add the
  `(category, subset)` group and the classification branch.
- `plot_planning_benchmark.py`: `BENCHMARK_SERIES["per_transition"|"per_episode"]`
  — add the explicit run-name list and a style entry if it needs a distinct
  marker/linestyle from its family.
- `submit_paper_epoch8_evaluations.sh` and `refresh_paper_epoch8_export.sh`:
  add the tag to their hardcoded tag lists.

`evaluate_planning.sh` is a simpler `SWEEP_TAG`+array-index-driven planning eval
(an alternative to the manifest-driven `evaluate_paper_epoch8_planning.sh`).

## Checking on jobs

```bash
squeue --me -o "%.12i %.20j %.9P %.2t %.10M %.6D %R" -r   # your queue
sacct -u $USER -j <jobid> --format=JobID,State,ExitCode,Elapsed -X
```

Training/eval stdout+stderr: `logs/<job-name>_<jobid>_<array-idx>.{out,err}`.
Evaluation/plotting stdout+stderr: `analysis_out/slurm_logs/<job-name>_<jobid>.{out,err}`.
