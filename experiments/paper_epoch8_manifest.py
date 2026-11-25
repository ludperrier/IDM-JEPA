#!/usr/bin/env python3
"""Discover the requested epoch-8 models and expose their paper categories.

The manifest is intentionally derived from checkpoint directories, so incomplete
training runs are recorded as unavailable instead of being silently evaluated at
another epoch.
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd


TAGS = ("v3", "v10", "v11", "v12", "v13")
PLANNING_SEEDS = (41, 42, 43, 44, 45)
RUN_PATTERN = re.compile(
    r"(?P<tag>v3|v10|v11|v12|v13)_(?P<arm>baseline_100|base_frac_\d+|idm_frac_\d+|cyc_ramp_frac_\d+)_s(?P<seed>\d+)"
)

PAPER_MODELS = (
    *(f"v10_idm_frac_{fraction}_s0" for fraction in (0, 5, 25, 50, 75, 100)),
    "v11_idm_frac_5_s0",
    "v11_idm_frac_25_s0",
    "v11_idm_frac_50_s0",
    "v11_idm_frac_75_s0",
    *(f"v12_idm_frac_{fraction}_s0" for fraction in (0, 5, 25, 50, 75, 100)),
    *(f"v13_idm_frac_{fraction}_s0" for fraction in (0, 5, 25, 50, 75, 100)),
    "v3_baseline_100_s0",
    *(f"v10_base_frac_{fraction}_s0" for fraction in (5, 25, 50, 75)),
    "v3_cyc_ramp_frac_0_s0",
    "v3_cyc_ramp_frac_100_s0",
    *(f"v10_cyc_ramp_frac_{fraction}_s0" for fraction in (5, 25, 50, 75)),
    *(f"v11_cyc_ramp_frac_{fraction}_s0" for fraction in (5, 25, 50, 75)),
)
CHECKPOINT_SOURCES = {
    "v11_idm_frac_75_s0": "v11_idm_frac_75_retry_s0_retry_20260817_151047",
    # frac_0/100 retry3 checkpoints stalled at epoch 5 and were superseded by
    # retry4, which only reran the middle four fractions.
    **{f"v13_idm_frac_{fraction}_s0": f"v13_idm_frac_{fraction}_s0_retry4" for fraction in (5, 25, 50, 75)},
    "v13_idm_frac_0_s0": "v13_idm_frac_0_s0_retry3",
    "v13_idm_frac_100_s0": "v13_idm_frac_100_s0_retry3",
}
CHECKPOINT_EPOCHS = {
    "v3_cyc_ramp_frac_100_s0": 7,
}


@dataclass(frozen=True)
class Model:
    run: str
    checkpoint_run: str
    checkpoint_epoch: int
    tag: str
    arm: str
    train_seed: int
    label_fraction_pct: int
    family: str
    supervision: str
    category: str
    subset: str


def default_checkpoint_root() -> Path:
    return Path(os.environ.get("STABLEWM_HOME", Path(os.environ.get("SCRATCH", Path.home() / "scratch")) / ".stable-wm")) / "checkpoints"


def parse_model(run: str, checkpoint_epoch: int = 8) -> Model | None:
    match = RUN_PATTERN.fullmatch(run)
    if match is None:
        return None
    tag = match["tag"]
    arm = match["arm"]
    if arm == "baseline_100":
        fraction = 100
        family = "JEPA baseline"
        supervision = "reference"
        category = "jepa-baseline"
        subset = "all"
    elif arm.startswith("base_frac_"):
        fraction = int(arm.rsplit("_", 1)[1])
        family = "JEPA baseline"
        supervision = "reference"
        category = "jepa-baseline"
        subset = "all"
    elif arm.startswith("idm_frac_"):
        fraction = int(arm.rsplit("_", 1)[1])
        if tag == "v13":
            family = "IDM-JEPA concat"
            supervision = "per-episode"
            category = "idm-planning"
            subset = "concat/per-episode"
        elif tag == "v12":
            family = "IDM-JEPA concat"
            supervision = "per-transition"
            category = "idm-planning"
            subset = "concat/per-transition"
        elif tag == "v11":
            family = "IDM-JEPA delta"
            supervision = "per-episode"
            category = "idm-planning"
            subset = "delta/per-episode"
        else:
            family = "IDM-JEPA delta"
            supervision = "per-transition"
            category = "idm-planning"
            subset = "delta/per-transition"
    else:
        fraction = int(arm.rsplit("_", 1)[1])
        family = "Cycle-JEPA"
        supervision = "per-episode" if tag == "v11" else "per-transition"
        category = "cycle-planning"
        subset = f"delta/{supervision}"
    return Model(
        run=run,
        checkpoint_run=CHECKPOINT_SOURCES.get(run, run),
        checkpoint_epoch=CHECKPOINT_EPOCHS.get(run, checkpoint_epoch),
        tag=tag,
        arm=arm,
        train_seed=int(match["seed"]),
        label_fraction_pct=fraction,
        family=family,
        supervision=supervision,
        category=category,
        subset=subset,
    )


def discover_models(checkpoint_root: Path, epoch: int) -> list[Model]:
    models = []
    for run in PAPER_MODELS:
        model = parse_model(run, epoch)
        if model is None:
            continue
        checkpoint = checkpoint_root / model.checkpoint_run / f"weights_epoch_{model.checkpoint_epoch}.pt"
        if checkpoint.is_file():
            models.append(model)
    return models


def missing_planning_tasks(models: list[Model], planning_dir: Path, solver: str) -> list[tuple[Model, int]]:
    tasks = []
    for model in models:
        for planning_seed in PLANNING_SEEDS:
            result = (
                planning_dir
                / model.run
                / f"epoch_{model.checkpoint_epoch}"
                / f"solver_{solver}"
                / f"eval_seed_{planning_seed}"
                / "pusht_results.txt"
            )
            if not result.is_file() or result.stat().st_size == 0:
                tasks.append((model, planning_seed))
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epoch", type=int, default=8)
    parser.add_argument("--checkpoint-root", type=Path, default=default_checkpoint_root())
    parser.add_argument("--tag", choices=TAGS)
    parser.add_argument("--checkpoint-epoch", type=int)
    parser.add_argument("--exclude-tag", action="append", choices=TAGS, default=[])
    parser.add_argument("--manifest-csv", type=Path, help="Use run names from a frozen manifest CSV.")
    parser.add_argument("--planning-task-csv", type=Path, help="Use frozen run/checkpoint/seed planning tasks from a CSV.")
    parser.add_argument("--output", type=Path, help="Write CSV manifest to this file.")
    parser.add_argument("--missing-planning", action="store_true", help="Restrict planning tasks to missing or empty result files.")
    parser.add_argument("--planning-dir", type=Path, default=Path(__file__).parent / "analysis_out" / "planning_eval")
    parser.add_argument("--solver", default="cem")
    parser.add_argument("--format", choices=("csv", "runs", "shell", "count"), default="csv")
    parser.add_argument("--index", type=int, help="Planning-task index when --format=shell.")
    args = parser.parse_args()

    frozen_tasks = None
    if args.planning_task_csv:
        task_frame = pd.read_csv(args.planning_task_csv)
        frozen_tasks = []
        for row in task_frame.itertuples(index=False):
            model = parse_model(row.run)
            if model is None:
                continue
            frozen_tasks.append((model, int(row.planning_seed)))
        models = [model for model, _ in frozen_tasks]
    elif args.manifest_csv:
        runs = pd.read_csv(args.manifest_csv, usecols=["run"])["run"]
        models = [model for run in runs if (model := parse_model(run)) is not None]
    else:
        models = discover_models(args.checkpoint_root, args.epoch)
    if args.tag:
        models = [model for model in models if model.tag == args.tag]
    if args.checkpoint_epoch:
        models = [model for model in models if model.checkpoint_epoch == args.checkpoint_epoch]
    if args.exclude_tag:
        excluded_tags = set(args.exclude_tag)
        models = [model for model in models if model.tag not in excluded_tags]

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(asdict(model) for model in models).to_csv(args.output, index=False)

    tasks = frozen_tasks if frozen_tasks is not None else (
        missing_planning_tasks(models, args.planning_dir, args.solver)
        if args.missing_planning
        else [(model, planning_seed) for model in models for planning_seed in PLANNING_SEEDS]
    )

    if args.format == "csv":
        pd.DataFrame(asdict(model) for model in models).to_csv(os.sys.stdout, index=False)
    elif args.format == "runs":
        print(",".join(model.run for model in models))
    elif args.format == "count":
        print(len(tasks))
    else:
        if args.index is None:
            parser.error("--index is required when --format=shell")
        task_count = len(tasks)
        if not 0 <= args.index < task_count:
            raise SystemExit(0)
        model, planning_seed = tasks[args.index]
        print(
            " ".join(
                f"{name}={shlex.quote(str(value))}"
                for name, value in {
                    "RUN_NAME": model.run,
                    "CHECKPOINT_RUN": model.checkpoint_run,
                    "EVAL_EPOCH": model.checkpoint_epoch,
                    "LABEL_FRAC": model.label_fraction_pct / 100,
                    "PLANNING_EVAL_SEED": planning_seed,
                }.items()
            )
        )


if __name__ == "__main__":
    main()