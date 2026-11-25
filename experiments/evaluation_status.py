#!/usr/bin/env python3
"""Report downstream-evaluation coverage for one training sweep.

Example:
    python evaluation_status.py --sweep-tag v10 --epoch 8
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

EXPERIMENTS_DIR = Path(__file__).resolve().parent


def has_csv_result(output_dir: Path, metric_name: str, run: str) -> bool:
    if (output_dir / f"{metric_name}_{run}.csv").exists():
        return True
    for path in output_dir.glob(f"{metric_name}*.csv"):
        try:
            if run in set(pd.read_csv(path, usecols=["run"])["run"]):
                return True
        except (OSError, ValueError, pd.errors.EmptyDataError):
            continue
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-tag", required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument(
        "--runs", help="Optional comma-separated run names; defaults to checkpoint directories."
    )
    parser.add_argument("--solver", default="cem")
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path(os.environ.get("SCRATCH", Path.home() / "scratch"))
        / ".stable-wm/checkpoints",
    )
    parser.add_argument(
        "--analysis-dir", type=Path, default=EXPERIMENTS_DIR / "analysis_out"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.runs:
        runs = sorted({run.strip() for run in args.runs.split(",") if run.strip()})
    else:
        runs = sorted(path.name for path in args.checkpoint_root.glob(f"{args.sweep_tag}_*") if path.is_dir())
    if not runs:
        raise FileNotFoundError(
            f"No checkpoint directories found for sweep tag {args.sweep_tag!r} in "
            f"{args.checkpoint_root}."
        )

    next_state_dir = args.analysis_dir / "next_state_eval" / args.sweep_tag / f"epoch_{args.epoch}"
    probe_dir = args.analysis_dir / "position_probe" / args.sweep_tag / f"epoch_{args.epoch}"
    geometry_dir = args.analysis_dir / "latent_geometry" / args.sweep_tag / f"epoch_{args.epoch}"
    records = []
    for run in runs:
        checkpoint_dir = args.checkpoint_root / run
        records.append(
            {
                "run": run,
                "checkpoint": (checkpoint_dir / f"weights_epoch_{args.epoch}.pt").exists(),
                "next_state": has_csv_result(next_state_dir, "next_state_metrics", run),
                "position_probe": has_csv_result(probe_dir, "position_probe_metrics", run),
                "action_probe": has_csv_result(probe_dir, "action_probe_metrics", run),
                "planning": (
                    args.analysis_dir
                    / "planning_eval"
                    / run
                    / f"epoch_{args.epoch}"
                    / f"solver_{args.solver}"
                    / "pusht_results.txt"
                ).is_file(),
                "geometry_pca": (geometry_dir / run / "latent_geometry.png").is_file(),
                "geometry_tsne": (geometry_dir / run / "latent_geometry_tsne.png").is_file(),
            }
        )

    status = pd.DataFrame(records)
    status["all_downstream"] = status.drop(columns=["run", "checkpoint"]).all(axis=1)
    output_dir = args.analysis_dir / "evaluation_status" / args.sweep_tag / f"epoch_{args.epoch}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "evaluation_status.csv"
    status.to_csv(output_path, index=False)

    print(f"Evaluation status: {args.sweep_tag} epoch {args.epoch} ({len(status)} discovered runs)")
    for column in status.columns.drop("run"):
        print(f"{column:>16}: {int(status[column].sum())}/{len(status)}")
    incomplete = status.loc[~status["all_downstream"], "run"].tolist()
    if incomplete:
        print("Incomplete runs:", ", ".join(incomplete))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
