#!/usr/bin/env python3
"""Build a portable, write-up-ready export for the available epoch-8 models."""

from __future__ import annotations

import argparse
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from paper_epoch8_manifest import PAPER_MODELS, Model, default_checkpoint_root, discover_models, parse_model
from plot_planning_benchmark import discover_results, summarize, summarize_repeats


EXPERIMENTS_DIR = Path(__file__).resolve().parent
ANALYSIS_DIR = EXPERIMENTS_DIR / "analysis_out"
METRIC_NAMES = (
    "next_state_metrics",
    "position_probe_metrics",
    "action_probe_metrics",
)
MODEL_COLORS = {
    "JEPA baseline": "#202124",
    "IDM-JEPA delta": "#007C91",
    "IDM-JEPA concat": "#3C6E71",
    "Cycle-JEPA": "#C94C12",
}


def model_frame(models: list[Model]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "run": model.run,
            "checkpoint_run": model.checkpoint_run,
            "checkpoint_epoch": model.checkpoint_epoch,
            "sweep_tag": model.tag,
            "arm": model.arm,
            "train_seed": model.train_seed,
            "label_fraction_pct": model.label_fraction_pct,
            "model_label": model.family,
            "supervision": model.supervision,
            "export_category": model.category,
            "export_subset": model.subset,
        }
        for model in models
    ).sort_values(["export_category", "export_subset", "label_fraction_pct", "run"])


def model_export_dir(export_dir: Path, model: Model) -> Path:
    return export_dir / model.category / Path(model.subset) / "models" / model.run


def paper_groups(models: list[Model]) -> dict[tuple[str, str], set[str]]:
    groups = {
        ("idm-planning", "delta/per-transition"): set(),
        ("idm-planning", "delta/per-episode"): set(),
        ("idm-planning", "concat/per-transition"): set(),
        ("idm-planning", "concat/per-episode"): set(),
        ("cycle-planning", "delta/per-transition"): set(),
        ("cycle-planning", "delta/per-episode"): set(),
        ("cycle-planning", "concat/per-transition"): set(),
        ("jepa-baseline", "all"): set(),
    }
    for model in models:
        if model.arm.startswith(("baseline_", "base_frac_")):
            groups[("jepa-baseline", "all")].add(model.run)
        elif model.tag == "v13" and model.arm.startswith("idm_"):
            groups[("idm-planning", "concat/per-episode")].add(model.run)
        elif model.tag == "v12" and model.arm.startswith("idm_"):
            groups[("idm-planning", "concat/per-transition")].add(model.run)
        elif model.tag == "v10" and model.arm.startswith("idm_"):
            groups[("idm-planning", "delta/per-transition")].add(model.run)
            if model.label_fraction_pct in {0, 100}:
                groups[("idm-planning", "delta/per-episode")].add(model.run)
        elif model.tag == "v11" and model.arm.startswith("idm_"):
            groups[("idm-planning", "delta/per-episode")].add(model.run)
        elif model.tag in {"v3", "v10"} and model.arm.startswith("cyc_ramp_"):
            groups[("cycle-planning", "delta/per-transition")].add(model.run)
            if model.tag == "v3" and model.label_fraction_pct in {0, 100}:
                groups[("cycle-planning", "delta/per-episode")].add(model.run)
        elif model.tag == "v11" and model.arm.startswith("cyc_ramp_"):
            groups[("cycle-planning", "delta/per-episode")].add(model.run)
    return groups


def copy_tree(source: Path, destination: Path) -> None:
    if source.exists():
        shutil.copytree(source, destination, dirs_exist_ok=True)


def read_model_metrics(models: list[Model], epoch: int, metric_name: str) -> pd.DataFrame:
    rows = []
    for model in models:
        path = ANALYSIS_DIR / ("next_state_eval" if metric_name == "next_state_metrics" else "position_probe") / model.tag / f"epoch_{model.checkpoint_epoch}" / f"{metric_name}_{model.run}.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        if not frame.empty:
            rows.append(frame.iloc[[0]])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["run", "epoch"])


def geometry_summary(models: list[Model], epoch: int) -> pd.DataFrame:
    rows = []
    for model in models:
        spectrum_path = ANALYSIS_DIR / "latent_geometry" / model.tag / f"epoch_{model.checkpoint_epoch}" / model.run / "pca_spectrum.csv"
        if not spectrum_path.exists():
            continue
        spectrum = pd.read_csv(spectrum_path)
        if spectrum.empty:
            continue
        row = {
            "run": model.run,
            "epoch": epoch,
            "pca_components": len(spectrum),
            "pc1_explained": spectrum.loc[0, "explained_variance_ratio"],
            "pc1_pc2_explained": spectrum.loc[:1, "explained_variance_ratio"].sum(),
        }
        explained = spectrum["explained_variance_ratio"].clip(lower=1e-12)
        entropy = -(explained * explained.map(math.log)).sum()
        row["effective_rank"] = math.exp(float(entropy))
        rows.append(row)
    return pd.DataFrame(rows)


def probe_comparison_table(position: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "run",
        "model_label",
        "supervision",
        "label_fraction_pct",
        "checkpoint_epoch",
        "block_orientation_ridge_r2",
        "block_displacement_ridge_r2",
        "agent_displacement_ridge_r2",
        "block_orientation_mlp_r2",
        "block_displacement_mlp_r2",
        "agent_displacement_mlp_r2",
    ]
    if position.empty:
        return pd.DataFrame(columns=columns)
    table = position.merge(
        manifest[["run", "model_label", "supervision", "label_fraction_pct", "checkpoint_epoch"]],
        on="run",
        how="left",
    )
    return table.reindex(columns=columns).sort_values(
        ["model_label", "supervision", "label_fraction_pct", "run"]
    )


def coverage_frame(models: list[Model], epoch: int) -> pd.DataFrame:
    rows = []
    for model in models:
        planning_root = ANALYSIS_DIR / "planning_eval" / model.run / f"epoch_{model.checkpoint_epoch}" / "solver_cem"
        seeds = sorted(
            int(path.parent.name.removeprefix("eval_seed_"))
            for path in planning_root.glob("eval_seed_*/pusht_results.txt")
            if path.stat().st_size > 0
        )
        rows.append(
            {
                "run": model.run,
                "checkpoint_run": model.checkpoint_run,
                "checkpoint_epoch": model.checkpoint_epoch,
                "latent_geometry": (ANALYSIS_DIR / "latent_geometry" / model.tag / f"epoch_{model.checkpoint_epoch}" / model.run / "pca_spectrum.csv").exists(),
                "planning_seed_count": len(seeds),
                "planning_seeds": ",".join(map(str, seeds)),
                "planning_complete_5_seeds": len(seeds) == 5,
                "next_state": (ANALYSIS_DIR / "next_state_eval" / model.tag / f"epoch_{model.checkpoint_epoch}" / f"next_state_metrics_{model.run}.csv").exists(),
                "position_probe": (ANALYSIS_DIR / "position_probe" / model.tag / f"epoch_{model.checkpoint_epoch}" / f"position_probe_metrics_{model.run}.csv").exists(),
                "action_probe": (ANALYSIS_DIR / "position_probe" / model.tag / f"epoch_{model.checkpoint_epoch}" / f"action_probe_metrics_{model.run}.csv").exists(),
            }
        )
    return pd.DataFrame(rows)


def write_group_csvs(
    export_dir: Path,
    manifest: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
    groups: dict[tuple[str, str], set[str]],
) -> None:
    for (category, subset), group_runs in groups.items():
        group_dir = export_dir / category / Path(subset) / "csv"
        group_dir.mkdir(parents=True, exist_ok=True)
        group = manifest.loc[manifest["run"].isin(group_runs)]
        group.to_csv(group_dir / "model_manifest.csv", index=False)
        for name, frame in frames.items():
            if "run" in frame:
                frame.loc[frame["run"].isin(group_runs)].to_csv(group_dir / f"{name}.csv", index=False)


def plot_metric(frame: pd.DataFrame, metric: str, output: Path, title: str, ylabel: str) -> None:
    if metric not in frame or frame[metric].dropna().empty:
        return
    figure, axis = plt.subplots(figsize=(8.3, 5.2))
    for label, group in frame.dropna(subset=[metric]).groupby("model_label", sort=False):
        grouped = group.groupby("label_fraction_pct", as_index=False)[metric].mean().sort_values("label_fraction_pct")
        axis.plot(
            grouped["label_fraction_pct"], grouped[metric], marker="o", linewidth=2,
            color=MODEL_COLORS.get(label, "#555555"), label=label,
        )
    axis.set(
        xlabel="Ground-truth action labels retained in training (%)",
        ylabel=ylabel,
        title=title,
        xlim=(-5, 105),
    )
    axis.set_xticks([0, 5, 25, 50, 75, 100])
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output, dpi=220)
    plt.close(figure)


def plot_idm_code_metrics(frame: pd.DataFrame, output: Path) -> None:
    metric_labels = {
        "idm_oracle_code_mse": "Oracle-code MSE",
        "idm_shuffled_next_state_code_mse": "Shuffled-next-state code MSE",
    }
    idm = frame.loc[frame["model_label"].str.startswith("IDM-JEPA", na=False)]
    if idm.empty or all(metric not in idm for metric in metric_labels):
        return
    figure, axis = plt.subplots(figsize=(8.3, 5.2))
    for metric, label in metric_labels.items():
        if metric not in idm:
            continue
        summary = idm.dropna(subset=[metric]).groupby("label_fraction_pct", as_index=False)[metric].mean()
        if not summary.empty:
            axis.plot(summary["label_fraction_pct"], summary[metric], marker="o", linewidth=2, label=label)
    axis.set(
        xlabel="Ground-truth action labels retained in training (%)",
        ylabel="MSE",
        title="IDM code prediction metrics",
        xlim=(-5, 105),
    )
    axis.set_xticks([0, 5, 25, 50, 75, 100])
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output, dpi=220)
    plt.close(figure)


def build_diagnostic_plots(analysis_dir: Path, manifest: pd.DataFrame, next_state: pd.DataFrame, position: pd.DataFrame, action: pd.DataFrame) -> None:
    analysis_dir.mkdir(parents=True, exist_ok=True)
    if not next_state.empty:
        next_state = next_state.merge(manifest[["run", "model_label", "label_fraction_pct"]], on="run", how="left")
        next_state["action_sensitivity_ratio"] = next_state["action_shuffled_mse"] / next_state["mse"]
        plot_metric(next_state, "mse", analysis_dir / "mse_vs_idm_frac.png", "Next-state prediction error", "MSE")
        plot_metric(next_state, "action_sensitivity_ratio", analysis_dir / "action_sensitivity.png", "Action sensitivity", "action-shuffled MSE / MSE")
        plot_idm_code_metrics(next_state, analysis_dir / "idm_code_metrics.png")
    if not position.empty:
        position = position.merge(manifest[["run", "model_label", "label_fraction_pct"]], on="run", how="left")
        plot_metric(position, "block_position_ridge_r2", analysis_dir / "position_probe_block_position_r2.png", "Position probe: block position", "Ridge $R^2$")
    if not action.empty:
        action = action.merge(manifest[["run", "model_label", "label_fraction_pct"]], on="run", how="left")
        plot_metric(action, "code_gain_over_state_ridge", analysis_dir / "action_probe_code_gain_ridge.png", "Action-code probe gain", "Ridge $R^2$ gain over state")


def expected_unavailable(checkpoint_root: Path, epoch: int) -> pd.DataFrame:
    rows = []
    for run in PAPER_MODELS:
        model = parse_model(run, epoch)
        if model is None:
            continue
        checkpoint = checkpoint_root / model.checkpoint_run / f"weights_epoch_{model.checkpoint_epoch}.pt"
        if not checkpoint.is_file():
            rows.append({"run": model.run, "reason": f"{checkpoint.name} is unavailable from {model.checkpoint_run}"})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epoch", type=int, default=8)
    parser.add_argument("--checkpoint-root", type=Path, default=default_checkpoint_root())
    parser.add_argument("--output-dir", type=Path, default=ANALYSIS_DIR / "paper_export" / "epoch_8")
    args = parser.parse_args()

    models = discover_models(args.checkpoint_root, args.epoch)
    if not models:
        raise FileNotFoundError(f"No requested epoch-{args.epoch} checkpoints in {args.checkpoint_root}")
    export_dir = args.output_dir
    csv_dir = export_dir / "csv"
    analysis_dir = export_dir / "updated_epoch8_analysis"
    csv_dir.mkdir(parents=True, exist_ok=True)
    if analysis_dir.exists():
        shutil.rmtree(analysis_dir)
    manifest = model_frame(models)
    next_state = read_model_metrics(models, args.epoch, "next_state_metrics")
    position = read_model_metrics(models, args.epoch, "position_probe_metrics")
    action = read_model_metrics(models, args.epoch, "action_probe_metrics")
    probe_table = probe_comparison_table(position, manifest)
    geometry = geometry_summary(models, args.epoch)
    coverage = coverage_frame(models, args.epoch)
    unavailable = expected_unavailable(args.checkpoint_root, args.epoch)
    groups = paper_groups(models)

    try:
        outcomes = discover_results(ANALYSIS_DIR / "planning_eval", {model.tag for model in models}, None, "cem", True)
        checkpoint_epochs = {model.run: model.checkpoint_epoch for model in models}
        outcomes = outcomes.loc[
            outcomes["run"].map(checkpoint_epochs).eq(outcomes["epoch"])
        ].copy()
    except FileNotFoundError:
        outcomes = pd.DataFrame()
    planning_seed_summary = summarize(outcomes) if not outcomes.empty else pd.DataFrame()
    planning_repeat_summary = summarize_repeats(planning_seed_summary) if not planning_seed_summary.empty else pd.DataFrame()

    frames = {
        "evaluation_coverage": coverage,
        "planning_episode_outcomes": outcomes,
        "planning_seed_summary": planning_seed_summary,
        "planning_repeat_summary": planning_repeat_summary,
        "latent_geometry_summary": geometry,
        "next_state_metrics": next_state,
        "position_probe_metrics": position,
        "action_probe_metrics": action,
        "probe_comparison_table": probe_table,
    }
    manifest.to_csv(csv_dir / "model_manifest.csv", index=False)
    unavailable.to_csv(csv_dir / "unavailable_epoch8_models.csv", index=False)
    for name, frame in frames.items():
        frame.to_csv(csv_dir / f"{name}.csv", index=False)
    write_group_csvs(export_dir, manifest, frames, groups)
    build_diagnostic_plots(analysis_dir, manifest, next_state, position, action)

    for model in models:
        for (category, subset), group_runs in groups.items():
            if model.run not in group_runs:
                continue
            destination = export_dir / category / Path(subset) / "models" / model.run
            destination.mkdir(parents=True, exist_ok=True)
            config = args.checkpoint_root / model.checkpoint_run / "config.yaml"
            if config.exists():
                shutil.copy2(config, destination / "training_config.yaml")
            copy_tree(ANALYSIS_DIR / "latent_geometry" / model.tag / f"epoch_{model.checkpoint_epoch}" / model.run, destination / "latent_geometry")
            copy_tree(ANALYSIS_DIR / "planning_eval" / model.run / f"epoch_{model.checkpoint_epoch}" / "solver_cem", destination / "planning")

    readme = export_dir / "README.md"
    readme.write_text(
        "# Epoch-8 Paper Export\n\n"
        f"Built {datetime.now(timezone.utc).isoformat()} from {len(models)} available epoch-{args.epoch} checkpoints.\n\n"
        "`csv/` contains the canonical consolidated tables. The top-level model-family folders contain copied raw planning outputs, latent-geometry artifacts, and training configs. `updated_epoch8_analysis/` contains regenerated next-state and probe figures. `unavailable_epoch8_models.csv` records trained runs without an epoch-8 checkpoint; they were not evaluated at another epoch.\n"
    )
    print(f"Wrote paper export for {len(models)} available models to {export_dir}")


if __name__ == "__main__":
    main()