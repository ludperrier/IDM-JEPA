"""Summarize PushT planning evaluations for a training sweep.

Examples:
    python plot_planning_benchmark.py --sweep-tag v10 --epoch 8 --solver cem
    python plot_planning_benchmark.py --sweep-tag v10 --epoch 8 --solver cem \
        --planning-dir analysis_out/planning_eval
    python plot_planning_benchmark.py --sweep-tag v3 --sweep-tag v10 \
        --sweep-tag v11 --sweep-tag v12 --epoch 8 --solver cem --benchmark both

The script accepts repeat-seeded planning outputs:
    planning_eval/<run>/epoch_<epoch>/solver_<solver>/eval_seed_<seed>/pusht_results.txt
as well as the previous layout:
    planning_eval/<run>/epoch_<epoch>/solver_<solver>/pusht_results.txt
and the earlier flat layout:
    planning_eval/<run>/pusht_results.txt
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import paper_style

paper_style.apply()


# Colour/marker per method, from the shared paper palette. The two IDM input
# modes are separate methods, so they get separate slots rather than a shared
# hue split by linestyle (which read as "same method" in the old figures).
FAMILY_STYLE = {
    "baseline": paper_style.SERIES_STYLE["baseline"],
    "baseline_step": paper_style.SERIES_STYLE["baseline_step"],
    "idm": paper_style.SERIES_STYLE["idm_delta"],
    "cycle": paper_style.SERIES_STYLE["cycle"],
    "cycle_ramp": paper_style.SERIES_STYLE["cycle"],
    "idm_concat": paper_style.SERIES_STYLE["idm_concat"],
    "idm_concat_episode": paper_style.SERIES_STYLE["idm_concat"],
}

# Each series lists (run, epoch) pairs; ``None`` means "the epoch requested on
# the command line, falling back to the run's latest checkpoint". An explicit
# epoch is pinned: if that checkpoint was not evaluated the point is dropped
# rather than silently substituted.
#
# The two baselines are the same training runs read at two different points.
# ``baseline`` is epoch-matched: every arm sees the same number of passes over
# its own (smaller) labelled subset. ``baseline_step`` is step-matched: the
# baseline is read at the epoch where its optimiser-step count matches what the
# IDM arms reach by epoch 8, i.e. epoch 8/rho. At 100% labels the two coincide
# by construction and share a single evaluation.
STEP_MATCHED_BASELINE = (
    ("v10_base_frac_5_s0", 160),
    ("v10_base_frac_25_s0", 32),
    ("v10_base_frac_50_s0", 16),
    ("v10_base_frac_75_s0", 11),
    ("v3_baseline_100_s0", None),
)
# Per-episode label allocation is a different training signal for the baseline
# too -- masking 25% of episodes is not masking 25% of transitions -- so the
# per-episode panel reads its own step-matched runs (v16) rather than reusing
# the per-transition ones. v16 has no 75% run; the entry below reuses the 50%
# evaluation at the 75% abscissa, which is a placeholder, not a measurement.
# The 100% point is shared with the per-transition panel: at full labels the
# two allocation schemes coincide.
PER_EPISODE_STEP_MATCHED_BASELINE = (
    ("v16_base_frac_5_s0", 160),
    ("v16_base_frac_25_s0", 32),
    ("v16_base_frac_50_s0", 16),
    ("v16_base_frac_50_s0", 16, 75),
    ("v3_baseline_100_s0", None),
)
EPOCH_MATCHED_BASELINE = (
    ("v3_baseline_100_s0", None),
    ("v10_base_frac_5_s0", None),
    ("v10_base_frac_25_s0", None),
    ("v10_base_frac_50_s0", None),
    ("v10_base_frac_75_s0", None),
)

BENCHMARK_SERIES = {
    "per_transition": (
        ("Baseline (epoch-matched)", EPOCH_MATCHED_BASELINE, "baseline"),
        ("Baseline (step-matched)", STEP_MATCHED_BASELINE, "baseline_step"),
        (
            "IDM-JEPA delta",
            tuple((f"v10_idm_frac_{fraction}_s0", None) for fraction in (0, 5, 25, 50, 75, 100)),
            "idm",
        ),
        (
            "IDM-JEPA concat",
            tuple((f"v12_idm_frac_{fraction}_s0", None) for fraction in (0, 5, 25, 50, 75, 100)),
            "idm_concat",
        ),
        (
            "Cycle-JEPA",
            tuple((run, None) for run in ("v3_cyc_ramp_frac_0_s0", "v10_cyc_ramp_frac_5_s0", "v10_cyc_ramp_frac_25_s0", "v10_cyc_ramp_frac_50_s0", "v10_cyc_ramp_frac_75_s0", "v3_cyc_ramp_frac_100_s0")),
            "cycle_ramp",
        ),
    ),
    "per_episode": (
        ("Baseline (epoch-matched)", EPOCH_MATCHED_BASELINE, "baseline"),
        ("Baseline (step-matched)", PER_EPISODE_STEP_MATCHED_BASELINE, "baseline_step"),
        (
            "IDM-JEPA delta",
            tuple((run, None) for run in ("v10_idm_frac_0_s0", "v11_idm_frac_5_s0", "v11_idm_frac_25_s0", "v11_idm_frac_50_s0", "v11_idm_frac_75_s0", "v10_idm_frac_100_s0")),
            "idm",
        ),
        (
            "Cycle-JEPA",
            tuple((run, None) for run in ("v3_cyc_ramp_frac_0_s0", "v11_cyc_ramp_frac_5_s0", "v11_cyc_ramp_frac_25_s0", "v11_cyc_ramp_frac_50_s0", "v11_cyc_ramp_frac_75_s0", "v3_cyc_ramp_frac_100_s0")),
            "cycle_ramp",
        ),
        (
            # v13 frac_0/100 training stalled before epoch 8 (retry4 dropped
            # them); at the label_frac extremes per-episode and per-transition
            # splits are equivalent, so reuse the v12 per-transition concat
            # results for those two points.
            "IDM-JEPA concat",
            tuple((run, None) for run in (("v12_idm_frac_0_s0",) + tuple(f"v13_idm_frac_{fraction}_s0" for fraction in (5, 25, 50, 75)) + ("v12_idm_frac_100_s0",))),
            "idm_concat_episode",
        ),
    ),
}

PER_EPISODE_PLACEHOLDERS = {"v11_idm_frac_75_s0": 75}


def parse_run_name(run: str) -> dict[str, object] | None:
    match = re.fullmatch(
        r"(?:(?P<tag>.+)_)?(?P<arm>baseline_100|base_frac_\d+|(?:idm|cyc_ramp|cyc)_frac_\d+)_s(?P<seed>\d+)",
        run,
    )
    if match is None:
        return None

    arm = match.group("arm")
    if arm == "baseline_100" or arm.startswith("base_frac_"):
        family = "baseline"
        fraction = 100 if arm == "baseline_100" else int(arm.rsplit("_", 1)[1])
    else:
        family = "cycle_ramp" if arm.startswith("cyc_ramp_") else arm.split("_", 1)[0]
        fraction = int(arm.rsplit("_", 1)[1])
    return {
        "run": run,
        "sweep_tag": match.group("tag") or "",
        "seed": int(match.group("seed")),
        "arm": arm,
        "family": family,
        "label_frac_pct": fraction,
    }


def parse_result_file(result_path: Path) -> list[list[bool]]:
    text = result_path.read_text()
    result_blocks = re.findall(r"==== RESULTS ====(.*?)(?===== CONFIG ====|\Z)", text, re.DOTALL)
    evaluations = []
    for block in result_blocks:
        episode_match = re.search(r"'episode_successes': array\(\[(.*?)\]\)", block, re.DOTALL)
        if episode_match is None:
            continue
        outcomes = re.findall(r"\b(True|False)\b", episode_match.group(1))
        if outcomes:
            evaluations.append([outcome == "True" for outcome in outcomes])
    return evaluations


def discover_results(
    planning_dir: Path, sweep_tags: set[str], epoch: int | None, solver: str | None, all_epochs: bool
) -> pd.DataFrame:
    records = []
    for result_path in sorted(planning_dir.glob("**/pusht_results.txt")):
        parts = result_path.relative_to(planning_dir).parts
        run = parts[0]
        metadata = parse_run_name(run)
        if metadata is None or metadata["sweep_tag"] not in sweep_tags:
            continue
        if metadata["sweep_tag"] == "v3" and metadata["family"] not in {"baseline", "cycle_ramp"}:
            continue

        result_epoch = None
        result_solver = None
        eval_seed = None
        if len(parts) >= 4:
            epoch_match = re.fullmatch(r"epoch_(\d+)", parts[1])
            solver_match = re.fullmatch(r"solver_(.+)", parts[2])
            if epoch_match is None or solver_match is None:
                continue
            result_epoch = int(epoch_match.group(1))
            result_solver = solver_match.group(1)
            if len(parts) >= 5:
                seed_match = re.fullmatch(r"eval_seed_(\d+)", parts[3])
                if seed_match is None:
                    continue
                eval_seed = int(seed_match.group(1))
        if eval_seed is None and any(result_path.parent.glob("eval_seed_*/pusht_results.txt")):
            continue
        if epoch is not None and result_epoch != epoch:
            continue
        if solver is not None and result_solver != solver:
            continue

        for evaluation_id, outcomes in enumerate(parse_result_file(result_path)):
            records.extend(
                {
                    **metadata,
                    "epoch": result_epoch,
                    "solver": result_solver,
                    "eval_seed": eval_seed,
                    "evaluation_id": evaluation_id,
                    "episode": episode,
                    "success": int(success),
                    "source": str(result_path),
                }
                for episode, success in enumerate(outcomes)
            )

    if not records:
        raise FileNotFoundError(
            f"No episode-level planning outcomes for sweep tags {sorted(sweep_tags)!r} in {planning_dir}."
        )
    results = pd.DataFrame(records)
    if epoch is None and not all_epochs and results["epoch"].nunique(dropna=False) > 1:
        raise ValueError("Multiple evaluation epochs found; pass --epoch.")
    if solver is None and results["solver"].nunique(dropna=False) > 1:
        raise ValueError("Multiple solvers found; pass --solver.")
    return results


def wilson_interval(successes: int, count: int, z: float = 1.96) -> tuple[float, float]:
    if count == 0:
        return (np.nan, np.nan)
    proportion = successes / count
    denominator = 1 + z**2 / count
    center = (proportion + z**2 / (2 * count)) / denominator
    half_width = z * np.sqrt(proportion * (1 - proportion) / count + z**2 / (4 * count**2)) / denominator
    return (100 * (center - half_width), 100 * (center + half_width))


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    summary = (
        results.groupby(
            [key for key in ("run", "sweep_tag", "arm", "family", "series", "series_family", "label_frac_pct", "seed", "epoch", "eval_seed") if key in results],
            dropna=False,
            as_index=False,
        )
        .agg(successes=("success", "sum"), episodes=("success", "size"))
        .sort_values(["family", "label_frac_pct", "seed"])
    )
    summary["planning_accuracy_pct"] = 100 * summary["successes"] / summary["episodes"]
    intervals = [wilson_interval(int(row.successes), int(row.episodes)) for row in summary.itertuples()]
    summary[["ci95_low", "ci95_high"]] = intervals
    return summary


def summarize_repeats(summary: pd.DataFrame) -> pd.DataFrame:
    grouped = summary.groupby(
        [key for key in ("run", "sweep_tag", "arm", "family", "series", "series_family", "label_frac_pct", "seed", "epoch") if key in summary],
        as_index=False,
    )
    repeat_summary = grouped.agg(
        repeats=("planning_accuracy_pct", "size"),
        planning_accuracy_pct=("planning_accuracy_pct", "mean"),
        planning_accuracy_std_pct=("planning_accuracy_pct", "std"),
        episodes_per_repeat=("episodes", "mean"),
    )
    repeat_summary["planning_accuracy_sem_pct"] = (
        repeat_summary["planning_accuracy_std_pct"]
        / np.sqrt(repeat_summary["repeats"])
    ).fillna(0.0)
    return repeat_summary


def select_benchmark_results(
    results: pd.DataFrame, benchmark: str, preferred_epoch: int
) -> tuple[pd.DataFrame, list[tuple[str, tuple[tuple[str, int | None], ...]]], dict[str, int]]:
    selected_frames = []
    missing_series = []
    fallback_epochs = {}
    for label, runs, family in BENCHMARK_SERIES[benchmark]:
        series_frames = []
        for run, pinned_epoch, *fraction_override in runs:
            # An optional third element plots the run at an abscissa other than
            # its own label fraction, standing in for a fraction that was never
            # trained. It is a placeholder, so say so at the terminal.
            tags = {"series": label, "series_family": family}
            if fraction_override:
                tags["label_frac_pct"] = fraction_override[0]
                print(f"note: {label}: {run} reused as the {fraction_override[0]}% point (placeholder).")
            run_results = results.loc[results["run"] == run]
            if run_results.empty:
                continue
            if pinned_epoch is not None:
                # Pinned checkpoints are exact: no fallback, because a
                # step-matched point read at the wrong epoch is not the
                # quantity the series claims to measure.
                pinned = run_results.loc[run_results["epoch"] == pinned_epoch]
                if pinned.empty:
                    print(f"warning: {label}: {run} has no epoch-{pinned_epoch} evaluation; point dropped.")
                    continue
                series_frames.append(pinned.assign(**tags))
                continue
            preferred_results = run_results.loc[run_results["epoch"] == preferred_epoch]
            if not preferred_results.empty:
                series_frames.append(preferred_results.assign(**tags))
                continue
            numeric_epochs = run_results["epoch"].dropna()
            if not numeric_epochs.empty:
                fallback_epoch = int(numeric_epochs.max())
                series_frames.append(
                    run_results.loc[run_results["epoch"] == fallback_epoch].assign(**tags)
                )
                fallback_epochs[run] = fallback_epoch
            else:
                series_frames.append(run_results.assign(**tags))
        if series_frames:
            selected_frames.extend(series_frames)
        else:
            missing_series.append((label, runs))
    if not selected_frames:
        raise FileNotFoundError(f"No planning results found for the {benchmark.replace('_', ' ')} benchmark.")
    return pd.concat(selected_frames, ignore_index=True), missing_series, fallback_epochs


def plot_benchmark_accuracy(
    summary: pd.DataFrame,
    output_dir: Path,
    benchmark: str,
    missing_series: list[tuple[str, tuple[str, ...]]],
) -> None:
    fig, ax = plt.subplots(figsize=paper_style.HALF_WIDTH)
    for label, runs, family in BENCHMARK_SERIES[benchmark]:
        frame = summary.loc[summary["series"] == label]
        if frame.empty:
            continue
        aggregate = frame.groupby("label_frac_pct", as_index=False).agg(
            accuracy=("planning_accuracy_pct", "mean"),
            std=("planning_accuracy_sem_pct", "mean"),
        ).sort_values("label_frac_pct")
        style = FAMILY_STYLE[family]
        ax.errorbar(
            aggregate["label_frac_pct"],
            aggregate["accuracy"],
            yerr=aggregate["std"].fillna(0.0),
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            elinewidth=0.7,
            capsize=1.8,
            capthick=0.7,
            label=style["label"],
            # hero method (IDM concat) draws last so it sits on top
            zorder=4 if family.startswith("idm_concat") else 3,
        )

    for label, _ in missing_series:
        ax.plot([], [], color=paper_style.MISSING_COLOR, linestyle="--", marker="x",
                label=f"{label} (not evaluated)")

    # No figure title: the LaTeX caption carries it. The panel title only says
    # which of the two supervision granularities this panel is.
    ax.set_title(
        "Per-transition supervision" if benchmark == "per_transition" else "Per-episode supervision",
        color=paper_style.INK,
    )
    ax.set(
        xlabel="Labelled transitions (%)" if benchmark == "per_transition" else "Labelled episodes (%)",
        ylabel="Planning success (%)",
        xlim=(-4, 104),
        ylim=(0, 100),
    )
    ax.set_xticks([0, 5, 25, 50, 75, 100])
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.grid(axis="y")
    # Identical legend order in every panel, driven by the shared palette order
    # rather than by whatever order the series happen to appear in.
    handles, labels = ax.get_legend_handles_labels()
    rank = {paper_style.SERIES_STYLE[key]["label"]: i for i, key in enumerate(paper_style.SERIES_ORDER)}
    ordered = sorted(zip(handles, labels), key=lambda hl: rank.get(hl[1], len(rank)))
    ax.legend([h for h, _ in ordered], [l for _, l in ordered], loc="lower right", ncol=1)
    fig.tight_layout(pad=0.2)
    paper_style.save(fig, output_dir, f"planning_accuracy_vs_label_fraction_{benchmark}")
    plt.close(fig)


def plot_accuracy_vs_fraction(summary: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    plot_summary = summary.assign(
        plot_series=lambda frame: np.where(
            frame["family"].isin(["baseline", "cycle_ramp"]),
            frame["family"],
            frame["sweep_tag"],
        )
    )
    for (plot_series, family), frame in plot_summary.groupby(["plot_series", "family"], sort=False):
        if family == "cycle_ramp":
            v10_fractions = frame.loc[frame["sweep_tag"] == "v10", "label_frac_pct"]
            frame = frame.loc[
                (frame["sweep_tag"] == "v10") | ~frame["label_frac_pct"].isin(v10_fractions)
            ]
        style = FAMILY_STYLE[family]
        aggregate = frame.groupby("label_frac_pct", as_index=False).agg(
            accuracy=("planning_accuracy_pct", "mean"),
            repeats=("planning_accuracy_pct", "size"),
            std=("planning_accuracy_std_pct", "mean"),
        ).sort_values("label_frac_pct")
        errors = aggregate["std"].fillna(0.0).to_numpy()
        label = {"baseline": "Baseline", "cycle_ramp": "Cycle-JEPA", "idm": "IDM-JEPA delta"}.get(
            family, style["label"]
        )
        ax.errorbar(
            aggregate["label_frac_pct"], aggregate["accuracy"], yerr=errors,
            color=style["color"], marker=style["marker"], linewidth=2.2,
            markersize=7, capsize=4, label=label, zorder=3,
        )

    ax.set(xlabel="Ground-truth action labels in training data (%)", ylabel="PushT planning accuracy (%)", xlim=(-5, 105), ylim=(0, 100))
    ax.set_xticks([0, 5, 25, 50, 75, 100])
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.8)
    ax.legend(frameon=False, title="Training objective")
    fig.tight_layout()
    fig.savefig(output_dir / "planning_accuracy_vs_label_fraction.png", dpi=220)
    plt.close(fig)


def plot_model_comparison(summary: pd.DataFrame, output_dir: Path) -> None:
    labels = summary.apply(
        lambda row: (
            f"{FAMILY_STYLE[row.family]['label']} ({row.sweep_tag}) {row.label_frac_pct:g}% "
            f"(e{int(row.epoch)}, s{row.seed})"
        ),
        axis=1,
    )
    ordered = summary.assign(label=labels).sort_values("planning_accuracy_pct")
    fig, ax = plt.subplots(figsize=(8.2, max(4.6, 0.42 * len(ordered) + 1.2)))
    colors = [FAMILY_STYLE[family]["color"] for family in ordered["family"]]
    ax.barh(ordered["label"], ordered["planning_accuracy_pct"], color=colors, height=0.65)
    for bar, row in zip(ax.patches, ordered.itertuples()):
        error = row.planning_accuracy_std_pct if row.repeats > 1 else 0.0
        ax.errorbar(
            row.planning_accuracy_pct,
            bar.get_y() + bar.get_height() / 2,
            xerr=error,
            fmt="none",
            color="#303030",
            capsize=3,
        )
        ax.text(
            min(row.planning_accuracy_pct + error + 1.5, 98),
            bar.get_y() + bar.get_height() / 2,
            f"{row.planning_accuracy_pct:.1f}%  (r={row.repeats:g})",
            va="center",
            fontsize=9,
        )
    ax.set(xlabel="PushT planning accuracy (%)", xlim=(0, 100))
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(output_dir / "planning_accuracy_by_model.png", dpi=220)
    plt.close(fig)


def plot_epoch_curves(results: pd.DataFrame, output_dir: Path) -> None:
    if results["epoch"].isna().all() or results["epoch"].nunique() < 2:
        return
    aggregate = results.groupby(["sweep_tag", "family", "label_frac_pct", "epoch"], as_index=False).agg(
        successes=("success", "sum"), episodes=("success", "size")
    )
    aggregate["accuracy"] = 100 * aggregate["successes"] / aggregate["episodes"]
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    for (sweep_tag, family, fraction), frame in aggregate.groupby(["sweep_tag", "family", "label_frac_pct"]):
        style = FAMILY_STYLE[family]
        ax.plot(frame["epoch"], frame["accuracy"], marker=style["marker"], color=style["color"], label=f"{style['label']} ({sweep_tag}) {fraction:g}%")
    ax.set(xlabel="Training epoch", ylabel="PushT planning accuracy (%)", ylim=(0, 100))
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.8)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "planning_accuracy_vs_training_epoch.png", dpi=220)
    plt.close(fig)


def plot_accuracy_vs_fraction_all_epochs(results: pd.DataFrame, output_dir: Path) -> None:
    """Plot each evaluated checkpoint, retaining the training-family and epoch identity."""
    summary = (
        results.groupby(
            ["run", "sweep_tag", "arm", "family", "label_frac_pct", "seed", "epoch", "eval_seed"],
            dropna=False,
            as_index=False,
        )
        .agg(successes=("success", "sum"), episodes=("success", "size"))
    )
    summary["planning_accuracy_pct"] = 100 * summary["successes"] / summary["episodes"]
    aggregate = (
        summary.groupby(["sweep_tag", "family", "label_frac_pct", "epoch"], as_index=False)
        .agg(
            planning_accuracy_pct=("planning_accuracy_pct", "mean"),
            planning_accuracy_std_pct=("planning_accuracy_pct", "std"),
            repeats=("planning_accuracy_pct", "size"),
        )
        .sort_values(["family", "epoch", "label_frac_pct"])
    )
    aggregate["planning_accuracy_std_pct"] = aggregate["planning_accuracy_std_pct"].fillna(0.0)
    aggregate.to_csv(output_dir / "planning_accuracy_by_fraction_and_epoch.csv", index=False)

    epochs = sorted(aggregate["epoch"].dropna().unique())
    markers = ("o", "s", "^", "P", "X", "D", "v", "<", ">")
    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    family_handles = []
    epoch_handles = []
    for (sweep_tag, family), frame in aggregate.groupby(["sweep_tag", "family"], sort=False):
        style = FAMILY_STYLE[family]
        base_color = np.asarray(plt.matplotlib.colors.to_rgb(style["color"]))
        family_handles.append(
            plt.Line2D([], [], color=style["color"], linewidth=2.4, label=f"{style['label']} ({sweep_tag})")
        )
        for epoch_index, epoch in enumerate(epochs):
            epoch_frame = frame[frame["epoch"] == epoch]
            if epoch_frame.empty:
                continue
            # Earlier checkpoints are lighter; newer checkpoints retain the family hue.
            mix = 0.48 + 0.52 * (epoch_index + 1) / max(len(epochs), 1)
            color = tuple(mix * base_color + (1 - mix) * np.ones(3))
            marker = markers[epoch_index % len(markers)]
            ax.errorbar(
                epoch_frame["label_frac_pct"],
                epoch_frame["planning_accuracy_pct"],
                yerr=epoch_frame["planning_accuracy_std_pct"],
                color=color,
                marker=marker,
                markeredgecolor=style["color"],
                markeredgewidth=0.9,
                linewidth=1.8,
                markersize=8,
                capsize=3,
                zorder=3,
            )
            if family == "baseline" and sweep_tag == sorted(aggregate["sweep_tag"].unique())[0]:
                epoch_handles.append(
                    plt.Line2D(
                        [], [], color="#555555", marker=marker, linestyle="None",
                        markersize=8, label=f"Epoch {int(epoch)}",
                    )
                )

    ax.set(
        xlabel="Ground-truth action labels retained from the original training data (%)",
        ylabel="PushT planning success (%)",
        xlim=(-5, 105),
        ylim=(0, 100),
    )
    ax.set_xticks([0, 5, 25, 50, 75, 100])
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.8)
    family_legend = ax.legend(handles=family_handles, frameon=False, title="Model type", loc="lower right")
    ax.add_artist(family_legend)
    ax.legend(handles=epoch_handles, frameon=False, title="Checkpoint", loc="upper left")
    fig.tight_layout()
    fig.savefig(output_dir / "planning_success_vs_label_fraction_all_epochs.png", dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot planning accuracy benchmarks for one sweep.")
    parser.add_argument("--sweep-tag", action="append", required=True, help="Training sweep namespace; repeat to compare tags, e.g. --sweep-tag v10 --sweep-tag v3.")
    parser.add_argument("--epoch", type=int, help="Training epoch evaluated by the planner.")
    parser.add_argument("--all-epochs", action="store_true", help="Also plot planning accuracy across evaluated epochs; headline plots use each model's latest epoch.")
    parser.add_argument("--solver", help="Planner solver name, e.g. cem.")
    parser.add_argument("--planning-dir", type=Path, default=Path(__file__).parent / "analysis_out" / "planning_eval")
    parser.add_argument("--output-dir", type=Path, help="Directory for CSVs and figures; defaults to analysis_out/planning_benchmark/<tag>.")
    parser.add_argument(
        "--benchmark",
        choices=("per_transition", "per_episode", "both"),
        help="Plot the prescribed per-transition and/or per-episode model comparison; epoch falls back per run when unavailable.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or Path(__file__).parent / "analysis_out" / "planning_benchmark" / "_".join(args.sweep_tag)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = discover_results(
        args.planning_dir,
        set(args.sweep_tag),
        None if args.benchmark else args.epoch,
        args.solver,
        args.all_epochs or args.benchmark is not None,
    )
    if args.benchmark:
        benchmarks = ("per_transition", "per_episode") if args.benchmark == "both" else (args.benchmark,)
        for benchmark in benchmarks:
            selected, missing_series, fallback_epochs = select_benchmark_results(results, benchmark, args.epoch)
            summary = summarize(selected)
            repeat_summary = summarize_repeats(summary)
            summary.to_csv(output_dir / f"planning_accuracy_summary_{benchmark}.csv", index=False)
            repeat_summary.to_csv(output_dir / f"planning_repeat_summary_{benchmark}.csv", index=False)
            plot_benchmark_accuracy(repeat_summary, output_dir, benchmark, missing_series)
            fallback_text = ", ".join(f"{run} (e{epoch})" for run, epoch in fallback_epochs.items())
            print(f"Wrote {benchmark.replace('_', ' ')} benchmark to {output_dir}" + (f"; fallback results: {fallback_text}" if fallback_text else ""))
        return
    latest_results = results
    if args.all_epochs:
        latest_epoch = results.groupby("run")["epoch"].transform("max")
        latest_results = results[results["epoch"] == latest_epoch]
    summary = summarize(latest_results)
    repeat_summary = summarize_repeats(summary)
    results.to_csv(output_dir / "planning_episode_outcomes.csv", index=False)
    summary.to_csv(output_dir / "planning_accuracy_summary.csv", index=False)
    repeat_summary.to_csv(output_dir / "planning_repeat_summary.csv", index=False)
    plot_accuracy_vs_fraction(repeat_summary, output_dir)
    plot_model_comparison(repeat_summary, output_dir)
    plot_epoch_curves(results, output_dir)
    if args.all_epochs:
        plot_accuracy_vs_fraction_all_epochs(results, output_dir)
    print(f"Wrote {len(summary)} model summaries and figures to {output_dir}")


if __name__ == "__main__":
    main()