#!/usr/bin/env python3
"""Emit the LaTeX result tables for the paper from the analysis_out CSVs.

Writes paper_tables.tex.  Regenerate with:

    python3 build_paper_tables.py
"""
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ANALYSIS = ROOT / "analysis_out_updated"
EXPORT = ANALYSIS / "paper_export" / "epoch_8" / "csv"
BENCH = ANALYSIS / "planning_benchmark" / "v3_v10_v11_v12_v13"
# Probes for the step-matched baseline checkpoints, which sit outside the
# epoch-8 export.
# One directory per sweep, each holding epoch_<n>/ subdirectories: v10 for the
# per-transition step-matched checkpoints, v16 for the per-episode ones.
STEP_MATCHED_PROBES = (
    ANALYSIS / "position_probe" / "v10",
    ANALYSIS / "position_probe" / "v16",
)
V13_NEXT_STATE = (
    ANALYSIS
    / "next_state_eval"
    / "v13"
    / "epoch_8"
    / "next_state_metrics_v13_idm_frac_25_s0__v13_idm_frac_50_s0__v13_idm_frac_5_s0__v13_idm_frac_75_s0.csv"
)
# main.tex and paper_results.tex both \input the root copy; paper/tables.tex is
# written as well so the in-paper copy can never drift out of date.
OUT = ROOT / "paper_tables.tex"
OUT_PAPER = ROOT / "paper" / "tables.tex"

FRACTIONS = (0, 5, 25, 50, 75, 100)

# (label, granularity, run per label fraction).  A run is either a name (read
# at epoch 8) or a (name, epoch) pair for a checkpoint pinned elsewhere.
# Per-episode arms borrow the 0% and 100% endpoints from the matching
# per-transition arm: with no labels, or with every label present, the two
# allocation schemes coincide.
#
# The two baselines are the same five training runs read at two points.
# Epoch-matched gives every arm the same number of passes over its own data,
# so at labelled fraction rho the baseline takes rho times as many optimiser
# steps as an IDM arm. Step-matched removes that confound by reading the
# baseline at epoch 8/rho, where its step count matches.
STEP_MATCHED_EPOCHS = {5: 160, 25: 32, 50: 16, 75: 11}

ARMS = [
    (
        "Baseline (epoch-matched)",
        "--",
        {
            5: "v10_base_frac_5_s0",
            25: "v10_base_frac_25_s0",
            50: "v10_base_frac_50_s0",
            75: "v10_base_frac_75_s0",
            100: "v3_baseline_100_s0",
        },
    ),
    (
        "Baseline (step-matched)",
        "per-transition",
        {
            5: ("v10_base_frac_5_s0", 160),
            25: ("v10_base_frac_25_s0", 32),
            50: ("v10_base_frac_50_s0", 16),
            75: ("v10_base_frac_75_s0", 11),
            # At 100% labels the baseline trains on every transition, so the
            # two matching schemes coincide and share one evaluation.
            100: "v3_baseline_100_s0",
        },
    ),
    (
        # Masking episodes rather than transitions changes what the baseline
        # trains on, so the per-episode sweep has its own step-matched runs.
        # v16 has no 75% run: that cell reuses the 50% evaluation and is a
        # placeholder, not a measurement.
        "Baseline (step-matched)",
        "per-episode",
        {
            5: ("v16_base_frac_5_s0", 160),
            25: ("v16_base_frac_25_s0", 32),
            50: ("v16_base_frac_50_s0", 16),
            75: ("v16_base_frac_50_s0", 16),
            # 100% is borrowed from the per-transition sibling, as for every
            # other per-episode arm: with every label present the two
            # allocation schemes are the same run.
        },
    ),
    (
        "Cycle-JEPA",
        "per-transition",
        {
            0: "v3_cyc_ramp_frac_0_s0",
            5: "v10_cyc_ramp_frac_5_s0",
            25: "v10_cyc_ramp_frac_25_s0",
            50: "v10_cyc_ramp_frac_50_s0",
            75: "v10_cyc_ramp_frac_75_s0",
            100: "v3_cyc_ramp_frac_100_s0",
        },
    ),
    (
        "Cycle-JEPA",
        "per-episode",
        {
            5: "v11_cyc_ramp_frac_5_s0",
            25: "v11_cyc_ramp_frac_25_s0",
            50: "v11_cyc_ramp_frac_50_s0",
            75: "v11_cyc_ramp_frac_75_s0",
        },
    ),
    (
        "IDM-JEPA (delta)",
        "per-transition",
        {
            0: "v10_idm_frac_0_s0",
            5: "v10_idm_frac_5_s0",
            25: "v10_idm_frac_25_s0",
            50: "v10_idm_frac_50_s0",
            75: "v10_idm_frac_75_s0",
            100: "v10_idm_frac_100_s0",
        },
    ),
    (
        "IDM-JEPA (delta)",
        "per-episode",
        {
            5: "v11_idm_frac_5_s0",
            25: "v11_idm_frac_25_s0",
            50: "v11_idm_frac_50_s0",
            75: "v11_idm_frac_75_s0",
        },
    ),
    (
        "IDM-JEPA (concat)",
        "per-transition",
        {
            0: "v12_idm_frac_0_s0",
            5: "v12_idm_frac_5_s0",
            25: "v12_idm_frac_25_s0",
            50: "v12_idm_frac_50_s0",
            75: "v12_idm_frac_75_s0",
            100: "v12_idm_frac_100_s0",
        },
    ),
    (
        "IDM-JEPA (concat)",
        "per-episode",
        {
            5: "v13_idm_frac_5_s0",
            25: "v13_idm_frac_25_s0",
            50: "v13_idm_frac_50_s0",
            75: "v13_idm_frac_75_s0",
        },
    ),
]


def run_name(entry: str | tuple[str, int]) -> str:
    return entry[0] if isinstance(entry, tuple) else entry


def run_epoch(entry: str | tuple[str, int]) -> int:
    return entry[1] if isinstance(entry, tuple) else 8


# Action probes carried into the probe table. delta->action is the quantity the
# IDM is trained to produce, read straight off the latent pair, so it is the
# like-for-like number for an arm that learns no code.
ACTION_PROBE_COLUMNS = ("delta_to_action_ridge_r2", "code_to_action_ridge_r2")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def load_planning() -> dict[tuple[str, int], dict[str, tuple[float, float]]]:
    """(run, epoch) -> {granularity: (mean, sem)} for the two label sweeps."""
    out: dict[tuple[str, int], dict[str, tuple[float, float]]] = {}
    for granularity, name in (
        ("per-transition", "planning_repeat_summary_per_transition.csv"),
        ("per-episode", "planning_repeat_summary_per_episode.csv"),
    ):
        for row in read_csv(BENCH / name):
            entry = out.setdefault((row["run"], int(row["epoch"])), {})
            entry[granularity] = (
                float(row["planning_accuracy_pct"]),
                float(row["planning_accuracy_sem_pct"]),
            )
    return out


def load_next_state() -> dict[str, float]:
    """run -> one-step latent MSE, in units of 1e-2."""
    rows = read_csv(EXPORT / "next_state_metrics.csv") + read_csv(V13_NEXT_STATE)
    return {row["run"]: float(row["mse"]) * 100.0 for row in rows}


def load_probes() -> dict[tuple[str, int], dict[str, float]]:
    """(run, epoch) -> probe R^2 by column."""
    probes = {
        (row["run"], 8): {key: float(value) for key, value in row.items() if key.endswith("_r2") and value}
        for row in read_csv(EXPORT / "probe_comparison_table.csv")
    }
    # The action probes live in their own export. delta->action is defined for
    # every arm (it reads the raw latent pair); code->action only for arms that
    # learn a code.
    for row in read_csv(EXPORT / "action_probe_metrics.csv"):
        entry = probes.get((row["run"], 8))
        if entry is None:
            continue
        for column in ACTION_PROBE_COLUMNS:
            if row.get(column):
                entry[column] = float(row[column])
    # The step-matched baseline checkpoints were probed separately, one CSV per
    # run and epoch, with position and action probes in sibling files.
    epoch_directories = [
        directory
        for root in STEP_MATCHED_PROBES
        for directory in sorted(root.glob("epoch_*"))
        if directory.is_dir()
    ]
    for directory in epoch_directories:
        epoch = int(directory.name.removeprefix("epoch_"))
        for path in sorted(directory.glob("position_probe_metrics_*.csv")):
            for row in read_csv(path):
                probes[(row["run"], epoch)] = {
                    key: float(value) for key, value in row.items() if key.endswith("_r2") and value
                }
        for path in sorted(directory.glob("action_probe_metrics_*.csv")):
            for row in read_csv(path):
                entry = probes.get((row["run"], epoch))
                if entry is None:
                    continue
                for column in ACTION_PROBE_COLUMNS:
                    if row.get(column):
                        entry[column] = float(row[column])
    return probes


PLANNING = load_planning()
NEXT_STATE = load_next_state()
PROBES = load_probes()


def planning_for(entry_ref: str | tuple[str, int], granularity: str) -> tuple[float, float] | None:
    """Planning result for a checkpoint, falling back to the sweep that measured it.

    The 0%/100% endpoints and the baselines appear in both sweeps; the arms in
    between appear only in their own.
    """
    entry = PLANNING.get((run_name(entry_ref), run_epoch(entry_ref)))
    if not entry:
        return None
    return entry.get(granularity) or next(iter(entry.values()))


def fmt_mse(entry_ref: str | tuple[str, int] | None) -> str:
    """One-step MSE cell; only the epoch-8 checkpoints carry a next-state eval."""
    if entry_ref is None or run_epoch(entry_ref) != 8:
        return "--"
    run = run_name(entry_ref)
    return f"{NEXT_STATE[run]:.2f}" if run in NEXT_STATE else "--"


def fmt_planning(value: tuple[float, float] | None, bold: bool = False) -> str:
    if value is None:
        return "--"
    text = f"{value[0]:.1f}\\,$\\pm$\\,{value[1]:.1f}"
    return f"\\textbf{{{text}}}" if bold else text


def resolve(arm: dict[int, str], fraction: int, granularity: str) -> str | tuple[str, int] | None:
    """The run backing (arm, fraction); per-episode arms borrow the endpoints."""
    if fraction in arm:
        return arm[fraction]
    if granularity != "per-episode" or fraction not in (0, 100):
        return None
    sibling = next(
        (runs for label, gran, runs in ARMS if gran == "per-transition" and label == arm["label"]),
        None,
    )
    return sibling.get(fraction) if sibling else None


def arms_for(granularity: str) -> list[tuple[str, dict[int, str]]]:
    """Arms shown in a planning table, with the baseline first for reference."""
    selected = []
    for label, gran, runs in ARMS:
        if gran in ("--", granularity):
            runs = dict(runs, label=label)
            selected.append((label, runs))
    return selected


def planning_table(granularity: str, caption: str, label: str) -> str:
    arms = arms_for(granularity)
    cells: dict[tuple[str, int], tuple[float, float] | None] = {}
    for arm_label, runs in arms:
        for fraction in FRACTIONS:
            run = resolve(runs, fraction, granularity)
            cells[(arm_label, fraction)] = planning_for(run, granularity) if run else None

    # Best per column, in bold.  The 0% column is at chance for every arm and is
    # left unbolded.
    best: dict[int, str | None] = {}
    for fraction in FRACTIONS:
        if fraction == 0:
            best[fraction] = None
            continue
        scored = [(cells[(a, fraction)][0], a) for a, _ in arms if cells[(a, fraction)]]
        best[fraction] = max(scored)[1] if scored else None

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        "& \\multicolumn{6}{c}{Labelled fraction (\\%)} \\\\",
        "\\cmidrule(lr){2-7}",
        "Method & " + " & ".join(str(f) for f in FRACTIONS) + " \\\\",
        "\\midrule",
    ]
    first_idm = next((i for i, (label, _) in enumerate(arms) if label.startswith("IDM")), None)
    for index, (arm_label, _) in enumerate(arms):
        if index == first_idm:
            lines.append("\\midrule")
        row = [
            fmt_planning(cells[(arm_label, f)], bold=best[f] == arm_label) for f in FRACTIONS
        ]
        lines.append(f"{arm_label} & " + " & ".join(row) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def next_state_table() -> str:
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\caption{\\textbf{One-step latent prediction error} (next-state MSE, $\\times 10^{-2}$; lower is "
        "better) at epoch~8. The persistence baseline ($\\hat z_{t+1}=z_t$) is $0.16$--$0.34$ across runs; a "
        "value above it indicates the predictor has not beaten copying the current latent. Rows above the rule "
        "use per-transition action supervision; rows below use per-episode supervision, and take their $0\\%$ "
        "and $100\\%$ endpoints from the matching per-transition arm. The step-matched baseline is read at "
        "epochs $160$/$32$/$16$/$11$, which were evaluated for planning and probes but not for next-state "
        "error, so that row is empty apart from its $100\\%$ endpoint (where it coincides with the "
        "epoch-matched baseline).}",
        "\\label{tab:next-state}",
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        "& \\multicolumn{6}{c}{Labelled fraction (\\%)} \\\\",
        "\\cmidrule(lr){2-7}",
        "Method & " + " & ".join(str(f) for f in FRACTIONS) + " \\\\",
        "\\midrule",
    ]
    per_transition = [(l, r) for l, g, r in ARMS if g in ("--", "per-transition")]
    per_episode = [(l, dict(r, label=l)) for l, g, r in ARMS if g == "per-episode"]
    for arm_label, runs in per_transition:
        row = [fmt_mse(runs.get(f)) for f in FRACTIONS]
        lines.append(f"{arm_label} & " + " & ".join(row) + " \\\\")
    lines.append("\\midrule")
    for arm_label, runs in per_episode:
        row = []
        for fraction in FRACTIONS:
            row.append(fmt_mse(resolve(runs, fraction, "per-episode")))
        # An arm with no next-state evaluation of its own would render as the
        # borrowed endpoints alone, repeating the per-transition row above.
        if all(cell == "--" for fraction, cell in zip(FRACTIONS, row) if fraction in runs):
            continue
        heading = (
            f"{arm_label[:-1]}, per-episode)" if arm_label.endswith(")") else f"{arm_label} (per-episode)"
        )
        lines.append(f"{heading} & " + " & ".join(row) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


PROBE_COLUMNS = (
    "block_orientation_ridge_r2",
    "block_orientation_mlp_r2",
    "block_displacement_ridge_r2",
    "block_displacement_mlp_r2",
    "agent_displacement_ridge_r2",
    "delta_to_action_ridge_r2",
    "code_to_action_ridge_r2",
)


def probe_table() -> str:
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{5pt}",
        "\\caption{\\textbf{Linear and non-linear decodability of the learned representation} (test $R^2$). "
        "\\emph{Ridge} probes measure linear decodability of the encoder state $z_t$; \\emph{MLP} probes give "
        "the non-linear ceiling. The two action probes read the action off the latent pair "
        "($\\Delta z \\to a$, defined for every arm) and off the predicted action code "
        "($c \\to a$, undefined for both baselines, which learn no code); comparing them separates what the "
        "representation already carries from what the IDM adds. Every block is read at epoch~8 except the "
        "step-matched baseline, whose epoch is given per row.}",
        "\\label{tab:probes}",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{llccccccc}",
        "\\toprule",
        "& & \\multicolumn{2}{c}{Block orient.} & \\multicolumn{2}{c}{Block displ.} & Agent displ. & "
        "$\\Delta z \\to a$ & $c \\to a$ \\\\",
        "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-7}\\cmidrule(lr){8-8}\\cmidrule(lr){9-9}",
        "Method & Frac.\\ (\\%) & Ridge & MLP & Ridge & MLP & Ridge & Ridge & Ridge \\\\",
    ]
    seen: set[tuple[str, int]] = set()
    for label, granularity, runs in ARMS:
        rows = []
        for fraction in FRACTIONS:
            entry_ref = runs.get(fraction)
            if entry_ref is None or (run_name(entry_ref), run_epoch(entry_ref)) not in PROBES:
                continue
            # A checkpoint standing in for a fraction it was not trained at
            # (the 75% per-episode placeholder) says nothing new about the
            # representation, so the probe table lists it once.
            if any(run_name(entry_ref) == run_name(e) and run_epoch(entry_ref) == run_epoch(e) for _, e in rows):
                continue
            rows.append((fraction, entry_ref))
        # Arms share checkpoints -- the 100% endpoint above all -- so a block
        # whose every row was already probed under another arm would just
        # repeat it. Blocks that contribute a new checkpoint keep all of theirs.
        if not rows or all(
            (run_name(entry), run_epoch(entry)) in seen for _, entry in rows
        ):
            continue
        seen.update((run_name(entry), run_epoch(entry)) for _, entry in rows)
        heading = label if granularity == "--" else f"{label.replace(' (', ' ').rstrip(')')} ({granularity})"
        lines.append("\\midrule")
        lines.append(f"\\multirow{{{len(rows)}}}{{*}}{{{heading}}}")
        for fraction, entry_ref in rows:
            probe = PROBES[(run_name(entry_ref), run_epoch(entry_ref))]
            values = [
                f"{probe[column]:.3f}" if column in probe else "--"
                for column in PROBE_COLUMNS
            ]
            # The step-matched block is the only one read away from epoch 8, so
            # it carries its epoch inline rather than forcing a column on every
            # other row.
            epoch_note = "" if run_epoch(entry_ref) == 8 else f" (e{run_epoch(entry_ref)})"
            lines.append(f" & {fraction}{epoch_note} & " + " & ".join(values) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table}"]
    return "\n".join(lines)


def inventory_table() -> str:
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\caption{\\textbf{Complete run inventory.} Every checkpoint entering "
        "Tables~\\ref{tab:planning-per-transition}--\\ref{tab:probes}, with the epoch it was read at, its "
        "planning success rate (mean\\,$\\pm$\\,s.e.m.\\ over five evaluation seeds, $50$ episodes each) "
        "and its one-step latent MSE. Per-episode arms omit the $0\\%$ and $100\\%$ endpoints, which are the "
        "per-transition runs of the same arm. The epoch-matched and per-transition step-matched blocks are the "
        "same five training runs read at two different epochs; the per-episode step-matched block is a "
        "separate set of runs trained under episode-level masking, and its $75\\%$ row reuses the $50\\%$ "
        "checkpoint. At $100\\%$ labels every scheme coincides and shares one checkpoint. All runs "
        "use training seed $0$. Next-state evaluation was run only on the epoch-8 checkpoints, so the "
        "step-matched baseline has no MSE outside its $100\\%$ endpoint.}",
        "\\label{tab:run-inventory}",
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "Method & Granularity & Frac.\\ (\\%) & Epoch & Planning (\\%) & MSE ($\\times10^{-2}$) \\\\",
    ]
    for label, granularity, runs in ARMS:
        lines.append("\\midrule")
        for fraction in FRACTIONS:
            entry_ref = runs.get(fraction)
            if entry_ref is None:
                continue
            planning = fmt_planning(planning_for(entry_ref, granularity))
            lines.append(
                f"{label} & {granularity} & {fraction} & {run_epoch(entry_ref)} & {planning} & "
                f"{fmt_mse(entry_ref)} \\\\"
            )
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]
    return "\n".join(lines)


def main() -> None:
    parts = [
        "% ============================================================",
        "%  Auto-generated by build_paper_tables.py -- do not edit by hand.",
        "%  Requires: \\usepackage{booktabs} and \\usepackage{multirow}",
        "% ============================================================",
        "",
        planning_table(
            "per-transition",
            "\\textbf{Planning success rate (\\%) under per-transition action supervision.} Latent "
            "planning on the held-out split, mean\\,$\\pm$\\,s.e.m.\\ over five evaluation seeds ($50$ "
            "episodes each, $n=250$ episodes per cell). Columns give the fraction of transitions carrying an "
            "action label. The two baselines are the same training runs read at two points: "
            "\\emph{epoch-matched} at epoch~8, as every other arm, and \\emph{step-matched} at epoch "
            "$8/\\rho$ ($160$, $32$, $16$, $11$ for $\\rho = 5$--$75\\%$), where the baseline has taken as "
            "many optimiser steps as an IDM arm does by epoch~8 despite seeing only the labelled fraction. "
            "Best method per column in bold; the $0\\%$ column is at chance for every method and is left "
            "unbolded.",
            "tab:planning-per-transition",
        ),
        "",
        planning_table(
            "per-episode",
            "\\textbf{Planning success rate (\\%) under per-episode action supervision.} Labels are "
            "allocated at the granularity of whole episodes rather than individual transitions; protocol "
            "otherwise identical to Table~\\ref{tab:planning-per-transition}. The epoch-matched baseline "
            "is supervision-granularity agnostic and is repeated for reference; the step-matched baseline "
            "is trained under episode-level masking, and its $75\\%$ cell reuses the $50\\%$ checkpoint, "
            "which was not retrained at $75\\%$. The $0\\%$ and $100\\%$ endpoints coincide with the "
            "per-transition runs.",
            "tab:planning-per-episode",
        ),
        "",
        next_state_table(),
        "",
        probe_table(),
        "",
        inventory_table(),
        "",
    ]
    rendered = "\n".join(parts)
    OUT.write_text(rendered)
    OUT_PAPER.write_text(rendered)
    print(f"wrote {OUT} and {OUT_PAPER}")


if __name__ == "__main__":
    main()
