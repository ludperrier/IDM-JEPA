"""Plot PushT planning accuracy vs. training epoch for v13 IDM frac_50 vs. the
v10 baseline frac_50 model (one-off comparison, epochs 1-8).
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PLANNING_DIR = Path(__file__).parent / "analysis_out" / "planning_eval"
OUTPUT_DIR = Path(__file__).parent / "analysis_out" / "planning_benchmark"
SOLVER = "cem"
RUNS = {
    "v13_idm_frac_50_s0": {"label": "IDM-JEPA delta (v13) 50%", "color": "#007C91", "marker": "o"},
    "v10_base_frac_50_s0": {"label": "JEPA baseline (v10) 50%", "color": "#202124", "marker": "D"},
}


def parse_result_file(path: Path) -> list[list[bool]]:
    text = path.read_text()
    evaluations = []
    for block in re.split(r"(?=\{')", text):
        match = re.search(r"'episode_successes': array\(\[(.*?)\]\)", block, re.DOTALL)
        if match is None:
            continue
        outcomes = re.findall(r"\b(True|False)\b", match.group(1))
        if outcomes:
            evaluations.append([outcome == "True" for outcome in outcomes])
    return evaluations


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    for run, style in RUNS.items():
        for epoch in range(1, 9):
            result_path = PLANNING_DIR / run / f"epoch_{epoch}" / f"solver_{SOLVER}" / "eval_seed_42" / "pusht_results.txt"
            if not result_path.exists():
                continue
            for outcomes in parse_result_file(result_path):
                for success in outcomes:
                    records.append({"run": run, "epoch": epoch, "success": int(success)})

    if not records:
        raise FileNotFoundError(f"No planning results found under {PLANNING_DIR} for {sorted(RUNS)}.")

    results = pd.DataFrame(records)
    accuracy = results.groupby(["run", "epoch"], as_index=False).agg(
        successes=("success", "sum"), episodes=("success", "size")
    )
    accuracy["accuracy"] = 100 * accuracy["successes"] / accuracy["episodes"]
    accuracy.to_csv(OUTPUT_DIR / "v13_frac50_vs_baseline_epoch_curve.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for run, style in RUNS.items():
        frame = accuracy[accuracy["run"] == run].sort_values("epoch")
        if frame.empty:
            continue
        ax.plot(frame["epoch"], frame["accuracy"], marker=style["marker"], color=style["color"], label=style["label"])
    ax.set(xlabel="Training epoch", ylabel="PushT planning accuracy (%)", xlim=(0.5, 8.5), ylim=(0, 100))
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.8)
    ax.legend(frameon=False)
    fig.tight_layout()
    output_path = OUTPUT_DIR / "v13_frac50_vs_baseline_epoch_curve.png"
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
