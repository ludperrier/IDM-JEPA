"""Combine PCA spectra from the selected v10 and v3 latent-geometry evaluations."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FAMILY_STYLE = {
    "baseline": ("JEPA baseline", "#202124", "-"),
    "idm": ("IDM", "#007C91", "-"),
    "cycle": ("Cycle", "#C94C12", "-"),
    "cycle_ramp": ("Cycle ramp", "#7B3F98", "-"),
}


def parse_run(run: str) -> tuple[str, str, int] | None:
    match = re.fullmatch(
        r"(?P<tag>v\d+)_(?P<arm>baseline_100|base_frac_\d+|(?:idm|cyc|cyc_ramp)_frac_\d+)_s(?P<seed>\d+)",
        run,
    )
    if match is None:
        return None
    arm = match.group("arm")
    if arm == "baseline_100" or arm.startswith("base_frac_"):
        family = "baseline"
    elif arm.startswith("cyc_ramp_"):
        family = "cycle_ramp"
    else:
        family = arm.split("_", 1)[0]
    return match.group("tag"), family, int(match.group("seed"))


def selected_run(run: str) -> bool:
    parsed = parse_run(run)
    if parsed is None:
        return False
    tag, _, _ = parsed
    if tag == "v10":
        return True
    return run == "v3_baseline_100_s0" or run.endswith("_frac_0_s0")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot selected v10/v3 latent PCA spectra together.")
    parser.add_argument("--geometry-dir", type=Path, required=True)
    parser.add_argument("--epoch", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    spectra = []
    for spectrum_path in sorted(args.geometry_dir.rglob("pca_spectrum.csv")):
        run = spectrum_path.parent.name
        if not selected_run(run):
            continue
        frame = pd.read_csv(spectrum_path)
        if frame.empty:
            continue
        spectra.append((run, frame))
    if not spectra:
        raise FileNotFoundError("No selected PCA spectra found.")

    figure, axis = plt.subplots(figsize=(10.2, 6.4), constrained_layout=True)
    for run, frame in spectra:
        tag, family, _ = parse_run(run)
        label, color, linestyle = FAMILY_STYLE[family]
        if tag == "v3":
            linestyle = "--"
        axis.semilogy(
            np.arange(1, len(frame) + 1),
            frame["explained_variance_ratio"].clip(lower=1e-12),
            color=color,
            linestyle=linestyle,
            linewidth=1.6,
            alpha=0.9,
            label=f"{label} ({run})",
        )

    axis.set(
        title=f"Held-out latent PCA spectra at epoch {args.epoch}",
        xlabel="Principal component",
        ylabel="Explained variance ratio",
    )
    axis.grid(alpha=0.2)
    axis.legend(fontsize=6.6, ncol=2, frameon=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=220)
    plt.close(figure)
    print(f"Wrote {args.output} from {len(spectra)} model spectra.")


if __name__ == "__main__":
    main()