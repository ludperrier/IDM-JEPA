"""Build a paginated PDF of planning and latent-geometry figures.

Example:
    python build_analysis_figure_book.py
    python build_analysis_figure_book.py --output analysis_out/figure_book/v10.pdf
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image


ROOT = Path(__file__).parent
MAX_IMAGE_PIXELS = 1_600
PDF_DPI = 120


def natural_key(path: Path) -> list[object]:
    return [int(token) if token.isdigit() else token.lower() for token in re.split(r"(\d+)", str(path))]


def display_name(path: Path) -> str:
    return path.stem.replace("_", " ")


def add_header(figure: plt.Figure, title: str, subtitle: str = "") -> None:
    figure.suptitle(title, x=0.04, y=0.97, ha="left", fontsize=18, fontweight="bold")
    if subtitle:
        figure.text(0.04, 0.935, subtitle, ha="left", va="top", fontsize=9, color="#555555")


def add_footer(figure: plt.Figure, page: int) -> None:
    figure.text(0.96, 0.025, f"Page {page}", ha="right", va="bottom", fontsize=8, color="#666666")


def draw_image(axis: plt.Axes, image_path: Path, caption: str) -> bool:
    try:
        with Image.open(image_path) as image:
            image.thumbnail((MAX_IMAGE_PIXELS, MAX_IMAGE_PIXELS))
            axis.imshow(image.copy())
    except (OSError, ValueError) as error:
        axis.text(0.5, 0.5, f"Could not read image\n{error}", ha="center", va="center")
        axis.set_axis_off()
        return False
    axis.set_title(caption, fontsize=10, pad=7)
    axis.set_axis_off()
    return True


def add_grid_pages(pdf: PdfPages, title: str, figure_paths: list[Path], source_root: Path, page: int) -> int:
    for start in range(0, len(figure_paths), 4):
        chunk = figure_paths[start : start + 4]
        figure, axes = plt.subplots(2, 2, figsize=(11.7, 8.3))
        add_header(figure, title, "Planning benchmark figures")
        for axis, image_path in zip(axes.flat, chunk):
            draw_image(axis, image_path, display_name(image_path))
            axis.text(
                0.5,
                -0.05,
                str(image_path.relative_to(source_root)),
                transform=axis.transAxes,
                ha="center",
                va="top",
                fontsize=7,
                color="#666666",
            )
        for axis in axes.flat[len(chunk) :]:
            axis.set_axis_off()
        add_footer(figure, page)
        figure.tight_layout(rect=(0.02, 0.05, 0.98, 0.91))
        pdf.savefig(figure, dpi=PDF_DPI)
        plt.close(figure)
        page += 1
    return page


def add_comparison_page(pdf: PdfPages, image_path: Path, source_root: Path, page: int) -> int:
    figure, axis = plt.subplots(figsize=(11.7, 8.3))
    add_header(figure, "Latent Geometry Comparison", str(image_path.relative_to(source_root)))
    draw_image(axis, image_path, display_name(image_path))
    add_footer(figure, page)
    figure.tight_layout(rect=(0.02, 0.05, 0.98, 0.91))
    pdf.savefig(figure, dpi=PDF_DPI)
    plt.close(figure)
    return page + 1


def add_run_page(pdf: PdfPages, run_dir: Path, image_paths: list[Path], source_root: Path, page: int) -> int:
    figure, axes = plt.subplots(1, 3, figsize=(11.7, 5.0))
    relative_run = run_dir.relative_to(source_root)
    add_header(figure, "Latent Geometry", str(relative_run))
    desired_order = {"latent_geometry.png": 0, "latent_geometry_3d.png": 1, "latent_geometry_tsne.png": 2}
    image_paths = sorted(image_paths, key=lambda path: desired_order.get(path.name, 99))
    for axis, image_path in zip(axes, image_paths):
        draw_image(axis, image_path, display_name(image_path))
    for axis in axes[len(image_paths) :]:
        axis.set_axis_off()
    add_footer(figure, page)
    figure.tight_layout(rect=(0.02, 0.05, 0.98, 0.89))
    pdf.savefig(figure, dpi=PDF_DPI)
    plt.close(figure)
    return page + 1


def build_figure_book(latent_root: Path, planning_root: Path, output: Path) -> tuple[int, int]:
    planning_figures = sorted(planning_root.rglob("*.png"), key=natural_key)
    latent_figures = sorted(latent_root.rglob("*.png"), key=natural_key)
    comparison_figures = [
        path for path in latent_figures if path.name == "pca_spectra_comparison.png"
    ]
    run_groups: dict[Path, list[Path]] = defaultdict(list)
    for image_path in latent_figures:
        if image_path not in comparison_figures:
            run_groups[image_path.parent].append(image_path)

    output.parent.mkdir(parents=True, exist_ok=True)
    page = 1
    with PdfPages(output) as pdf:
        figure = plt.figure(figsize=(11.7, 8.3))
        add_header(figure, "Latent-JEPA Analysis Figure Book")
        figure.text(
            0.05,
            0.78,
            "A consolidated visual record of generated planning benchmarks\nand latent-geometry analyses.",
            fontsize=19,
            va="top",
        )
        figure.text(
            0.05,
            0.54,
            f"Planning figures: {len(planning_figures)}\n"
            f"Latent-geometry comparisons: {len(comparison_figures)}\n"
            f"Latent-geometry model panels: {len(run_groups)}",
            fontsize=13,
            va="top",
        )
        figure.text(
            0.05,
            0.20,
            "Sources\n"
            f"{planning_root}\n"
            f"{latent_root}",
            fontsize=9,
            va="top",
            color="#555555",
        )
        add_footer(figure, page)
        pdf.savefig(figure, dpi=PDF_DPI)
        plt.close(figure)
        page += 1

        page = add_grid_pages(pdf, "Planning Benchmarks", planning_figures, planning_root, page)
        for image_path in comparison_figures:
            page = add_comparison_page(pdf, image_path, latent_root, page)
        for run_dir in sorted(run_groups, key=natural_key):
            page = add_run_page(pdf, run_dir, run_groups[run_dir], latent_root, page)

    return len(planning_figures), len(latent_figures)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a global PDF figure book for evaluation outputs.")
    parser.add_argument("--latent-geometry-dir", type=Path, default=ROOT / "analysis_out" / "latent_geometry")
    parser.add_argument("--planning-benchmark-dir", type=Path, default=ROOT / "analysis_out" / "planning_benchmark")
    parser.add_argument("--output", type=Path, default=ROOT / "analysis_out" / "figure_book" / "analysis_figure_book.pdf")
    args = parser.parse_args()

    planning_count, latent_count = build_figure_book(
        args.latent_geometry_dir, args.planning_benchmark_dir, args.output
    )
    print(f"Wrote {args.output} from {planning_count} planning and {latent_count} latent-geometry figures.")


if __name__ == "__main__":
    main()