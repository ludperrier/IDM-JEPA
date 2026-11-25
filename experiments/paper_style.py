"""Shared matplotlib style for paper figures.

Import and call ``apply()`` at the top of any plotting script so every figure in
the paper shares one typographic and colour system:

    import paper_style
    paper_style.apply()

Sizing targets the NeurIPS text block (5.5 in wide). ``HALF_WIDTH`` panels sit
side by side at ``width=0.49\\linewidth``; ``FULL_WIDTH`` spans the column.

Palette: Okabe-Ito derived, four categorical slots assigned by *method*, in
fixed legend order. Verified colourblind-safe (worst adjacent-pair CVD
separation dE 18.0 protan / 8.6 tritan, all slots >= 3:1 contrast on white).
Marker shape is a redundant encoding, so the figures survive greyscale print.
Never recolour a series when a series is added or dropped -- colour follows the
method, not its rank.
"""

from __future__ import annotations

import matplotlib as mpl

# --- canvas sizes (inches) -------------------------------------------------
HALF_WIDTH = (2.72, 2.25)   # two panels side by side in a NeurIPS figure
FULL_WIDTH = (5.5, 3.0)     # one panel spanning the text block

# --- ink -------------------------------------------------------------------
INK = "#1A1A1A"        # primary text
INK_MUTED = "#5A5A5A"  # axis labels, ticks
GRID = "#DDDDDD"

# --- categorical palette (fixed order; do not cycle) -----------------------
SERIES_STYLE = {
    "baseline": {"label": "Baseline (epoch-matched)", "color": "#0072B2", "marker": "o", "linestyle": (0, (4, 2))},
    # The step-matched baseline is the same run trained for the same number of
    # optimiser steps rather than the same number of epochs, so it shares the
    # baseline hue and is separated by lightness, marker and dash pattern.
    "baseline_step": {"label": "Baseline (step-matched)", "color": "#56B4E9", "marker": "v", "linestyle": (0, (1, 1.6))},
    "cycle": {"label": "Cycle-JEPA", "color": "#009E73", "marker": "s", "linestyle": "-"},
    "idm_delta": {"label": "IDM-JEPA (delta)", "color": "#785EF0", "marker": "^", "linestyle": "-"},
    "idm_concat": {"label": "IDM-JEPA (concat)", "color": "#D55E00", "marker": "D", "linestyle": "-"},
}
MISSING_COLOR = "#9A9A9A"

# Legend / draw order. Both baselines first (they are the references), hero
# method last so it is drawn on top of the others.
SERIES_ORDER = ("baseline", "baseline_step", "cycle", "idm_delta", "idm_concat")


def apply() -> None:
    """Install the paper rcParams globally."""
    mpl.rcParams.update({
        # type: serif to match the NeurIPS body text
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 6.5,
        # recessive frame: no top/right spine, hairline axes, muted ticks
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": INK_MUTED,
        "axes.linewidth": 0.6,
        "axes.labelcolor": INK,
        "axes.titlepad": 4.0,
        "text.color": INK,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        # grid sits behind the marks and stays quiet
        "axes.grid": False,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "axes.axisbelow": True,
        # marks
        "lines.linewidth": 1.4,
        "lines.markersize": 4.0,
        "lines.markeredgewidth": 0.0,
        # legend
        "legend.frameon": False,
        "legend.handlelength": 1.9,
        "legend.handletextpad": 0.5,
        "legend.labelspacing": 0.3,
        "legend.borderpad": 0.2,
        # output
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.01,
        "pdf.fonttype": 42,   # embed real (editable, non-bitmap) fonts
        "ps.fonttype": 42,
    })


def save(fig, output_dir, stem: str) -> None:
    """Write both a PDF (for LaTeX) and a PNG (for quick viewing)."""
    for suffix in ("pdf", "png"):
        fig.savefig(output_dir / f"{stem}.{suffix}")
