"""Shared conference-style plotting: crisp, grayscale, seaborn-based.

Every figure in the repo goes through `set_style()` so the look is consistent and
publication-ready: serif type, despined axes, ticks inward, no chartjunk. Series
are distinguished by grayscale shade + line style + marker (and hatch for bars),
so the figures survive black-and-white printing.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# Per-mode style: (gray shade, linestyle, marker, hatch, label). Black-and-white safe.
MODE_STYLE = {
    "none":               ("0.0",  "-",  "o", "",   "no noise"),
    "uniform":            ("0.45", "--", "s", "//",  "uniform (absolute, naive)"),
    "uniform-calibrated": ("0.25", ":",  "^", "..",  "uniform (calibrated)"),
    "realistic":          ("0.10", "-.", "D", "xx",  "realistic (structured)"),
    "vel-flat":           ("0.45", "--", "s", "\\\\", "velocity noise, flat (state-indep.)"),
    "vel-statedep":       ("0.0",  "-",  "D", "xx",  "velocity noise, state-dependent"),
}


def set_style():
    sns.set_theme(context="paper", style="ticks", font="serif")
    mpl.rcParams.update({
        "figure.dpi": 200,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "axes.linewidth": 0.9,
        "lines.linewidth": 1.8,
        "lines.markersize": 4.5,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "image.cmap": "Greys",
    })


def style(mode):
    """Return (shade, linestyle, marker, hatch, label) for a mode, with a fallback."""
    return MODE_STYLE.get(mode, ("0.5", "-", "o", "", mode))


def finish(fig, ax):
    """Despine + tighten — call before saving."""
    if isinstance(ax, plt.Axes):
        sns.despine(ax=ax)
    fig.tight_layout()
