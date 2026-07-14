"""Shared plotting style: serif type, despined axes, grayscale series.

Series are distinguished by shade + line style + marker (and hatch for bars)
so the figures survive black-and-white printing.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# per-mode style: (gray shade, linestyle, marker, hatch, label)
MODE_STYLE = {
    "none":               ("0.0",  "-",  "o", "",   "no noise"),
    "uniform":            ("0.45", "--", "s", "//",  "uniform (absolute, naive)"),
    "uniform-calibrated": ("0.25", ":",  "^", "..",  "uniform (calibrated)"),
    "realistic":          ("0.10", "-.", "D", "xx",  "realistic (structured)"),
    "vel-flat":           ("0.45", "--", "s", "\\\\", "velocity noise, flat (state-indep.)"),
    "vel-statedep":       ("0.0",  "-",  "D", "xx",  "velocity noise, state-dependent"),
    "pos-flat":           ("0.45", "--", "s", "\\\\", "position noise, flat (state-indep.)"),
    "pos-statedep":       ("0.0",  "-",  "D", "xx",  "position noise, state-dependent"),
    "both-ff":            ("0.55", "--", "s", "//",  "both flat"),
    "both-sf":            ("0.30", ":",  "^", "..",  "velocity state-dep, position flat"),
    "both-fs":            ("0.15", "-.", "v", "\\\\", "velocity flat, position state-dep"),
    "both-ss":            ("0.0",  "-",  "D", "xx",  "both state-dependent"),
    "fixed-p25":          ("0.60", ":",  "v", "..",  "fixed = 25th pct of scales"),
    "fixed-median":       ("0.45", "--", "^", "//",  "fixed = median of scales"),
    "fixed-mean":         ("0.30", "-.", "s", "\\\\", "fixed = mean of scales"),
    "fixed-p75":          ("0.15", ":",  "o", "xx",  "fixed = 75th pct of scales"),
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
    """Despine and tighten; call before saving."""
    if isinstance(ax, plt.Axes):
        sns.despine(ax=ax)
    fig.tight_layout()
