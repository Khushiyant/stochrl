"""Final return vs where the fixed noise amount is pinned on the sensor-scale
spread. Reads a results dir with fixed-<stat> runs plus 'none' and
'uniform-calibrated' anchors.

  uv run python scripts/plot_fixed.py --outdir results_fixed --figdir assets
"""

from __future__ import annotations

import json
import os

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import tyro
from matplotlib.ticker import FuncFormatter, NullFormatter

from stochrl import collect_signal_stats
from stochrl.plotting import finish, set_style
from stochrl.presets import SCALE_STATS
from stochrl.stats import bootstrap_ci, iqm

STATS = ["p25", "median", "mean", "p75"]


def finals(outdir, mode):
    m = json.load(open(os.path.join(outdir, "manifest.json")))
    out = []
    for r in m["runs"]:
        if r["mode"] == mode and r["returncode"] == 0 and os.path.exists(r["csv"]):
            d = np.genfromtxt(r["csv"], delimiter=",", names=True)
            out.append(np.atleast_1d(d["eval_return"])[-1])
    return out


def main(outdir: str = "results_fixed", figdir: str = "assets", prefix: str = "fixed",
         env_id: str = "HalfCheetah-v5", calib_steps: int = 10_000, calib_seed: int = 0):
    set_style()
    os.makedirs(figdir, exist_ok=True)
    # same calibration rollout the training runs used, so the x positions are
    # the actual pinned values
    scale = collect_signal_stats(gym.make(env_id), steps=calib_steps, seed=calib_seed).scale

    xs, pts, err = [], [], [[], []]
    for stat in STATS:
        pt, lo, hi = bootstrap_ci(finals(outdir, f"fixed-{stat}"))
        xs.append(SCALE_STATS[stat](scale))
        pts.append(pt)
        err[0].append(pt - lo)
        err[1].append(hi - pt)

    clean = iqm(finals(outdir, "none"))
    calib = iqm(finals(outdir, "uniform-calibrated"))

    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.axhline(clean, color="0.0", ls="-", lw=1.0, alpha=0.5, label="no noise")
    ax.axhline(calib, color="0.35", ls="--", lw=1.2, label="scaled per sensor")
    ax.errorbar(xs, pts, yerr=err, color="black", marker="o", capsize=3,
                label="fixed amount (IQM, 95% CI)")
    for x, pt, stat in zip(xs, pts, STATS):
        ax.annotate(stat, (x, pt), textcoords="offset points", xytext=(6, 6), fontsize=9)
    ax.set_xscale("log")
    ax.set_xticks([0.3, 1, 3, 6])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set(xlabel="sensor-scale value the fixed amount is pinned to (log)",
           ylabel="final clean-eval return")
    ax.set_title(f"Fixed noise amount: performance by pinning point ({env_id})")
    ax.legend(loc="lower left")
    finish(fig, ax)
    fig.savefig(f"{figdir}/{prefix}_pinning.png")
    print(f"Saved {figdir}/{prefix}_pinning.png")


if __name__ == "__main__":
    tyro.cli(main)
