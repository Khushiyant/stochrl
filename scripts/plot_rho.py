"""Final return vs noise level (rho), paper-style. Reads several per-rho sweep dirs.

  uv run python scripts/plot_rho.py --pairs 0.0:results_cleanrl 0.05:results_rho005 \
      0.1:results_cleanrl 0.2:results_rho020
"""

from __future__ import annotations

import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import tyro

from stochrl.plotting import finish, set_style, style

MODES = ["uniform-calibrated", "realistic"]


def finals(outdir, mode):
    m = json.load(open(os.path.join(outdir, "manifest.json")))
    out = []
    for r in m["runs"]:
        if r["mode"] == mode and r["returncode"] == 0 and os.path.exists(r["csv"]):
            d = np.genfromtxt(r["csv"], delimiter=",", names=True)
            out.append(np.atleast_1d(d["eval_return"])[-1])
    return out


def main(pairs: list[str], figdir: str = "assets", prefix: str = "rho",
         clean_dir: str = "results_cleanrl"):
    """pairs: list of 'rho:outdir'. clean_dir supplies the rho=0 (no-noise) anchor."""
    set_style()
    rho_dirs = sorted(((float(p.split(":")[0]), p.split(":")[1]) for p in pairs), key=lambda x: x[0])
    clean = np.mean(finals(clean_dir, "none"))

    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    for mode in MODES:
        shade, ls, marker, _, label = style(mode)
        xs, mean, sd = [0.0], [clean], [np.std(finals(clean_dir, "none"))]
        for rho, d in rho_dirs:
            vals = finals(d, mode)
            if vals:
                xs.append(rho); mean.append(np.mean(vals)); sd.append(np.std(vals))
        order = np.argsort(xs)
        xs, mean, sd = np.array(xs)[order], np.array(mean)[order], np.array(sd)[order]
        ax.errorbar(xs, mean, yerr=sd, color=shade, ls=ls, marker=marker, capsize=3, label=label)
    ax.axhline(clean, color="0.0", ls="-", lw=1, alpha=0.5)
    ax.set(xlabel=r"observation noise level $\rho$", ylabel="final clean-eval return")
    ax.set_title("Performance vs noise level (HalfCheetah-v5)")
    ax.legend()
    finish(fig, ax)
    fig.savefig(f"{figdir}/{prefix}_sweep.png")
    print(f"Saved {figdir}/{prefix}_sweep.png")


if __name__ == "__main__":
    tyro.cli(main)
