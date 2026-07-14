# usage: uv run python scripts/plot_fixed.py --figdir assets

from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import tyro

from stochrl.plotting import finish, set_style
from stochrl.stats import bootstrap_ci, iqm

CONDITIONS = [("fixed-p25", "25th pct"), ("fixed-median", "median"), ("fixed-mean", "mean"),
              ("fixed-p75", "75th pct"), ("uniform-calibrated", "scaled\nper sensor")]
ENV_STYLE = [("0.15", ""), ("0.5", "//"), ("0.8", "xx")]


def finals(outdir, mode):
    m = json.load(open(os.path.join(outdir, "manifest.json")))
    out = []
    for r in m["runs"]:
        if r["mode"] == mode and r["returncode"] == 0 and os.path.exists(r["csv"]):
            d = np.genfromtxt(r["csv"], delimiter=",", names=True)
            out.append(np.atleast_1d(d["eval_return"])[-1])
    return out


def main(figdir: str = "assets", prefix: str = "fixed",
         pairs: list[str] = ["HalfCheetah-v5:results_fixed", "Hopper-v5:results_fixed_hopper",
                             "Walker2d-v5:results_fixed_walker2d"]):
    set_style()
    os.makedirs(figdir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    x = np.arange(len(CONDITIONS))
    width = 0.8 / len(pairs)
    for i, pair in enumerate(pairs):
        env_id, outdir = pair.split(":")
        clean = iqm(finals(outdir, "none"))
        ys, err = [], [[], []]
        for mode, _ in CONDITIONS:
            pt, lo, hi = bootstrap_ci(finals(outdir, mode))
            ys.append(100 * pt / clean)
            err[0].append(100 * (pt - lo) / clean)
            err[1].append(100 * (hi - pt) / clean)
        shade, hatch = ENV_STYLE[i % len(ENV_STYLE)]
        ax.bar(x + (i - (len(pairs) - 1) / 2) * width, ys, width, yerr=err, capsize=2.5,
               color=shade, hatch=hatch, edgecolor="black", lw=0.7,
               label=env_id.replace("-v5", ""))
    ax.axhline(100, color="black", ls="--", lw=1, label="no noise")
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in CONDITIONS])
    ax.set(xlabel="where the fixed amount is pinned on the spread of sensor scales",
           ylabel="final return, % of clean score")
    ax.set_title("One fixed noise amount: the pinning choice decides the outcome")
    ax.legend(loc="upper right")
    finish(fig, ax)
    fig.savefig(f"{figdir}/{prefix}_pinning.png")
    print(f"Saved {figdir}/{prefix}_pinning.png")


if __name__ == "__main__":
    tyro.cli(main)
