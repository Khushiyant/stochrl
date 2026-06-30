"""Aggregate benchmark CSVs into conference-style learning-curve + summary figures.

Reads <outdir>/manifest.json and each run's CSV (columns: step,eval_return),
groups by noise mode, and writes grayscale, black-and-white-safe figures:
  <figdir>/<prefix>_curves.png   clean-eval return vs steps, mean +/- std over seeds
  <figdir>/<prefix>_final.png    final-performance bar chart per noise mode

  uv run python scripts/plot_results.py --outdir results --prefix benchmark
"""

from __future__ import annotations

import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import tyro

from stochrl.plotting import finish, set_style, style

# Plot modes in this canonical order when present.
ORDER = ["none", "uniform", "uniform-calibrated", "realistic", "vel-flat", "vel-statedep"]


def load(manifest_path):
    with open(manifest_path) as f:
        manifest = json.load(f)
    by_mode = defaultdict(list)
    for run in manifest["runs"]:
        if run["returncode"] != 0 or not os.path.exists(run["csv"]):
            continue
        data = np.genfromtxt(run["csv"], delimiter=",", names=True)
        if data.size:
            by_mode[run["mode"]].append((np.atleast_1d(data["step"]),
                                         np.atleast_1d(data["eval_return"])))
    return manifest, by_mode


def stack_on_common_grid(runs):
    """Align seeds on the shortest common step grid and stack returns."""
    n = min(len(steps) for steps, _ in runs)
    grid = runs[0][0][:n]
    return grid, np.stack([rets[:n] for _, rets in runs])


def main(outdir: str = "results", figdir: str = "figures", prefix: str = "benchmark",
         modes: list[str] | None = None):
    """`modes` optionally restricts which noise modes to plot (default: all present)."""
    set_style()
    os.makedirs(figdir, exist_ok=True)
    manifest, by_mode = load(os.path.join(outdir, "manifest.json"))
    if modes:
        by_mode = {m: by_mode[m] for m in modes if m in by_mode}
    modes = [m for m in ORDER if m in by_mode] + [m for m in by_mode if m not in ORDER]

    # ---- Learning curves ---------------------------------------------------- #
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    table = []
    for mode in modes:
        shade, ls, marker, _, label = style(mode)
        grid, mat = stack_on_common_grid(by_mode[mode])
        mean, sd = mat.mean(0), mat.std(0)
        ax.plot(grid, mean, color=shade, ls=ls, marker=marker, markevery=2, label=label)
        ax.fill_between(grid, mean - sd, mean + sd, color=shade, alpha=0.12, lw=0)
        table.append((mode, mat[:, -1].mean(), mat[:, -1].std(), mat.shape[0]))
    ax.set(xlabel="environment steps", ylabel="clean-eval episodic return")
    ax.set_title(f"SAC on {manifest['args']['env_id']} under observation noise "
                 fr"($\rho={manifest['args']['rho']}$, mean$\pm$std over seeds)")
    ax.legend(loc="upper left")
    finish(fig, ax)
    fig.savefig(f"{figdir}/{prefix}_curves.png")

    # ---- Final-performance bar chart ---------------------------------------- #
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for i, (mode, mean, sd, _) in enumerate(table):
        shade, _, _, hatch, _ = style(mode)
        ax.bar(i, mean, yerr=sd, color=shade, edgecolor="black", hatch=hatch,
               capsize=4, width=0.7)
    ax.set_xticks(range(len(table)))
    ax.set_xticklabels([style(t[0])[4] for t in table], rotation=25, ha="right")
    ax.set(ylabel="final clean-eval return")
    ax.set_title(f"Final performance by noise mode ({manifest['args']['env_id']})")
    finish(fig, ax)
    fig.savefig(f"{figdir}/{prefix}_final.png")

    # ---- Console table ------------------------------------------------------ #
    base = next((t[1] for t in table if t[0] == "none"), None)
    print(f"\nResults — {manifest['args']['env_id']}, rho={manifest['args']['rho']}, "
          f"{manifest['args']['total_timesteps']} steps:\n")
    print(f"  {'mode':24s} {'final return':>14s} {'+/- std':>10s} {'seeds':>6s} {'vs clean':>10s}")
    for mode, mean, sd, n in table:
        rel = f"{100 * mean / base:4.0f}%" if base else "n/a"
        print(f"  {mode:24s} {mean:14.0f} {sd:10.0f} {n:6d} {rel:>10s}")
    print(f"\nSaved {figdir}/{prefix}_curves.png and {figdir}/{prefix}_final.png")


if __name__ == "__main__":
    tyro.cli(main)
