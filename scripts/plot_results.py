# usage: uv run python scripts/plot_results.py --outdir results --prefix benchmark

from __future__ import annotations

import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import tyro

from stochrl.plotting import finish, set_style, style
from stochrl.stats import bootstrap_ci, estimator_name

# plot modes in this order when present
ORDER = ["none", "uniform", "fixed-p25", "fixed-median", "fixed-mean", "fixed-p75",
         "uniform-calibrated", "realistic",
         "vel-flat", "vel-statedep", "pos-flat", "pos-statedep",
         "both-ff", "both-sf", "both-fs", "both-ss"]


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

    # learning curves
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    finals = []  # (mode, per-seed final returns)
    for mode in modes:
        shade, ls, marker, _, label = style(mode)
        grid, mat = stack_on_common_grid(by_mode[mode])
        mean, sd = mat.mean(0), mat.std(0)
        ax.plot(grid, mean, color=shade, ls=ls, marker=marker, markevery=2, label=label)
        ax.fill_between(grid, mean - sd, mean + sd, color=shade, alpha=0.12, lw=0)
        finals.append((mode, mat[:, -1]))
    ax.set(xlabel="environment steps", ylabel="clean-eval episodic return")
    ax.set_title(f"SAC on {manifest['args']['env_id']} under observation noise "
                 fr"($\rho={manifest['args']['rho']}$, mean$\pm$std over seeds)")
    ax.legend(loc="upper left")
    finish(fig, ax)
    fig.savefig(f"{figdir}/{prefix}_curves.png")

    # final performance: IQM + 95% bootstrap CI over seeds
    summary = [(mode, *bootstrap_ci(f), len(f)) for mode, f in finals]
    est = estimator_name(min(s[4] for s in summary)) if summary else "IQM"
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for i, (mode, pt, lo, hi, _) in enumerate(summary):
        shade, _, _, hatch, _ = style(mode)
        ax.bar(i, pt, yerr=[[pt - lo], [hi - pt]], color=shade, edgecolor="black",
               hatch=hatch, capsize=4, width=0.7)
    ax.set_xticks(range(len(summary)))
    ax.set_xticklabels([style(s[0])[4] for s in summary], rotation=25, ha="right")
    ax.set(ylabel=f"final clean-eval return ({est}, 95% CI)")
    ax.set_title(f"Final performance by noise mode ({manifest['args']['env_id']})")
    finish(fig, ax)
    fig.savefig(f"{figdir}/{prefix}_final.png")

    # console table
    base = next((s[1] for s in summary if s[0] == "none"), None)
    print(f"\nResults — {manifest['args']['env_id']}, rho={manifest['args']['rho']}, "
          f"{manifest['args']['total_timesteps']} steps ({est} [95% CI] over seeds):\n")
    print(f"  {'mode':24s} {est:>8s} {'95% CI':>18s} {'seeds':>6s} {'vs clean':>9s}")
    for mode, pt, lo, hi, n in summary:
        rel = f"{100 * pt / base:4.0f}%" if base else "n/a"
        print(f"  {mode:24s} {pt:8.0f} {f'[{lo:.0f}, {hi:.0f}]':>18s} {n:6d} {rel:>9s}")
    print(f"\nSaved {figdir}/{prefix}_curves.png and {figdir}/{prefix}_final.png")


if __name__ == "__main__":
    tyro.cli(main)
