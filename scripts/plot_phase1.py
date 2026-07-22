"""Phase-1 figure (scaled noise: obs vs transition), original 2-panel style.
usage: uv run python scripts/plot_phase1.py <cheetah|walker|quadruped>
Panel A: clean-env learning curves (IQM over seeds, IQR band). Panel B: final % of clean, 95% bootstrap CI."""
import glob
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

rng = np.random.default_rng(0)


def iqm(x):
    x = np.sort(np.asarray(x, float))
    k = int(np.floor(0.25 * len(x)))
    return float(np.mean(x[k:len(x) - k] if len(x) - 2 * k > 0 else x))


def boot_ratio(num, den, n=10000):
    num, den = np.asarray(num, float), np.asarray(den, float)
    s = [100 * iqm(rng.choice(num, len(num), True)) / iqm(rng.choice(den, len(den), True)) for _ in range(n)]
    return np.percentile(s, 2.5), np.percentile(s, 97.5)


# condition: (label, dir, color)  Okabe-Ito; hue=type, shade=level
CONDS = [("clean (ρ0)", "clean", "#444444"),
         ("obs ρ0.05", "obs_r005", "#56B4E9"),
         ("obs ρ0.10", "obs_r010", "#0072B2"),
         ("transition ρ0.05", "trans_r005", "#E69F00"),
         ("transition ρ0.10", "trans_r010", "#D55E00")]

TITLES = {"cheetah": "cheetah-run", "walker": "walker-run", "quadruped": "quadruped-run"}
CAVEATS = {
    "cheetah": "Wide 95% CIs reflect only 5 seeds + a weak clean seed; treat as a strong direction, not a tight effect.",
    "walker": "5 seeds; transition ρ0.10 collapses because the state kicks topple the balancing walker (a task-difficulty, not observability, effect).",
    "quadruped": "5 seeds; preliminary.",
}


def main(env="cheetah"):
    curves, finals = {}, {}
    for label, d, c in CONDS:
        files = sorted(glob.glob(f"results/{env}/{d}/*.csv"))
        if not files:
            continue
        per_step, cf, nf = {}, [], []
        for f in files:
            a = np.atleast_1d(np.genfromtxt(f, delimiter=",", names=True))
            for row in a:
                per_step.setdefault(int(row["step"]), []).append(float(row["eval_return"]))
            cf.append(float(a[-1]["eval_return"]))
            nf.append(float(a[-1]["eval_return_noisy"]))
        steps = sorted(s for s, v in per_step.items() if len(v) >= 3)
        curves[d] = (np.array(steps), np.array([iqm(per_step[s]) for s in steps]),
                     np.array([np.percentile(per_step[s], 25) for s in steps]),
                     np.array([np.percentile(per_step[s], 75) for s in steps]))
        finals[d] = (cf, nf)
    if "clean" not in finals:
        raise SystemExit(f"no clean results under results/{env}/clean")
    base = finals["clean"][0]
    present = [(label, d, c) for label, d, c in CONDS if d in finals]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1.35, 1]})

    # Panel A: learning curves
    for label, d, c in present:
        s, ic, p25, p75 = curves[d]
        ax1.fill_between(s / 1e6, p25, p75, color=c, alpha=0.12, linewidth=0)
        ax1.plot(s / 1e6, ic, color=c, lw=2.2, label=label)
    ax1.set_xlabel("training steps (millions)")
    ax1.set_ylabel("clean-env return (IQM over 5 seeds)")
    ax1.set_title("A. Learning curves — return on a clean env", fontsize=11, loc="left")
    ax1.grid(True, alpha=0.25)
    ax1.legend(frameon=False, fontsize=9, loc="lower right")
    ax1.spines[["top", "right"]].set_visible(False)

    # Panel B: final % of clean with 95% bootstrap CI
    labels = [x[0] for x in present]
    colors = [x[2] for x in present]
    pcts, los, his = [], [], []
    for label, d, c in present:
        cf = finals[d][0]
        pcts.append(100 * iqm(cf) / iqm(base))
        if d == "clean":
            los.append(0)
            his.append(0)
        else:
            lo, hi = boot_ratio(cf, base)
            los.append(pcts[-1] - lo)
            his.append(hi - pcts[-1])
    y = np.arange(len(present))[::-1]
    ax2.barh(y, pcts, color=colors, height=0.6, zorder=2)
    ax2.errorbar(pcts, y, xerr=[los, his], fmt="none", ecolor="#222222", elinewidth=1.3, capsize=4, zorder=3)
    ax2.axvline(100, color="#888888", ls="--", lw=1, zorder=1)
    for yi, p in zip(y, pcts):
        ax2.text(min(p + 3, max(pcts) + 18), yi, f"{p:.0f}%", va="center", fontsize=9)
    ax2.set_yticks(y)
    ax2.set_yticklabels(labels, fontsize=9)
    ax2.set_xlabel("final return, % of clean baseline (IQM, 95% CI)")
    ax2.set_title("B. Damage to learning (higher = less damage)", fontsize=11, loc="left")
    ax2.set_xlim(0, max(his[i] + pcts[i] for i in range(len(pcts))) + 22)
    ax2.grid(True, axis="x", alpha=0.25)
    ax2.spines[["top", "right"]].set_visible(False)

    fig.suptitle(f"StochRL — dm_control {TITLES.get(env, env)}, SAC 1M steps, 5 seeds  "
                 f"(scaled noise: obs vs transition)", fontsize=12, y=0.99)
    fig.text(0.5, 0.005, CAVEATS.get(env, "5 seeds."), ha="center", fontsize=8, style="italic", color="#666666")
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    fig.savefig(f"assets/{env}_phase1.png", dpi=140)
    print(f"saved assets/{env}_phase1.png  pcts:", [f"{lb}:{p:.0f}%" for (lb, _, _), p in zip(present, pcts)])


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "cheetah")
