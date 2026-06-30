"""Explore and visualise noise patterns on a continuous-control env (conference B&W).

Answers, with pictures:
  1. Why is a single uniform sigma wrong?  -> channels differ in scale by ~40x.
  2. What does calibrated per-channel noise look like vs the baseline?
  3. What does *state-dependent* noise look like? -> velocity-channel noise that
     grows with speed.

  uv run python scripts/explore_noise.py --env HalfCheetah-v5 --outdir figures
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import tyro

from stochrl import collect_signal_stats, presets
from stochrl.plotting import finish, set_style


@dataclass
class Args:
    env: str = "HalfCheetah-v5"
    rho: float = 0.15
    """relative noise level (fraction of each channel's signal std)."""
    calib_steps: int = 10_000
    rollout_steps: int = 400
    seed: int = 0
    outdir: str = "figures"


def rollout_clean(env, steps, seed):
    obs, _ = env.reset(seed=seed)
    env.action_space.seed(seed)
    seq = [np.asarray(obs, np.float64)]
    for _ in range(steps):
        obs, _, term, trunc, _ = env.step(env.action_space.sample())
        seq.append(np.asarray(obs, np.float64))
        if term or trunc:
            obs, _ = env.reset()
    return np.stack(seq)


def apply_model(model, clean_seq, seed):
    rng = np.random.default_rng(seed)
    model.reset(rng)
    return np.stack([model(o, rng) for o in clean_seq])


def main(args: Args):
    set_style()
    os.makedirs(args.outdir, exist_ok=True)
    env = gym.make(args.env)
    dim = env.observation_space.shape[0]
    print(f"env={args.env}  obs_dim={dim}  calibrating ({args.calib_steps} steps)...")
    stats = collect_signal_stats(env, steps=args.calib_steps, seed=args.seed)

    # ---- Figure 1: per-channel signal scale (the case for calibration) ------ #
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    ax.bar(range(dim), stats.scale, color="0.6", edgecolor="black", lw=0.6)
    ax.axhline(stats.scale.mean(), color="black", ls="--", lw=1.2,
               label="mean (what one uniform $\\sigma$ assumes)")
    ax.set(xlabel="observation channel", ylabel="signal std")
    ax.set_title(f"{args.env}: per-channel signal magnitude "
                 f"(max/min $=$ {stats.scale.max() / stats.scale.min():.0f}$\\times$)")
    ax.legend(loc="upper left")
    finish(fig, ax)
    fig.savefig(f"{args.outdir}/01_signal_scale.png")
    print(f"  channel scales span {stats.scale.min():.3g} .. {stats.scale.max():.3g}")

    clean = rollout_clean(env, args.rollout_steps, args.seed + 1)
    base = apply_model(presets.uniform_gaussian(stats, rho=args.rho, calibrated=False), clean, args.seed)
    real = apply_model(presets.realistic_sensors(stats, rho=args.rho), clean, args.seed)

    n_pos = dim // 2
    pos_ch = int(np.argmax(stats.scale[:n_pos]))
    vel_ch = n_pos + int(np.argmax(stats.scale[n_pos:]))

    # ---- Figure 2: clean vs noisy, position vs velocity channel ------------- #
    fig, axes = plt.subplots(2, 1, figsize=(6.6, 4.4), sharex=True)
    for ax, ch, name in [(axes[0], pos_ch, "position"), (axes[1], vel_ch, "velocity")]:
        t = np.arange(len(clean))
        ax.plot(t, clean[:, ch], color="black", lw=2.0, label="clean")
        ax.plot(t, base[:, ch], color="0.6", ls="--", lw=1.2, label="uniform (uncalibrated)")
        ax.plot(t, real[:, ch], color="0.0", ls=":", lw=1.4, label="realistic (calibrated)")
        ax.set_ylabel(f"ch {ch}\n({name})")
    axes[0].set_title("Clean vs noisy: correlated drift + state-dependent spread on velocity")
    axes[0].legend(loc="upper right", ncol=3)
    axes[1].set_xlabel("step")
    finish(fig, axes[0])
    finish(fig, axes[1])
    fig.savefig(f"{args.outdir}/02_trajectories.png")

    # ---- Figure 3: state-dependent degree (the headline claim) -------------- #
    speed = np.abs(clean[:, vel_ch])
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    ax.scatter(speed, np.abs(base[:, vel_ch] - clean[:, vel_ch]), s=18,
               facecolors="none", edgecolors="0.55", linewidths=0.8,
               label="uniform: noise independent of state")
    ax.scatter(speed, np.abs(real[:, vel_ch] - clean[:, vel_ch]), s=16,
               color="black", marker="x", linewidths=0.9,
               label="realistic: noise grows with $|$velocity$|$")
    ax.set(xlabel=f"$|$velocity$|$ of channel {vel_ch}", ylabel="applied noise magnitude")
    ax.set_title("Which state should be noisier? State-dependent degree")
    ax.legend(loc="upper left")
    finish(fig, ax)
    fig.savefig(f"{args.outdir}/03_state_dependent.png")

    env.close()
    print(f"Saved 3 figures to ./{args.outdir}/")


if __name__ == "__main__":
    main(tyro.cli(Args))
