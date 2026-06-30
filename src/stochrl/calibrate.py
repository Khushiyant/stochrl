"""Estimate per-channel signal magnitude so noise levels are comparable.

The whole point: a noise level of "0.05" is meaningless on its own, because a
joint-angle channel (range ~1 rad) and a velocity channel (range ~20 rad/s) live
on totally different scales. Adding sigma=0.05 to both perturbs the first one
massively and the second one not at all. That is exactly the "indiscriminate of
the observation dimension" flaw the paper calls out.

Fix: roll out a reference policy (random by default), measure each channel's own
spread (std, robust MAD, and range), and express noise *relative* to that. Then
one knob `rho` means "perturb every channel by rho of its natural variation",
which is comparable across channels and across environments.
"""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np


@dataclass
class SignalStats:
    mean: np.ndarray
    std: np.ndarray
    mad: np.ndarray  # median absolute deviation (robust scale)
    span: np.ndarray  # 1st..99th percentile range
    n: int

    @property
    def scale(self) -> np.ndarray:
        """Default scale used for calibration: std, floored away from zero."""
        s = self.std.copy()
        floor = np.median(s[s > 0]) * 1e-3 if np.any(s > 0) else 1.0
        return np.maximum(s, floor)


def collect_signal_stats(env: gym.Env, steps: int = 20_000, seed: int = 0) -> SignalStats:
    """Run a random policy and summarise the per-channel observation distribution."""
    obs, _ = env.reset(seed=seed)
    env.action_space.seed(seed)
    buf = [np.asarray(obs, dtype=np.float64)]
    for _ in range(steps):
        obs, _, term, trunc, _ = env.step(env.action_space.sample())
        buf.append(np.asarray(obs, dtype=np.float64))
        if term or trunc:
            obs, _ = env.reset()
    data = np.stack(buf)
    return SignalStats(
        mean=data.mean(0),
        std=data.std(0),
        mad=np.median(np.abs(data - np.median(data, 0)), 0),
        span=np.percentile(data, 99, 0) - np.percentile(data, 1, 0),
        n=len(data),
    )
