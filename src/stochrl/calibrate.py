"""Estimate per-channel signal magnitude for noise calibration.

Observation channels differ in scale by orders of magnitude (a joint angle
spans ~1 rad, a joint velocity ~20 rad/s), so one absolute sigma perturbs
them very unevenly. Rolling out a random policy and measuring each channel's
spread lets noise be expressed relative to that spread: a single level rho
then perturbs every channel by the same fraction of its natural variation.
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
