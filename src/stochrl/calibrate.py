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
        """Per-channel std, floored away from zero."""
        s = self.std.copy()
        floor = np.median(s[s > 0]) * 1e-3 if np.any(s > 0) else 1.0
        return np.maximum(s, floor)


def _rollout_stats(env, steps, seed, read):
    obs, _ = env.reset(seed=seed)
    env.action_space.seed(seed)
    buf = [read(obs)]
    for _ in range(steps):
        obs, _, term, trunc, _ = env.step(env.action_space.sample())
        buf.append(read(obs))
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


def collect_signal_stats(env: gym.Env, steps: int = 20_000, seed: int = 0) -> SignalStats:
    """Per-channel observation stats from a random-policy rollout."""
    return _rollout_stats(env, steps, seed, lambda obs: np.asarray(obs, dtype=np.float64))


def _state_reader(env):
    """Reader for the full MuJoCo state: qpos+qvel (gym) or physics.get_state (dm_control)."""
    base = env.unwrapped
    if hasattr(base, "data") and hasattr(base, "set_state"):
        return lambda _obs: np.concatenate([base.data.qpos, base.data.qvel])
    if hasattr(base, "physics"):
        return lambda _obs: base.physics.get_state().copy()
    raise TypeError("transition noise needs a Gymnasium MuJoCo or dm_control env")


def collect_state_stats(env: gym.Env, steps: int = 20_000, seed: int = 0) -> SignalStats:
    """Per-channel state (qpos+qvel) stats from a random-policy rollout (for transition noise)."""
    return _rollout_stats(env, steps, seed, _state_reader(env))
