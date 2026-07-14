"""Gymnasium wrappers that inject observation or action noise via a NoiseModel.

Noise lives in a wrapper around an unmodified environment, so the same spec
applies to any continuous-control task without touching the physics.
Observation noise (which makes the task a POMDP) and action noise (which the
dynamics absorb into a transformed MDP) are separate wrappers so they can be
studied independently.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from .noise import NoiseModel


class ObservationNoise(gym.ObservationWrapper):
    """Apply a NoiseModel to observations returned to the agent."""

    def __init__(self, env, model: NoiseModel, seed: int = 0):
        super().__init__(env)
        self.model = model
        self._rng = np.random.default_rng(seed)

    def reset(self, **kwargs):
        self.model.reset(self._rng)
        return super().reset(**kwargs)

    def observation(self, obs):
        return self.model(obs, self._rng)


class ActionNoise(gym.ActionWrapper):
    """Apply a NoiseModel to actions before they reach the environment."""

    def __init__(self, env, model: NoiseModel, seed: int = 0):
        super().__init__(env)
        self.model = model
        self._rng = np.random.default_rng(seed)

    def reset(self, **kwargs):
        self.model.reset(self._rng)
        return super().reset(**kwargs)

    def action(self, action):
        return self.model(action, self._rng)
