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


class TransitionNoise(gym.Wrapper):
    """Perturb qvel after each step, so the dynamics themselves are stochastic.

    The model runs over the nv velocity DOFs. Gymnasium MuJoCo envs only.
    Termination is read from the pre-jolt state, which is consistent only when
    the env's health check does not depend on velocity (true for the study
    envs HalfCheetah/Ant/Walker2d; Hopper/Humanoid would need it re-evaluated).
    """

    def __init__(self, env, model: NoiseModel, seed: int = 0):
        super().__init__(env)
        base = env.unwrapped
        if not (hasattr(base, "set_state") and hasattr(base, "data") and hasattr(base, "_get_obs")):
            raise TypeError("TransitionNoise needs a Gymnasium MuJoCo env (set_state/data/_get_obs)")
        self.model = model
        self._rng = np.random.default_rng(seed)

    def reset(self, **kwargs):
        self.model.reset(self._rng)
        return super().reset(**kwargs)

    def step(self, action):
        _, reward, terminated, truncated, info = self.env.step(action)
        base = self.env.unwrapped
        qvel = self.model(base.data.qvel, self._rng)
        base.set_state(base.data.qpos.copy(), qvel)
        return base._get_obs(), reward, terminated, truncated, info
