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
    """Jolt the velocity DOFs before each physics step, making the dynamics
    stochastic. The env's own step then produces the observation and reward, so
    the agent always sees the true resulting state (this is process noise, not
    observation noise). Works on Gymnasium MuJoCo and dm_control (shimmy).
    """

    def __init__(self, env, model: NoiseModel, seed: int = 0):
        super().__init__(env)
        base = env.unwrapped
        self._gym_mujoco = hasattr(base, "set_state") and hasattr(base, "data")
        self._dmc = hasattr(base, "physics")
        if not (self._gym_mujoco or self._dmc):
            raise TypeError("TransitionNoise needs a Gymnasium MuJoCo or dm_control env")
        self.model = model
        self._rng = np.random.default_rng(seed)

    def reset(self, **kwargs):
        self.model.reset(self._rng)
        return super().reset(**kwargs)

    def step(self, action):
        base = self.env.unwrapped
        if self._gym_mujoco:
            qvel = self.model(base.data.qvel, self._rng)
            base.set_state(base.data.qpos.copy(), qvel)
        else:
            base.physics.data.qvel[:] = self.model(base.physics.data.qvel, self._rng)
            base.physics.forward()
        return self.env.step(action)
