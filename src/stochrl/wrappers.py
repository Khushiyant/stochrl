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
    """Process noise: after each step, perturb the full MuJoCo state (qpos+qvel;
    dm_control get_state) and re-derive the observation, so the agent sees the
    noised next state and the dynamics carry it forward. Reward and termination
    come from the true step. Works on Gymnasium MuJoCo and dm_control (shimmy).
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
        obs, r, term, trunc, info = self.env.step(action)
        if term or trunc:  # don't set_state into a terminated/reset-pending env
            return obs, r, term, trunc, info
        base = self.env.unwrapped
        if self._gym_mujoco:
            nq = base.model.nq
            state = self.model(np.concatenate([base.data.qpos, base.data.qvel]), self._rng)
            base.set_state(state[:nq], state[nq:])
            obs = base._get_obs()
        else:
            from shimmy.utils.dm_env import dm_obs2gym_obs
            base.physics.set_state(self.model(base.physics.get_state(), self._rng))
            base.physics.forward()
            dict_obs = dm_obs2gym_obs(base.task.get_observation(base.physics))
            obs = gym.spaces.flatten(base.observation_space, dict_obs)
        return obs, r, term, trunc, info
