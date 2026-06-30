"""Ready-made noise specifications, from baseline to realistic.

These are the concrete answers to "which state -> which pattern -> what degree".
Start from `uniform_gaussian` (reproduces prior work) and move toward
`realistic_sensors` (heterogeneous, calibrated, partly state-dependent), then
invent your own. Each returns a `NoiseModel` ready to hand to a wrapper.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from . import noise as N
from .calibrate import SignalStats


def _all(dim):
    return list(range(dim))


def uniform_gaussian(stats: SignalStats, rho: float = 0.05, calibrated: bool = True,
                     record: bool = False) -> N.NoiseModel:
    """The prior-work baseline: one Gaussian on every channel.

    calibrated=False reproduces the paper exactly (same absolute sigma=rho on
    every channel). calibrated=True scales sigma per channel by its signal std,
    which is the minimal fix the paper itself asks for.
    """
    dim = len(stats.std)
    scale = stats.scale if calibrated else np.ones(dim)
    specs = [N.ChannelNoise(_all(dim), N.Gaussian(relative=rho))]
    return N.NoiseModel(dim, scale, specs, record=record)


def realistic_sensors(stats: SignalStats, rho: float = 0.05, n_pos: int | None = None,
                      record: bool = False) -> N.NoiseModel:
    """Heterogeneous, signal-calibrated sensor model for MuJoCo-style obs.

    MuJoCo observations are roughly [positions/angles | velocities]. We give them
    physically different noise patterns:

      * positions  -> white Gaussian + encoder quantization (clean-ish, discretized)
      * velocities -> Ornstein-Uhlenbeck drift (correlated) + a small dropout,
                      and the degree GROWS WITH SPEED (state-dependent gain) --
                      the concrete "which state should be noisier" claim.

    Pass `n_pos` to set the split; defaults to the first half of the obs vector.
    """
    dim = len(stats.std)
    n_pos = dim // 2 if n_pos is None else n_pos
    pos_idx, vel_idx = _all(n_pos), list(range(n_pos, dim))
    scale = stats.scale

    # Velocity channels: noise scales with |velocity| relative to its own std.
    vel_scale = scale[vel_idx]

    def speed_gain(ref):
        v = np.abs(ref[vel_idx]) / vel_scale
        return 0.5 + v  # at rest -> 0.5x, fast -> grows with speed

    specs = [
        N.ChannelNoise(pos_idx, N.Compose([
            N.Gaussian(relative=0.5 * rho),
            N.Quantization(levels=128),
        ])),
        N.ChannelNoise(vel_idx, N.Compose([
            N.OrnsteinUhlenbeck(relative=rho, theta=0.1),
            N.Dropout(prob=0.01, mode="hold"),
        ]), gain_fn=speed_gain),
    ]
    return N.NoiseModel(dim, scale, specs, record=record)


def _speed_gain_fn(vel_idx, vel_scale):
    """gain(state) = 0.5 + |velocity| / signal_scale, per velocity channel."""
    def gain_fn(ref):
        return 0.5 + np.abs(np.asarray(ref)[vel_idx]) / vel_scale
    return gain_fn


def measure_matched_gain(env, vel_idx, vel_scale, steps=10_000, seed=0):
    """RMS of the state-dependent speed gain over a random rollout, per channel.

    Used to build a state-INDEPENDENT noise with the SAME average injected variance:
    injected var per step ~ (rho*scale*gain)^2, so a constant gain c = sqrt(E[gain^2])
    matches the time-averaged variance of the state-dependent version exactly.
    """
    obs, _ = env.reset(seed=seed)
    env.action_space.seed(seed)
    gain_fn = _speed_gain_fn(vel_idx, vel_scale)
    acc, n = np.zeros(len(vel_idx)), 0
    for _ in range(steps):
        obs, _, term, trunc, _ = env.step(env.action_space.sample())
        acc += gain_fn(obs) ** 2
        n += 1
        if term or trunc:
            obs, _ = env.reset()
    return np.sqrt(acc / n)


def velocity_isolation(stats: SignalStats, rho: float, env_id: str, mode: str,
                       calib_steps: int = 10_000, calib_seed: int = 0,
                       record: bool = False) -> N.NoiseModel:
    """Isolate the EFFECT of state-dependence: identical Gaussian noise on the
    velocity channels, either spread evenly (vel-flat) or concentrated at high
    speed (vel-statedep), at MATCHED average injected variance.

    The only difference between the two is *where* the noise lands, so any learning
    difference is attributable to state-dependence, not to total noise amount.
    """
    dim = len(stats.std)
    n_pos = dim // 2
    vel_idx = list(range(n_pos, dim))
    vel_scale = stats.scale[vel_idx]

    if mode == "vel-statedep":
        gain_fn = _speed_gain_fn(vel_idx, vel_scale)
    elif mode == "vel-flat":
        c = measure_matched_gain(gym.make(env_id), vel_idx, vel_scale, calib_steps, calib_seed)
        gain_fn = lambda ref: c  # constant, matched to RMS of the state-dependent gain
    else:
        raise ValueError(f"unknown velocity-isolation mode: {mode}")

    specs = [N.ChannelNoise(vel_idx, N.Gaussian(relative=rho), gain_fn=gain_fn)]
    return N.NoiseModel(dim, stats.scale, specs, record=record)


def actuator_noise(action_dim: int, rho: float = 0.1, record: bool = False) -> N.NoiseModel:
    """Action-side noise: gain error (multiplicative) + small additive jitter.

    Actions are already normalised to ~[-1, 1], so scale=1 is the natural unit.
    """
    scale = np.ones(action_dim)
    specs = [N.ChannelNoise(_all(action_dim), N.Compose([
        N.MultiplicativeGaussian(relative=rho),
        N.Gaussian(relative=0.3 * rho),
    ]))]
    return N.NoiseModel(action_dim, scale, specs, record=record)
