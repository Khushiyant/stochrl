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


def _deflection_gain(idx, scale, center):
    """gain(state) = 0.5 + |x - center| / signal_scale, per channel.

    Concentrates noise where a channel deviates most from its typical value:
    fast motion for velocities (center ~ 0), large deflections for positions.
    """
    idx, scale, center = list(idx), np.asarray(scale), np.asarray(center)

    def gain_fn(ref):
        return 0.5 + np.abs(np.asarray(ref)[idx] - center) / scale
    return gain_fn


def _matched_constant_gain(env, gain_fn, n, steps=10_000, seed=0):
    """RMS of a state-dependent gain over a random rollout, per channel.

    Lets a state-INDEPENDENT ('flat') noise match the state-dependent one's average
    injected variance: injected var per step ~ (rho*scale*gain)^2, so a constant
    gain c = sqrt(E[gain^2]) reproduces the time-averaged variance exactly. The two
    then differ only in *where* the noise lands, not in how much.
    """
    obs, _ = env.reset(seed=seed)
    env.action_space.seed(seed)
    acc, count = np.zeros(n), 0
    for _ in range(steps):
        obs, _, term, trunc, _ = env.step(env.action_space.sample())
        acc += gain_fn(obs) ** 2
        count += 1
        if term or trunc:
            obs, _ = env.reset()
    return np.sqrt(acc / count)


def channel_isolation(stats: SignalStats, rho: float, env_id: str, mode: str,
                      calib_steps: int = 10_000, calib_seed: int = 0, n_pos: int | None = None,
                      n_root_pos: int = 2, record: bool = False) -> N.NoiseModel:
    """State-dependence isolation on one channel group, at matched noise energy.

    Applies identical calibrated Gaussian noise to either the position or velocity
    channels, placed either evenly ('*-flat') or concentrated where the channel
    deviates most from its typical value ('*-statedep'). Flat and statedep inject the
    same average variance (matched via `_matched_constant_gain`), so any learning gap
    is attributable to *where* the noise lands, not to how much.

    Position noise targets the joint-angle sensors only, skipping the first
    `n_root_pos` channels (root height/pitch): under a random policy those are
    non-stationary, so their calibration mean doesn't transfer and would break the
    matched-energy control. (For planar MuJoCo — HalfCheetah/Hopper/Walker — the root
    pose is the first 2 observation channels.)

    mode: 'vel-flat' | 'vel-statedep' | 'pos-flat' | 'pos-statedep'
      velocity rule: noisier when moving fast; position rule: noisier at large deflection.
    """
    dim = len(stats.std)
    n_pos = dim // 2 if n_pos is None else n_pos
    group, kind = mode.split("-")
    idx = list(range(n_root_pos, n_pos)) if group == "pos" else list(range(n_pos, dim))
    if not idx or kind not in ("flat", "statedep") or group not in ("pos", "vel"):
        raise ValueError(f"unknown isolation mode: {mode}")

    gain_fn = _deflection_gain(idx, stats.scale[idx], stats.mean[idx])
    if kind == "flat":
        c = _matched_constant_gain(gym.make(env_id), gain_fn, len(idx), calib_steps, calib_seed)
        gain_fn = lambda ref: c  # constant, matched to RMS of the state-dependent gain

    specs = [N.ChannelNoise(idx, N.Gaussian(relative=rho), gain_fn=gain_fn)]
    return N.NoiseModel(dim, stats.scale, specs, record=record)


def _const_gain(c):
    """Constant (state-independent) gain, with c bound at definition time."""
    return lambda ref: c


def combined_isolation(stats: SignalStats, rho: float, env_id: str, vel_kind: str, pos_kind: str,
                       calib_steps: int = 10_000, calib_seed: int = 0, n_pos: int | None = None,
                       n_root_pos: int = 2, record: bool = False) -> N.NoiseModel:
    """Noise on BOTH velocity and (joint) position channels at once, each independently
    'flat' or 'statedep', at matched per-group energy. Tests whether state-dependence on
    the two groups interacts — only the timing per group changes, not the amount.
    """
    dim = len(stats.std)
    n_pos = dim // 2 if n_pos is None else n_pos
    groups = [("vel", list(range(n_pos, dim)), vel_kind),
              ("pos", list(range(n_root_pos, n_pos)), pos_kind)]
    specs = []
    for _, idx, kind in groups:
        gain_fn = _deflection_gain(idx, stats.scale[idx], stats.mean[idx])
        if kind == "flat":
            gain_fn = _const_gain(_matched_constant_gain(gym.make(env_id), gain_fn, len(idx),
                                                         calib_steps, calib_seed))
        elif kind != "statedep":
            raise ValueError(f"kind must be flat|statedep, got {kind}")
        specs.append(N.ChannelNoise(idx, N.Gaussian(relative=rho), gain_fn=gain_fn))
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
