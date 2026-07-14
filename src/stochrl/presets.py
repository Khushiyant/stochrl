"""Ready-made noise specifications, from the uniform baseline to structured
sensor models. Each function returns a NoiseModel ready to hand to a wrapper.
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
    """One Gaussian on every channel.

    calibrated=False uses the same absolute sigma=rho everywhere, as in prior
    work. calibrated=True scales sigma per channel by its signal std.
    """
    dim = len(stats.std)
    scale = stats.scale if calibrated else np.ones(dim)
    specs = [N.ChannelNoise(_all(dim), N.Gaussian(relative=rho))]
    return N.NoiseModel(dim, scale, specs, record=record)


SCALE_STATS = {
    "mean": np.mean,
    "median": np.median,
    "p25": lambda x: np.percentile(x, 25),
    "p75": lambda x: np.percentile(x, 75),
    "min": np.min,
    "max": np.max,
}


def fixed_gaussian(stats: SignalStats, rho: float = 0.1, stat: str = "median",
                   record: bool = False) -> N.NoiseModel:
    """The same absolute sigma on every channel: rho * stat(scales), where
    `stat` is a summary of the per-channel scale spread (mean, median, p25,
    p75, ...). Shows how the fixed-vs-calibrated comparison depends on which
    point of the spread the fixed value is pinned to.
    """
    dim = len(stats.std)
    value = SCALE_STATS[stat](stats.scale)
    scale = np.full(dim, value)
    specs = [N.ChannelNoise(_all(dim), N.Gaussian(relative=rho))]
    return N.NoiseModel(dim, scale, specs, record=record)


def realistic_sensors(stats: SignalStats, rho: float = 0.05, n_pos: int | None = None,
                      record: bool = False) -> N.NoiseModel:
    """Heterogeneous sensor model for MuJoCo-style observations.

    MuJoCo observations are roughly [positions/angles | velocities], and the
    two groups get physically different noise: positions get white Gaussian
    noise plus encoder quantization, velocities get Ornstein-Uhlenbeck drift
    plus a small dropout, with strength growing with speed via a
    state-dependent gain. `n_pos` sets the split; defaults to the first half
    of the vector.
    """
    dim = len(stats.std)
    n_pos = dim // 2 if n_pos is None else n_pos
    pos_idx, vel_idx = _all(n_pos), list(range(n_pos, dim))
    scale = stats.scale

    # velocity noise scales with |velocity| relative to the channel's own std
    vel_scale = scale[vel_idx]

    def speed_gain(ref):
        v = np.abs(ref[vel_idx]) / vel_scale
        return 0.5 + v  # 0.5x at rest, growing with speed

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
    """gain(state) = 0.5 + |x - center| / signal_scale per channel, so noise
    concentrates where a channel deviates most from its typical value: fast
    motion for velocities (center ~ 0), large deflections for positions.
    """
    idx, scale, center = list(idx), np.asarray(scale), np.asarray(center)

    def gain_fn(ref):
        return 0.5 + np.abs(np.asarray(ref)[idx] - center) / scale
    return gain_fn


def _matched_constant_gain(env, gain_fn, n, steps=10_000, seed=0):
    """Per-channel RMS of a state-dependent gain over a random rollout.

    Injected variance per step is (rho*scale*gain)^2, so a constant gain
    c = sqrt(E[gain^2]) reproduces the state-dependent gain's time-averaged
    variance exactly. Flat and state-dependent noise then differ only in
    where the noise lands, not in how much.
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
    """Calibrated Gaussian noise on one channel group, at matched noise energy.

    mode is 'vel-flat', 'vel-statedep', 'pos-flat' or 'pos-statedep': noise
    goes on the velocity or position channels, either spread evenly over time
    ('flat') or concentrated where the channel deviates most from its typical
    value ('statedep': fast motion for velocities, large deflections for
    positions). Both variants inject the same average variance (matched via
    `_matched_constant_gain`), so any learning gap comes from where the noise
    lands, not from how much.

    Position noise skips the first `n_root_pos` channels (root height/pitch):
    under a random policy those are non-stationary, so their calibration mean
    doesn't transfer and would break the matched-energy control. For planar
    MuJoCo (HalfCheetah/Hopper/Walker) the root pose is the first 2 channels.
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
        gain_fn = lambda ref: c  # constant, matched to the state-dependent gain's RMS

    specs = [N.ChannelNoise(idx, N.Gaussian(relative=rho), gain_fn=gain_fn)]
    return N.NoiseModel(dim, stats.scale, specs, record=record)


def _const_gain(c):
    """Constant gain, with c bound at definition time (not at call time)."""
    return lambda ref: c


def combined_isolation(stats: SignalStats, rho: float, env_id: str, vel_kind: str, pos_kind: str,
                       calib_steps: int = 10_000, calib_seed: int = 0, n_pos: int | None = None,
                       n_root_pos: int = 2, record: bool = False) -> N.NoiseModel:
    """Noise on both the velocity and joint-position channels at once, each
    independently 'flat' or 'statedep', at matched per-group energy. Only the
    timing per group changes, not the amount, which tests whether
    state-dependence on the two groups interacts.
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
    """Action-side noise: multiplicative gain error plus small additive jitter.

    Actions are already normalised to ~[-1, 1], so scale=1 is the natural unit.
    """
    scale = np.ones(action_dim)
    specs = [N.ChannelNoise(_all(action_dim), N.Compose([
        N.MultiplicativeGaussian(relative=rho),
        N.Gaussian(relative=0.3 * rho),
    ]))]
    return N.NoiseModel(action_dim, scale, specs, record=record)
