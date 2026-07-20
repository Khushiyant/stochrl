from __future__ import annotations

import numpy as np

from . import noise as N
from .calibrate import SignalStats
from .envs import make_flat


def _all(dim):
    return list(range(dim))


# PB: If I got it correctly this is noise calibrated per channel
def uniform_gaussian(stats: SignalStats, rho: float = 0.05, calibrated: bool = True, record: bool = False) -> N.NoiseModel:
    """Gaussian on every channel: absolute sigma=rho if not calibrated, rho*std per channel if so."""
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


def fixed_gaussian(stats: SignalStats, rho: float = 0.1, stat: str = "median", record: bool = False) -> N.NoiseModel:
    """One absolute sigma = rho * stat(scales) on every channel."""
    dim = len(stats.std)
    value = SCALE_STATS[stat](stats.scale)
    scale = np.full(dim, value)
    specs = [N.ChannelNoise(_all(dim), N.Gaussian(relative=rho))]
    return N.NoiseModel(dim, scale, specs, record=record)


def realistic_sensors(stats: SignalStats, rho: float = 0.05, n_pos: int | None = None, record: bool = False) -> N.NoiseModel:
    """Positions: Gaussian + quantization. Velocities: OU drift + dropout, noisier at speed."""
    dim = len(stats.std)
    n_pos = dim // 2 if n_pos is None else n_pos
    pos_idx, vel_idx = _all(n_pos), list(range(n_pos, dim))
    scale = stats.scale

    vel_scale = scale[vel_idx]

    def speed_gain(ref):
        v = np.abs(ref[vel_idx]) / vel_scale
        return 0.5 + v

    specs = [
        N.ChannelNoise(
            pos_idx,
            N.Compose(
                [
                    N.Gaussian(relative=0.5 * rho),
                    N.Quantization(levels=128),
                ]
            ),
        ),
        N.ChannelNoise(
            vel_idx,
            N.Compose(
                [
                    N.OrnsteinUhlenbeck(relative=rho, theta=0.1),
                    N.Dropout(prob=0.01, mode="hold"),
                ]
            ),
            gain_fn=speed_gain,
        ),
    ]
    return N.NoiseModel(dim, scale, specs, record=record)


def _deflection_gain(idx, scale, center):
    """gain(state) = 0.5 + |x - center| / scale: noise concentrates away from typical values."""
    idx, scale, center = list(idx), np.asarray(scale), np.asarray(center)

    def gain_fn(ref):
        return 0.5 + np.abs(np.asarray(ref)[idx] - center) / scale

    return gain_fn


def _matched_constant_gain(env, gain_fn, n, steps=10_000, seed=0):
    """RMS of the gain over a random rollout; as a constant gain it injects the
    same average variance as the state-dependent one."""
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


def channel_isolation(
    stats: SignalStats,
    rho: float,
    env_id: str,
    mode: str,
    calib_steps: int = 10_000,
    calib_seed: int = 0,
    n_pos: int | None = None,
    n_root_pos: int = 2,
    record: bool = False,
) -> N.NoiseModel:
    """Gaussian noise on the velocity or position group, spread 'flat' over time or
    'statedep' (concentrated away from typical values), at matched average variance.
    mode: {vel,pos}-{flat,statedep}."""
    dim = len(stats.std)
    n_pos = dim // 2 if n_pos is None else n_pos
    group, kind = mode.split("-")
    # position group skips the root pose channels: non-stationary under a random policy
    idx = list(range(n_root_pos, n_pos)) if group == "pos" else list(range(n_pos, dim))
    if not idx or kind not in ("flat", "statedep") or group not in ("pos", "vel"):
        raise ValueError(f"unknown isolation mode: {mode}")

    gain_fn = _deflection_gain(idx, stats.scale[idx], stats.mean[idx])
    if kind == "flat":
        c = _matched_constant_gain(make_flat(env_id), gain_fn, len(idx), calib_steps, calib_seed)
        gain_fn = lambda ref: c  # constant, matched to the state-dependent gain's RMS

    specs = [N.ChannelNoise(idx, N.Gaussian(relative=rho), gain_fn=gain_fn)]
    return N.NoiseModel(dim, stats.scale, specs, record=record)


def _const_gain(c):
    # binds c at definition time, not call time
    return lambda ref: c


def combined_isolation(
    stats: SignalStats,
    rho: float,
    env_id: str,
    vel_kind: str,
    pos_kind: str,
    calib_steps: int = 10_000,
    calib_seed: int = 0,
    n_pos: int | None = None,
    n_root_pos: int = 2,
    record: bool = False,
) -> N.NoiseModel:
    """Noise on both groups at once, each 'flat' or 'statedep', at matched per-group variance."""
    dim = len(stats.std)
    n_pos = dim // 2 if n_pos is None else n_pos
    groups = [("vel", list(range(n_pos, dim)), vel_kind), ("pos", list(range(n_root_pos, n_pos)), pos_kind)]
    specs = []
    for _, idx, kind in groups:
        gain_fn = _deflection_gain(idx, stats.scale[idx], stats.mean[idx])
        if kind == "flat":
            gain_fn = _const_gain(_matched_constant_gain(make_flat(env_id), gain_fn, len(idx), calib_steps, calib_seed))
        elif kind != "statedep":
            raise ValueError(f"kind must be flat|statedep, got {kind}")
        specs.append(N.ChannelNoise(idx, N.Gaussian(relative=rho), gain_fn=gain_fn))
    return N.NoiseModel(dim, stats.scale, specs, record=record)


def actuator_noise(action_dim: int, rho: float = 0.1, record: bool = False) -> N.NoiseModel:
    """Multiplicative gain error plus small additive jitter; actions are ~[-1, 1] so scale=1."""
    scale = np.ones(action_dim)
    specs = [
        N.ChannelNoise(
            _all(action_dim),
            N.Compose(
                [
                    N.MultiplicativeGaussian(relative=rho),
                    N.Gaussian(relative=0.3 * rho),
                ]
            ),
        )
    ]
    return N.NoiseModel(action_dim, scale, specs, record=record)
